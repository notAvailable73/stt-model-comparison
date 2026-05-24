"""Download OpenSLR-104, parse transcripts, stratified-sample 50 clips, build manifest.

Plan §2 + §6 Cell 3. The stratified sampler uses adjacency spillover per
DECISIONS.md §A and logs the final per-stratum counts so the actual
distribution is visible in the manifest summary.
"""

from __future__ import annotations

import itertools
import logging
import random
import tarfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

OPENSLR_URL = "https://openslr.trmal.net/resources/104/Bengali-English_test.tar.gz"

LENGTH_BUCKETS = ("short", "medium", "long")
CS_BUCKETS = ("low", "medium", "high")

MANIFEST_COLUMNS = [
    "clip_id", "audio_path", "reference_transcript",
    "length_sec", "code_switch_density", "length_bucket", "cs_bucket",
]


# --- Download + extract ------------------------------------------------------

def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> Path:
    """Stream a URL to disk with a tqdm progress bar."""
    from tqdm.auto import tqdm

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length") or 0)
        with open(dest, "wb") as f, tqdm(
            total=total or None, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    logger.info("Downloaded %s (%d bytes)", dest, dest.stat().st_size)
    return dest


def extract_tarball(archive: Path, dest_dir: Path) -> Path:
    logger.info("Extracting %s -> %s", archive, dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(dest_dir)
    logger.info("Extraction complete")
    return dest_dir


# --- Transcript + audio discovery -------------------------------------------

TRANSCRIPT_FILENAMES = (
    "transcripts.txt", "transcripts.tsv",
    "transcript.txt", "transcript.tsv",
    "text",
)


def _parse_transcript_file(path: Path) -> dict[str, str]:
    """Parse a transcript file as {utt_id: text}.

    Accepts tab-separated or first-whitespace-separated rows. Each line should
    be `<utt_id> <text>` or `<utt_id>\\t<text>`.
    """
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n").strip()
            if not line:
                continue
            parts = line.split("\t", 1) if "\t" in line else line.split(None, 1)
            if len(parts) != 2:
                continue
            utt_id, text = parts[0].strip(), parts[1].strip()
            if utt_id and text:
                out[utt_id] = text
    return out


def find_transcripts(root: Path) -> dict[str, str]:
    """Scan the dataset root for any known transcript file and merge them."""
    merged: dict[str, str] = {}
    for name in TRANSCRIPT_FILENAMES:
        for path in root.rglob(name):
            parsed = _parse_transcript_file(path)
            logger.info("Parsed %d transcripts from %s", len(parsed), path)
            merged.update(parsed)
    return merged


def find_audio_files(root: Path) -> dict[str, Path]:
    """Map recording_id (= filename stem) -> .wav path."""
    return {p.stem: p for p in root.rglob("*.wav")}


# --- Kaldi-style segments + wav.scp -----------------------------------------
# OpenSLR-104 ships ~40 long lecture recordings plus a `segments` file that
# slices each recording into ~4275 utterances by (start, end) timestamps.
# So utt_id in `text` does NOT match a wav stem — it references a segment of
# a parent recording. We parse segments + wav.scp, then extract per-utterance
# sub-clips for the 50 sampled rows.

def _parse_segments_file(path: Path) -> dict[str, tuple[str, float, float]]:
    """Parse Kaldi `segments`: utt_id -> (recording_id, start_sec, end_sec)."""
    out: dict[str, tuple[str, float, float]] = {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            parts = raw.split()
            if len(parts) != 4:
                continue
            utt_id, rec_id, start, end = parts
            try:
                out[utt_id] = (rec_id, float(start), float(end))
            except ValueError:
                continue
    return out


def find_segments(root: Path) -> dict[str, tuple[str, float, float]]:
    merged: dict[str, tuple[str, float, float]] = {}
    for path in root.rglob("segments"):
        if path.is_file():
            parsed = _parse_segments_file(path)
            logger.info("Parsed %d segments from %s", len(parsed), path)
            merged.update(parsed)
    return merged


def _parse_wav_scp(path: Path) -> dict[str, str]:
    """Parse Kaldi `wav.scp`: recording_id -> wav path or filename token."""
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            parts = raw.strip().split(None, 1)
            if len(parts) == 2:
                out[parts[0]] = parts[1].strip()
    return out


def find_wav_scp(root: Path) -> dict[str, str]:
    merged: dict[str, str] = {}
    for path in root.rglob("wav.scp"):
        if path.is_file():
            merged.update(_parse_wav_scp(path))
    return merged


def extract_segment(src_wav: Path, start_sec: float, end_sec: float, dest: Path) -> None:
    """Slice [start, end] from src_wav and write a 16-bit PCM WAV to dest.

    Idempotent: skips if dest already exists.
    """
    import soundfile as sf

    if dest.exists():
        return
    info = sf.info(str(src_wav))
    sr = info.samplerate
    start_frame = max(0, int(start_sec * sr))
    n_frames = max(1, int((end_sec - start_sec) * sr))
    data, _ = sf.read(str(src_wav), start=start_frame, frames=n_frames, dtype="int16")
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), data, sr, subtype="PCM_16")


# --- Per-clip features -------------------------------------------------------

def audio_length_sec(path: Path) -> float:
    """Read clip duration in seconds without loading the samples."""
    import soundfile as sf

    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def code_switch_density(text: str) -> float:
    """Fraction of tokens that look like English/ASCII words.

    A token counts as English-like if it's pure ASCII and contains at least
    one letter. Pure-punctuation tokens are ignored in both numerator and
    denominator so they don't skew the ratio.
    """
    tokens = [t for t in text.strip().split() if any(c.isalpha() for c in t)]
    if not tokens:
        return 0.0
    english_like = sum(1 for t in tokens if t.isascii())
    return english_like / len(tokens)


def length_bucket(sec: float) -> str:
    if sec <= 10:
        return "short"
    if sec <= 30:
        return "medium"
    return "long"


def cs_bucket(density: float) -> str:
    if density < 0.15:
        return "low"
    if density <= 0.40:
        return "medium"
    return "high"


# --- Stratified sampling -----------------------------------------------------

def _neighbors(stratum: tuple[str, str]) -> list[tuple[str, str]]:
    """Adjacency rule from DECISIONS.md §A: same length + neighboring CS, or
    same CS + neighboring length. No diagonals."""
    lb, cb = stratum
    li, ci = LENGTH_BUCKETS.index(lb), CS_BUCKETS.index(cb)
    out: list[tuple[str, str]] = []
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = li + di, ci + dj
        if 0 <= ni < 3 and 0 <= nj < 3:
            out.append((LENGTH_BUCKETS[ni], CS_BUCKETS[nj]))
    return out


def stratified_sample(
    records: list[dict],
    n: int = 50,
    seed: int = 42,
) -> tuple[list[dict], dict[tuple[str, str], int]]:
    """Sample n records with one-step adjacency spillover.

    Returns (selected, final_per_stratum_counts).
    """
    rng = random.Random(seed)
    all_strata = list(itertools.product(LENGTH_BUCKETS, CS_BUCKETS))

    pools: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        pools[(r["length_bucket"], r["cs_bucket"])].append(r)
    for k in pools:
        rng.shuffle(pools[k])

    base = n // len(all_strata)
    extras = n - base * len(all_strata)
    targets = {k: base for k in all_strata}
    # Distribute the leftover slots to the fullest strata so we don't immediately
    # need spillover for buckets that happen to be popular.
    for k in sorted(all_strata, key=lambda k: -len(pools[k]))[:extras]:
        targets[k] += 1

    selected: list[dict] = []
    deficit: dict[tuple[str, str], int] = {}
    for k, target in targets.items():
        pool = pools[k]
        take = min(target, len(pool))
        selected.extend(pool[:take])
        pools[k] = pool[take:]
        if take < target:
            deficit[k] = target - take
            logger.info("Stratum %s short by %d (had %d, wanted %d)", k, target - take, take, target)

    for k, need in list(deficit.items()):
        for nb in _neighbors(k):
            if need <= 0:
                break
            pool = pools[nb]
            take = min(need, len(pool))
            if take > 0:
                logger.info("Spillover: %d clips %s -> %s", take, nb, k)
                selected.extend(pool[:take])
                pools[nb] = pool[take:]
                need -= take
        if need > 0:
            logger.warning("Stratum %s still short %d clips after spillover", k, need)

    selected = selected[:n]
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for r in selected:
        counts[(r["length_bucket"], r["cs_bucket"])] += 1
    return selected, dict(counts)


# --- Manifest builder --------------------------------------------------------

def build_manifest(
    raw_dir: Path = Path("data/raw"),
    manifest_path: Path = Path("data/manifest.csv"),
    clips_dir: Path = Path("data/clips"),
    n_samples: int = 50,
    seed: int = 42,
    archive_url: str = OPENSLR_URL,
) -> Path:
    """Download + sample + write manifest. Idempotent: skips download/extract if present."""
    raw_dir = Path(raw_dir)
    manifest_path = Path(manifest_path)
    clips_dir = Path(clips_dir)

    archive_path = raw_dir / "Bengali-English_test.tar.gz"
    if not archive_path.exists():
        download_file(archive_url, archive_path)
    else:
        logger.info("Archive already present at %s — skipping download", archive_path)

    # Heuristic: if the raw dir already has .wav files, assume extraction is done.
    if not any(raw_dir.rglob("*.wav")):
        extract_tarball(archive_path, raw_dir)
    else:
        logger.info("WAV files already present under %s — skipping extraction", raw_dir)

    transcripts = find_transcripts(raw_dir)
    audio_paths = find_audio_files(raw_dir)
    segments = find_segments(raw_dir)
    wav_scp = find_wav_scp(raw_dir)
    if not transcripts:
        raise RuntimeError(
            f"No transcripts found under {raw_dir}. Check archive layout — expected one of: "
            + ", ".join(TRANSCRIPT_FILENAMES)
        )
    if not audio_paths:
        raise RuntimeError(f"No .wav files found under {raw_dir}")
    logger.info(
        "Found %d transcripts, %d recordings, %d segments",
        len(transcripts), len(audio_paths), len(segments),
    )

    def resolve_recording(rec_id: str) -> Path | None:
        # Prefer wav.scp entry; fall back to stem lookup in audio_paths.
        scp_entry = wav_scp.get(rec_id)
        if scp_entry:
            stem = Path(scp_entry).stem
            if stem in audio_paths:
                return audio_paths[stem]
        return audio_paths.get(rec_id)

    records: list[dict] = []
    if segments:
        # Kaldi-style: each utt_id is a (recording_id, start, end) slice.
        for utt_id, text in transcripts.items():
            seg = segments.get(utt_id)
            if seg is None:
                continue
            rec_id, start, end = seg
            src = resolve_recording(rec_id)
            if src is None:
                continue
            length = end - start
            if length <= 0:
                continue
            density = code_switch_density(text)
            records.append({
                "utt_id": utt_id,
                "_source_wav": str(src.resolve()),
                "_start_sec": start,
                "_end_sec": end,
                "reference_transcript": text,
                "length_sec": round(length, 3),
                "code_switch_density": round(density, 3),
                "length_bucket": length_bucket(length),
                "cs_bucket": cs_bucket(density),
            })
    else:
        # Fallback: utt_id matches a wav stem directly (non-Kaldi layout).
        for utt_id, text in transcripts.items():
            wav = audio_paths.get(utt_id)
            if wav is None:
                continue
            try:
                length = audio_length_sec(wav)
            except (RuntimeError, OSError) as e:
                logger.warning("Skipping %s — couldn't read audio length: %s", wav, e)
                continue
            density = code_switch_density(text)
            records.append({
                "utt_id": utt_id,
                "_source_wav": str(wav.resolve()),
                "_start_sec": 0.0,
                "_end_sec": length,
                "reference_transcript": text,
                "length_sec": round(length, 3),
                "code_switch_density": round(density, 3),
                "length_bucket": length_bucket(length),
                "cs_bucket": cs_bucket(density),
            })
    logger.info("Matched %d (transcript, audio) pairs", len(records))
    if len(records) < n_samples:
        raise RuntimeError(
            f"Only {len(records)} matched pairs available — need at least {n_samples}"
        )

    selected, counts = stratified_sample(records, n=n_samples, seed=seed)
    # Stable clip_id assignment so reruns produce the same IDs.
    selected = sorted(selected, key=lambda r: r["utt_id"])
    for i, r in enumerate(selected, start=1):
        r["clip_id"] = f"clip_{i:03d}"

    # Extract sub-clips so downstream transcribers see one self-contained wav per row.
    clips_dir.mkdir(parents=True, exist_ok=True)
    for r in selected:
        dest = clips_dir / f"{r['clip_id']}.wav"
        extract_segment(Path(r["_source_wav"]), r["_start_sec"], r["_end_sec"], dest)
        r["audio_path"] = str(dest.resolve())

    df = pd.DataFrame(selected, columns=MANIFEST_COLUMNS)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)

    logger.info("Final per-stratum counts (length × cs):")
    for lb in LENGTH_BUCKETS:
        for cb in CS_BUCKETS:
            logger.info("  %-6s × %-6s : %d", lb, cb, counts.get((lb, cb), 0))
    logger.info("Wrote manifest with %d clips -> %s", len(selected), manifest_path)
    return manifest_path


# --- Notebook helper ---------------------------------------------------------

def print_sample_transcripts(manifest_path: Path, k: int = 10, seed: int = 0) -> None:
    """Print k random reference transcripts so the user can eyeball the script.

    Plan §2.4 — user inspects whether transcripts are Bangla / Roman / mixed.
    """
    df = pd.read_csv(manifest_path)
    sample = df.sample(n=min(k, len(df)), random_state=seed)
    print(f"\n--- {len(sample)} random reference transcripts from {manifest_path} ---")
    for _, row in sample.iterrows():
        print(f"[{row['clip_id']}] ({row['length_bucket']}, cs={row['cs_bucket']}) "
              f"len={row['length_sec']:.1f}s  cs={row['code_switch_density']:.2f}")
        print(f"  {row['reference_transcript']}")
    print()
