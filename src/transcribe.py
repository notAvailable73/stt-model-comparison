"""Cell-4 driver: run every model in config/models.yaml over the manifest.

Per plan §3.2 and §6 Cell 4:
- Load failures mark the model FAILED in the returned list; the loop continues.
- Per-clip errors are logged and recorded as empty text + NaN latency so we
  don't lose progress on the rest of the clips.
- Predictions CSV is flushed after every clip (resumability — DECISIONS.md §E).
- `decoder_variant` is populated for model 4 (hybrid Conformer) and blank
  for everything else, per DECISIONS.md §B.
"""

from __future__ import annotations

import csv
import importlib
import logging
from pathlib import Path

import pandas as pd

from src.utils import read_yaml, slugify_model_id

logger = logging.getLogger(__name__)

PREDICTION_COLUMNS = ["clip_id", "model_id", "predicted_text", "latency_sec", "decoder_variant"]

FAMILY_ADAPTERS: dict[str, str] = {
    "whisper_hf":       "src.transcribers.whisper_hf.WhisperHFTranscriber",
    "nemo_conformer":   "src.transcribers.nemo_conformer.NemoConformerTranscriber",
    "omni_asr":         "src.transcribers.omni_asr.OmniASRTranscriber",
    "openrouter_stt":   "src.transcribers.openrouter_stt.OpenRouterSTTTranscriber",
}


def _load_adapter_class(family: str):
    """Resolve a family name to its adapter class via importlib (lazy)."""
    if family not in FAMILY_ADAPTERS:
        raise ValueError(f"Unknown family '{family}'. Known: {list(FAMILY_ADAPTERS)}")
    module_path, class_name = FAMILY_ADAPTERS[family].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _existing_clip_ids(csv_path: Path) -> set[str]:
    """Return clip_ids already present in the predictions CSV (resumability)."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    df = pd.read_csv(csv_path)
    return set(df["clip_id"].astype(str)) if "clip_id" in df.columns else set()


def _append_prediction(csv_path: Path, row: dict, write_header: bool) -> None:
    """Append one prediction row to the CSV and flush, so a crash mid-run keeps it."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in PREDICTION_COLUMNS})


def run_transcription(
    manifest_csv: str | Path,
    models_yaml: str | Path,
    predictions_dir: str | Path = "predictions",
    device: str = "cuda",
) -> list[dict]:
    """Run all 9 models. Returns the list of failed-model records.

    Each failed entry: {"model_id": str, "status": "FAILED", "error": str}.
    """
    manifest = pd.read_csv(manifest_csv)
    models_cfg = read_yaml(models_yaml)
    predictions_dir = Path(predictions_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    failed: list[dict] = []
    for spec in models_cfg["models"]:
        model_id: str = spec["id"]
        family: str = spec["family"]
        # Anything other than id/family is passed as adapter kwargs.
        adapter_kwargs = {k: v for k, v in spec.items() if k not in ("id", "family")}

        slug = slugify_model_id(model_id)
        out_csv = predictions_dir / f"{slug}.csv"
        already_done = _existing_clip_ids(out_csv)
        write_header = not out_csv.exists() or out_csv.stat().st_size == 0
        logger.info(
            "=== Model %s (family=%s) — %d clips already in %s ===",
            model_id, family, len(already_done), out_csv,
        )

        # --- Load ---
        try:
            AdapterCls = _load_adapter_class(family)
            transcriber = AdapterCls(model_id=model_id, device=device, **adapter_kwargs)
        except Exception as e:  # noqa: BLE001 — load can fail many ways; we always log.
            logger.exception("Model %s FAILED during load", model_id)
            failed.append({"model_id": model_id, "status": "FAILED", "error": f"load: {e!r}"})
            continue

        # --- Per-clip loop ---
        try:
            for _, row in manifest.iterrows():
                clip_id = str(row["clip_id"])
                if clip_id in already_done:
                    continue
                audio_path = str(row["audio_path"])

                try:
                    text, latency = transcriber.transcribe(audio_path)
                except Exception as e:  # noqa: BLE001 — log and continue with NaN.
                    logger.exception("clip=%s model=%s transcribe failed", clip_id, model_id)
                    text, latency = "", float("nan")

                _append_prediction(
                    out_csv,
                    {
                        "clip_id": clip_id,
                        "model_id": model_id,
                        "predicted_text": text,
                        "latency_sec": latency,
                        "decoder_variant": getattr(transcriber, "current_decoder", None) or "",
                    },
                    write_header=write_header,
                )
                write_header = False
                already_done.add(clip_id)
                logger.info(
                    "  %s | clip=%s | latency=%.2fs | %s",
                    model_id, clip_id, latency if latency == latency else float("nan"),
                    (text[:60] + "…") if len(text) > 60 else text,
                )
        finally:
            try:
                transcriber.cleanup()
            except Exception:  # noqa: BLE001
                logger.exception("Cleanup failed for %s", model_id)

    logger.info("Transcription complete. Failed models: %s", [f["model_id"] for f in failed])
    return failed
