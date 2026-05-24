"""Regenerate the 9 per-model + 1 judge Kaggle notebooks in notebooks/.

Run:
    python scripts/build_notebooks.py

All notebooks are deterministic from this file — edit here, re-run, commit
the regenerated .ipynb files. Don't hand-edit the .ipynb files directly.

Why a generator: every notebook shares ~80% of its cells (clone repo,
build manifest, run one model, download CSV). Hand-maintaining 10 JSON
files would guarantee drift between siblings.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

GITHUB_REPO_URL = "https://github.com/notAvailable73/stt-model-comparison.git"
REPO_DIR = "/kaggle/working/banglish-stt-benchmark"


# ---------------------------------------------------------------------------
# Notebook scaffolding
# ---------------------------------------------------------------------------

def _md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def _code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def _notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _write_notebook(filename: str, cells: list[dict]) -> None:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOKS_DIR / filename
    path.write_text(json.dumps(_notebook(cells), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Shared cell snippets
# ---------------------------------------------------------------------------

def clone_repo_cell() -> dict:
    return _code(f"""# === Clone the benchmark repo ===========================================
# Kaggle paths: /kaggle/working is the writable home. We clone the project
# repo there and cd into it so the relative paths used by src/ resolve.

!apt-get -qq install -y libsndfile1

GITHUB_REPO_URL = "{GITHUB_REPO_URL}"
REPO_DIR = "{REPO_DIR}"

import os, sys, subprocess
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", GITHUB_REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=False)
os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
print("cwd:", os.getcwd())
""")


def build_manifest_cell() -> dict:
    return _code("""# === Build the 50-clip manifest =========================================
# Downloads OpenSLR-104 if not already cached, extracts, parses transcripts,
# computes length + code-switch density, and stratified-samples 50 clips.
#
# CRITICAL: every notebook in this benchmark calls this with seed=42 so
# they all see the EXACT SAME 50 clips. Do not change the seed.

from src.utils import setup_logging, set_seeds
from src.data_prep import build_manifest

setup_logging("logs")
set_seeds(42)

manifest_path = build_manifest(
    raw_dir="data/raw",
    manifest_path="data/manifest.csv",
    n_samples=50,
    seed=42,
)
print("manifest:", manifest_path)
""")


def transcribe_cell(model_id: str, device: str = "cuda") -> dict:
    return _code(f'''# === Transcribe with this model =========================================
# Reads the spec for THIS notebook's model from config/models.yaml,
# then runs the 50-clip loop. Resumable: rerun this cell to pick up
# where it stopped (clips already in the CSV are skipped).

from src.utils import read_yaml
from src.transcribe import run_single_model

MODEL_ID = "{model_id}"
spec = next(s for s in read_yaml("config/models.yaml")["models"] if s["id"] == MODEL_ID)

failure = run_single_model(
    spec,
    manifest_csv="data/manifest.csv",
    predictions_dir="predictions",
    device="{device}",
)
print("FAILED:", failure) if failure else print("OK — see predictions/")
''')


def download_cell(model_id: str) -> dict:
    slug = model_id.replace("/", "__")
    return _code(f'''# === Download the predictions CSV =======================================
# Right-click the link below and save to your local machine. You will
# upload all 9 CSVs to the judge notebook (notebook 10) at the end.
#
# File:   predictions/{slug}.csv
# Schema: clip_id, model_id, predicted_text, latency_sec, decoder_variant

from pathlib import Path
import pandas as pd
from IPython.display import FileLink, display

csv_path = Path("predictions/{slug}.csv")
if csv_path.exists():
    df = pd.read_csv(csv_path)
    print(f"{{len(df)}} rows in {{csv_path}}")
    display(df.head())
    display(FileLink(str(csv_path)))
else:
    print(f"NOT FOUND: {{csv_path}} — transcription cell failed or was skipped.")
''')


# ---------------------------------------------------------------------------
# Per-family install cells
# ---------------------------------------------------------------------------

def whisper_install_cell() -> dict:
    return _code("""# === Install transformers + base deps ===================================
# These versions are the same across all whisper_hf notebooks (models
# 1, 2, 3, 8). Pinned to match what the project's adapters expect.

!pip install -q transformers==4.44.2
!pip install -q accelerate==0.34.2
!pip install -q librosa==0.10.2.post1
!pip install -q soundfile==0.12.1
!pip install -q pandas==2.2.3
!pip install -q pyyaml==6.0.2
!pip install -q tqdm==4.67.1
!pip install -q requests==2.32.3
""")


def python_version_check_cell(required: str = "3.11") -> dict:
    return _code(f'''# === Python version check ===============================================
# This notebook needs Python {required}. Kaggle's current default is 3.12,
# which has cp312 wheel gaps that make the NeMo install painful (see
# DECISIONS.md §I). Switch images: right sidebar → Settings →
# Environment → "Pin to original environment" (or pick a 3.11 image).

import sys
v = f"{{sys.version_info.major}}.{{sys.version_info.minor}}"
if v != "{required}":
    print("=" * 70)
    print(f"WARNING: detected Python {{v}}, this notebook expects {required}.")
    print("Settings → Environment → Pin to original environment (3.11),")
    print("then Run → Restart & Clear, then re-run from the top.")
    print("=" * 70)
else:
    print(f"Python {{v}} OK.")
''')


def nemo_install_cell() -> dict:
    return _code("""# === Install AI4Bharat NeMo fork ========================================
# Heavy cell — 5-10 min if both workarounds apply, hours otherwise.
# Reason for the fork: model 4 needs the multi-softmax aggregate tokenizer
# + `language_id="bn"` kwarg that stock nemo_toolkit lacks. The fork
# installs editable under the same `nemo_toolkit` name, so hishab models
# 5/6 load on top — no second NeMo install. (DECISIONS.md §B)
#
# Two workarounds applied below (DECISIONS.md §I):
#
# (1) Pre-uninstall + legacy resolver. The fork pins huggingface_hub
#     ==0.23.2, but Kaggle ships transformers/datasets/diffusers built
#     against a much newer hf_hub. With pip's new resolver this triggers
#     multi-hour backtracking. Removing the conflict seeds first +
#     PIP_USE_DEPRECATED=legacy-resolver brings resolution back to seconds.
#
# (2) Relax tensorstore pin. The fork pins `tensorstore<0.1.46`, but
#     cp312 wheels only start at ~0.1.50 (cp311 wheels exist further back
#     but this guard catches both). Without it, pip falls back to a Bazel
#     source build that takes 30-60+ min.

!rm -rf /kaggle/working/NeMo
!git clone https://github.com/AI4Bharat/NeMo.git /kaggle/working/NeMo
!cd /kaggle/working/NeMo && git checkout nemo-v2

!pip uninstall -y transformers datasets diffusers huggingface_hub tokenizers
!pip install -q "pip<24.2"

!sed -i 's/tensorstore<0.1.46/tensorstore/g' /kaggle/working/NeMo/setup.py 2>/dev/null || true
!find /kaggle/working/NeMo/requirements -name '*.txt' -exec sed -i 's/tensorstore<0.1.46/tensorstore/g' {} +
!pip install -q "tensorstore>=0.1.71"

!cd /kaggle/working/NeMo && PIP_USE_DEPRECATED=legacy-resolver bash reinstall.sh

# Project's own pinned deps (numpy + torchvision ABI realignment is needed
# after the NeMo install above because reinstall.sh downgrades both).
!pip install -q transformers==4.44.2 accelerate==0.34.2 librosa==0.10.2.post1 \\
    soundfile==0.12.1 pandas==2.2.3 pyyaml==6.0.2 tqdm==4.67.1 requests==2.32.3
!pip install -q --force-reinstall --no-deps numpy==2.0.2
!pip install -q --upgrade --force-reinstall --no-deps torchvision \\
    --index-url https://download.pytorch.org/whl/cu128
""")


def nemo_path_cell() -> dict:
    return _code('''# === Make NeMo importable in this kernel ================================
# `pip install -e` writes a .pth file that's only read at interpreter
# startup. The kernel that ran the install above never sees it → import
# nemo fails. Add the source dir to sys.path manually for this session.
# (DECISIONS.md §I)

import sys
if "/kaggle/working/NeMo" not in sys.path:
    sys.path.append("/kaggle/working/NeMo")

# Verify
import nemo.collections.asr  # noqa: F401
print("nemo.collections.asr import OK")
''')


def restart_check_cell() -> dict:
    return _code('''# === Restart check (numpy ABI realignment) ==============================
# If the on-disk numpy differs from the in-memory numpy, restart the
# kernel and re-run all cells. Pip caches persist within a session, so
# the rerun is fast.

import importlib, subprocess, sys
_disk = subprocess.check_output(
    [sys.executable, "-c", "import numpy; print(numpy.__version__)"]
).decode().strip()
_runtime = importlib.import_module("numpy").__version__
print(f"numpy on disk: {_disk} | numpy in kernel: {_runtime}")
if _disk != _runtime:
    print("=" * 70)
    print("RESTART REQUIRED: Run → Restart, then re-run from the top.")
    print("=" * 70)
else:
    print("numpy aligned.")
''')


def omni_install_cell() -> dict:
    return _code('''# === Install omnilingual-asr (expected to FAIL) =========================
# DECISIONS.md §D: omnilingual-asr depends on fairseq2n which ships only
# a CUDA-13 wheel. Kaggle provides CUDA 12.x → libcudart.so.13 missing.
# The plan locks the model list, so we install + try anyway; the
# transcription cell catches the load error and writes a FAILED status.
#
# If you happen to be on a CUDA-13 runtime, this might actually work —
# the rest of the notebook is identical to the other model notebooks.

!pip install -q omnilingual-asr || true
!pip install -q transformers==4.44.2 accelerate==0.34.2 librosa==0.10.2.post1 \\
    soundfile==0.12.1 pandas==2.2.3 pyyaml==6.0.2 tqdm==4.67.1 requests==2.32.3
''')


def openrouter_install_cell() -> dict:
    return _code('''# === Install openai SDK (used as OpenRouter client) =====================
# Model 9 (google/chirp-3) is a cloud model; no torch, no transformers.

!pip install -q openai==1.55.0 librosa==0.10.2.post1 soundfile==0.12.1 \\
    pandas==2.2.3 pyyaml==6.0.2 tqdm==4.67.1 requests==2.32.3
''')


def openrouter_secret_cell() -> dict:
    return _code('''# === OPENROUTER_API_KEY ================================================
# Recommended: Add-ons → Secrets → add OPENROUTER_API_KEY (never logged).
# Fallback: getpass prompt.

import os
if not os.environ.get("OPENROUTER_API_KEY"):
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["OPENROUTER_API_KEY"] = (
            UserSecretsClient().get_secret("OPENROUTER_API_KEY")
        )
        print("Loaded OPENROUTER_API_KEY from Kaggle Secrets.")
    except Exception:
        from getpass import getpass
        os.environ["OPENROUTER_API_KEY"] = getpass("OPENROUTER_API_KEY: ")
print("OPENROUTER_API_KEY set:", bool(os.environ.get("OPENROUTER_API_KEY")))
''')


# ---------------------------------------------------------------------------
# Per-notebook builders
# ---------------------------------------------------------------------------

def whisper_notebook(n: int, model_id: str, title: str) -> tuple[str, list[dict]]:
    filename = f"{n}_{title}.ipynb"
    cells = [
        _md(f"# Notebook {n}: `{model_id}`\n\n"
            f"Family: `whisper_hf` · Python 3.12 (Kaggle default) · GPU recommended.\n\n"
            f"Produces `predictions/{model_id.replace('/', '__')}.csv`. "
            f"Download at the end; upload to notebook 10 for judging.\n"),
        clone_repo_cell(),
        whisper_install_cell(),
        build_manifest_cell(),
        transcribe_cell(model_id),
        download_cell(model_id),
    ]
    return filename, cells


def nemo_notebook(n: int, model_id: str, title: str) -> tuple[str, list[dict]]:
    filename = f"{n}_{title}.ipynb"
    cells = [
        _md(f"# Notebook {n}: `{model_id}`\n\n"
            f"Family: `nemo_conformer` · **Python 3.11 required** · GPU recommended.\n\n"
            f"Settings → Environment → **Pin to original environment** before running.\n"
            f"The NeMo install cell is heavy (5-10 min) and the recipe is documented in "
            f"DECISIONS.md §I. After the install, if Cell 4 prints `RESTART REQUIRED`, "
            f"do that and re-run from the top.\n\n"
            f"Produces `predictions/{model_id.replace('/', '__')}.csv`.\n"),
        python_version_check_cell("3.11"),
        clone_repo_cell(),
        nemo_install_cell(),
        restart_check_cell(),
        nemo_path_cell(),
        build_manifest_cell(),
        transcribe_cell(model_id),
        download_cell(model_id),
    ]
    return filename, cells


def omni_notebook(n: int, model_id: str, title: str) -> tuple[str, list[dict]]:
    filename = f"{n}_{title}.ipynb"
    cells = [
        _md(f"# Notebook {n}: `{model_id}`\n\n"
            f"Family: `omni_asr` · **Expected to FAIL** on Kaggle (DECISIONS.md §D — "
            f"fairseq2n CUDA-13 wheel vs Kaggle's CUDA 12.x).\n\n"
            f"Still run the notebook end to end: the transcription cell will mark the "
            f"model FAILED and produce an empty `predictions/{model_id.replace('/', '__')}.csv` "
            f"with a header row. Upload it to notebook 10 like the others — the judge "
            f"will surface it as FAILED in the master JSON.\n"),
        clone_repo_cell(),
        omni_install_cell(),
        build_manifest_cell(),
        transcribe_cell(model_id),
        _code(f'''# === Ensure the predictions CSV exists (even if FAILED) =================
# If the load failed, run_single_model returned a failure dict but did not
# write a CSV. Write a header-only file so the judge notebook sees the
# expected slug and marks the model FAILED via its existing missing-model
# path.

from pathlib import Path
from src.transcribe import PREDICTION_COLUMNS

csv_path = Path("predictions/{model_id.replace("/", "__")}.csv")
if not csv_path.exists():
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(",".join(PREDICTION_COLUMNS) + "\\n", encoding="utf-8")
    print(f"Wrote header-only {{csv_path}} (model FAILED).")
else:
    print(f"{{csv_path}} already exists ({{csv_path.stat().st_size}} bytes).")
'''),
        download_cell(model_id),
    ]
    return filename, cells


def chirp_notebook(n: int, model_id: str, title: str) -> tuple[str, list[dict]]:
    filename = f"{n}_{title}.ipynb"
    cells = [
        _md(f"# Notebook {n}: `{model_id}`\n\n"
            f"Family: `openrouter_stt` · CPU only · network required.\n\n"
            f"Set `OPENROUTER_API_KEY` in Kaggle Secrets before running.\n\n"
            f"Produces `predictions/{model_id.replace('/', '__')}.csv`.\n"),
        clone_repo_cell(),
        openrouter_install_cell(),
        openrouter_secret_cell(),
        build_manifest_cell(),
        transcribe_cell(model_id, device="cpu"),
        download_cell(model_id),
    ]
    return filename, cells


def judge_notebook() -> tuple[str, list[dict]]:
    cells = [
        _md("""# Notebook 10: Judge + Analyze

Consumes the 9 CSVs produced by notebooks 1–9. Runs the Gemini 3 Flash
judge over every (clip, model) pair, then builds the master JSON,
comparison table, plots, and final report.

**Flow:**

1. Clone repo, install minimal deps.
2. Rebuild the manifest deterministically (same `seed=42` as the model notebooks).
3. **Upload your 9 prediction CSVs as a Kaggle dataset** (instructions below).
4. Aggregate uploaded CSVs into `./predictions/`.
5. Set `OPENROUTER_API_KEY` in Kaggle Secrets.
6. Judge → master JSON → comparison table → plots → report → bundle zip.
"""),
        clone_repo_cell(),
        _code('''# === Install minimal deps (no torch, no transformers) ===================
!pip install -q openai==1.55.0 pandas==2.2.3 pyyaml==6.0.2 \\
    matplotlib==3.9.2 seaborn==0.13.2 librosa==0.10.2.post1 \\
    soundfile==0.12.1 tqdm==4.67.1 requests==2.32.3
'''),
        build_manifest_cell(),
        _md("""## UPLOAD YOUR PREDICTIONS

After running notebooks 1–9, you have 9 CSVs saved locally. They are named
exactly:

```
openai__whisper-large-v3.csv
openai__whisper-large-v3-turbo.csv
KhushiDS__whisper-large-v3-Bengali.csv
ai4bharat__indicconformer_stt_bn_hybrid_ctc_rnnt_large.csv
hishab__titu_stt_bn_fastconformer.csv
hishab__titu_stt_bn_conformer_large.csv
facebook__omniASR-CTC-300M.csv
bangla-speech-processing__BanglaASR.csv
google__chirp-3.csv
```

**Steps on this Kaggle notebook:**

1. Right sidebar → **+ Add Data** → **Upload** (the tab to the right of "Search Datasets").
2. Drag in all 9 CSVs. Give the dataset a name like `banglish-stt-predictions`.
3. After upload finishes, the files appear at `/kaggle/input/<dataset-slug>/`.
4. Set `UPLOADED_DIR` in the next cell to that path.

The next cell will validate that all 9 expected slugs are present and copy them
into `./predictions/`. Any missing CSV is reported and the corresponding model
is marked `FAILED` in the final JSON — the judge / analysis still runs over the
models that ARE present.
"""),
        _code('''# === Aggregate uploaded CSVs into ./predictions/ ========================

from pathlib import Path
import shutil
from src.utils import read_yaml, slugify_model_id

UPLOADED_DIR = "/kaggle/input/banglish-stt-predictions"   # <- change to match your upload

local_dir = Path("predictions")
local_dir.mkdir(parents=True, exist_ok=True)

expected_slugs = {
    slugify_model_id(spec["id"]): spec["id"]
    for spec in read_yaml("config/models.yaml")["models"]
}

found, missing = [], []
upload_path = Path(UPLOADED_DIR)
for slug, model_id in expected_slugs.items():
    src = upload_path / f"{slug}.csv"
    dst = local_dir / f"{slug}.csv"
    if src.exists():
        shutil.copyfile(src, dst)
        found.append(model_id)
    else:
        missing.append(model_id)

print(f"Found {len(found)}/{len(expected_slugs)} prediction CSVs.")
if missing:
    print("Missing (will be marked FAILED):")
    for m in missing:
        print(" -", m)

# `failed_models` is passed to the analysis cells below.
failed_models = [
    {"model_id": m, "status": "FAILED", "error": "no predictions CSV uploaded"}
    for m in missing
]
'''),
        openrouter_secret_cell(),
        _code('''# === Judging ============================================================
# One Gemini 3 Flash call per (clip, model). Resumable: rows already in
# judgments.csv are skipped, so reruns continue from the last judged pair.

from src.judge import judge_predictions

judge_predictions(
    manifest_csv="data/manifest.csv",
    predictions_dir="predictions",
    judgments_csv="judgments.csv",
    prompts_yaml="config/prompts.yaml",
)
'''),
        _code('''# === Master JSON ========================================================
from src.analyze import build_master_json

master_json = build_master_json(
    manifest_csv="data/manifest.csv",
    predictions_dir="predictions",
    judgments_csv="judgments.csv",
    models_yaml="config/models.yaml",
    failed_models=failed_models,
    output_json="results/benchmark_results.json",
)
print("Master JSON:", master_json)
'''),
        _code('''# === Comparison table ===================================================
from src.analyze import build_comparison_table
from IPython.display import Image, display
import pandas as pd

comp_csv, comp_png = build_comparison_table(
    manifest_csv="data/manifest.csv",
    predictions_dir="predictions",
    judgments_csv="judgments.csv",
    models_yaml="config/models.yaml",
    failed_models=failed_models,
    out_csv="results/comparison_table.csv",
    out_png="results/comparison_table.png",
)
display(pd.read_csv(comp_csv))
display(Image(str(comp_png)))
'''),
        _code('''# === Analytical plots ===================================================
from src.analyze import make_all_plots
from IPython.display import Image, display

plot_paths = make_all_plots(
    manifest_csv="data/manifest.csv",
    predictions_dir="predictions",
    judgments_csv="judgments.csv",
    models_yaml="config/models.yaml",
    failed_models=failed_models,
    plots_dir="results/plots",
)
for p in plot_paths:
    print(p)
    display(Image(str(p)))
'''),
        _code('''# === Final report (markdown) ============================================
from pathlib import Path
from src.analyze import write_report_md
from IPython.display import Markdown, display

report_path = write_report_md(
    comparison_csv="results/comparison_table.csv",
    master_json="results/benchmark_results.json",
    out_md="results/report.md",
)
display(Markdown(Path(report_path).read_text(encoding="utf-8")))
'''),
        _code('''# === Bundle results =====================================================
import zipfile
from pathlib import Path
from src.analyze import bundle_results

zip_path = bundle_results("results", "banglish_stt_benchmark_results.zip")
size_mb = zip_path.stat().st_size / (1 << 20)
print(f"Bundle: {zip_path}  ({size_mb:.2f} MB)")
with zipfile.ZipFile(zip_path) as zf:
    for name in zf.namelist():
        print(" ", name)

from IPython.display import FileLink, display
display(FileLink(str(zip_path)))
'''),
    ]
    return "10_judge_and_analyze.ipynb", cells


# ---------------------------------------------------------------------------
# Notebook registry (matches config/models.yaml order)
# ---------------------------------------------------------------------------

NOTEBOOKS = [
    whisper_notebook(1, "openai/whisper-large-v3",                                  "whisper_large_v3"),
    whisper_notebook(2, "openai/whisper-large-v3-turbo",                            "whisper_large_v3_turbo"),
    whisper_notebook(3, "KhushiDS/whisper-large-v3-Bengali",                        "whisper_large_v3_bengali"),
    nemo_notebook(   4, "ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large",    "indicconformer_bn"),
    nemo_notebook(   5, "hishab/titu_stt_bn_fastconformer",                         "titu_fastconformer"),
    nemo_notebook(   6, "hishab/titu_stt_bn_conformer_large",                       "titu_conformer_large"),
    omni_notebook(   7, "facebook/omniASR-CTC-300M",                                "omni_asr"),
    whisper_notebook(8, "bangla-speech-processing/BanglaASR",                       "bangla_asr"),
    chirp_notebook(  9, "google/chirp-3",                                           "chirp_3"),
    judge_notebook(),
]


def main() -> None:
    for filename, cells in NOTEBOOKS:
        _write_notebook(filename, cells)
    print(f"\n{len(NOTEBOOKS)} notebooks written to {NOTEBOOKS_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
