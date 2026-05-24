# Banglish STT Benchmark

Compare 9 Speech-to-Text models on Banglish (Bengali–English code-switched)
audio from OpenSLR-104. A single Gemini 3 Flash judge scores each prediction
on a 3-dimension rubric. Output is a master JSON + comparison table + plots.

See [`plan.md`](plan.md) for the full spec and [`DECISIONS.md`](DECISIONS.md)
for non-obvious defaults.

## Repo layout

```
.
├── plan.md                     # Spec (locked)
├── DECISIONS.md                # Defaults / judgment calls not in the spec
├── requirements.txt
├── notebooks/                  # Kaggle entry points — one per model + judge
│   ├── 1_whisper_large_v3.ipynb
│   ├── 2_whisper_large_v3_turbo.ipynb
│   ├── 3_whisper_large_v3_bengali.ipynb
│   ├── 4_indicconformer_bn.ipynb           # NeMo, Python 3.11
│   ├── 5_titu_fastconformer.ipynb          # NeMo, Python 3.11
│   ├── 6_titu_conformer_large.ipynb        # NeMo, Python 3.11
│   ├── 7_omni_asr.ipynb                    # expected FAIL (DECISIONS.md §D)
│   ├── 8_bangla_asr.ipynb
│   ├── 9_chirp_3.ipynb                     # OpenRouter
│   └── 10_judge_and_analyze.ipynb          # consumes uploaded CSVs
├── src/
│   ├── data_prep.py            # Download OpenSLR-104, sample 50, build manifest
│   ├── transcribe.py           # run_single_model / run_transcription
│   ├── judge.py                # Gemini 3 Flash via OpenRouter
│   ├── analyze.py              # Aggregation + 6 plots
│   ├── utils.py                # Logging, JSON/YAML I/O, slugify
│   └── transcribers/
│       ├── base.py             # BaseTranscriber abstract class (§3.1)
│       ├── whisper_hf.py       # Models 1, 2, 3, 8
│       ├── nemo_conformer.py   # Models 4, 5, 6
│       ├── omni_asr.py         # Model 7
│       └── openrouter_stt.py   # Model 9 (Chirp 3)
├── config/
│   ├── models.yaml             # Locked list of 9 models
│   └── prompts.yaml            # Judge prompt template
└── scripts/
    ├── build_notebooks.py      # Regenerate notebooks/ from this single source
    └── smoke_whisper.py        # One-clip sanity check for whisper_hf adapter
```

## Run it (Kaggle)

The single-notebook approach doesn't work on Kaggle — the 9 models' Python
dependencies cannot share an environment. See [DECISIONS.md §I](DECISIONS.md)
for the full diagnosis. The new workflow is one Kaggle notebook per model.

1. Fork this repo and set the `GITHUB_REPO_URL` constant at the top of
   `scripts/build_notebooks.py` to your fork's clone URL, then re-run
   `python scripts/build_notebooks.py`. Commit and push.
2. For each `notebooks/N_*.ipynb`:
   - Create a new Kaggle notebook (File → Upload notebook, or paste cells).
   - **For notebooks 4, 5, 6 (NeMo):** Settings → Environment → "Pin to
     original environment" (Python 3.11). For everything else, Kaggle's
     current default is fine.
   - For notebooks 9 and 10: add `OPENROUTER_API_KEY` under Add-ons → Secrets.
   - Run all cells. Download the predictions CSV at the end (a `FileLink`
     appears in the last cell's output).
3. Run notebook 10 (`10_judge_and_analyze.ipynb`) last:
   - Upload all 9 CSVs as a Kaggle dataset (right sidebar → + Add Data →
     Upload).
   - Set `UPLOADED_DIR` in the aggregation cell to that dataset's path.
   - Run all cells. The final cell produces a downloadable zip.

The notebooks are intentionally thin — almost all logic lives in `src/`,
and the install cells are model-family-specific (see DECISIONS.md §I for
the NeMo recipe in particular).

### Don't hand-edit the .ipynb files

`scripts/build_notebooks.py` is the source of truth. Edit there, re-run,
commit the regenerated notebooks.

## Local smoke test (one model, one clip)

```bash
pip install -r requirements.txt
python scripts/smoke_whisper.py /path/to/clip.wav
```

This loads `openai/whisper-large-v3-turbo` (small enough for a quick check)
and prints the transcript + latency. Used to confirm the adapter shape before
fanning out across all 9 models.

## Resumability

Transcription and judging both skip work that's already in the output CSVs.
Colab sessions die — this is mandatory, not a bonus feature. See
[DECISIONS.md](DECISIONS.md) §E.
