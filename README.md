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
├── requirements.txt            # Colab pip install
├── notebook.ipynb              # Entry point — clones repo, runs cells 1–10
├── src/
│   ├── data_prep.py            # Download OpenSLR-104, sample 50, build manifest
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
    └── smoke_whisper.py        # One-clip sanity check for whisper_hf adapter
```

## Run it (Colab)

1. Fork this repo.
2. Open `notebook.ipynb` in Colab.
3. Set `GITHUB_REPO_URL` in Cell 1 to your fork's clone URL.
4. Run cells 1–10. Cell 10 produces the downloadable zip.

The notebook is intentionally thin — almost all logic lives in `src/`.

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
