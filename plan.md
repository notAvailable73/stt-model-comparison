# Banglish STT Benchmark — Full Plan

> **Purpose of this document**
> This is a complete, self-contained specification for building a benchmark that compares 9 Speech-to-Text (STT) models on Banglish (Bengali-English code-switched) audio. It is written so that an AI coding assistant (Claude Code) or a developer can execute it without needing prior context. All decisions are locked in. Where something is left as a placeholder, it is clearly marked `<<TODO>>`.

---

## 1. Project Goal

Compare 9 STT models on Banglish (Bengali–English code-switched) speech and produce a ranked recommendation for use in **Version3's B2B chatbot platform**, which receives voice messages from end-users on WhatsApp/Messenger/Instagram.

The benchmark must produce:

1. **Per-clip, per-model transcription text + judge score** stored as a single JSON file.
2. **Comparison table** of all models across all metrics.
3. **Analytical plots** (PNGs) for visual comparison.
4. A final **downloadable bundle** containing the JSON, the comparison table, and all plot images.

---

## 2. Test Data

### 2.1 Source
- **Dataset:** OpenSLR-104 — Bengali-English Code-Switched Test Set
- **URL:** https://www.openslr.org/104/
- **Direct download:** https://openslr.trmal.net/resources/104/Bengali-English_test.tar.gz
- **Size:** 606 MB
- **License:** CC BY-SA 4.0
- **Specs:** 16 kHz, 16-bit, 7.02 hours, vocabulary 13,656 words. Extracted from Bengali educational spoken tutorials (Indian Bengali speakers).

### 2.2 Sampling
- Sample **exactly 50 clips** from the test set, stratified by:
  - **Clip length:** mix of short (≤10s), medium (10–30s), long (>30s)
  - **Code-switching density:** computed as percentage of ASCII (English) tokens in the reference transcript. Bucket into low (<15%), medium (15–40%), high (>40%).
- Save the sampled set as `data/manifest.csv` with columns:
  - `clip_id` (string, e.g., `clip_001`)
  - `audio_path` (absolute path to .wav file)
  - `reference_transcript` (string, as-is from the dataset)
  - `length_sec` (float)
  - `code_switch_density` (float, 0–1)
  - `length_bucket` (short/medium/long)
  - `cs_bucket` (low/medium/high)

### 2.3 Honest caveats to acknowledge in the final report
- OpenSLR-104 is **Indian Bengali educational speech**, not Bangladeshi casual voice messages. Results indicate model capability but may not directly predict Version3 production performance.
- Audio is clean (16 kHz studio-quality). Noisy phone audio results may differ.
- Reference transcript script (Bangla / Roman / mixed) must be inspected after download — see §2.4.

### 2.4 Reference transcript script — must verify post-download
The original OpenSLR-104 transcripts may be in Bangla script, Roman, or mixed. The notebook must:
1. After download, print 10 random reference transcripts.
2. Report the apparent script choice.
3. The LLM judge prompt must be designed to **ignore script differences** when scoring (so a Roman-script prediction vs. a Bangla-script reference is not penalized if the meaning matches).

---

## 3. Models Under Evaluation (Locked List of 9)

| # | Model ID | Family | Inference Library | Notes |
|---|---|---|---|---|
| 1 | `openai/whisper-large-v3` | Whisper | `transformers` | Multilingual baseline |
| 2 | `openai/whisper-large-v3-turbo` | Whisper | `transformers` | Faster variant |
| 3 | `KhushiDS/whisper-large-v3-Bengali` | Whisper fine-tune | `transformers` | Bangla-specialized |
| 4 | `ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large` | Conformer | NVIDIA NeMo | Academic Bangla Conformer |
| 5 | `hishab/titu_stt_bn_fastconformer` | Conformer | NVIDIA NeMo | Bangladeshi, fast |
| 6 | `hishab/titu_stt_bn_conformer_large` | Conformer | NVIDIA NeMo | Bangladeshi, large |
| 7 | `facebook/omniASR-CTC-300M` | OmniASR | `omnilingual-asr` pkg | Meta, 1600+ languages; pass `lang="ben_Beng"` |
| 8 | `bangla-speech-processing/BanglaASR` | Older Bangla model | `transformers` | Older baseline |
| 9 | `google/chirp-3` (via OpenRouter) | Cloud | OpenRouter `/v1/audio/transcriptions` | Cloud reference |

### 3.1 Per-model adapter requirements
Because the 4 inference libraries above are mutually incompatible, the code must implement **one adapter class per family**, all conforming to:

```python
class BaseTranscriber:
    def __init__(self, model_id: str, device: str = "cuda"): ...
    def transcribe(self, audio_path: str) -> tuple[str, float]:
        """Returns (predicted_text, latency_seconds)."""
```

### 3.2 Failure policy
If a model fails to load or transcribe within **30 minutes of debugging**, skip it and mark it as `FAILED` in the output JSON with the error message. Do not block the benchmark on one broken model.

---

## 4. LLM Judge

### 4.1 Judge model
- **Single judge:** `google/gemini-3-flash-preview` via OpenRouter
- **Endpoint:** OpenRouter `/api/v1/chat/completions`
- **No repetition** — one judge call per (clip × model) pair = 50 × 9 = 450 calls total
- **Temperature:** 0
- **Pricing (as of plan date):** ~$0.50/M input tokens, ~$3.00/M output tokens

### 4.2 Rubric (1–5 per dimension, **3 dimensions** — code-switching dimension removed per user instruction)

| Dimension | What it measures |
|---|---|
| **Semantic accuracy** | Does the prediction convey the same meaning as the reference? Script differences are NOT penalized. |
| **Word-level faithfulness** | Are key content words (nouns, product names, numbers, technical terms) captured? |
| **Fluency & readability** | Is the output grammatical, well-punctuated, readable as text? |

Plus an **overall score** (1–10) and a **1-line justification**.

### 4.3 Judge prompt (use this exact structure)

```
You are an expert evaluator for Bengali-English code-switched (Banglish) speech transcription.

You will be shown a REFERENCE transcript (the ground truth) and a PREDICTED transcript (from an STT model). Score the PREDICTED transcript on three dimensions.

IMPORTANT RULES:
- IGNORE script differences. "ami office jacchi" and "আমি অফিস যাচ্ছি" express the same meaning — score them equally on semantic accuracy.
- IGNORE minor punctuation and capitalization differences.
- Focus on whether the PREDICTED captures what was said.

REFERENCE:
{reference_transcript}

PREDICTED:
{predicted_transcript}

Score on each dimension (integers 1-5, where 5 is best):

1. SEMANTIC_ACCURACY: Does the prediction convey the same meaning as the reference?
2. WORD_FAITHFULNESS: Are key content words (nouns, numbers, product names, technical terms) correctly captured?
3. FLUENCY: Is the prediction grammatically coherent and readable?

Then give an OVERALL score (1-10) and a one-line JUSTIFICATION.

Return ONLY valid JSON in this exact schema, no markdown fences, no preamble:
{
  "semantic_accuracy": <int 1-5>,
  "word_faithfulness": <int 1-5>,
  "fluency": <int 1-5>,
  "overall": <int 1-10>,
  "justification": "<one sentence>"
}
```

### 4.4 Judge output parsing
- Parse the JSON response. If parsing fails, retry once with `temperature=0`.
- If second attempt fails, log the raw response and mark that judgment as `null` in the output (do not crash the run).

---

## 5. Code Structure (GitHub-based modular approach)

Because execution is on Google Colab, the strategy is:
- **Modular Python code lives in a GitHub repo** that the user owns.
- The Colab notebook **clones the repo** in the first cell, then imports from it.
- This keeps the notebook short and lets the user version their code on GitHub.

### 5.1 GitHub repo placeholder
The notebook will reference this URL — user fills it in:
```
GITHUB_REPO_URL = "<<TODO: user fills in, e.g. https://github.com/Mainul/banglish-stt-benchmark.git>>"
```

### 5.2 Repository layout (to be created by user)

```
banglish-stt-benchmark/
├── pyproject.toml              # uv-managed dependencies (optional, for local dev)
├── requirements.txt            # for Colab pip install
├── README.md
├── src/
│   ├── __init__.py
│   ├── data_prep.py            # Download OpenSLR-104, sample 50 clips, build manifest
│   ├── transcribers/
│   │   ├── __init__.py
│   │   ├── base.py             # BaseTranscriber abstract class
│   │   ├── whisper_hf.py       # For models 1, 2, 3, 8
│   │   ├── nemo_conformer.py   # For models 4, 5, 6
│   │   ├── omni_asr.py         # For model 7
│   │   └── openrouter_stt.py   # For model 9 (Chirp 3)
│   ├── judge.py                # Gemini 3 Flash judging via OpenRouter
│   ├── analyze.py              # Aggregation + plots
│   └── utils.py                # Logging, JSON I/O helpers
└── config/
    ├── models.yaml             # Model registry (id, family, params)
    └── prompts.yaml            # Judge prompt template
```

### 5.3 requirements.txt (for Colab)
```
torch>=2.1
torchaudio
transformers>=4.45
accelerate
librosa
soundfile
pandas
pyyaml
matplotlib
seaborn
tqdm
requests
nemo_toolkit[asr]>=2.0
omnilingual-asr
openai  # used for OpenRouter (OpenAI-compatible API)
```

---

## 6. Notebook Structure (single Colab notebook)

The notebook is the **entry point** the user runs. It has these cells in order:

### Cell 1: Setup & repo clone
- Install system deps (`libsndfile1`).
- Clone the GitHub repo.
- `cd` into it, `pip install -r requirements.txt`.

### Cell 2: Secrets
- Prompt for OpenRouter API key (use `getpass` so it isn't logged).
- Store as env var.
- Optionally mount Google Drive for persistence.

### Cell 3: Data preparation
- Call `src/data_prep.py` functions to:
  1. Download OpenSLR-104 Bengali-English test set (if not already present).
  2. Extract.
  3. Parse transcripts.
  4. Compute length + code-switching density per clip.
  5. Stratified-sample 50 clips.
  6. Save `data/manifest.csv`.
- Print summary stats and 10 sample reference transcripts (so user can verify script).

### Cell 4: Per-model transcription
- For each model in the locked list of 9:
  1. Load the model.
  2. Iterate over the 50 clips, transcribe, record latency.
  3. Save predictions to `predictions/<model_id_slug>.csv` (columns: `clip_id`, `predicted_text`, `latency_sec`).
  4. Release GPU memory before loading the next model.
- If a model fails to load/run in ≤30 min, mark `FAILED` and continue.

### Cell 5: LLM judging
- For each (clip, model) where prediction exists:
  - Send to Gemini 3 Flash via OpenRouter.
  - Parse JSON response.
  - Store in `judgments.csv` (columns: `clip_id`, `model_id`, `semantic_accuracy`, `word_faithfulness`, `fluency`, `overall`, `justification`).

### Cell 6: Build the master JSON
Generate a single JSON file `results/benchmark_results.json` with this exact structure:

```json
{
  "metadata": {
    "benchmark_date": "2026-05-21",
    "dataset": "OpenSLR-104 Bengali-English Test (50 clips sampled)",
    "judge_model": "google/gemini-3-flash-preview",
    "models_evaluated": 9,
    "models_failed": []
  },
  "clips": [
    {
      "clip_id": "clip_001",
      "audio_path": "data/raw/.../audio_001.wav",
      "reference_transcript": "...",
      "length_sec": 12.4,
      "code_switch_density": 0.22,
      "predictions": {
        "openai/whisper-large-v3": {
          "text": "...",
          "latency_sec": 1.8,
          "judgment": {
            "semantic_accuracy": 4,
            "word_faithfulness": 4,
            "fluency": 5,
            "overall": 8,
            "justification": "Accurate meaning, minor word miss on a product name."
          }
        },
        "hishab/titu_stt_bn_conformer_large": {
          "text": "...",
          "latency_sec": 2.3,
          "judgment": {
            "semantic_accuracy": 5,
            "word_faithfulness": 5,
            "fluency": 5,
            "overall": 9,
            "justification": "Excellent on the Bangla portion, handled English words well."
          }
        }
        // ... all 9 models per clip
      }
    }
    // ... 50 clips
  ]
}
```

### Cell 7: Comparison table
Compute aggregate per-model stats and save as both:
- `results/comparison_table.csv`
- `results/comparison_table.png` (rendered as a styled matplotlib table)

Columns:
| Model | Avg Overall | Avg Semantic | Avg Word | Avg Fluency | Avg Latency (s/clip) | Latency per Audio-Sec | Status |
|---|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... | OK / FAILED |

Sort by `Avg Overall` descending.

### Cell 8: Analytical plots
Generate the following PNGs in `results/plots/`:

1. `01_overall_score_bar.png` — Bar chart, models on x-axis, average overall score on y-axis, error bars = std dev across clips.
2. `02_per_dimension_radar.png` — Radar/spider chart, one polygon per model, 3 axes (semantic, word, fluency).
3. `03_score_vs_latency_scatter.png` — Scatter plot: x=avg latency, y=avg overall score, one point per model, labeled. The "ideal" model sits top-left.
4. `04_score_by_length_bucket.png` — Grouped bar chart: how each model performs on short vs. medium vs. long clips.
5. `05_score_by_codeswitching_bucket.png` — Grouped bar chart: how each model performs on low/medium/high code-switching clips.
6. `06_heatmap_clip_vs_model.png` — Heatmap: rows=clips (sorted by avg score across models), cols=models, cell=overall score. Reveals which clips are universally hard.

Use a consistent color palette and clear titles/labels.

### Cell 9: Final report (markdown summary)
Write `results/report.md` containing:
- Title, date, summary stats.
- The comparison table (markdown formatted).
- Top 3 models with one-paragraph rationale each.
- Honest caveats (dataset is Indian Bengali, clean audio, single judge).
- Pointers to the JSON for full details.

### Cell 10: Download bundle (the "download reports and images cell")
- Zip everything in `results/` (the JSON, CSVs, all PNGs, report.md) into `banglish_stt_benchmark_results.zip`.
- Use `google.colab.files.download()` to trigger browser download.
- Print the zip's size and contents list so user knows what they're getting.

---

## 7. Execution Order

1. **User creates GitHub repo** with the layout in §5.2 (Claude Code can scaffold this).
2. **User pushes the modular code** to that repo.
3. **User opens the notebook on Colab**, fills in `GITHUB_REPO_URL`.
4. **User runs cells 1–10 in order**.
5. **Cell 10 downloads the final zip** to the user's machine.

---

## 8. Failure Modes & Mitigations

| Risk | Mitigation |
|---|---|
| NeMo install on Colab fails | Pin version; have a fallback `pip install nemo_toolkit[asr]==<<TODO: pin after first run>>` |
| OmniASR package not installable | Mark model as FAILED, document, continue |
| OpenRouter API returns non-JSON | Retry once; if still fails, mark judgment as null |
| Colab session times out mid-transcription | Save predictions after every clip (not at end); resumable: skip clips that already have a prediction in CSV |
| Big model OOM on T4 (free Colab) | Suggest Colab Pro or use CPU fallback for one model |
| Reference script ≠ predicted script biases judge | Judge prompt explicitly tells it to ignore script |
| Gemini 3 Flash output schema drifts | Validate JSON with a schema; one retry on failure |

---

## 9. Cost Estimate

| Item | Estimate |
|---|---|
| OpenRouter Chirp 3 (50 clips × ~20s avg = ~17 min audio) | ~$1–2 |
| OpenRouter Gemini 3 Flash (450 judge calls, ~500 in + ~150 out tokens each) | ~$0.40 |
| Colab Pro (optional) | $10/month |
| **Total** | **~$15 max** |

---

## 10. Out of Scope (Explicitly)

- Real-time / streaming evaluation
- Speaker diarization
- Fine-tuning any model
- Production deployment, latency optimization, quantization
- Statistical significance testing (n=50 is too small for that anyway)
- Phase 2 custom Bangladeshi dataset (deferred — add only if Phase 1 reveals a tight contest)

---

## 11. What the Final Deliverable Looks Like

A zip file containing:

```
banglish_stt_benchmark_results.zip
├── benchmark_results.json          # The single master JSON (§6 Cell 6)
├── comparison_table.csv
├── comparison_table.png
├── report.md
└── plots/
    ├── 01_overall_score_bar.png
    ├── 02_per_dimension_radar.png
    ├── 03_score_vs_latency_scatter.png
    ├── 04_score_by_length_bucket.png
    ├── 05_score_by_codeswitching_bucket.png
    └── 06_heatmap_clip_vs_model.png
```

---

## 12. Open Items the User Must Fill In

| Placeholder | Where | What to do |
|---|---|---|
| `GITHUB_REPO_URL` | Notebook Cell 1 | Create a GitHub repo with the §5.2 structure and put the clone URL here |
| `OPENROUTER_API_KEY` | Notebook Cell 2 | User pastes when prompted (uses `getpass`) |
| Reference transcript script verification | After Cell 3 | User looks at sample transcripts, confirms script, no code change needed |

---

## 13. Notes for the Coding Assistant Building This

- **Use `uv` for any local dev** (user preference). On Colab, plain `pip` is fine.
- **Use Python 3.11 syntax** if possible.
- **Type-hint everything**.
- **Logging:** Use Python `logging` module; INFO level by default. Save logs to `logs/run_<timestamp>.log`.
- **Reproducibility:** Set seeds where applicable (numpy, torch, random).
- **No `try: ... except: pass`** — always log the exception.
- **Resumability:** Every long-running step (transcription, judging) should check for existing partial output and resume.
- **Style:** Simple, well-organized code with short functions. Comments for non-obvious logic. Match the user's preference for plain, child-understandable explanations in comments.
- **Do not over-engineer.** This is a benchmark, not a production system. No fancy abstractions beyond what §3.1 already specifies.