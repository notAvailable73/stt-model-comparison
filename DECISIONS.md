# Decisions Log

Locked-in choices for this benchmark that are NOT spelled out in `plan.md`.
Read this if you come back to the project in 3 months and want to know "why
does the code do X." The plan is the spec; this file is the addendum.

## A. Stratified sampling

- 3 length buckets × 3 code-switching density buckets = 9 strata.
- Target ~5–6 clips per stratum to reach 50 total.
- If a stratum is underpopulated in the source set, fill the shortfall from an
  **adjacent** stratum, where adjacent means: same length bucket + neighboring
  CS bucket, OR same CS bucket + neighboring length bucket.
- The final per-stratum counts are logged so the actual distribution is
  visible in the manifest summary.

## B. Model 4 decoder (`ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large`)

- Default decoder: **RNN-T** (usually higher accuracy on hybrid Conformers).
- After the first 5 clips, compare RNN-T median latency to a one-clip CTC
  probe. If RNN-T is more than **~3× slower** than CTC, log a warning and
  switch to CTC for the remaining 45 clips.
- The chosen decoder for each clip is recorded in `predictions/<slug>.csv`
  under a `decoder_variant` column. The column is null for all other models.

### Why AI4Bharat's NeMo fork (not stock nemo_toolkit)

- Stock `nemo_toolkit[asr]` doesn't ship the multi-softmax aggregate
  tokenizer or the `language_id` transcribe kwarg that the indicconformer
  model requires. Loading the model against stock NeMo fails at the
  tokenizer-setup stage.
- Reference: AI4Bharat publishes the patched fork at
  `https://github.com/AI4Bharat/NeMo` on the `nemo-v2` branch. Their
  `reinstall.sh` installs it in editable mode under the same
  `nemo_toolkit` package name, so the hishab Conformer models (5 and 6)
  also load on top without any extra install.
- Cell 1 of the notebook runs `git clone + bash reinstall.sh` for this
  fork. This adds ~5–10 min to first-time setup, cached thereafter.
- The adapter passes `language_id="bn"` and (when in CTC mode for the
  hybrid model) `logprobs=False` for AI4Bharat models only. Hishab models
  reject these kwargs, so the adapter sends them only when `language_id`
  is set in `config/models.yaml`.

## C. Chirp 3 via OpenRouter (model 9)

- Use the `openai` Python SDK pointed at `https://openrouter.ai/api/v1`.
- Call `client.audio.transcriptions.create(model="google/chirp-3", file=...)`.
- If OpenRouter rejects the endpoint or model at runtime, the adapter raises
  and the model is marked `FAILED` per §3.2. **Do not** fall back to a
  different model, endpoint, or chat-completions hack.

## D. OmniASR (model 7, `facebook/omniASR-CTC-300M`)

- The model ID string is used **exactly as written in the plan**. No
  substitution, no "closest match" lookup on the HF Hub.
- If the repo doesn't resolve or the `omnilingual-asr` package can't load it,
  the model is marked `FAILED`. That's the correct behavior — the plan locks
  the model list.

### Realized failure on the first Colab run (2026-05-21)

- `omnilingual-asr` installs cleanly, but its dep `fairseq2n` ships only a
  CUDA-13 wheel: `libcudart.so.13: cannot open shared object file`.
- Colab provides CUDA 12.8. There is no `fairseq2n` wheel built for
  CUDA 12 at the version pinned by omnilingual-asr, and we are not
  pinning model versions ourselves (the plan locks model IDs, not their
  dep tree).
- Per the failure policy this run reports model 7 as FAILED in
  `metadata.models_failed`. The remaining 8 models still run.
- A future runtime with CUDA 13 (or a fairseq2n CUDA-12 wheel published
  upstream) would let this model run without any code change.

## E. Resumability granularity

- Transcription: per-model CSV is flushed **after every clip** (not at end).
  On resume, clips already present in the CSV are skipped.
- Judging: `judgments.csv` is appended **after every judged (clip, model)
  pair**. On resume, pairs already present are skipped.
- This is intentionally slower than batched writes. Colab dying mid-run and
  losing 30 minutes of work is the failure mode we are paying to avoid.

## F. Comparison-table latency columns

- `Avg Latency (s/clip)` — mean of `latency_sec` across the 50 clips (plan spec).
- `Latency per Audio-Sec` — real-time factor (RTF): mean over clips of
  `latency_sec / length_sec`. Lower is better. Reported to 3 decimal places.
- Both columns appear in `results/comparison_table.csv` and the PNG.

## G. No `pyproject.toml`

- `requirements.txt` is the only dependency manifest. The plan marks
  `pyproject.toml` optional; we skip it to keep the file tree tight.

## H. Why this file exists

- The plan says "do not over-engineer" and "no fancy abstractions beyond §3.1".
- The defaults above (sampling spillover, decoder auto-switch, etc.) are
  judgment calls that aren't visible in the code without context. This file
  is the context.

## I. Per-model Kaggle notebooks (replaces single Colab notebook)

`plan.md` §6 specifies one Colab notebook that runs every model. On Kaggle
this turns out to be unworkable — the 9 models' dep trees do not share a
solvable Python environment. We split into one notebook per model under
`notebooks/`, each owning its install profile, all writing the same
`predictions/<slug>.csv` schema. A separate `notebooks/10_judge_and_analyze.ipynb`
consumes the 9 CSVs (uploaded by the user) and runs the existing judge +
analysis pipeline unchanged.

### Why the single-notebook approach failed

- **NeMo fork pins `huggingface_hub==0.23.2`**, but stock Kaggle
  `transformers>=5`, `datasets>=4.x`, `diffusers>=0.34`, and `openai>=2.x`
  all require `hf_hub>=0.25` or `>=1.5`. Impossible to satisfy together.
- **Pip's new resolver thrashes** on the conflict — observed >1h
  backtracking through every published version of `transformers` and
  `datasets`.
- **`tensorstore<0.1.46`** (NeMo's pin) has no cp312 wheel. Falls back to
  a Bazel source build that takes 30–60+ min on Kaggle CPU.
- **Editable install race.** `pip install -e` writes a `.pth` file that
  Python reads only at interpreter startup. Kaggle's "Restart & Clear"
  resets the kernel before the new `.pth` is picked up → `import nemo`
  fails even though `pip show nemo_toolkit` reports it installed.

### NeMo notebook install recipe (notebooks 4, 5, 6)

- **Python 3.11**, not 3.12. Kaggle: Settings → Environment → "Pin to
  original environment", or pick a 3.11 image. Most cp312 wheel gaps
  vanish on 3.11.
- Pre-uninstall `transformers / datasets / diffusers / huggingface_hub
  / tokenizers` before installing NeMo (removes the resolver's
  conflict seeds).
- `pip install "pip<24.2"` (faster legacy resolver available).
- Patch `tensorstore<0.1.46` → unconstrained, pre-install
  `tensorstore>=0.1.71`.
- Run `PIP_USE_DEPRECATED=legacy-resolver bash reinstall.sh`.
- Then `sys.path.append("/kaggle/working/NeMo")` in the kernel that
  needs to import nemo — survives the kernel-restart race above.

The recipe is baked into the Cell 3 of every NeMo notebook; the
generator at `scripts/build_notebooks.py` is the source of truth.

### Manifest sharing across notebooks

Each notebook calls `build_manifest(seed=42)` independently.
`src/data_prep.py` produces an identical clip set given the same source
archive, so all 10 notebooks see the same 50 clips with the same
`clip_id`s — no manual dataset upload step for the manifest. The cost is
that each notebook downloads OpenSLR-104 (606 MB) on first run.

### Predictions flow

Each model notebook ends with an `IPython.display.FileLink` to its
`predictions/<slug>.csv`. The user downloads all 9 locally, then uploads
them as a Kaggle dataset attached to notebook 10. Notebook 10 aggregates
them into its local `predictions/` and runs `judge_predictions` +
`build_master_json` + `build_comparison_table` + `make_all_plots` +
`write_report_md` + `bundle_results` (all unchanged from §6).

### Why `run_single_model` was extracted

`src/transcribe.py` previously only exposed `run_transcription`, which
loops over every model in `config/models.yaml`. Per-model notebooks need
to run exactly one model from the same config. The inner per-model loop
was extracted into a public `run_single_model(spec, ...)` and
`run_transcription` is now a thin wrapper that calls it for each spec.
No behaviour change for callers of `run_transcription`.
