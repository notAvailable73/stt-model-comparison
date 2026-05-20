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
