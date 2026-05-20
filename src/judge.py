"""Gemini 3 Flash judge via OpenRouter — plan.md §4.

- One judge call per (clip, model) pair, temperature 0.
- Prompt comes verbatim from config/prompts.yaml (plan §4.3 wording is locked).
- JSON parsing with one retry. If both attempts fail, the row is written
  with status=JUDGE_NULL and the raw response is logged — we do NOT crash
  the whole run on a single bad judgment.
- Append-after-each-pair for resumability (DECISIONS.md §E).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import time
from pathlib import Path

import pandas as pd

from src.utils import read_yaml

logger = logging.getLogger(__name__)

JUDGE_MODEL = "google/gemini-3-flash-preview"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

JUDGMENT_COLUMNS = [
    "clip_id", "model_id",
    "semantic_accuracy", "word_faithfulness", "fluency",
    "overall", "justification", "status",
]


def load_judge_prompt(prompts_yaml: str | Path) -> str:
    return read_yaml(prompts_yaml)["judge_prompt"]


def _existing_pairs(csv_path: Path) -> set[tuple[str, str]]:
    """Return (clip_id, model_id) pairs already judged (used for resume)."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    df = pd.read_csv(csv_path)
    if "clip_id" not in df.columns or "model_id" not in df.columns:
        return set()
    return set(zip(df["clip_id"].astype(str), df["model_id"].astype(str)))


def _strip_code_fences(text: str) -> str:
    """Some judge models still wrap JSON in ```json fences despite the prompt."""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s.strip()


def _call_openrouter(client, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content or ""


def _judge_once(
    client,
    prompt_tpl: str,
    reference: str,
    predicted: str,
) -> tuple[dict | None, str]:
    """Return (parsed_dict_or_None, raw_response_of_last_attempt).

    Retries exactly once on JSONDecodeError per plan §4.4.
    """
    prompt = prompt_tpl.format(
        reference_transcript=reference,
        predicted_transcript=predicted,
    )

    raw = _call_openrouter(client, prompt)
    try:
        return json.loads(_strip_code_fences(raw)), raw
    except json.JSONDecodeError as e:
        logger.warning("Judge JSON parse failed (will retry once): %s | raw=%r", e, raw[:200])

    raw2 = _call_openrouter(client, prompt)
    try:
        return json.loads(_strip_code_fences(raw2)), raw2
    except json.JSONDecodeError as e:
        logger.error(
            "Judge JSON parse failed twice — marking null. err=%s | raw=%r",
            e, raw2[:500],
        )
        return None, raw2


def _append_judgment(csv_path: Path, row: dict, write_header: bool) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=JUDGMENT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in JUDGMENT_COLUMNS})


def judge_predictions(
    manifest_csv: str | Path,
    predictions_dir: str | Path,
    judgments_csv: str | Path,
    prompts_yaml: str | Path,
    api_key_env: str = "OPENROUTER_API_KEY",
    sleep_between_calls_sec: float = 0.0,
) -> Path:
    """Judge every (clip, model) prediction that hasn't been judged yet.

    Skips rows already in judgments_csv. Empty predicted_text still gets judged
    (the judge will score it low, but the score is informative).
    """
    from openai import OpenAI

    manifest_csv = Path(manifest_csv)
    predictions_dir = Path(predictions_dir)
    judgments_csv = Path(judgments_csv)
    prompts_yaml = Path(prompts_yaml)

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set in the environment")

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    prompt_tpl = load_judge_prompt(prompts_yaml)

    manifest = pd.read_csv(manifest_csv).set_index("clip_id")
    done_pairs = _existing_pairs(judgments_csv)
    write_header = not judgments_csv.exists() or judgments_csv.stat().st_size == 0

    pred_files = sorted(predictions_dir.glob("*.csv"))
    if not pred_files:
        raise RuntimeError(f"No prediction CSVs found in {predictions_dir}")

    for pred_file in pred_files:
        preds = pd.read_csv(pred_file)
        if preds.empty or "model_id" not in preds.columns:
            logger.warning("Skipping %s — empty or missing model_id column", pred_file)
            continue
        model_id = str(preds["model_id"].iloc[0])
        logger.info("--- Judging %s (%d rows) ---", model_id, len(preds))

        for _, row in preds.iterrows():
            clip_id = str(row["clip_id"])
            if (clip_id, model_id) in done_pairs:
                continue
            if clip_id not in manifest.index:
                logger.warning("clip=%s in predictions but not manifest — skipping", clip_id)
                continue

            reference = str(manifest.loc[clip_id, "reference_transcript"])
            predicted = str(row.get("predicted_text", "") or "")

            parsed, _raw = _judge_once(client, prompt_tpl, reference, predicted)

            if parsed is None:
                out_row = {
                    "clip_id": clip_id,
                    "model_id": model_id,
                    "semantic_accuracy": "",
                    "word_faithfulness": "",
                    "fluency": "",
                    "overall": "",
                    "justification": "",
                    "status": "JUDGE_NULL",
                }
            else:
                out_row = {
                    "clip_id": clip_id,
                    "model_id": model_id,
                    "semantic_accuracy": parsed.get("semantic_accuracy", ""),
                    "word_faithfulness": parsed.get("word_faithfulness", ""),
                    "fluency": parsed.get("fluency", ""),
                    "overall": parsed.get("overall", ""),
                    "justification": parsed.get("justification", ""),
                    "status": "OK",
                }

            _append_judgment(judgments_csv, out_row, write_header)
            write_header = False
            done_pairs.add((clip_id, model_id))
            logger.info(
                "  judged clip=%s model=%s overall=%s",
                clip_id, model_id, out_row["overall"] or "NULL",
            )

            if sleep_between_calls_sec > 0:
                time.sleep(sleep_between_calls_sec)

    logger.info("Judging complete -> %s", judgments_csv)
    return judgments_csv
