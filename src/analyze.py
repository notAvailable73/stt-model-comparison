"""Aggregation, plotting, report, and bundling — plan.md §6 Cells 6–10.

Public entry points used by the notebook:
- build_master_json
- build_comparison_table
- make_all_plots
- write_report_md
- bundle_results
- run_analysis      (orchestrates all of the above)
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import read_yaml, slugify_model_id, write_json

logger = logging.getLogger(__name__)

JUDGE_MODEL_ID = "google/gemini-3-flash-preview"

# Plot styling — applied lazily in make_all_plots so importing this module is cheap.
def _apply_style() -> None:
    sns.set_theme(context="notebook", style="whitegrid", palette="deep")


# ============================================================================
# §6 Cell 6 — master JSON
# ============================================================================

def build_master_json(
    manifest_csv: str | Path,
    predictions_dir: str | Path,
    judgments_csv: str | Path,
    models_yaml: str | Path,
    failed_models: list[dict] | None,
    output_json: str | Path,
    benchmark_date: str | None = None,
) -> Path:
    """Merge manifest + predictions + judgments into the §6 Cell 6 schema."""
    manifest = pd.read_csv(manifest_csv)
    judgments = (
        pd.read_csv(judgments_csv)
        if Path(judgments_csv).exists() and Path(judgments_csv).stat().st_size > 0
        else pd.DataFrame(columns=["clip_id", "model_id"])
    )
    models_cfg = read_yaml(models_yaml)
    expected_model_ids = [m["id"] for m in models_cfg["models"]]

    # Index judgments for O(1) lookup.
    if not judgments.empty:
        judgments_idx = judgments.set_index(["clip_id", "model_id"]).to_dict(orient="index")
    else:
        judgments_idx = {}

    # Load all prediction CSVs, key by model_id, then clip_id.
    predictions_by_model: dict[str, dict[str, dict]] = {}
    predictions_dir = Path(predictions_dir)
    for csv_path in sorted(predictions_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if df.empty or "model_id" not in df.columns:
            continue
        model_id = str(df["model_id"].iloc[0])
        predictions_by_model[model_id] = {
            str(r["clip_id"]): r.to_dict() for _, r in df.iterrows()
        }

    failed_ids = {f["model_id"] for f in (failed_models or [])}

    clips_out: list[dict] = []
    for _, row in manifest.iterrows():
        clip_id = str(row["clip_id"])
        clip_entry: dict = {
            "clip_id": clip_id,
            "audio_path": row["audio_path"],
            "reference_transcript": row["reference_transcript"],
            "length_sec": float(row["length_sec"]),
            "code_switch_density": float(row["code_switch_density"]),
            "predictions": {},
        }
        for model_id in expected_model_ids:
            if model_id in failed_ids:
                continue  # Failed models are listed in metadata, not per clip.
            pred = predictions_by_model.get(model_id, {}).get(clip_id)
            if pred is None:
                continue
            judgment = judgments_idx.get((clip_id, model_id))
            clip_entry["predictions"][model_id] = {
                "text": "" if pd.isna(pred.get("predicted_text")) else str(pred["predicted_text"]),
                "latency_sec": _to_float_or_none(pred.get("latency_sec")),
                "judgment": _judgment_payload(judgment),
            }
        clips_out.append(clip_entry)

    out = {
        "metadata": {
            "benchmark_date": benchmark_date or _dt.date.today().isoformat(),
            "dataset": "OpenSLR-104 Bengali-English Test (50 clips sampled)",
            "judge_model": JUDGE_MODEL_ID,
            "models_evaluated": len(expected_model_ids),
            "models_failed": failed_models or [],
        },
        "clips": clips_out,
    }
    write_json(output_json, out)
    logger.info("Wrote master JSON -> %s", output_json)
    return Path(output_json)


def _to_float_or_none(v) -> float | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _judgment_payload(j: dict | None) -> dict | None:
    if j is None or str(j.get("status", "OK")) == "JUDGE_NULL":
        return None
    out: dict = {}
    for k in ("semantic_accuracy", "word_faithfulness", "fluency", "overall"):
        v = j.get(k)
        if pd.isna(v):
            return None
        out[k] = int(v)
    out["justification"] = "" if pd.isna(j.get("justification")) else str(j.get("justification", ""))
    return out


# ============================================================================
# §6 Cell 7 — comparison table
# ============================================================================

def _aggregate_per_model(
    manifest: pd.DataFrame,
    predictions_by_model: dict[str, pd.DataFrame],
    judgments: pd.DataFrame,
    expected_model_ids: list[str],
    failed_ids: set[str],
) -> pd.DataFrame:
    """Per-model means over the 50 clips + RTF + status."""
    rows: list[dict] = []
    for model_id in expected_model_ids:
        if model_id in failed_ids:
            rows.append({
                "Model": model_id,
                "Avg Overall": np.nan,
                "Avg Semantic": np.nan,
                "Avg Word": np.nan,
                "Avg Fluency": np.nan,
                "Avg Latency (s/clip)": np.nan,
                "Latency per Audio-Sec": np.nan,
                "Status": "FAILED",
            })
            continue

        preds = predictions_by_model.get(model_id, pd.DataFrame())
        judged = judgments[judgments["model_id"] == model_id] if not judgments.empty else pd.DataFrame()

        # Latency
        if not preds.empty:
            avg_lat = pd.to_numeric(preds["latency_sec"], errors="coerce").mean()
            merged = preds.merge(
                manifest[["clip_id", "length_sec"]],
                on="clip_id", how="left",
            )
            merged["rtf"] = pd.to_numeric(merged["latency_sec"], errors="coerce") / merged["length_sec"]
            rtf = merged["rtf"].mean()
        else:
            avg_lat = np.nan
            rtf = np.nan

        # Scores
        if not judged.empty:
            for col in ("semantic_accuracy", "word_faithfulness", "fluency", "overall"):
                judged[col] = pd.to_numeric(judged[col], errors="coerce")
            row = {
                "Model": model_id,
                "Avg Overall": judged["overall"].mean(),
                "Avg Semantic": judged["semantic_accuracy"].mean(),
                "Avg Word": judged["word_faithfulness"].mean(),
                "Avg Fluency": judged["fluency"].mean(),
            }
        else:
            row = {
                "Model": model_id,
                "Avg Overall": np.nan,
                "Avg Semantic": np.nan,
                "Avg Word": np.nan,
                "Avg Fluency": np.nan,
            }
        row.update({
            "Avg Latency (s/clip)": avg_lat,
            "Latency per Audio-Sec": rtf,
            "Status": "OK",
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["Avg Overall", "Avg Latency (s/clip)"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
    return df


def build_comparison_table(
    manifest_csv: str | Path,
    predictions_dir: str | Path,
    judgments_csv: str | Path,
    models_yaml: str | Path,
    failed_models: list[dict] | None,
    out_csv: str | Path,
    out_png: str | Path,
) -> tuple[Path, Path]:
    manifest = pd.read_csv(manifest_csv)
    judgments = (
        pd.read_csv(judgments_csv)
        if Path(judgments_csv).exists() and Path(judgments_csv).stat().st_size > 0
        else pd.DataFrame()
    )
    predictions_by_model: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(Path(predictions_dir).glob("*.csv")):
        df = pd.read_csv(csv_path)
        if df.empty or "model_id" not in df.columns:
            continue
        predictions_by_model[str(df["model_id"].iloc[0])] = df

    models_cfg = read_yaml(models_yaml)
    expected_model_ids = [m["id"] for m in models_cfg["models"]]
    failed_ids = {f["model_id"] for f in (failed_models or [])}

    df = _aggregate_per_model(
        manifest, predictions_by_model, judgments, expected_model_ids, failed_ids
    )

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, float_format="%.3f")
    logger.info("Wrote comparison table CSV -> %s", out_csv)

    _render_table_png(df, Path(out_png))
    logger.info("Wrote comparison table PNG -> %s", out_png)
    return Path(out_csv), Path(out_png)


def _render_table_png(df: pd.DataFrame, out_path: Path) -> None:
    display = df.copy()
    for col in ("Avg Overall", "Avg Semantic", "Avg Word", "Avg Fluency", "Avg Latency (s/clip)"):
        display[col] = display[col].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    display["Latency per Audio-Sec"] = display["Latency per Audio-Sec"].apply(
        lambda v: f"{v:.3f}" if pd.notna(v) else "—"
    )

    fig, ax = plt.subplots(figsize=(14, 0.6 + 0.45 * len(display)))
    ax.axis("off")
    tbl = ax.table(
        cellText=display.values.tolist(),
        colLabels=display.columns.tolist(),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.4)
    for j in range(display.shape[1]):
        cell = tbl[0, j]
        cell.set_text_props(weight="bold", color="white")
        cell.set_facecolor("#3b4252")
    for i, status in enumerate(display["Status"].tolist(), start=1):
        face = "#fde2e2" if status == "FAILED" else ("#e8f0fe" if i % 2 == 0 else "white")
        for j in range(display.shape[1]):
            tbl[i, j].set_facecolor(face)
    ax.set_title("Banglish STT Benchmark — Comparison Table", pad=12, fontsize=12, weight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# §6 Cell 8 — six plots
# ============================================================================

def make_all_plots(
    manifest_csv: str | Path,
    predictions_dir: str | Path,
    judgments_csv: str | Path,
    models_yaml: str | Path,
    failed_models: list[dict] | None,
    plots_dir: str | Path,
) -> list[Path]:
    _apply_style()
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_csv)
    judgments = (
        pd.read_csv(judgments_csv)
        if Path(judgments_csv).exists() and Path(judgments_csv).stat().st_size > 0
        else pd.DataFrame()
    )
    if judgments.empty:
        logger.warning("No judgments found — plots will be sparse")

    predictions_by_model: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(Path(predictions_dir).glob("*.csv")):
        df = pd.read_csv(csv_path)
        if df.empty or "model_id" not in df.columns:
            continue
        predictions_by_model[str(df["model_id"].iloc[0])] = df

    models_cfg = read_yaml(models_yaml)
    expected_model_ids = [m["id"] for m in models_cfg["models"]]
    failed_ids = {f["model_id"] for f in (failed_models or [])}
    ok_models = [m for m in expected_model_ids if m not in failed_ids]

    # Numeric coerce on judgments once.
    for col in ("semantic_accuracy", "word_faithfulness", "fluency", "overall"):
        if col in judgments.columns:
            judgments[col] = pd.to_numeric(judgments[col], errors="coerce")

    # Merge clip metadata into judgments for bucket-aware plots.
    if not judgments.empty:
        judgments_meta = judgments.merge(
            manifest[["clip_id", "length_bucket", "cs_bucket", "length_sec"]],
            on="clip_id", how="left",
        )
    else:
        judgments_meta = judgments

    paths: list[Path] = []
    paths.append(_plot_01_overall_bar(judgments_meta, ok_models, plots_dir / "01_overall_score_bar.png"))
    paths.append(_plot_02_radar(judgments_meta, ok_models, plots_dir / "02_per_dimension_radar.png"))
    paths.append(_plot_03_scatter(predictions_by_model, manifest, judgments_meta, ok_models, plots_dir / "03_score_vs_latency_scatter.png"))
    paths.append(_plot_04_by_length(judgments_meta, ok_models, plots_dir / "04_score_by_length_bucket.png"))
    paths.append(_plot_05_by_cs(judgments_meta, ok_models, plots_dir / "05_score_by_codeswitching_bucket.png"))
    paths.append(_plot_06_heatmap(judgments_meta, ok_models, plots_dir / "06_heatmap_clip_vs_model.png"))
    return paths


def _short(model_id: str) -> str:
    """Compact label for plot axes."""
    return model_id.split("/")[-1]


def _plot_01_overall_bar(judgments: pd.DataFrame, models: list[str], out: Path) -> Path:
    if judgments.empty:
        return _write_empty_plot(out, "Overall score by model (no data)")
    agg = (
        judgments[judgments["model_id"].isin(models)]
        .groupby("model_id")["overall"]
        .agg(["mean", "std"])
        .reindex(models)
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(agg))
    ax.bar(x, agg["mean"], yerr=agg["std"].fillna(0), capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([_short(m) for m in agg.index], rotation=30, ha="right")
    ax.set_ylabel("Overall score (1–10)")
    ax.set_ylim(0, 10)
    ax.set_title("Average Overall Score by Model (error bars = std dev across clips)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_02_radar(judgments: pd.DataFrame, models: list[str], out: Path) -> Path:
    if judgments.empty:
        return _write_empty_plot(out, "Per-dimension radar (no data)")
    dims = ["semantic_accuracy", "word_faithfulness", "fluency"]
    means = (
        judgments[judgments["model_id"].isin(models)]
        .groupby("model_id")[dims]
        .mean()
        .reindex(models)
    )
    angles = [n / len(dims) * 2 * math.pi for n in range(len(dims))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for model in means.index:
        vals = means.loc[model].tolist()
        if any(pd.isna(v) for v in vals):
            continue
        vals += vals[:1]
        ax.plot(angles, vals, label=_short(model), linewidth=1.6)
        ax.fill(angles, vals, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["Semantic", "Word", "Fluency"])
    ax.set_ylim(0, 5)
    ax.set_title("Per-Dimension Mean Scores (1–5)", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_03_scatter(
    predictions_by_model: dict[str, pd.DataFrame],
    manifest: pd.DataFrame,
    judgments: pd.DataFrame,
    models: list[str],
    out: Path,
) -> Path:
    if judgments.empty:
        return _write_empty_plot(out, "Score vs latency (no data)")
    points: list[tuple[str, float, float]] = []
    for m in models:
        preds = predictions_by_model.get(m)
        if preds is None or preds.empty:
            continue
        lat = pd.to_numeric(preds["latency_sec"], errors="coerce").mean()
        score = judgments[judgments["model_id"] == m]["overall"].mean()
        if pd.notna(lat) and pd.notna(score):
            points.append((m, lat, score))
    if not points:
        return _write_empty_plot(out, "Score vs latency (no data)")

    fig, ax = plt.subplots(figsize=(10, 6))
    for m, lat, score in points:
        ax.scatter(lat, score, s=80)
        ax.annotate(_short(m), (lat, score), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Avg latency per clip (s) — lower is better")
    ax.set_ylabel("Avg overall score (1–10) — higher is better")
    ax.set_ylim(0, 10)
    ax.set_title("Score vs Latency  (ideal model: top-left)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_grouped_bar_by_bucket(
    judgments: pd.DataFrame, models: list[str], bucket_col: str,
    bucket_order: list[str], title: str, out: Path,
) -> Path:
    if judgments.empty:
        return _write_empty_plot(out, f"{title} (no data)")
    sub = judgments[judgments["model_id"].isin(models)]
    pivot = (
        sub.groupby(["model_id", bucket_col])["overall"]
        .mean()
        .unstack(bucket_col)
        .reindex(models)
        .reindex(columns=bucket_order)
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.25
    x = np.arange(len(pivot.index))
    for i, bucket in enumerate(bucket_order):
        ax.bar(x + i * width, pivot[bucket].values, width=width, label=bucket)
    ax.set_xticks(x + width * (len(bucket_order) - 1) / 2)
    ax.set_xticklabels([_short(m) for m in pivot.index], rotation=30, ha="right")
    ax.set_ylabel("Avg overall score (1–10)")
    ax.set_ylim(0, 10)
    ax.set_title(title)
    ax.legend(title=bucket_col)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_04_by_length(judgments: pd.DataFrame, models: list[str], out: Path) -> Path:
    return _plot_grouped_bar_by_bucket(
        judgments, models, "length_bucket",
        ["short", "medium", "long"],
        "Score by Clip Length Bucket", out,
    )


def _plot_05_by_cs(judgments: pd.DataFrame, models: list[str], out: Path) -> Path:
    return _plot_grouped_bar_by_bucket(
        judgments, models, "cs_bucket",
        ["low", "medium", "high"],
        "Score by Code-Switching Density Bucket", out,
    )


def _plot_06_heatmap(judgments: pd.DataFrame, models: list[str], out: Path) -> Path:
    if judgments.empty:
        return _write_empty_plot(out, "Clip × Model heatmap (no data)")
    sub = judgments[judgments["model_id"].isin(models)]
    pivot = sub.pivot_table(index="clip_id", columns="model_id", values="overall", aggfunc="mean")
    pivot = pivot.reindex(columns=models)
    pivot["__row_mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("__row_mean", ascending=False).drop(columns="__row_mean")

    fig, ax = plt.subplots(figsize=(11, max(8, 0.18 * len(pivot))))
    sns.heatmap(
        pivot, ax=ax, cmap="RdYlGn", vmin=1, vmax=10,
        cbar_kws={"label": "Overall (1–10)"},
        xticklabels=[_short(m) for m in pivot.columns],
        yticklabels=pivot.index,
    )
    ax.set_title("Per-Clip Score Heatmap (clips sorted by mean across models)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Clip")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _write_empty_plot(path: Path, msg: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ============================================================================
# §6 Cell 9 — markdown report
# ============================================================================

CAVEATS_MD = """\
## Honest caveats

- The dataset is **OpenSLR-104**, Indian Bengali educational tutorials read in
  studio-quality audio. Production WhatsApp/Messenger voice notes are
  Bangladeshi, casual, and noisy. Results indicate model *capability*, not
  guaranteed production performance.
- N=50 clips is too small for statistical significance testing.
- Judge is a single LLM (Gemini 3 Flash). No human spot-check.
- Code-switching density is computed from the reference transcript; if the
  reference is in pure Bangla script, the metric will under-count English use.
"""


def write_report_md(
    comparison_csv: str | Path,
    master_json: str | Path,
    out_md: str | Path,
    benchmark_date: str | None = None,
) -> Path:
    import json

    df = pd.read_csv(comparison_csv)
    with open(master_json, encoding="utf-8") as f:
        master = json.load(f)

    date = benchmark_date or master.get("metadata", {}).get("benchmark_date", _dt.date.today().isoformat())
    n_models = master.get("metadata", {}).get("models_evaluated", len(df))
    n_failed = len(master.get("metadata", {}).get("models_failed", []))

    ok_rows = df[df["Status"] == "OK"].head(3)
    top_section_lines: list[str] = []
    for i, (_, r) in enumerate(ok_rows.iterrows(), start=1):
        rtf = r["Latency per Audio-Sec"]
        rtf_str = f"{rtf:.3f}" if pd.notna(rtf) else "—"
        top_section_lines.append(
            f"**{i}. {r['Model']}** — avg overall **{r['Avg Overall']:.2f}/10**, "
            f"avg latency {r['Avg Latency (s/clip)']:.2f}s/clip (RTF {rtf_str}). "
            f"Semantic {r['Avg Semantic']:.2f}, Word {r['Avg Word']:.2f}, "
            f"Fluency {r['Avg Fluency']:.2f}."
        )
    if not top_section_lines:
        top_section_lines.append("_No model produced judged predictions._")

    md = [
        "# Banglish STT Benchmark Report",
        "",
        f"- **Date:** {date}",
        f"- **Dataset:** OpenSLR-104 Bengali-English Test (50 clips, stratified)",
        f"- **Judge:** {master.get('metadata', {}).get('judge_model', JUDGE_MODEL_ID)}",
        f"- **Models evaluated:** {n_models}  (**failed:** {n_failed})",
        "",
        "## Comparison table",
        "",
        df.to_markdown(index=False, floatfmt=".2f"),
        "",
        "## Top 3 models",
        "",
        *top_section_lines,
        "",
        CAVEATS_MD,
        "## Where to find everything",
        "",
        "- Full per-clip results: `benchmark_results.json`",
        "- Plots: `plots/`",
        "- Comparison: `comparison_table.csv` / `.png`",
        "",
    ]
    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md), encoding="utf-8")
    logger.info("Wrote report -> %s", out_md)
    return out_md


# ============================================================================
# §6 Cell 10 — bundle
# ============================================================================

def bundle_results(results_dir: str | Path, zip_path: str | Path) -> Path:
    results_dir = Path(results_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(results_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(results_dir))
    size_mb = zip_path.stat().st_size / (1 << 20)
    logger.info("Wrote zip bundle %s (%.2f MB)", zip_path, size_mb)
    return zip_path


# ============================================================================
# Orchestrator (used by notebook Cell 7–10)
# ============================================================================

def run_analysis(
    manifest_csv: str | Path = "data/manifest.csv",
    predictions_dir: str | Path = "predictions",
    judgments_csv: str | Path = "judgments.csv",
    models_yaml: str | Path = "config/models.yaml",
    results_dir: str | Path = "results",
    failed_models: list[dict] | None = None,
    benchmark_date: str | None = None,
) -> dict[str, Path]:
    """One-shot orchestration: master JSON, table, plots, report, zip.

    Returns a dict of {name: path} for things the notebook may want to show.
    """
    results_dir = Path(results_dir)
    plots_dir = results_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)

    master_json = build_master_json(
        manifest_csv=manifest_csv,
        predictions_dir=predictions_dir,
        judgments_csv=judgments_csv,
        models_yaml=models_yaml,
        failed_models=failed_models,
        output_json=results_dir / "benchmark_results.json",
        benchmark_date=benchmark_date,
    )
    comp_csv, comp_png = build_comparison_table(
        manifest_csv=manifest_csv,
        predictions_dir=predictions_dir,
        judgments_csv=judgments_csv,
        models_yaml=models_yaml,
        failed_models=failed_models,
        out_csv=results_dir / "comparison_table.csv",
        out_png=results_dir / "comparison_table.png",
    )
    plot_paths = make_all_plots(
        manifest_csv=manifest_csv,
        predictions_dir=predictions_dir,
        judgments_csv=judgments_csv,
        models_yaml=models_yaml,
        failed_models=failed_models,
        plots_dir=plots_dir,
    )
    report_md = write_report_md(
        comparison_csv=comp_csv,
        master_json=master_json,
        out_md=results_dir / "report.md",
        benchmark_date=benchmark_date,
    )
    zip_path = bundle_results(results_dir, "banglish_stt_benchmark_results.zip")

    return {
        "master_json": master_json,
        "comparison_csv": comp_csv,
        "comparison_png": comp_png,
        "plots": plots_dir,
        "report_md": report_md,
        "zip": zip_path,
        **{f"plot_{i+1}": p for i, p in enumerate(plot_paths)},
    }
