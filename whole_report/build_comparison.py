"""Produce cross-model comparison artefacts for whole_report/.

Reads the four per-model `results/` and `results_gradual_training/` trees
and writes:
  - A combined CSV of headline + gradual-peak numbers (`comparison_table.csv`).
  - Four PNGs in whole_report/images/ used by main.tex / presentation.tex:
      compare_models_pooled_auc.png         main-pipeline pooled AUC vs horizon
      compare_models_peak_auc.png           gradual peak AUC vs horizon
      compare_models_balacc.png             main-pipeline balanced accuracy
      compare_models_brier.png              main-pipeline pooled Brier
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = Path(__file__).resolve().parent / "images"
IMG_DIR.mkdir(exist_ok=True)

FAMILIES = [
    ("Random Forest",       "Random_Forest"),
    ("XGBoost",             "XGBoost"),
    ("Logistic Regression", "LogisticRegression"),
    ("SVM-RBF",             "SVM"),
]
COLORS = {
    "Random Forest":       "#3b6ea8",
    "XGBoost":             "#cc6633",
    "Logistic Regression": "#1f7a1f",
    "SVM-RBF":             "#7b3a8a",
}
HORIZONS = [f"Within{k}Yr" for k in range(1, 8)]


def _load_main(model_dir: str, outcome: str) -> dict:
    with open(ROOT / model_dir / "results" / outcome / "metrics.json") as f:
        return json.load(f)


def _load_peak(model_dir: str, outcome: str) -> dict:
    p = ROOT / model_dir / "results_gradual_training" / f"{outcome}_results" / "peak_summary.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


# ---------- table ---------- #
rows = []
for label, mdir in FAMILIES:
    for outcome in HORIZONS:
        m = _load_main(mdir, outcome)
        pooled = m["pooled_metrics_threshold_0.5"]
        agg = m["fold_aggregate_mean_std"]
        peak = _load_peak(mdir, outcome).get("peak_per_metric", {})

        rows.append({
            "model": label,
            "outcome": outcome,
            "n_features_main": m["n_selected_features_final"],
            "pooled_auc": pooled["roc_auc"],
            "pooled_ap": pooled["average_precision"],
            "pooled_brier": pooled["brier_score"],
            "fold_auc_mean": agg["roc_auc"]["mean"],
            "fold_auc_std": agg["roc_auc"]["std"],
            "fold_balacc_mean": agg["balanced_accuracy"]["mean"],
            "fold_balacc_std": agg["balanced_accuracy"]["std"],
            "fold_f1_mean": agg["f1"]["mean"],
            "youden_threshold": m["youden_threshold"],
            "peak_auc_value": peak.get("roc_auc", {}).get("best_value_mean", np.nan),
            "peak_auc_k":     peak.get("roc_auc", {}).get("best_n_features", np.nan),
            "peak_balacc_value": peak.get("balanced_accuracy", {}).get("best_value_mean", np.nan),
            "peak_balacc_k":     peak.get("balanced_accuracy", {}).get("best_n_features", np.nan),
            "peak_f1_value":  peak.get("f1", {}).get("best_value_mean", np.nan),
            "peak_f1_k":      peak.get("f1", {}).get("best_n_features", np.nan),
        })

df = pd.DataFrame(rows)
csv_path = Path(__file__).resolve().parent / "comparison_table.csv"
df.to_csv(csv_path, index=False)
print(f"Wrote {csv_path}")


# ---------- shared helpers ---------- #
def _line_plot(metric_col: str, ylabel: str, title: str, fname: str, ylim=None):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.arange(len(HORIZONS))
    for label, _ in FAMILIES:
        sub = df[df["model"] == label].set_index("outcome").reindex(HORIZONS)
        ax.plot(x, sub[metric_col].values, "o-", lw=2, color=COLORS[label], label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(HORIZONS, rotation=20)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = IMG_DIR / fname
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")


def _bar_plot(metric_col: str, ylabel: str, title: str, fname: str, ylim=None, error_col=None):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(HORIZONS))
    width = 0.20
    for i, (label, _) in enumerate(FAMILIES):
        sub = df[df["model"] == label].set_index("outcome").reindex(HORIZONS)
        offset = (i - 1.5) * width
        if error_col is not None:
            ax.bar(x + offset, sub[metric_col].values, width,
                   yerr=sub[error_col].values, color=COLORS[label],
                   ecolor="grey", capsize=2, label=label, alpha=0.85)
        else:
            ax.bar(x + offset, sub[metric_col].values, width,
                   color=COLORS[label], label=label, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(HORIZONS, rotation=20)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = IMG_DIR / fname
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------- main-pipeline pooled AUC ---------- #
_line_plot(
    "pooled_auc",
    "Pooled OOF ROC-AUC",
    "Main pipeline: pooled OOF AUC by model and horizon (15-fold nested CV)",
    "compare_models_pooled_auc.png",
    ylim=(0.5, 0.8),
)

# ---------- gradual peak AUC ---------- #
_line_plot(
    "peak_auc_value",
    "Gradual-sweep peak per-fold mean AUC",
    "Gradual sweep: peak per-fold mean AUC by model and horizon",
    "compare_models_peak_auc.png",
    ylim=(0.55, 1.02),
)

# ---------- balanced accuracy bar (main pipeline, with SD) ---------- #
_bar_plot(
    "fold_balacc_mean",
    "Per-fold balanced accuracy (mean $\\pm$ SD)",
    "Main pipeline: balanced accuracy (15 outer folds) by model and horizon",
    "compare_models_balacc.png",
    ylim=(0.4, 0.85),
    error_col="fold_balacc_std",
)

# ---------- pooled Brier (main pipeline) ---------- #
_line_plot(
    "pooled_brier",
    "Pooled OOF Brier score (lower is better)",
    "Main pipeline: pooled OOF Brier by model and horizon",
    "compare_models_brier.png",
    ylim=(0.10, 0.30),
)

# ---------- gradual peak vs main pooled gap (illustrates leakage caveat) -- #
fig, ax = plt.subplots(figsize=(8.0, 4.8))
x = np.arange(len(HORIZONS))
width = 0.20
for i, (label, _) in enumerate(FAMILIES):
    sub = df[df["model"] == label].set_index("outcome").reindex(HORIZONS)
    gap = sub["peak_auc_value"].values - sub["pooled_auc"].values
    offset = (i - 1.5) * width
    ax.bar(x + offset, gap, width, color=COLORS[label], label=label, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(HORIZONS, rotation=20)
ax.set_ylabel("Peak AUC (gradual)  −  Pooled AUC (main)")
ax.set_title("Gradual peak − main-pipeline AUC gap by model and horizon")
ax.axhline(0, color="grey", lw=0.8)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="best", fontsize=9)
fig.tight_layout()
out = IMG_DIR / "compare_models_peak_minus_pooled.png"
fig.savefig(out, dpi=200)
plt.close(fig)
print(f"Wrote {out}")
