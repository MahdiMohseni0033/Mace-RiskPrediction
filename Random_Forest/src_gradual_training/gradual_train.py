"""Gradual-training sweep over the published feature ranking.

For each outcome configured in `config.py`, this script:

  1. Reads `final_selected_features.csv` produced by the main pipeline
     (sorted by Random-Forest impurity importance).
  2. For k = 1, 2, ..., N — where N is the size of that ranking — fits a
     Random-Forest classifier under the *same* nested cross-validation
     protocol as the main pipeline, but using only the top-k features.
  3. Records every classification metric per outer fold and pooled across
     out-of-fold predictions, plus the inner-CV chosen hyperparameters.
  4. Writes per-k metrics, all OOF predictions, and a rich plot suite that
     answers: how many features does the model actually need?

Run:
    python Random_Forest/src_gradual_training/gradual_train.py
    python Random_Forest/src_gradual_training/gradual_train.py --outcomes Within3Yr
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)

# Local config (src_gradual_training/config.py) — must be imported BEFORE we
# add the main `src/` directory to sys.path, since both folders contain a
# `config.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    ALL_OUTCOMES,
    DATA_PATH,
    HYPERPARAM_STRATEGY,
    ID_COLUMN,
    INNER_N_SPLITS,
    INNER_SCORING,
    K_MAX,
    K_MIN,
    K_STEP,
    MAIN_RESULTS_DIR,
    N_JOBS,
    OUTCOMES_TO_RUN,
    OUTER_N_REPEATS,
    OUTER_N_SPLITS,
    RANDOM_STATE,
    RESULTS_DIR,
    RF_PARAM_GRID,
    VERBOSE,
)

# Reuse the main pipeline's metric helpers — guarantees identical definitions
# across the two analyses.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics_utils import compute_metrics, write_json, youden_threshold  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Metrics tracked at every k.
METRIC_COLUMNS = [
    "roc_auc",
    "average_precision",
    "balanced_accuracy",
    "accuracy",
    "f1",
    "precision",
    "recall",
    "sensitivity",
    "specificity",
    "brier_score",
]
HIGHER_IS_BETTER = {m: True for m in METRIC_COLUMNS}
HIGHER_IS_BETTER["brier_score"] = False  # Brier: lower is better

PRETTY_LABEL = {
    "roc_auc": "ROC AUC",
    "average_precision": "Average Precision",
    "balanced_accuracy": "Balanced Accuracy",
    "accuracy": "Accuracy",
    "f1": "F1 Score",
    "precision": "Precision",
    "recall": "Recall",
    "sensitivity": "Sensitivity (TPR)",
    "specificity": "Specificity (TNR)",
    "brier_score": "Brier Score",
}


# --------------------------------------------------------------------------- #
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if ID_COLUMN not in df.columns:
        raise ValueError(f"ID column '{ID_COLUMN}' not in dataset")
    return df


def load_ranked_features(outcome: str) -> pd.DataFrame:
    path = MAIN_RESULTS_DIR / outcome / "final_selected_features.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find '{path}'. Run Random_Forest/src/train_rf.py first."
        )
    df = pd.read_csv(path).sort_values("rf_importance", ascending=False).reset_index(drop=True)
    return df


def load_main_best_params(outcome: str) -> dict:
    """Most-frequent best hyperparameters from the main pipeline."""
    path = MAIN_RESULTS_DIR / outcome / "best_hyperparameters.json"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find '{path}'.")
    with open(path) as f:
        records = json.load(f)
    counter = Counter(tuple(sorted(r["best_params"].items())) for r in records)
    return dict(counter.most_common(1)[0][0])


def fit_one_k(
    X_k: np.ndarray, y: np.ndarray, outer, fixed_best_params: dict | None
):
    """Run the outer CV at a single feature-count k.

    Returns per-fold metric records, list of (y_true, y_prob) for each fold,
    pooled metrics, and the inner-CV best params per fold.
    """
    fold_records = []
    per_fold_probs = []
    per_fold_best_params = []
    cv_predictions = []

    for fold_idx, (tr, te) in enumerate(outer.split(X_k, y)):
        fold_seed = RANDOM_STATE + fold_idx
        Xtr, Xte = X_k[tr], X_k[te]
        ytr, yte = y[tr], y[te]

        if HYPERPARAM_STRATEGY == "per_fold_grid":
            inner = StratifiedKFold(
                n_splits=INNER_N_SPLITS, shuffle=True, random_state=fold_seed
            )
            base_rf = RandomForestClassifier(
                class_weight="balanced", random_state=fold_seed, n_jobs=1
            )
            gs = GridSearchCV(
                base_rf, RF_PARAM_GRID, scoring=INNER_SCORING, cv=inner,
                n_jobs=N_JOBS, refit=True, error_score=np.nan,
            )
            gs.fit(Xtr, ytr)
            model = gs.best_estimator_
            best_params = gs.best_params_
        elif HYPERPARAM_STRATEGY == "fixed_best":
            model = RandomForestClassifier(
                class_weight="balanced",
                random_state=fold_seed,
                n_jobs=N_JOBS,
                **fixed_best_params,
            )
            model.fit(Xtr, ytr)
            best_params = fixed_best_params
        else:
            raise ValueError(f"Unknown HYPERPARAM_STRATEGY={HYPERPARAM_STRATEGY!r}")

        y_prob = model.predict_proba(Xte)[:, 1]
        m = compute_metrics(yte, y_prob, threshold=0.5)
        m["fold"] = fold_idx
        fold_records.append(m)
        per_fold_probs.append((np.asarray(yte), y_prob))
        per_fold_best_params.append(best_params)
        for yt, yp in zip(yte, y_prob):
            cv_predictions.append({
                "fold": fold_idx,
                "y_true": int(yt),
                "y_prob": float(yp),
                "y_pred_0.5": int(yp >= 0.5),
            })

    y_true_pool = np.concatenate([yt for yt, _ in per_fold_probs])
    y_prob_pool = np.concatenate([yp for _, yp in per_fold_probs])
    pooled_05 = compute_metrics(y_true_pool, y_prob_pool, threshold=0.5)
    youden_thr = youden_threshold(y_true_pool, y_prob_pool)
    pooled_youden = compute_metrics(y_true_pool, y_prob_pool, threshold=youden_thr)

    return {
        "fold_records": fold_records,
        "per_fold_probs": per_fold_probs,
        "per_fold_best_params": per_fold_best_params,
        "cv_predictions": cv_predictions,
        "pooled_05": pooled_05,
        "pooled_youden": pooled_youden,
        "youden_threshold": youden_thr,
    }


def aggregate_fold(fold_records, col):
    arr = np.array([r[col] for r in fold_records], dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0


# --------------------------------------------------------------------------- #
def run_outcome(outcome: str, df: pd.DataFrame) -> dict:
    print(f"\n=== Gradual training: {outcome} ===")
    out_dir = RESULTS_DIR / f"{outcome}_results"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    valid = df[outcome].notna()
    sub = df.loc[valid].reset_index(drop=True)
    y = sub[outcome].astype(int).to_numpy()
    n_pos = int(y.sum()); n_neg = int((1 - y).sum())
    print(f"  n={len(y)}  positives={n_pos}  negatives={n_neg}  prevalence={y.mean():.3f}")

    ranked = load_ranked_features(outcome)
    ranked.to_csv(out_dir / "final_selected_features_used.csv", index=False)
    feature_order = ranked["feature"].tolist()
    n_total = len(feature_order)
    k_max = min(K_MAX, n_total) if K_MAX is not None else n_total
    k_values = list(range(K_MIN, k_max + 1, K_STEP))
    print(f"  feature ranking size={n_total}  evaluating k in [{k_values[0]}..{k_values[-1]}] step {K_STEP}  ({len(k_values)} models)")

    fixed_best = None
    if HYPERPARAM_STRATEGY == "fixed_best":
        fixed_best = load_main_best_params(outcome)
        print(f"  using fixed hyperparameters from main run: {fixed_best}")

    outer = RepeatedStratifiedKFold(
        n_splits=OUTER_N_SPLITS,
        n_repeats=OUTER_N_REPEATS,
        random_state=RANDOM_STATE,
    )

    # --- the sweep --- #
    rows = []                                # one row per k
    fold_long_rows = []                      # per-fold per-k details
    cv_pred_rows = []                        # all OOF predictions
    pooled_pairs_per_k = {}                  # k -> (y_true, y_prob)
    inner_params_per_k = {}                  # k -> list of best params per fold

    t0 = time.time()
    for idx, k in enumerate(k_values):
        feats = feature_order[:k]
        X_k = sub[feats].to_numpy(dtype=float)
        res = fit_one_k(X_k, y, outer, fixed_best)
        pooled_pairs_per_k[k] = (
            np.concatenate([yt for yt, _ in res["per_fold_probs"]]),
            np.concatenate([yp for _, yp in res["per_fold_probs"]]),
        )
        inner_params_per_k[k] = res["per_fold_best_params"]

        row = {"n_features": k}
        for col in METRIC_COLUMNS:
            mean, std = aggregate_fold(res["fold_records"], col)
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_pooled"] = res["pooled_05"][col]
        row["pooled_youden_balanced_accuracy"] = res["pooled_youden"]["balanced_accuracy"]
        row["pooled_youden_f1"] = res["pooled_youden"]["f1"]
        row["youden_threshold"] = res["youden_threshold"]
        rows.append(row)

        for r in res["fold_records"]:
            d = {"n_features": k, "fold": r["fold"]}
            for col in METRIC_COLUMNS:
                d[col] = r[col]
            fold_long_rows.append(d)
        for d in res["cv_predictions"]:
            d["n_features"] = k
            cv_pred_rows.append(d)

        if VERBOSE:
            print(
                f"  k={k:>3}/{n_total}  AUC={row['roc_auc_mean']:.3f}±{row['roc_auc_std']:.3f}  "
                f"BalAcc={row['balanced_accuracy_mean']:.3f}  "
                f"F1={row['f1_mean']:.3f}  "
                f"AP={row['average_precision_mean']:.3f}  "
                f"({idx + 1}/{len(k_values)})"
            )

    elapsed = time.time() - t0
    print(f"  sweep finished in {elapsed:.1f}s")

    # ----- save tables -----
    metrics_df = pd.DataFrame(rows).sort_values("n_features").reset_index(drop=True)
    metrics_df.to_csv(out_dir / "metrics_per_k.csv", index=False)
    pd.DataFrame(fold_long_rows).to_csv(out_dir / "cv_fold_metrics_per_k.csv", index=False)
    pd.DataFrame(cv_pred_rows).to_csv(out_dir / "cv_predictions_per_k.csv", index=False)

    # Hyperparameter records
    hp_records = []
    for k, params_list in inner_params_per_k.items():
        for fold_idx, p in enumerate(params_list):
            hp_records.append({"n_features": k, "fold": fold_idx,
                               **{f"hp_{kk}": vv for kk, vv in p.items()}})
    pd.DataFrame(hp_records).to_csv(out_dir / "best_hyperparameters_per_k.csv", index=False)

    # ----- peak summary -----
    peak = {}
    for col in METRIC_COLUMNS:
        series = metrics_df[f"{col}_mean"].values
        better = HIGHER_IS_BETTER[col]
        idx = int(np.nanargmax(series) if better else np.nanargmin(series))
        peak[col] = {
            "best_n_features": int(metrics_df.loc[idx, "n_features"]),
            "best_value_mean": float(metrics_df.loc[idx, f"{col}_mean"]),
            "best_value_std": float(metrics_df.loc[idx, f"{col}_std"]),
            "pooled_at_best_k": float(metrics_df.loc[idx, f"{col}_pooled"]),
            "higher_is_better": better,
        }
    peak_summary = {
        "outcome": outcome,
        "n_samples": int(len(y)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "prevalence": float(y.mean()),
        "n_total_features_in_ranking": n_total,
        "k_values_evaluated": k_values,
        "hyperparam_strategy": HYPERPARAM_STRATEGY,
        "rf_param_grid": RF_PARAM_GRID if HYPERPARAM_STRATEGY == "per_fold_grid" else None,
        "fixed_best_params": fixed_best,
        "outer_cv": f"RepeatedStratifiedKFold(n_splits={OUTER_N_SPLITS}, n_repeats={OUTER_N_REPEATS})",
        "inner_cv": f"StratifiedKFold(n_splits={INNER_N_SPLITS})",
        "peak_per_metric": peak,
        "elapsed_seconds": elapsed,
        "random_state": RANDOM_STATE,
    }
    write_json(peak_summary, out_dir / "peak_summary.json")

    # ----- plots -----
    make_plots(outcome, metrics_df, fold_long_rows, pooled_pairs_per_k,
               feature_order, ranked, peak, plots_dir)

    # ----- per-outcome report -----
    write_outcome_report(out_dir, outcome, metrics_df, peak, ranked, n_pos, n_neg, len(y),
                        elapsed, k_values)

    # Cross-outcome row
    return {
        "outcome": outcome,
        "n_samples": int(len(y)),
        "n_positive": n_pos,
        "prevalence": float(y.mean()),
        "n_total_features": n_total,
        "best_k_auc": peak["roc_auc"]["best_n_features"],
        "best_auc_mean": peak["roc_auc"]["best_value_mean"],
        "best_k_balanced_accuracy": peak["balanced_accuracy"]["best_n_features"],
        "best_balanced_accuracy_mean": peak["balanced_accuracy"]["best_value_mean"],
        "best_k_f1": peak["f1"]["best_n_features"],
        "best_f1_mean": peak["f1"]["best_value_mean"],
        "best_k_average_precision": peak["average_precision"]["best_n_features"],
        "best_average_precision_mean": peak["average_precision"]["best_value_mean"],
        "elapsed_seconds": elapsed,
    }


# --------------------------------------------------------------------------- #
# ----------------------------- plotting ------------------------------------ #
def _mark_peak(ax, x, y, color, label):
    ax.axvline(x, linestyle=":", color=color, lw=1, alpha=0.8)
    ax.scatter([x], [y], color=color, zorder=5, s=55, edgecolor="black",
               linewidth=0.6, label=label)


def _single_metric_plot(metric, metrics_df, peak, plots_dir, outcome):
    mean = metrics_df[f"{metric}_mean"].values
    std = metrics_df[f"{metric}_std"].values
    pooled = metrics_df[f"{metric}_pooled"].values
    ks = metrics_df["n_features"].values

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.fill_between(ks, mean - std, mean + std, alpha=0.18, color="#3b6ea8",
                    label="±1 SD across outer folds")
    ax.plot(ks, mean, lw=2.0, color="#3b6ea8", label="Mean across outer folds")
    ax.plot(ks, pooled, lw=1.4, color="#cc6633", linestyle="--",
            label="Pooled OOF")
    p = peak[metric]
    _mark_peak(ax, p["best_n_features"], p["best_value_mean"],
               "#1f7a1f",
               f"Peak: k={p['best_n_features']}, "
               f"{PRETTY_LABEL[metric]}={p['best_value_mean']:.3f}")
    ax.set_xlabel("Number of top features used (ranked by RF importance)")
    ax.set_ylabel(PRETTY_LABEL[metric])
    ax.set_title(f"{PRETTY_LABEL[metric]} vs. number of features — {outcome}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{metric}_vs_n_features.png", dpi=200)
    plt.close(fig)


def _multi_panel_plot(metrics_df, peak, plots_dir, outcome):
    metrics = ["roc_auc", "average_precision", "balanced_accuracy", "f1",
               "sensitivity", "specificity", "precision", "brier_score"]
    n = len(metrics)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.0 * rows), sharex=True)
    axes = axes.ravel()
    ks = metrics_df["n_features"].values
    for ax, metric in zip(axes, metrics):
        mean = metrics_df[f"{metric}_mean"].values
        std = metrics_df[f"{metric}_std"].values
        ax.fill_between(ks, mean - std, mean + std, alpha=0.18, color="#3b6ea8")
        ax.plot(ks, mean, lw=1.7, color="#3b6ea8")
        p = peak[metric]
        ax.axvline(p["best_n_features"], linestyle=":", color="#1f7a1f", lw=1)
        ax.scatter([p["best_n_features"]], [p["best_value_mean"]],
                   color="#1f7a1f", zorder=5, s=40, edgecolor="black", lw=0.5)
        ax.set_title(f"{PRETTY_LABEL[metric]}  "
                     f"(peak k={p['best_n_features']}, val={p['best_value_mean']:.3f})",
                     fontsize=10)
        ax.grid(True, alpha=0.3)
    for ax in axes[n:]:
        ax.axis("off")
    for i, ax in enumerate(axes[:n]):
        if i // cols == rows - 1:
            ax.set_xlabel("Number of features")
    fig.suptitle(f"All metrics vs. number of features — {outcome}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(plots_dir / "all_metrics_vs_n_features.png", dpi=200)
    plt.close(fig)


def _per_fold_lines_plot(metric, fold_rows, plots_dir, outcome):
    df = pd.DataFrame(fold_rows)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for fold, sub in df.groupby("fold"):
        sub = sub.sort_values("n_features")
        ax.plot(sub["n_features"], sub[metric], color="grey", alpha=0.3, lw=0.9)
    mean_curve = df.groupby("n_features")[metric].mean().sort_index()
    ax.plot(mean_curve.index, mean_curve.values, lw=2.2, color="#cc3333",
            label=f"Mean across folds")
    ax.set_xlabel("Number of features")
    ax.set_ylabel(PRETTY_LABEL[metric])
    ax.set_title(f"Per-fold {PRETTY_LABEL[metric]} vs. number of features — {outcome}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / f"per_fold_{metric}_vs_n_features.png", dpi=200)
    plt.close(fig)


def _heatmap_plot(fold_rows, metric, plots_dir, outcome):
    df = pd.DataFrame(fold_rows)
    pivot = df.pivot(index="fold", columns="n_features", values=metric)
    fig, ax = plt.subplots(figsize=(min(13, 0.18 * pivot.shape[1] + 4), 4.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis",
                   origin="lower")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([f"fold {i}" for i in pivot.index], fontsize=8)
    step = max(1, pivot.shape[1] // 20)
    xt = list(range(0, pivot.shape[1], step))
    ax.set_xticks(xt)
    ax.set_xticklabels([str(pivot.columns[i]) for i in xt], fontsize=8)
    ax.set_xlabel("Number of features")
    ax.set_title(f"{PRETTY_LABEL[metric]} per outer fold — {outcome}")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(plots_dir / f"heatmap_{metric}_per_fold.png", dpi=200)
    plt.close(fig)


def _peak_marker_summary_plot(peak, plots_dir, outcome):
    metrics = ["roc_auc", "average_precision", "balanced_accuracy", "f1",
               "sensitivity", "specificity", "precision", "brier_score"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ks = [peak[m]["best_n_features"] for m in metrics]
    vals = [peak[m]["best_value_mean"] for m in metrics]
    bars = ax.bar(metrics, ks, color="#3b6ea8", alpha=0.8)
    ax.set_ylabel("Best k (number of features)")
    ax.set_title(f"Best feature count per metric — {outcome}")
    for bar, k, v in zip(bars, ks, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"k={k}\n{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "peak_marker_summary.png", dpi=200)
    plt.close(fig)


def _ranked_features_plot(ranked, plots_dir, outcome, top_n=30):
    sub = ranked.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(sub) + 1)))
    ax.barh(sub["feature"], sub["rf_importance"], color="#3b6ea8")
    ax.set_xlabel("RF impurity importance (final model on all data)")
    ax.set_title(f"Feature ranking used for the gradual sweep — {outcome}")
    fig.tight_layout()
    fig.savefig(plots_dir / "feature_ranking_used.png", dpi=200)
    plt.close(fig)


def _pooled_roc_at_peak_plot(pooled_pairs_per_k, peak, plots_dir, outcome):
    """ROC overlay: compare pooled curves at k=1, peak-AUC k, and full k."""
    from sklearn.metrics import roc_auc_score, roc_curve
    ks = sorted(pooled_pairs_per_k.keys())
    peak_k = peak["roc_auc"]["best_n_features"]
    chosen = sorted({ks[0], peak_k, ks[-1]})
    colors = ["#999999", "#cc3333", "#3b6ea8"]
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for k, color in zip(chosen, colors):
        yt, yp = pooled_pairs_per_k[k]
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, yp)
        auc = roc_auc_score(yt, yp)
        ax.plot(fpr, tpr, color=color, lw=2.0,
                label=f"k={k}  AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Pooled OOF ROC at selected k values — {outcome}")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / "roc_pooled_selected_k.png", dpi=200)
    plt.close(fig)


def make_plots(outcome, metrics_df, fold_long_rows, pooled_pairs_per_k,
               feature_order, ranked, peak, plots_dir):
    for metric in METRIC_COLUMNS:
        _single_metric_plot(metric, metrics_df, peak, plots_dir, outcome)
    _multi_panel_plot(metrics_df, peak, plots_dir, outcome)
    for metric in ["roc_auc", "balanced_accuracy", "f1"]:
        _per_fold_lines_plot(metric, fold_long_rows, plots_dir, outcome)
        _heatmap_plot(fold_long_rows, metric, plots_dir, outcome)
    _peak_marker_summary_plot(peak, plots_dir, outcome)
    _ranked_features_plot(ranked, plots_dir, outcome)
    _pooled_roc_at_peak_plot(pooled_pairs_per_k, peak, plots_dir, outcome)


# --------------------------------------------------------------------------- #
def write_outcome_report(out_dir, outcome, metrics_df, peak, ranked,
                         n_pos, n_neg, n_total, elapsed, k_values):
    """Per-outcome short report."""
    n_total_feats = len(ranked)
    lines = [
        f"# Gradual training report — {outcome}",
        "",
        "## 1. Setup",
        "",
        f"- Cohort: n={n_total} (positives={n_pos}, negatives={n_neg}, "
        f"prevalence={n_pos / n_total:.3f}).",
        f"- Feature ranking source: `Random_Forest/results/{outcome}/"
        f"final_selected_features.csv` (sorted by Random-Forest impurity importance).",
        f"- Number of ranked features: **{n_total_feats}**.",
        f"- Sweep range: k = {k_values[0]} … {k_values[-1]} (step {K_STEP}), "
        f"{len(k_values)} models trained.",
        f"- Hyperparameter strategy: **{HYPERPARAM_STRATEGY}**.",
        f"- Outer CV: RepeatedStratifiedKFold(n_splits={OUTER_N_SPLITS}, "
        f"n_repeats={OUTER_N_REPEATS}) — 15 held-out folds per k.",
        f"- Inner CV: StratifiedKFold(n_splits={INNER_N_SPLITS}), scoring={INNER_SCORING}.",
        f"- Random seed: {RANDOM_STATE}. Wall time: {elapsed:.1f}s.",
        "",
        "## 2. Peak performance per metric",
        "",
        "Each row reports the feature count that maximised (or minimised, for Brier) "
        "the per-fold mean of the metric. `Pooled @ k*` is the metric computed on the "
        "concatenated out-of-fold predictions at that same k.",
        "",
        "| Metric | Best k | Mean ± SD across folds | Pooled @ k* |",
        "|---|---:|---|---:|",
    ]
    for m in ["roc_auc", "average_precision", "balanced_accuracy", "f1",
              "sensitivity", "specificity", "precision", "accuracy", "brier_score"]:
        p = peak[m]
        direction = "min" if not p["higher_is_better"] else "max"
        lines.append(
            f"| {PRETTY_LABEL[m]} ({direction}) | {p['best_n_features']} | "
            f"{p['best_value_mean']:.3f} ± {p['best_value_std']:.3f} | "
            f"{p['pooled_at_best_k']:.3f} |"
        )
    lines += [
        "",
        "## 3. Top-ranked features (used in the sweep)",
        "",
        "These are the features that the published main-pipeline model selected via "
        "stability selection on the full data, ranked by Random-Forest impurity "
        "importance. The k-th model in the sweep uses the first k of them.",
        "",
        "| Rank | Feature | Stability frequency | RF importance |",
        "|---:|---|---:|---:|",
    ]
    for i, row in ranked.head(30).iterrows():
        lines.append(
            f"| {i + 1} | `{row['feature']}` | "
            f"{row['stability_frequency']:.2f} | {row['rf_importance']:.4f} |"
        )
    if len(ranked) > 30:
        lines.append(f"| … | _{len(ranked) - 30} more_ | | |")
    lines += [
        "",
        "## 4. Files in this directory",
        "",
        "- `metrics_per_k.csv` — one row per k with mean/std across folds + pooled metrics.",
        "- `cv_fold_metrics_per_k.csv` — long format, one row per (k, fold).",
        "- `cv_predictions_per_k.csv` — every OOF prediction at every k.",
        "- `best_hyperparameters_per_k.csv` — inner-CV winning hyperparameters.",
        "- `final_selected_features_used.csv` — exact feature ranking that drove the sweep.",
        "- `peak_summary.json` — machine-readable peak-per-metric table.",
        "- `plots/` — every metric vs. k, multi-panel summary, per-fold curves, "
        "heatmaps, peak-marker bar chart, feature-ranking bar chart, ROC overlay at "
        "selected k.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
def write_master_report(summary_rows, root_report_path):
    df = pd.DataFrame(summary_rows)
    df.to_csv(RESULTS_DIR / "summary_gradual_training.csv", index=False)

    lines = [
        "# Gradual Training — MACE Risk Prediction",
        "",
        "This document explains the **gradual training** analysis added on top of the "
        "main Random-Forest pipeline and reports its findings across all configured "
        "outcome horizons.",
        "",
        "## 1. Motivation",
        "",
        "The main pipeline (`Random_Forest/src/train_rf.py`) selects between roughly "
        "30 and 50 features per horizon via stability selection and trains a Random "
        "Forest on those. A natural follow-up question for a publication is: **how "
        "many of those features are actually doing the work?** If the curve of "
        "performance vs. feature count plateaus (or peaks) early, the published "
        "model could be slimmed down, both for biological interpretability and to "
        "reduce the chance that the larger feature set is over-fitting noise on n=94 "
        "patients. This analysis answers that question with a held-out, "
        "cross-validated sweep.",
        "",
        "## 2. Method",
        "",
        "### 2.1 Inputs",
        "",
        "For every configured horizon `Within{k}Yr` we read the published feature "
        "ranking from",
        "",
        "```",
        "Random_Forest/results/Within{k}Yr/final_selected_features.csv",
        "```",
        "",
        "produced by the main pipeline. The file lists every feature that survived "
        "stability selection on the full dataset, with two scores: "
        "`stability_frequency` (the L1-logistic stability-selection score, "
        "Meinshausen & Bühlmann 2010) and `rf_importance` (mean decrease in impurity "
        "from the final Random Forest fitted on all n=94 patients with the most "
        "frequent winning hyperparameters from the outer CV). **We sort by "
        "`rf_importance` (descending)** and call this the *feature ranking*.",
        "",
        "### 2.2 The sweep",
        "",
        "Let N be the size of the feature ranking for that outcome. For "
        "k = 1, 2, …, N we train a separate Random-Forest classifier using only the "
        "top-k features. Each training run uses the **same nested cross-validation "
        "protocol as the main pipeline**:",
        "",
        "- Outer loop: `RepeatedStratifiedKFold(n_splits=5, n_repeats=3)` — 15 "
        "  held-out estimates per k.",
        "- Inner loop: `StratifiedKFold(n_splits=5)` for hyperparameter tuning.",
        "- Random seed: 42 (fixed end-to-end).",
        "",
        "Class imbalance is handled by `class_weight=\"balanced\"` inside the Random "
        "Forest, exactly as in the main pipeline.",
        "",
        "### 2.3 Hyperparameter tuning",
        "",
        "Two strategies are supported in `config.py` (`HYPERPARAM_STRATEGY`):",
        "",
        "1. **`per_fold_grid` (default).** Inside each outer fold, run a "
        "   `GridSearchCV` over the same Random-Forest grid used by the main pipeline "
        "   (`n_estimators ∈ {300}`, `max_depth ∈ {None, 5}`, "
        "   `min_samples_leaf ∈ {1, 3}`, `max_features ∈ {\"sqrt\", 0.3}`), scored on "
        "   `roc_auc`, on the 5-fold stratified inner CV. The best estimator is "
        "   refit and used to score the held-out outer fold. This is the most "
        "   rigorous option and matches the main pipeline exactly.",
        "2. **`fixed_best`.** Reuse the most-frequent winning combination from the "
        "   main pipeline's `best_hyperparameters.json` — no inner-CV search. Useful "
        "   for isolating the feature-count effect from hyperparameter variance, and "
        "   considerably faster.",
        "",
        "Per-fold winning hyperparameters at every k are saved to "
        "`best_hyperparameters_per_k.csv` so the choices can be inspected.",
        "",
        "### 2.4 Why use the full-data feature ranking, not refit it inside each fold?",
        "",
        "Strictly speaking, computing the feature ordering on the full dataset before "
        "the outer CV introduces a small amount of look-ahead — the ordering is "
        "informed by every patient's outcome. We use this ordering deliberately, for "
        "two reasons. First, this is the *published* ordering: a clinician asking "
        "\"do I need to measure all 50 of these proteins, or just the top five?\" is "
        "asking about that exact list. Second, we are not making a generalisation "
        "claim about the ranking itself; for each k the *model* is still trained and "
        "evaluated under fold-respecting nested CV, so the k-vs-performance curve is "
        "a held-out estimate. The interaction is mild — at small k the ranking has "
        "almost no leverage to memorise individual patients — but it should be "
        "interpreted as **\"how performance varies as we walk down the published "
        "ranking\"**, not as fully nested feature selection.",
        "",
        "### 2.5 Metrics recorded",
        "",
        "At every k we record per-fold and pooled values of:",
        "",
        "- ROC AUC, Average Precision (PR AUC),",
        "- Balanced accuracy, accuracy, F1, precision, recall,",
        "- Sensitivity (TPR) and specificity (TNR) at threshold 0.5,",
        "- Brier score (lower is better),",
        "- Pooled balanced accuracy and F1 at the Youden-optimal threshold.",
        "",
        "All metric definitions are imported from the main pipeline's "
        "`metrics_utils.py` to guarantee identical maths across the two analyses.",
        "",
        "### 2.6 Definition of \"best k\"",
        "",
        "For each metric we report the k at which the **mean across the 15 outer "
        "folds** is best (highest, except for Brier where lowest). Per-fold "
        "standard deviation, the pooled OOF value at that same k, and the inner-CV "
        "winning hyperparameters are stored alongside.",
        "",
        "## 3. Results",
        "",
        "### 3.1 Cross-outcome summary",
        "",
        "| Outcome | n | prev. | total feats | Best k (AUC) | Best AUC | "
        "Best k (BalAcc) | Best BalAcc | Best k (F1) | Best F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['outcome']} | {r['n_samples']} | {r['prevalence']:.2f} | "
            f"{r['n_total_features']} | {r['best_k_auc']} | {r['best_auc_mean']:.3f} | "
            f"{r['best_k_balanced_accuracy']} | {r['best_balanced_accuracy_mean']:.3f} | "
            f"{r['best_k_f1']} | {r['best_f1_mean']:.3f} |"
        )
    lines += [
        "",
        "### 3.2 Per-outcome detail",
        "",
        "Each `Within{k}Yr_results/` folder contains:",
        "",
        "- `report.md` — same numbers as above, plus the actual top-30 features used.",
        "- `metrics_per_k.csv` and `cv_fold_metrics_per_k.csv` — every metric, "
        "  every k, every fold.",
        "- `cv_predictions_per_k.csv` — every patient's OOF probability at every k.",
        "- `best_hyperparameters_per_k.csv` — inner-CV winners at every (k, fold).",
        "- `peak_summary.json` — machine-readable peak-per-metric.",
        "- `plots/` — the 20+ plots described below.",
        "",
        "### 3.3 Plots produced (per outcome)",
        "",
        "- `{metric}_vs_n_features.png` — for every metric, mean ± SD band, pooled "
        "  curve overlay, peak marker. Ten metrics × one plot each.",
        "- `all_metrics_vs_n_features.png` — eight-panel summary on one figure.",
        "- `per_fold_{metric}_vs_n_features.png` — every outer fold's curve in grey, "
        "  mean curve in red. Shows fold-to-fold variability.",
        "- `heatmap_{metric}_per_fold.png` — fold × k heatmap; colour shows where the "
        "  signal is concentrated.",
        "- `peak_marker_summary.png` — bar chart of best k per metric.",
        "- `feature_ranking_used.png` — the top-30 features fed into the sweep.",
        "- `roc_pooled_selected_k.png` — pooled ROC at k=1, peak-AUC k, and full k.",
        "",
        "## 4. How to reproduce",
        "",
        "```bash",
        "# all configured outcomes (see Random_Forest/src_gradual_training/config.py)",
        ".venv/bin/python Random_Forest/src_gradual_training/gradual_train.py",
        "",
        "# subset",
        ".venv/bin/python Random_Forest/src_gradual_training/gradual_train.py "
        "--outcomes Within3Yr Within5Yr",
        "```",
        "",
        "Total runtime depends on `HYPERPARAM_STRATEGY` and the number of features in "
        "each ranking. With `per_fold_grid` and the default grid, expect a few "
        "minutes per outcome on a modern multi-core machine.",
        "",
        "## 5. Caveats",
        "",
        "- **n = 94 is small.** The k-vs-performance curves are bumpy because each "
        "  point is the mean of only 15 outer-fold estimates. Standard-deviation "
        "  bands are wide; a difference of 0.02 AUC between two adjacent k values "
        "  should not be over-interpreted.",
        "- **Feature ordering uses full-data RF importance.** See §2.4. The intent is "
        "  to characterise the published ranking, not to perform fully nested "
        "  selection.",
        "- **No external validation.** The same caveats from `Report.md` apply.",
        "",
    ]
    root_report_path.write_text("\n".join(lines))


def make_master_summary_plot(summary_rows, out_dir):
    df = pd.DataFrame(summary_rows)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(df))
    width = 0.25
    ax.bar(x - width, df["best_k_auc"], width, label="best-k AUC", color="#3b6ea8")
    ax.bar(x,         df["best_k_balanced_accuracy"], width,
           label="best-k Balanced Accuracy", color="#cc6633")
    ax.bar(x + width, df["best_k_f1"], width, label="best-k F1", color="#1f7a1f")
    ax.set_xticks(x); ax.set_xticklabels(df["outcome"])
    ax.set_ylabel("Best k (number of features)")
    ax.set_title("Optimal feature count per metric, all horizons")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_best_k.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(df["outcome"], df["best_auc_mean"], "o-", lw=2, color="#3b6ea8",
            label="Best AUC")
    ax.plot(df["outcome"], df["best_balanced_accuracy_mean"], "o-", lw=2,
            color="#cc6633", label="Best Balanced Accuracy")
    ax.plot(df["outcome"], df["best_f1_mean"], "o-", lw=2, color="#1f7a1f",
            label="Best F1")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Metric value")
    ax.set_title("Peak metric value per horizon (gradual sweep)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_best_metric_value.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcomes", nargs="+", default=None,
                        help=f"Subset of outcomes to run (default from config: "
                             f"{OUTCOMES_TO_RUN}).")
    args = parser.parse_args()

    targets = args.outcomes if args.outcomes else OUTCOMES_TO_RUN
    bad = [o for o in targets if o not in ALL_OUTCOMES]
    if bad:
        raise ValueError(f"Unknown outcomes requested: {bad}; allowed: {ALL_OUTCOMES}")

    df = load_dataset()
    print(f"Loaded {DATA_PATH}: shape={df.shape}")
    print(f"Outcomes to run: {targets}")
    print(f"Hyperparameter strategy: {HYPERPARAM_STRATEGY}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for outcome in targets:
        try:
            summary_rows.append(run_outcome(outcome, df))
        except FileNotFoundError as e:
            print(f"SKIPPING {outcome}: {e}")
        except Exception as e:
            print(f"FAILED on {outcome}: {e}")
            raise

    if summary_rows:
        write_master_report(summary_rows, RESULTS_DIR / "Report.md")
        make_master_summary_plot(summary_rows, RESULTS_DIR)
        print(f"\nWrote master summary to {RESULTS_DIR}")
        print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
