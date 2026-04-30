"""Logistic-Regression gradual-training sweep.

Mirrors Random_Forest/src_gradual_training/gradual_train.py one-for-one;
only the classifier (a StandardScaler -> LogisticRegression Pipeline) and
its grid differ. Reads the per-outcome ranking from
LogisticRegression/results/Within{k}Yr/final_selected_features.csv (sorted
by `lr_importance`).
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    LR_MAX_ITER,
    LR_PARAM_GRID,
    MAIN_RESULTS_DIR,
    MODEL_FAMILY,
    N_JOBS,
    OUTCOMES_TO_RUN,
    OUTER_N_REPEATS,
    OUTER_N_SPLITS,
    RANDOM_STATE,
    RANKING_COLUMN,
    RESULTS_DIR,
    VERBOSE,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics_utils import compute_metrics, write_json, youden_threshold  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

METRIC_COLUMNS = [
    "roc_auc", "average_precision", "balanced_accuracy", "accuracy",
    "f1", "precision", "recall", "sensitivity", "specificity", "brier_score",
]
HIGHER_IS_BETTER = {m: True for m in METRIC_COLUMNS}
HIGHER_IS_BETTER["brier_score"] = False

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


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if ID_COLUMN not in df.columns:
        raise ValueError(f"ID column '{ID_COLUMN}' not in dataset")
    return df


def load_ranked_features(outcome: str) -> pd.DataFrame:
    path = MAIN_RESULTS_DIR / outcome / "final_selected_features.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find '{path}'. Run LogisticRegression/src/train_lr.py first."
        )
    return pd.read_csv(path).sort_values(RANKING_COLUMN, ascending=False).reset_index(drop=True)


def load_main_best_params(outcome: str) -> dict:
    path = MAIN_RESULTS_DIR / outcome / "best_hyperparameters.json"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find '{path}'.")
    with open(path) as f:
        records = json.load(f)
    counter = Counter(tuple(sorted(r["best_params"].items())) for r in records)
    return dict(counter.most_common(1)[0][0])


def make_lr_pipeline(*, random_state: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=LR_MAX_ITER,
            random_state=random_state,
        )),
    ])


def fit_one_k(X_k, y, outer, fixed_best_params):
    fold_records = []
    per_fold_probs = []
    per_fold_best_params = []
    cv_predictions = []

    for fold_idx, (tr, te) in enumerate(outer.split(X_k, y)):
        fold_seed = RANDOM_STATE + fold_idx
        Xtr, Xte = X_k[tr], X_k[te]
        ytr, yte = y[tr], y[te]

        if HYPERPARAM_STRATEGY == "per_fold_grid":
            inner = StratifiedKFold(n_splits=INNER_N_SPLITS, shuffle=True, random_state=fold_seed)
            base = make_lr_pipeline(random_state=fold_seed)
            gs = GridSearchCV(
                base, LR_PARAM_GRID, scoring=INNER_SCORING, cv=inner,
                n_jobs=N_JOBS, refit=True, error_score=np.nan,
            )
            gs.fit(Xtr, ytr)
            model = gs.best_estimator_
            best_params = gs.best_params_
        elif HYPERPARAM_STRATEGY == "fixed_best":
            model = make_lr_pipeline(random_state=fold_seed)
            model.set_params(**fixed_best_params)
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
                "fold": fold_idx, "y_true": int(yt),
                "y_prob": float(yp), "y_pred_0.5": int(yp >= 0.5),
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


def run_outcome(outcome: str, df: pd.DataFrame) -> dict:
    print(f"\n=== Gradual training ({MODEL_FAMILY}): {outcome} ===")
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
        n_splits=OUTER_N_SPLITS, n_repeats=OUTER_N_REPEATS, random_state=RANDOM_STATE,
    )

    rows, fold_long_rows, cv_pred_rows = [], [], []
    pooled_pairs_per_k, inner_params_per_k = {}, {}

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
                f"F1={row['f1_mean']:.3f}  AP={row['average_precision_mean']:.3f}  "
                f"({idx + 1}/{len(k_values)})"
            )

    elapsed = time.time() - t0
    print(f"  sweep finished in {elapsed:.1f}s")

    metrics_df = pd.DataFrame(rows).sort_values("n_features").reset_index(drop=True)
    metrics_df.to_csv(out_dir / "metrics_per_k.csv", index=False)
    pd.DataFrame(fold_long_rows).to_csv(out_dir / "cv_fold_metrics_per_k.csv", index=False)
    pd.DataFrame(cv_pred_rows).to_csv(out_dir / "cv_predictions_per_k.csv", index=False)

    hp_records = []
    for k, params_list in inner_params_per_k.items():
        for fold_idx, p in enumerate(params_list):
            hp_records.append({"n_features": k, "fold": fold_idx,
                               **{f"hp_{kk}": vv for kk, vv in p.items()}})
    pd.DataFrame(hp_records).to_csv(out_dir / "best_hyperparameters_per_k.csv", index=False)

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
        "model_family": MODEL_FAMILY,
        "outcome": outcome,
        "n_samples": int(len(y)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "prevalence": float(y.mean()),
        "n_total_features_in_ranking": n_total,
        "k_values_evaluated": k_values,
        "hyperparam_strategy": HYPERPARAM_STRATEGY,
        "param_grid": LR_PARAM_GRID if HYPERPARAM_STRATEGY == "per_fold_grid" else None,
        "fixed_best_params": fixed_best,
        "outer_cv": f"RepeatedStratifiedKFold(n_splits={OUTER_N_SPLITS}, n_repeats={OUTER_N_REPEATS})",
        "inner_cv": f"StratifiedKFold(n_splits={INNER_N_SPLITS})",
        "peak_per_metric": peak,
        "elapsed_seconds": elapsed,
        "random_state": RANDOM_STATE,
    }
    write_json(peak_summary, out_dir / "peak_summary.json")

    make_plots(outcome, metrics_df, fold_long_rows, pooled_pairs_per_k,
               feature_order, ranked, peak, plots_dir)
    write_outcome_report(out_dir, outcome, metrics_df, peak, ranked, n_pos, n_neg, len(y),
                        elapsed, k_values)

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


# --------- plotting (identical structure to RF/XGB versions) ---------- #
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
    ax.plot(ks, pooled, lw=1.4, color="#cc6633", linestyle="--", label="Pooled OOF")
    p = peak[metric]
    _mark_peak(ax, p["best_n_features"], p["best_value_mean"], "#1f7a1f",
               f"Peak: k={p['best_n_features']}, "
               f"{PRETTY_LABEL[metric]}={p['best_value_mean']:.3f}")
    ax.set_xlabel(f"Number of top features used (ranked by {MODEL_FAMILY} importance)")
    ax.set_ylabel(PRETTY_LABEL[metric])
    ax.set_title(f"{PRETTY_LABEL[metric]} vs. number of features — {outcome} ({MODEL_FAMILY})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{metric}_vs_n_features.png", dpi=200)
    plt.close(fig)


def _multi_panel_plot(metrics_df, peak, plots_dir, outcome):
    metrics = ["roc_auc", "average_precision", "balanced_accuracy", "f1",
               "sensitivity", "specificity", "precision", "brier_score"]
    n = len(metrics); cols = 2; rows = (n + cols - 1) // cols
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
    fig.suptitle(f"All metrics vs. number of features — {outcome} ({MODEL_FAMILY})", fontsize=13)
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
    ax.set_title(f"Per-fold {PRETTY_LABEL[metric]} vs. number of features — {outcome} ({MODEL_FAMILY})")
    ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / f"per_fold_{metric}_vs_n_features.png", dpi=200)
    plt.close(fig)


def _heatmap_plot(fold_rows, metric, plots_dir, outcome):
    df = pd.DataFrame(fold_rows)
    pivot = df.pivot(index="fold", columns="n_features", values=metric)
    fig, ax = plt.subplots(figsize=(min(13, 0.18 * pivot.shape[1] + 4), 4.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", origin="lower")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([f"fold {i}" for i in pivot.index], fontsize=8)
    step = max(1, pivot.shape[1] // 20)
    xt = list(range(0, pivot.shape[1], step))
    ax.set_xticks(xt)
    ax.set_xticklabels([str(pivot.columns[i]) for i in xt], fontsize=8)
    ax.set_xlabel("Number of features")
    ax.set_title(f"{PRETTY_LABEL[metric]} per outer fold — {outcome} ({MODEL_FAMILY})")
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
    ax.set_title(f"Best feature count per metric — {outcome} ({MODEL_FAMILY})")
    for bar, k, v in zip(bars, ks, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"k={k}\n{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "peak_marker_summary.png", dpi=200)
    plt.close(fig)


def _ranked_features_plot(ranked, plots_dir, outcome, top_n=30):
    sub = ranked.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(sub) + 1)))
    ax.barh(sub["feature"], sub[RANKING_COLUMN], color="#3b6ea8")
    ax.set_xlabel(f"{MODEL_FAMILY} importance (final model on all data)")
    ax.set_title(f"Feature ranking used for the gradual sweep — {outcome} ({MODEL_FAMILY})")
    fig.tight_layout()
    fig.savefig(plots_dir / "feature_ranking_used.png", dpi=200)
    plt.close(fig)


def _pooled_roc_at_peak_plot(pooled_pairs_per_k, peak, plots_dir, outcome):
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
        ax.plot(fpr, tpr, color=color, lw=2.0, label=f"k={k}  AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Pooled OOF ROC at selected k values — {outcome} ({MODEL_FAMILY})")
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


def write_outcome_report(out_dir, outcome, metrics_df, peak, ranked,
                         n_pos, n_neg, n_total, elapsed, k_values):
    n_total_feats = len(ranked)
    lines = [
        f"# Gradual training report ({MODEL_FAMILY}) — {outcome}",
        "",
        "## 1. Setup",
        "",
        f"- Cohort: n={n_total} (positives={n_pos}, negatives={n_neg}, "
        f"prevalence={n_pos / n_total:.3f}).",
        f"- Feature ranking source: `LogisticRegression/results/{outcome}/"
        f"final_selected_features.csv` (sorted by `{RANKING_COLUMN}`).",
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
        f"| Rank | Feature | Stability frequency | {RANKING_COLUMN} |",
        "|---:|---|---:|---:|",
    ]
    for i, row in ranked.head(30).iterrows():
        lines.append(
            f"| {i + 1} | `{row['feature']}` | "
            f"{row['stability_frequency']:.2f} | {row[RANKING_COLUMN]:.4f} |"
        )
    if len(ranked) > 30:
        lines.append(f"| … | _{len(ranked) - 30} more_ | | |")
    lines += ["", "## 4. Files in this directory", "",
              "- `metrics_per_k.csv`",
              "- `cv_fold_metrics_per_k.csv`",
              "- `cv_predictions_per_k.csv`",
              "- `best_hyperparameters_per_k.csv`",
              "- `final_selected_features_used.csv`",
              "- `peak_summary.json`",
              "- `plots/`",
              ""]
    (out_dir / "report.md").write_text("\n".join(lines))


def write_master_report(summary_rows, root_report_path):
    df = pd.DataFrame(summary_rows)
    df.to_csv(RESULTS_DIR / "summary_gradual_training.csv", index=False)
    lines = [
        f"# Gradual Training ({MODEL_FAMILY}) — MACE Risk Prediction",
        "",
        f"This document mirrors the Random-Forest gradual-training analysis but uses "
        f"**{MODEL_FAMILY}** as the classifier.",
        "",
        "### Cross-outcome summary",
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
    lines += ["", "Per-outcome detail is in each `Within{k}Yr_results/report.md`.", ""]
    root_report_path.write_text("\n".join(lines))


def make_master_summary_plot(summary_rows, out_dir):
    df = pd.DataFrame(summary_rows)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(df)); width = 0.25
    ax.bar(x - width, df["best_k_auc"], width, label="best-k AUC", color="#3b6ea8")
    ax.bar(x,         df["best_k_balanced_accuracy"], width,
           label="best-k Balanced Accuracy", color="#cc6633")
    ax.bar(x + width, df["best_k_f1"], width, label="best-k F1", color="#1f7a1f")
    ax.set_xticks(x); ax.set_xticklabels(df["outcome"])
    ax.set_ylabel("Best k (number of features)")
    ax.set_title(f"Optimal feature count per metric, all horizons ({MODEL_FAMILY})")
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_best_k.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(df["outcome"], df["best_auc_mean"], "o-", lw=2, color="#3b6ea8", label="Best AUC")
    ax.plot(df["outcome"], df["best_balanced_accuracy_mean"], "o-", lw=2,
            color="#cc6633", label="Best Balanced Accuracy")
    ax.plot(df["outcome"], df["best_f1_mean"], "o-", lw=2, color="#1f7a1f", label="Best F1")
    ax.set_ylim(0, 1.0); ax.set_ylabel("Metric value")
    ax.set_title(f"Peak metric value per horizon (gradual sweep, {MODEL_FAMILY})")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_best_metric_value.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcomes", nargs="+", default=None)
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
