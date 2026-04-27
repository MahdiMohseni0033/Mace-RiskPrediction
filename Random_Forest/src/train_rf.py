"""Train one Random Forest classifier per outcome horizon (Within1Yr ... Within7Yr).

The pipeline implements:
  * Nested cross-validation
      - outer: RepeatedStratifiedKFold (unbiased generalisation estimate)
      - inner: StratifiedKFold for RF hyperparameter tuning
  * Stability selection via L1-logistic regression executed *inside* every
    outer-training fold (no leakage from held-out data into selection)
  * Class-weight balancing for imbalanced horizons
  * Publication-relevant metrics, ROC/PR/CM/feature plots, model artifacts

Run:
    python Random_Forest/src/train_rf.py
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_PATH,
    ID_COLUMN,
    INNER_N_SPLITS,
    MAX_SELECTED_FEATURES,
    MIN_SELECTED_FEATURES,
    N_JOBS,
    OUTCOME_COLUMNS,
    OUTER_N_REPEATS,
    OUTER_N_SPLITS,
    RANDOM_STATE,
    RESULTS_DIR,
    RF_PARAM_GRID,
    STABILITY_C_GRID,
    STABILITY_N_SUBSAMPLES,
    STABILITY_SUBSAMPLE_FRAC,
    STABILITY_THRESHOLD,
)
from metrics_utils import (
    compute_metrics,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curves,
    plot_top_features,
    write_json,
    youden_threshold,
)
from stability_selection import stability_selection

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
os.environ.setdefault("PYTHONWARNINGS", "ignore")


# --------------------------------------------------------------------------- #
def load_data():
    df = pd.read_csv(DATA_PATH)
    if ID_COLUMN not in df.columns:
        raise ValueError(f"ID column '{ID_COLUMN}' not in dataset")
    for c in OUTCOME_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"Outcome column '{c}' not in dataset")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {ID_COLUMN, *OUTCOME_COLUMNS}
    feats = [c for c in df.columns if c not in drop]
    non_num = [c for c in feats if not np.issubdtype(df[c].dtype, np.number)]
    if non_num:
        raise ValueError(f"Non-numeric features remain: {non_num[:5]}...")
    return feats


# --------------------------------------------------------------------------- #
def fit_one_outer_fold(
    X_train, y_train, X_test, y_test, feature_names, fold_seed
):
    """Run stability selection + inner-CV-tuned RF on one outer fold."""
    scaler = StandardScaler().fit(X_train)
    Xtr_s = scaler.transform(X_train)

    sel_idx, sel_freq = stability_selection(
        Xtr_s, y_train,
        n_subsamples=STABILITY_N_SUBSAMPLES,
        subsample_frac=STABILITY_SUBSAMPLE_FRAC,
        c_grid=STABILITY_C_GRID,
        threshold=STABILITY_THRESHOLD,
        min_features=MIN_SELECTED_FEATURES,
        max_features=MAX_SELECTED_FEATURES,
        random_state=fold_seed,
    )
    sel_names = [feature_names[i] for i in sel_idx]

    # RF works on the unscaled selected columns directly (trees are scale-invariant).
    Xtr_sel = X_train[:, sel_idx]
    Xte_sel = X_test[:, sel_idx]

    inner = StratifiedKFold(n_splits=INNER_N_SPLITS, shuffle=True, random_state=fold_seed)
    base_rf = RandomForestClassifier(
        class_weight="balanced", random_state=fold_seed, n_jobs=1,
    )
    gs = GridSearchCV(
        base_rf, RF_PARAM_GRID, scoring="roc_auc", cv=inner,
        n_jobs=N_JOBS, refit=True, error_score=np.nan,
    )
    gs.fit(Xtr_sel, y_train)

    best_rf = gs.best_estimator_
    y_prob = best_rf.predict_proba(Xte_sel)[:, 1]

    # OOB proxy for "training loss" (Brier on training fold)
    train_prob = best_rf.predict_proba(Xtr_sel)[:, 1]
    train_brier = float(np.mean((train_prob - y_train) ** 2))
    val_brier = float(np.mean((y_prob - y_test) ** 2))

    return {
        "selected_idx": sel_idx,
        "selected_names": sel_names,
        "selection_freq": sel_freq,
        "best_params": gs.best_params_,
        "inner_best_score": float(gs.best_score_),
        "y_prob": y_prob,
        "y_true": y_test,
        "train_brier": train_brier,
        "val_brier": val_brier,
        "feature_importances": best_rf.feature_importances_,
        "scaler": scaler,
    }


# --------------------------------------------------------------------------- #
def train_outcome(df: pd.DataFrame, outcome: str, feature_names: list[str]) -> dict:
    print(f"\n=== Training Random Forest for {outcome} ===")
    out_dir = RESULTS_DIR / outcome
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    valid = df[outcome].notna()
    sub = df.loc[valid].reset_index(drop=True)
    y = sub[outcome].astype(int).to_numpy()
    ids = sub[ID_COLUMN].to_numpy()
    X = sub[feature_names].to_numpy(dtype=float)
    n_pos = int(y.sum()); n_neg = int((1 - y).sum())
    print(f"  n={len(y)}  positives={n_pos}  negatives={n_neg}  prevalence={y.mean():.3f}")

    outer = RepeatedStratifiedKFold(
        n_splits=OUTER_N_SPLITS,
        n_repeats=OUTER_N_REPEATS,
        random_state=RANDOM_STATE,
    )

    fold_records = []
    cv_predictions = []
    per_fold_probs = []
    feature_freq_acc = np.zeros(len(feature_names))
    feature_freq_n = 0
    selection_count = Counter()

    t0 = time.time()
    for k, (tr_idx, te_idx) in enumerate(outer.split(X, y)):
        fold_seed = RANDOM_STATE + k
        res = fit_one_outer_fold(
            X[tr_idx], y[tr_idx], X[te_idx], y[te_idx],
            feature_names, fold_seed,
        )
        per_fold_probs.append((res["y_true"], res["y_prob"]))
        feature_freq_acc += res["selection_freq"]
        feature_freq_n += 1
        for name in res["selected_names"]:
            selection_count[name] += 1

        m = compute_metrics(res["y_true"], res["y_prob"], threshold=0.5)
        m["fold"] = k
        m["best_params"] = res["best_params"]
        m["inner_best_score"] = res["inner_best_score"]
        m["n_selected_features"] = len(res["selected_names"])
        m["train_brier"] = res["train_brier"]
        m["val_brier"] = res["val_brier"]
        fold_records.append(m)

        for pid, yt, yp in zip(ids[te_idx], res["y_true"], res["y_prob"]):
            cv_predictions.append({
                "fold": k, ID_COLUMN: pid,
                "y_true": int(yt), "y_prob": float(yp),
                "y_pred_0.5": int(yp >= 0.5),
            })
        print(f"  fold {k+1:2d}/{outer.get_n_splits()}  AUC={m['roc_auc']:.3f}  "
              f"BalAcc={m['balanced_accuracy']:.3f}  selected={m['n_selected_features']}")

    elapsed = time.time() - t0
    print(f"  outer CV finished in {elapsed:.1f}s")

    # ----- pooled metrics (concatenate all fold predictions) -----
    y_true_pool = np.concatenate([yt for yt, _ in per_fold_probs])
    y_prob_pool = np.concatenate([yp for _, yp in per_fold_probs])

    youden_thr = youden_threshold(y_true_pool, y_prob_pool)
    pooled_05 = compute_metrics(y_true_pool, y_prob_pool, threshold=0.5)
    pooled_youden = compute_metrics(y_true_pool, y_prob_pool, threshold=youden_thr)

    # ----- per-fold aggregate (mean ± sd) -----
    df_folds = pd.DataFrame(fold_records)
    agg_cols = ["roc_auc", "average_precision", "balanced_accuracy", "accuracy",
                "precision", "recall", "sensitivity", "specificity", "f1",
                "brier_score", "train_brier", "val_brier"]
    fold_aggregate = {
        col: {"mean": float(df_folds[col].mean()),
              "std": float(df_folds[col].std(ddof=1)) if len(df_folds) > 1 else 0.0}
        for col in agg_cols
    }

    # ----- selection frequency tables -----
    mean_freq = feature_freq_acc / max(feature_freq_n, 1)
    sel_table = pd.DataFrame({
        "feature": feature_names,
        "mean_selection_freq": mean_freq,
        "selected_in_n_folds": [selection_count.get(n, 0) for n in feature_names],
    }).sort_values(["selected_in_n_folds", "mean_selection_freq"], ascending=False)
    sel_table.to_csv(out_dir / "selected_features_cv.csv", index=False)

    # ----- final model: stability-select on all data, refit RF with best CV params -----
    scaler_full = StandardScaler().fit(X)
    Xs_full = scaler_full.transform(X)
    sel_idx_full, sel_freq_full = stability_selection(
        Xs_full, y,
        n_subsamples=STABILITY_N_SUBSAMPLES,
        subsample_frac=STABILITY_SUBSAMPLE_FRAC,
        c_grid=STABILITY_C_GRID,
        threshold=STABILITY_THRESHOLD,
        min_features=MIN_SELECTED_FEATURES,
        max_features=MAX_SELECTED_FEATURES,
        random_state=RANDOM_STATE,
    )
    sel_names_full = [feature_names[i] for i in sel_idx_full]

    # Pick the most-frequent best-hyperparameter combo across folds.
    param_counter = Counter(tuple(sorted(rec["best_params"].items())) for rec in fold_records)
    most_common = dict(param_counter.most_common(1)[0][0])
    final_rf = RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=N_JOBS,
        **most_common,
    )
    final_rf.fit(X[:, sel_idx_full], y)

    final_features_df = pd.DataFrame({
        "feature": sel_names_full,
        "stability_frequency": sel_freq_full[sel_idx_full],
        "rf_importance": final_rf.feature_importances_,
    }).sort_values("rf_importance", ascending=False)
    final_features_df.to_csv(out_dir / "final_selected_features.csv", index=False)

    # ----- write outputs -----
    metrics_out = {
        "outcome": outcome,
        "n_samples": int(len(y)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "prevalence": float(y.mean()),
        "outer_cv": {
            "n_splits": OUTER_N_SPLITS,
            "n_repeats": OUTER_N_REPEATS,
            "n_total_folds": int(outer.get_n_splits()),
        },
        "inner_cv": {"n_splits": INNER_N_SPLITS},
        "youden_threshold": youden_thr,
        "fold_aggregate_mean_std": fold_aggregate,
        "pooled_metrics_threshold_0.5": pooled_05,
        "pooled_metrics_youden_threshold": pooled_youden,
        "n_selected_features_final": int(len(sel_names_full)),
        "final_best_params": most_common,
        "elapsed_seconds": elapsed,
    }
    write_json(metrics_out, out_dir / "metrics.json")
    df_folds.to_csv(out_dir / "cv_fold_metrics.csv", index=False)
    pd.DataFrame(cv_predictions).to_csv(out_dir / "cv_predictions.csv", index=False)
    write_json(
        [{"fold": rec["fold"], "best_params": rec["best_params"],
          "inner_best_score": rec["inner_best_score"]} for rec in fold_records],
        out_dir / "best_hyperparameters.json",
    )

    # ----- plots -----
    plot_roc_curves(per_fold_probs, y_true_pool, y_prob_pool,
                    plots_dir / "roc_curves.png",
                    f"ROC — {outcome} (Random Forest, nested CV)")
    plot_pr_curve(y_true_pool, y_prob_pool,
                  plots_dir / "pr_curve.png",
                  f"Precision-Recall — {outcome}")
    cm = np.array([[pooled_05["confusion_matrix"]["tn"], pooled_05["confusion_matrix"]["fp"]],
                   [pooled_05["confusion_matrix"]["fn"], pooled_05["confusion_matrix"]["tp"]]])
    plot_confusion_matrix(cm, plots_dir / "confusion_matrix_thr0.5.png",
                          f"Confusion matrix — {outcome} (thr=0.50)")
    cm_y = np.array([[pooled_youden["confusion_matrix"]["tn"], pooled_youden["confusion_matrix"]["fp"]],
                     [pooled_youden["confusion_matrix"]["fn"], pooled_youden["confusion_matrix"]["tp"]]])
    plot_confusion_matrix(cm_y, plots_dir / "confusion_matrix_youden.png",
                          f"Confusion matrix — {outcome} (thr={youden_thr:.2f})")
    plot_top_features(final_features_df["feature"].tolist(),
                      final_features_df["rf_importance"].tolist(),
                      plots_dir / "rf_feature_importance.png",
                      f"Top RF feature importance — {outcome}",
                      "Mean decrease in impurity")
    top_stab = sel_table.head(20)
    plot_top_features(top_stab["feature"].tolist(),
                      top_stab["mean_selection_freq"].tolist(),
                      plots_dir / "stability_selection_frequency.png",
                      f"Stability selection frequency — {outcome}",
                      "Mean L1-logistic selection frequency")

    # ----- model artifact -----
    joblib.dump({
        "model": final_rf,
        "scaler_for_l1": scaler_full,   # only used during selection; RF uses raw values
        "selected_feature_names": sel_names_full,
        "all_feature_names": feature_names,
        "outcome": outcome,
        "best_params": most_common,
        "youden_threshold": youden_thr,
    }, out_dir / "model.joblib")

    print(f"  pooled AUC={pooled_05['roc_auc']:.3f}  pooled BalAcc={pooled_05['balanced_accuracy']:.3f}  "
          f"final features={len(sel_names_full)}")
    return {
        "outcome": outcome,
        "n": len(y),
        "n_positive": n_pos,
        "prevalence": float(y.mean()),
        "auc_mean": fold_aggregate["roc_auc"]["mean"],
        "auc_std": fold_aggregate["roc_auc"]["std"],
        "balanced_accuracy_mean": fold_aggregate["balanced_accuracy"]["mean"],
        "balanced_accuracy_std": fold_aggregate["balanced_accuracy"]["std"],
        "f1_mean": fold_aggregate["f1"]["mean"],
        "sensitivity_mean": fold_aggregate["sensitivity"]["mean"],
        "specificity_mean": fold_aggregate["specificity"]["mean"],
        "pooled_auc": pooled_05["roc_auc"],
        "pooled_average_precision": pooled_05["average_precision"],
        "n_selected_features_final": len(sel_names_full),
    }


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcomes", nargs="+", default=None,
                        help="Subset of outcome columns to train (default: all 7).")
    args = parser.parse_args()

    df = load_data()
    feats = feature_columns(df)
    print(f"Loaded {DATA_PATH}: shape={df.shape}, candidate features={len(feats)}")

    targets = args.outcomes or OUTCOME_COLUMNS
    summary = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for outcome in targets:
        s = train_outcome(df, outcome, feats)
        summary.append(s)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)
    write_json({"summary": summary,
                "config": {
                    "outer_cv": f"RepeatedStratifiedKFold(n_splits={OUTER_N_SPLITS}, n_repeats={OUTER_N_REPEATS})",
                    "inner_cv": f"StratifiedKFold(n_splits={INNER_N_SPLITS})",
                    "stability_selection": {
                        "n_subsamples": STABILITY_N_SUBSAMPLES,
                        "subsample_frac": STABILITY_SUBSAMPLE_FRAC,
                        "c_grid": list(STABILITY_C_GRID),
                        "threshold": STABILITY_THRESHOLD,
                    },
                    "rf_param_grid": RF_PARAM_GRID,
                    "random_state": RANDOM_STATE,
                }},
               RESULTS_DIR / "summary_metrics.json")
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print(f"\nResults written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
