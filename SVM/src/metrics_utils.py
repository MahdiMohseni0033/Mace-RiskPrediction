"""Classification metrics + plotting helpers for publication-quality output."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _safe(fn, *a, **kw):
    try:
        return float(fn(*a, **kw))
    except Exception:
        return float("nan")


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return probability cut-off maximising sensitivity + specificity - 1."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    if len(j) == 0:
        return 0.5
    k = int(np.argmax(j))
    return float(thr[k])


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "roc_auc": _safe(roc_auc_score, y_true, y_prob),
        "average_precision": _safe(average_precision_score, y_true, y_prob),
        "brier_score": _safe(brier_score_loss, y_true, y_prob),
        "balanced_accuracy": _safe(balanced_accuracy_score, y_true, y_pred),
        "accuracy": float((y_true == y_pred).mean()),
        "precision": _safe(precision_score, y_true, y_pred, zero_division=0),
        "recall": _safe(recall_score, y_true, y_pred, zero_division=0),
        "sensitivity": float(sensitivity) if sensitivity == sensitivity else float("nan"),
        "specificity": float(specificity) if specificity == specificity else float("nan"),
        "f1": _safe(f1_score, y_true, y_pred, zero_division=0),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


# ----------------------------- plotting ----------------------------- #

def plot_roc_curves(per_fold_probs, y_true_pooled, y_prob_pooled, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    aucs = []
    for i, (yt, yp) in enumerate(per_fold_probs):
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, yp)
        auc = roc_auc_score(yt, yp)
        aucs.append(auc)
        ax.plot(fpr, tpr, alpha=0.35, lw=1)
    if len(np.unique(y_true_pooled)) >= 2:
        fpr, tpr, _ = roc_curve(y_true_pooled, y_prob_pooled)
        auc = roc_auc_score(y_true_pooled, y_prob_pooled)
        ax.plot(fpr, tpr, color="black", lw=2.2, label=f"Pooled AUC = {auc:.3f}")
    if aucs:
        ax.plot([], [], " ", label=f"Per-fold AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1)
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    if len(np.unique(y_true)) >= 2:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, lw=2, label=f"AP = {ap:.3f}")
        baseline = float(np.mean(y_true))
        ax.hlines(baseline, 0, 1, linestyle="--", color="grey", lw=1, label=f"Prevalence = {baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_top_features(names, values, out_path: Path, title: str, xlabel: str, top_n: int = 20) -> None:
    df = pd.DataFrame({"feature": names, "value": values}).sort_values("value", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(df) + 1)))
    ax.barh(df["feature"][::-1], df["value"][::-1], color="#3b6ea8")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
