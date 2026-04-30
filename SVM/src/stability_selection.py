"""Stability selection via L1-regularised logistic regression.

Implements a simplified Meinshausen & Buhlmann (2010) procedure:
- repeatedly subsample the training set without replacement
- fit an L1-logistic regression at each value of a regularisation grid
- record, for every feature, the fraction of subsamples in which its
  coefficient was non-zero (per regularisation level)
- a feature's stability score is the maximum of those fractions across the
  C-grid; features whose score exceeds a threshold are retained
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression


def _fit_one(X_sub: np.ndarray, y_sub: np.ndarray, C: float, seed: int) -> np.ndarray | None:
    if len(np.unique(y_sub)) < 2:
        return None
    clf = LogisticRegression(
        solver="saga",
        l1_ratio=1.0,
        C=C,
        class_weight="balanced",
        max_iter=2000,
        tol=1e-3,
        random_state=seed,
    )
    try:
        clf.fit(X_sub, y_sub)
    except Exception:
        return None
    return (np.abs(clf.coef_.ravel()) > 1e-8).astype(np.int64)


def stability_selection(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_subsamples: int,
    subsample_frac: float,
    c_grid,
    threshold: float,
    min_features: int,
    max_features: int,
    random_state: int,
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (selected_indices, selection_frequency_per_feature).

    Parameters
    ----------
    X : (n_samples, n_features) standardised feature matrix
    y : (n_samples,) binary labels
    """
    rng = np.random.default_rng(random_state)
    n_samples, n_features = X.shape
    sub_size = max(10, int(round(subsample_frac * n_samples)))

    # Per-C selection counts so we can take the max across the grid: a feature is
    # "stable" if it is reliably selected at any one regularisation level
    # (closer to the randomised-lasso variant of stability selection).
    counts_per_C = np.zeros((len(c_grid), n_features), dtype=np.int64)
    fits_per_C = np.zeros(len(c_grid), dtype=np.int64)

    # Pre-generate every (subsample, C) job and run in parallel.
    jobs = []
    for b in range(n_subsamples):
        idx = rng.choice(n_samples, size=sub_size, replace=False)
        Xs, ys = X[idx], y[idx]
        seed = int(random_state + b)
        for ci, C in enumerate(c_grid):
            jobs.append((ci, Xs, ys, float(C), seed))

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_fit_one)(Xs, ys, C, seed) for (_, Xs, ys, C, seed) in jobs
    )

    for (ci, _, _, _, _), res in zip(jobs, results):
        if res is None:
            continue
        counts_per_C[ci] += res
        fits_per_C[ci] += 1

    if fits_per_C.sum() == 0:
        # Fallback: no usable subsamples — keep all features
        freq = np.zeros(n_features)
        order = np.arange(n_features)
        return order[:max_features], freq

    # Frequency = max over C-grid of (selected / fits) at that C
    with np.errstate(divide="ignore", invalid="ignore"):
        per_C_freq = np.where(
            fits_per_C[:, None] > 0,
            counts_per_C / np.maximum(fits_per_C, 1)[:, None],
            0.0,
        )
    freq = per_C_freq.max(axis=0)
    above = np.where(freq >= threshold)[0]

    if len(above) >= min_features:
        order = above[np.argsort(-freq[above])]
    else:
        # Not enough features cleared the threshold — fall back to top-k by frequency
        order = np.argsort(-freq)[:min_features]

    if len(order) > max_features:
        order = order[:max_features]

    return order, freq
