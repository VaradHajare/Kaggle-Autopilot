"""Phase 6 — Ensembling.

Strategies: weighted average, rank average, stacking, or none.

Stacking (corrected two-level CV scheme):
  - Meta-learner TRAIN features = the OOF prediction matrix from the k-fold models
    (no row ever sees its own training prediction).
  - Meta-learner TEST features = each base model's fold-AVERAGED test predictions
    (the `<model>_foldtest.npy` arrays) — NOT the full-train model's predictions.
  - Full-train models are not used for stacking inference.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import cross_val_predict


def softmax_weights(scores: list[float], higher_is_better: bool) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    if not higher_is_better:
        s = -s
    s = s - s.max()
    e = np.exp(s)
    return e / e.sum()


def weighted_average(arrays: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    return sum(w * a for w, a in zip(weights, arrays))


def rank_average(arrays: list[np.ndarray]) -> np.ndarray:
    """Mean of per-array percentile ranks (1D arrays only)."""
    ranks = [rankdata(a) / len(a) for a in arrays]
    return np.mean(ranks, axis=0)


def _as2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def build_meta_features(
    oof_list: list[np.ndarray], foldtest_list: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Meta train = horizontally-stacked OOF arrays; meta test = stacked
    fold-averaged test arrays. Both stacked in the same model order."""
    x_meta = np.hstack([_as2d(o) for o in oof_list])
    x_test = np.hstack([_as2d(t) for t in foldtest_list])
    return x_meta, x_test


def stacking(
    oof_list: list[np.ndarray],
    foldtest_list: list[np.ndarray],
    y: np.ndarray,
    *,
    task_kind: str,
    seed: int = 42,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a meta-learner on the OOF stack; return (oof_meta_pred, test_pred).

    oof_meta_pred is produced via cross_val_predict on the OOF stack so the
    blended score itself is leakage-free.
    """
    x_meta, x_test = build_meta_features(oof_list, foldtest_list)
    if task_kind == "classification":
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

        meta = LogisticRegression(max_iter=1000)
        meta.fit(x_meta, y)
        oof_meta = cross_val_predict(
            LogisticRegression(max_iter=1000), x_meta, y, cv=n_splits, method="predict_proba"
        )
        oof_meta = oof_meta[:, 1] if oof_meta.shape[1] == 2 else oof_meta
        test_pred = meta.predict_proba(x_test)
        test_pred = test_pred[:, 1] if test_pred.shape[1] == 2 else test_pred
    else:
        from sklearn.linear_model import Ridge  # noqa: PLC0415

        meta = Ridge()
        meta.fit(x_meta, y)
        oof_meta = cross_val_predict(Ridge(), x_meta, y, cv=n_splits)
        test_pred = meta.predict(x_test)
    return oof_meta, test_pred


def run_ensemble(
    method: str,
    *,
    oof_list: list[np.ndarray],
    test_list: list[np.ndarray],
    foldtest_list: list[np.ndarray],
    scores: list[float],
    y: np.ndarray,
    task_kind: str,
    higher_is_better: bool,
    seed: int = 42,
    n_splits: int = 5,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Dispatch to a strategy. Returns (method_used, blended_oof, test_pred).

    Falls back to weighted average if stacking/rank inputs are unusable.
    """
    if method == "stacking" and len(oof_list) >= 2 and all(t is not None for t in foldtest_list):
        oof_meta, test_pred = stacking(
            oof_list, foldtest_list, y, task_kind=task_kind, seed=seed, n_splits=n_splits
        )
        return "stacking", oof_meta, test_pred

    if method == "rank_average" and all(o.ndim == 1 for o in oof_list):
        return "rank_average", rank_average(oof_list), rank_average(test_list)

    if method == "none":
        best = _best_index(scores, higher_is_better)
        return "none", oof_list[best], test_list[best]

    # Default: weighted average.
    w = softmax_weights(scores, higher_is_better)
    return "weighted_average", weighted_average(oof_list, w), weighted_average(test_list, w)


def _best_index(scores: list[float], higher_is_better: bool) -> int:
    return int(np.argmax(scores) if higher_is_better else np.argmin(scores))
