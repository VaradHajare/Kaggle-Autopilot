"""Phase 7 — Submission generation: map predictions into the sample_submission
format, apply metric-appropriate post-processing, and validate before writing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agent.errors import SubmissionError

PROBA_CLIP = (1e-6, 1.0 - 1e-6)


def postprocess(
    preds: np.ndarray,
    *,
    metric_kind: str,
    label_domain: list | None = None,
) -> np.ndarray:
    """Apply metric-driven post-processing.

    proba  -> clip to [1e-6, 1-1e-6]
    value  -> raw floats
    label  -> argmax/threshold mapped into label_domain
    """
    preds = np.asarray(preds)
    if metric_kind == "proba":
        return np.clip(preds, *PROBA_CLIP)
    if metric_kind == "value":
        return preds
    # label
    if preds.ndim == 2:
        idx = preds.argmax(axis=1)
    else:
        idx = (preds > 0.5).astype(int)
    if label_domain is not None:
        domain = list(label_domain)
        return np.array([domain[i] if i < len(domain) else domain[-1] for i in idx])
    return idx


def build_submission(
    sample: pd.DataFrame,
    *,
    id_column: str,
    target_columns: list[str],
    preds: np.ndarray,
) -> pd.DataFrame:
    """Construct a submission DataFrame matching `sample`'s columns and row order."""
    out = pd.DataFrame()
    out[id_column] = sample[id_column].values
    preds = np.asarray(preds)

    if len(target_columns) == 1:
        if preds.ndim == 2 and preds.shape[1] == 1:
            preds = preds.ravel()
        if preds.ndim == 2:
            raise SubmissionError(
                "Single target column but 2D predictions — apply label/proba "
                "reduction before build_submission."
            )
        out[target_columns[0]] = preds
    else:
        if preds.ndim != 2 or preds.shape[1] != len(target_columns):
            raise SubmissionError(
                f"Expected predictions of shape (n, {len(target_columns)}), "
                f"got {preds.shape}."
            )
        for i, col in enumerate(target_columns):
            out[col] = preds[:, i]
    return out[sample.columns.tolist()]


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Raise SubmissionError if the submission doesn't match the sample format."""
    if list(submission.columns) != list(sample.columns):
        raise SubmissionError(
            f"Column mismatch: {list(submission.columns)} != {list(sample.columns)}"
        )
    if len(submission) != len(sample):
        raise SubmissionError(
            f"Row count {len(submission)} != expected {len(sample)}"
        )
    numeric = submission.select_dtypes(include=[np.number])
    if numeric.isna().any().any():
        raise SubmissionError("Submission contains NaN values.")
    if np.isinf(numeric.to_numpy()).any():
        raise SubmissionError("Submission contains infinite values.")


def submission_filename(timestamp: str, cv_score: float) -> str:
    """Filename encodes the CV score for easy identification."""
    return f"submission_{timestamp}_cv{cv_score:.5f}.csv"
