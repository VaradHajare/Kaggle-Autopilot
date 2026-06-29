"""Phase 7 submission generation: post-processing, build, validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent.errors import SubmissionError
from agent.tools import submitter as sub


def test_postprocess_proba_clips():
    out = sub.postprocess(np.array([0.0, 1.0, 0.5]), metric_kind="proba")
    assert out.min() >= 1e-6 and out.max() <= 1 - 1e-6


def test_postprocess_value_raw():
    arr = np.array([1.5, -3.2])
    assert np.array_equal(sub.postprocess(arr, metric_kind="value"), arr)


def test_postprocess_label_binary_with_domain():
    out = sub.postprocess(np.array([0.2, 0.8]), metric_kind="label",
                          label_domain=["no", "yes"])
    assert out.tolist() == ["no", "yes"]


def test_postprocess_label_multiclass_argmax():
    preds = np.array([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]])
    out = sub.postprocess(preds, metric_kind="label", label_domain=[10, 20, 30])
    assert out.tolist() == [20, 10]


def test_build_submission_single_target():
    sample = pd.DataFrame({"id": [1, 2], "y": [0, 0]})
    out = sub.build_submission(sample, id_column="id", target_columns=["y"],
                               preds=np.array([0.3, 0.7]))
    assert list(out.columns) == ["id", "y"]
    assert out["y"].tolist() == [0.3, 0.7]


def test_build_submission_multi_target():
    sample = pd.DataFrame({"id": [1, 2], "a": [0, 0], "b": [0, 0]})
    preds = np.array([[0.1, 0.2], [0.3, 0.4]])
    out = sub.build_submission(sample, id_column="id", target_columns=["a", "b"], preds=preds)
    assert out["a"].tolist() == [0.1, 0.3]
    assert out["b"].tolist() == [0.2, 0.4]


def test_build_submission_rejects_2d_for_single_target():
    sample = pd.DataFrame({"id": [1, 2], "y": [0, 0]})
    with pytest.raises(SubmissionError):
        sub.build_submission(sample, id_column="id", target_columns=["y"],
                             preds=np.array([[0.1, 0.9], [0.2, 0.8]]))


def test_validate_column_mismatch():
    sample = pd.DataFrame({"id": [1], "y": [0]})
    bad = pd.DataFrame({"id": [1], "z": [0]})
    with pytest.raises(SubmissionError, match="Column mismatch"):
        sub.validate_submission(bad, sample)


def test_validate_row_count():
    sample = pd.DataFrame({"id": [1, 2], "y": [0, 0]})
    bad = pd.DataFrame({"id": [1], "y": [0.5]})
    with pytest.raises(SubmissionError, match="Row count"):
        sub.validate_submission(bad, sample)


def test_validate_nan_and_inf():
    sample = pd.DataFrame({"id": [1, 2], "y": [0, 0]})
    with pytest.raises(SubmissionError, match="NaN"):
        sub.validate_submission(pd.DataFrame({"id": [1, 2], "y": [0.5, np.nan]}), sample)
    with pytest.raises(SubmissionError, match="infinite"):
        sub.validate_submission(pd.DataFrame({"id": [1, 2], "y": [0.5, np.inf]}), sample)


def test_submission_filename_encodes_score():
    assert sub.submission_filename("20260101_000000", 0.9321).endswith("cv0.93210.csv")
