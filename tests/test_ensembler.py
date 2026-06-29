"""Phase 6 ensembling: weighted avg, rank avg, and stacking correctness."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from agent.tools import ensembler as ens


def test_softmax_weights_higher_better():
    w = ens.softmax_weights([0.9, 0.8], higher_is_better=True)
    assert w[0] > w[1]
    assert pytest.approx(w.sum()) == 1.0


def test_softmax_weights_lower_better():
    w = ens.softmax_weights([0.1, 0.5], higher_is_better=False)
    assert w[0] > w[1]  # lower error -> more weight


def test_weighted_average():
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    out = ens.weighted_average([a, b], np.array([0.5, 0.5]))
    assert out.tolist() == [2.0, 3.0]


def test_rank_average():
    a = np.array([0.1, 0.2, 0.3])
    b = np.array([0.3, 0.2, 0.1])
    out = ens.rank_average([a, b])
    # Symmetric inputs -> equal averaged ranks.
    assert pytest.approx(out[0]) == out[2]


def test_build_meta_features_shapes():
    oof = [np.zeros(10), np.ones(10)]
    foldtest = [np.zeros(4), np.ones(4)]
    x_meta, x_test = ens.build_meta_features(oof, foldtest)
    assert x_meta.shape == (10, 2)
    assert x_test.shape == (4, 2)


def test_stacking_uses_foldtest_not_fulltrain():
    """Meta TEST features must come from foldtest_list. If we corrupt foldtest,
    the stacked test prediction must change — proving fulltrain preds aren't used."""
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, size=n)
    # Two base OOF signals correlated with y.
    oof1 = y * 0.6 + rng.normal(0, 0.1, n)
    oof2 = y * 0.4 + rng.normal(0, 0.1, n)
    foldtest_a = [np.full(5, 0.1), np.full(5, 0.1)]
    foldtest_b = [np.full(5, 0.9), np.full(5, 0.9)]

    _, test_a = ens.stacking([oof1, oof2], foldtest_a, y, task_kind="classification")
    _, test_b = ens.stacking([oof1, oof2], foldtest_b, y, task_kind="classification")
    assert not np.allclose(test_a, test_b)


def test_stacking_oof_is_cross_validated():
    rng = np.random.default_rng(1)
    n = 150
    y = rng.integers(0, 2, size=n)
    oof1 = y * 0.7 + rng.normal(0, 0.1, n)
    oof2 = y * 0.5 + rng.normal(0, 0.1, n)
    foldtest = [np.full(3, 0.5), np.full(3, 0.5)]
    oof_meta, test_pred = ens.stacking([oof1, oof2], foldtest, y, task_kind="classification")
    assert len(oof_meta) == n
    assert len(test_pred) == 3


def test_run_ensemble_dispatch_weighted():
    oof = [np.array([0.2, 0.8]), np.array([0.3, 0.7])]
    test = [np.array([0.5]), np.array([0.6])]
    used, blended, pred = ens.run_ensemble(
        "weighted_average", oof_list=oof, test_list=test, foldtest_list=[None, None],
        scores=[0.9, 0.8], y=np.array([0, 1]), task_kind="classification",
        higher_is_better=True,
    )
    assert used == "weighted_average"
    assert len(pred) == 1


def test_run_ensemble_none_picks_best():
    oof = [np.array([0.0, 1.0]), np.array([0.0, 1.0])]
    test = [np.array([0.1]), np.array([0.9])]
    used, blended, pred = ens.run_ensemble(
        "none", oof_list=oof, test_list=test, foldtest_list=[None, None],
        scores=[0.95, 0.80], y=np.array([0, 1]), task_kind="classification",
        higher_is_better=True,
    )
    assert used == "none"
    assert pred[0] == 0.1  # best model (index 0) chosen


def test_run_ensemble_stacking_falls_back_without_foldtest():
    oof = [np.array([0.2, 0.8]), np.array([0.3, 0.7])]
    test = [np.array([0.5]), np.array([0.6])]
    used, _, _ = ens.run_ensemble(
        "stacking", oof_list=oof, test_list=test, foldtest_list=[None, None],
        scores=[0.9, 0.8], y=np.array([0, 1]), task_kind="classification",
        higher_is_better=True,
    )
    assert used == "weighted_average"  # fell back
