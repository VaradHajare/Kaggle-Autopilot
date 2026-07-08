"""Phase 5 trainer: metrics, CV, full train loop on sklearn toy datasets,
in-fold deferred encoding, early stopping, and the importance probe.

Per FIX F1 the regression fixture is california_housing (boston was removed in
scikit-learn 1.2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import fetch_california_housing, load_iris, make_classification
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

from agent.config import load_model_search_spaces
from agent.memory import FEOperation
from agent.tools import trainer as tr

SPACES = load_model_search_spaces()


# ----------------------------------------------------------------- metrics / cv
def test_resolve_metric():
    assert tr.resolve_metric("AUC", "classification").name == "auc"
    assert tr.resolve_metric("RMSE", "regression").name == "rmse"
    assert tr.resolve_metric("weird", "regression").name == "rmse"
    assert tr.resolve_metric("weird", "classification").name == "auc"


def test_catboost_estimator_disables_file_writing():
    # CatBoost must not dump catboost_info/ into the cwd (repo root); all
    # artifacts belong under runs/<slug>/.
    pytest.importorskip("catboost")
    for name in ("CatBoostClassifier", "CatBoostRegressor"):
        est = tr.build_estimator(name, {"iterations": 10})
        assert est.get_params().get("allow_writing_files") is False


def test_resolve_metric_kaggle_phrasings_map_to_label_kind():
    # Regression test for the Titanic LB=0.0 bug: "Categorization Accuracy"
    # must resolve to the label metric, not fall through to proba/AUC.
    m = tr.resolve_metric("Categorization Accuracy", "classification")
    assert m.name == "accuracy"
    assert m.kind == "label"
    assert tr.resolve_metric("Mean F1 Score", "classification").name == "f1"
    assert tr.resolve_metric("Mean F1 Score", "classification").kind == "label"
    assert tr.resolve_metric("Area Under Curve", "classification").name == "auc"
    assert tr.resolve_metric("Root Mean Squared Error", "regression").name == "rmse"


def test_make_cv_types():
    assert isinstance(tr.make_cv("classification", False, 5, 42), StratifiedKFold)
    assert isinstance(tr.make_cv("regression", False, 5, 42), KFold)
    assert isinstance(tr.make_cv("classification", True, 5, 42), TimeSeriesSplit)


# ----------------------------------------------------------------- binary lgbm
def test_train_model_binary(tmp_path):
    X_arr, y_arr = make_classification(n_samples=200, n_features=8, n_informative=5,
                                       random_state=0)
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(8)])
    y = pd.Series(y_arr)
    metric = tr.resolve_metric("AUC", "classification")
    cv = tr.make_cv("classification", False, 5, 42)

    art = tr.train_model(
        name="LGBMClassifier", space=SPACES["LGBMClassifier"], X=X, y=y,
        raw_train=X, X_test=X.head(10), raw_test=X.head(10), deferred_ops=[],
        task_kind="classification", metric=metric, cv=cv, models_dir=tmp_path,
        n_trials=2, timeout=None,
    )
    assert art.cv_result.oof_score > 0.7
    assert (tmp_path / "LGBMClassifier_oof.npy").exists()
    assert (tmp_path / "LGBMClassifier_test.npy").exists()
    assert (tmp_path / "LGBMClassifier_fulltrain.joblib").exists()
    assert (tmp_path / "LGBMClassifier_fold0.joblib").exists()
    assert len(art.fold_test_preds) == 5  # one per fold (for stacking)


# ----------------------------------------------------------------- regression
def test_train_model_regression_california(tmp_path):
    data = fetch_california_housing()
    idx = np.arange(300)
    X = pd.DataFrame(data.data[idx], columns=data.feature_names)
    y = pd.Series(data.target[idx])
    metric = tr.resolve_metric("RMSE", "regression")
    cv = tr.make_cv("regression", False, 5, 42)

    art = tr.train_model(
        name="Ridge", space=SPACES["Ridge"], X=X, y=y, raw_train=X,
        X_test=X.head(5), raw_test=X.head(5), deferred_ops=[],
        task_kind="regression", metric=metric, cv=cv, models_dir=tmp_path,
        n_trials=2, timeout=None,
    )
    assert art.cv_result.oof_score > 0  # finite RMSE
    assert art.test_pred is not None and len(art.test_pred) == 5


# ----------------------------------------------------------------- multiclass
def test_train_model_multiclass_iris(tmp_path):
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    metric = tr.resolve_metric("accuracy", "classification")
    cv = tr.make_cv("classification", False, 5, 42)

    art = tr.train_model(
        name="RandomForestClassifier", space=SPACES["RandomForestClassifier"],
        X=X, y=y, raw_train=X, X_test=None, raw_test=None, deferred_ops=[],
        task_kind="classification", metric=metric, cv=cv, models_dir=tmp_path,
        n_trials=2, timeout=None,
    )
    assert art.cv_result.oof_score > 0.8
    assert art.oof.ndim == 2  # multiclass proba OOF


# ----------------------------------------------------------------- in-fold encode
def test_infold_target_encoding_runs(tmp_path):
    rng = np.random.default_rng(0)
    cat = rng.integers(0, 4, size=200)
    y = pd.Series((cat >= 2).astype(int))  # category predicts target
    X = pd.DataFrame({"noise": rng.normal(size=200)})
    raw = pd.DataFrame({"cat": cat})
    op = FEOperation(operation="target_encode", columns=["cat"], output_name="cat_te")
    metric = tr.resolve_metric("AUC", "classification")
    cv = tr.make_cv("classification", False, 5, 42)

    art = tr.train_model(
        name="LGBMClassifier", space=SPACES["LGBMClassifier"], X=X, y=y,
        raw_train=raw, X_test=X.head(5), raw_test=raw.head(5), deferred_ops=[op],
        task_kind="classification", metric=metric, cv=cv, models_dir=tmp_path,
        n_trials=2, timeout=None,
    )
    # The encoded feature carries signal -> better than chance.
    assert art.cv_result.oof_score > 0.7
    assert len(art.oof) == 200


def test_foldtest_encoding_uses_fold_train_rows(tmp_path, monkeypatch):
    """Regression guard for the stacking meta-test features: each per-fold test
    prediction must encode the test set with THAT fold's own training rows — the
    same distribution the fold model saw at fit time — never the full-train
    encoding. Spies on apply_deferred_op and inspects the fit-row counts used for
    test-set augmentation."""
    rng = np.random.default_rng(0)
    n = 200
    cat = rng.integers(0, 4, size=n)
    y = pd.Series((cat >= 2).astype(int))
    X = pd.DataFrame({"noise": rng.normal(size=n)})
    raw = pd.DataFrame({"cat": cat})
    op = FEOperation(operation="target_encode", columns=["cat"], output_name="cat_te")
    metric = tr.resolve_metric("AUC", "classification")
    cv = tr.make_cv("classification", False, 5, 42)

    real = tr.apply_deferred_op
    test_fit_lengths: list[int] = []

    def spy(op, fit_df, fit_target, transform_df, **kw):
        if len(transform_df) == 5:  # a test-set augmentation call (X_test has 5 rows)
            test_fit_lengths.append(len(fit_df))
        return real(op, fit_df, fit_target, transform_df, **kw)

    monkeypatch.setattr(tr, "apply_deferred_op", spy)

    tr.train_model(
        name="LGBMClassifier", space=SPACES["LGBMClassifier"], X=X, y=y,
        raw_train=raw, X_test=X.head(5), raw_test=raw.head(5), deferred_ops=[op],
        task_kind="classification", metric=metric, cv=cv, models_dir=tmp_path,
        n_trials=1, timeout=None,
    )

    # Five fold-test-pred calls fit on 160 fold-train rows each; one full-train
    # refit call fits on all 200. Before the fix every call fit on all 200.
    fold_train_calls = [ln for ln in test_fit_lengths if ln < n]
    assert len(fold_train_calls) == 5
    assert all(ln == 160 for ln in fold_train_calls)
    assert n in test_fit_lengths  # full-train refit still encodes on every row


# ------------------------------------------------------- time-series OOF coverage
def test_covered_mask_kfold_vs_timeseries():
    y = pd.Series(np.arange(120) % 2)
    kf = tr.make_cv("classification", False, 5, 42)
    assert tr.covered_mask(kf, y).all()  # k-fold validates every row
    ts = tr.make_cv("classification", True, 5, 42)
    m = tr.covered_mask(ts, y)
    assert m.sum() < len(y)  # TimeSeriesSplit leaves the initial block unvalidated
    assert not m[0]          # first row is never in a validation fold
    assert m[-1]             # last row is


def test_timeseries_oof_score_excludes_unvalidated_rows(tmp_path):
    """With TimeSeriesSplit the initial block is never validated, so its OOF
    entries stay at init (0). The reported score must equal a covered-only
    recompute, not the full-array score that includes those zeros."""
    X_arr, y_arr = make_classification(n_samples=180, n_features=6, n_informative=4,
                                       random_state=0)
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(6)])
    y = pd.Series(y_arr)
    metric = tr.resolve_metric("AUC", "classification")
    cv = tr.make_cv("classification", is_time_series=True, n_splits=5, seed=42)

    art = tr.train_model(
        name="LGBMClassifier", space=SPACES["LGBMClassifier"], X=X, y=y,
        raw_train=X, X_test=None, raw_test=None, deferred_ops=[],
        task_kind="classification", metric=metric, cv=cv, models_dir=tmp_path,
        n_trials=2, timeout=None,
    )
    oof = np.load(tmp_path / "LGBMClassifier_oof.npy")
    covered = tr.covered_mask(cv, y)
    assert covered.sum() < len(y)
    full_score = metric.score(y.to_numpy(), oof)
    cov_score = metric.score(y.to_numpy()[covered], oof[covered])
    assert abs(art.cv_result.oof_score - cov_score) < 1e-9  # scored over covered only
    assert abs(full_score - cov_score) > 1e-6  # the zero rows really would change it


# ----------------------------------------------------------------- early stop
def test_early_stopping_prunes_weak_model(tmp_path):
    # Pure noise -> AUC ~0.5. With an unbeatable prior best (1.0), median of the
    # first 10 trials is well below 0.85 -> the study halts early.
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(120, 4)), columns=[f"f{i}" for i in range(4)])
    y = pd.Series(rng.integers(0, 2, size=120))
    metric = tr.resolve_metric("AUC", "classification")
    cv = tr.make_cv("classification", False, 5, 42)

    art = tr.train_model(
        name="LogisticRegression", space=SPACES["LogisticRegression"], X=X, y=y,
        raw_train=X, X_test=None, raw_test=None, deferred_ops=[],
        task_kind="classification", metric=metric, cv=cv, models_dir=tmp_path,
        n_trials=20, timeout=None, best_prior_score=1.0,
    )
    assert art.cv_result.status == "PRUNED_EARLY"
    assert art.cv_result.n_trials <= 12  # stopped shortly after the 10-trial check


def test_probe_importance_sorted(tmp_path):
    X_arr, y_arr = make_classification(n_samples=150, n_features=6, n_informative=4,
                                       random_state=0)
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(6)])
    y = pd.Series(y_arr)
    imp = tr.probe_feature_importance("LGBMClassifier", X, y, task_kind="classification")
    assert list(imp.index) == list(imp.sort_values(ascending=False).index)
    assert len(imp) == 6
