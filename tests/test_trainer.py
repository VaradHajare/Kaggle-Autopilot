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
