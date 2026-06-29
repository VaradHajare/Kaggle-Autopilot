"""Phase 5b — Training & Cross-Validation.

Per model: an Optuna study (study-level timeout, not a manual clock) with an
early-stopping heuristic, CV with deferred target/group encoding fit IN-FOLD,
then a full-train refit. Artifacts written per model:

  <name>_oof.npy        OOF predictions (k-fold)
  <name>_test.npy       test predictions from the full-train model
  <name>_fold<k>.joblib one model per fold (used for correct stacking)
  <name>_fulltrain.joblib

The in-fold encoding path is mandatory: deferred ops are fit on each fold's
training rows only — never on full train inside the CV loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

from agent.memory import CVResult, FEOperation
from agent.tools.feature_engineering import apply_deferred_op

optuna.logging.set_verbosity(optuna.logging.WARNING)

EARLY_STOP_MIN_TRIALS = 10
EARLY_STOP_REL = 0.15


# ----------------------------------------------------------------- metrics
@dataclass
class Metric:
    name: str
    higher_is_better: bool
    kind: str  # "value" | "proba" | "label"

    def score(self, y_true: np.ndarray, preds: np.ndarray) -> float:
        n = self.name
        if n == "rmse":
            return float(np.sqrt(mean_squared_error(y_true, preds)))
        if n == "mae":
            return float(mean_absolute_error(y_true, preds))
        if n == "r2":
            return float(r2_score(y_true, preds))
        if n == "auc":
            if preds.ndim == 1:
                return float(roc_auc_score(y_true, preds))
            return float(roc_auc_score(y_true, preds, multi_class="ovr"))
        if n == "logloss":
            return float(log_loss(y_true, preds))
        labels = _to_labels(preds)
        if n == "f1":
            avg = "binary" if preds.ndim == 1 else "macro"
            return float(f1_score(y_true, labels, average=avg))
        return float(accuracy_score(y_true, labels))  # accuracy default


_METRICS = {
    "rmse": Metric("rmse", False, "value"),
    "mae": Metric("mae", False, "value"),
    "r2": Metric("r2", True, "value"),
    "auc": Metric("auc", True, "proba"),
    "logloss": Metric("logloss", False, "proba"),
    "accuracy": Metric("accuracy", True, "label"),
    "f1": Metric("f1", True, "label"),
}


def resolve_metric(eval_metric: str, task_kind: str) -> Metric:
    key = (eval_metric or "").strip().lower().replace(" ", "")
    aliases = {
        "rootmeansquarederror": "rmse", "meanabsoluteerror": "mae",
        "logarithmicloss": "logloss", "areaunderthecurve": "auc",
        "rocauc": "auc",
    }
    key = aliases.get(key, key)
    if key in _METRICS:
        return _METRICS[key]

    # Kaggle phrases metric names freely ("Categorization Accuracy", "Mean F1
    # Score", "Area Under Curve", ...). Detect the family by substring so a
    # label-based metric is never mis-resolved to a proba metric — which would
    # write probabilities into an accuracy/F1 submission and score ~0 on the LB.
    substrings = (
        ("rmse", "rmse"), ("rootmeansquared", "rmse"),
        ("mae", "mae"), ("meanabsolute", "mae"),
        ("r2", "r2"),
        ("logloss", "logloss"), ("logarithmicloss", "logloss"),
        ("crossentropy", "logloss"),
        ("auc", "auc"), ("areaunder", "auc"),
        ("f1", "f1"),
        ("accuracy", "accuracy"),
    )
    for needle, name in substrings:
        if needle in key:
            return _METRICS[name]

    return _METRICS["rmse"] if task_kind == "regression" else _METRICS["auc"]


def _to_labels(preds: np.ndarray) -> np.ndarray:
    if preds.ndim == 1:
        return (preds > 0.5).astype(int)
    return preds.argmax(axis=1)


# ----------------------------------------------------------------- estimators
def build_estimator(name: str, params: dict, *, seed: int = 42):
    p = dict(params)
    if name == "LGBMClassifier":
        from lightgbm import LGBMClassifier  # noqa: PLC0415
        return LGBMClassifier(random_state=seed, n_jobs=1, verbose=-1, **p)
    if name == "LGBMRegressor":
        from lightgbm import LGBMRegressor  # noqa: PLC0415
        return LGBMRegressor(random_state=seed, n_jobs=1, verbose=-1, **p)
    if name == "XGBClassifier":
        from xgboost import XGBClassifier  # noqa: PLC0415
        return XGBClassifier(random_state=seed, n_jobs=1, **p)
    if name == "XGBRegressor":
        from xgboost import XGBRegressor  # noqa: PLC0415
        return XGBRegressor(random_state=seed, n_jobs=1, **p)
    if name == "CatBoostClassifier":
        from catboost import CatBoostClassifier  # noqa: PLC0415
        # allow_writing_files=False stops CatBoost dumping a catboost_info/ dir
        # into the cwd (repo root) — all artifacts must stay under runs/<slug>/.
        return CatBoostClassifier(
            random_state=seed, verbose=0, allow_writing_files=False, **p
        )
    if name == "CatBoostRegressor":
        from catboost import CatBoostRegressor  # noqa: PLC0415
        return CatBoostRegressor(
            random_state=seed, verbose=0, allow_writing_files=False, **p
        )
    if name == "RandomForestClassifier":
        from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415
        return RandomForestClassifier(random_state=seed, n_jobs=1, **p)
    if name == "RandomForestRegressor":
        from sklearn.ensemble import RandomForestRegressor  # noqa: PLC0415
        return RandomForestRegressor(random_state=seed, n_jobs=1, **p)
    if name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
        return LogisticRegression(max_iter=1000, **p)
    if name == "Ridge":
        from sklearn.linear_model import Ridge  # noqa: PLC0415
        return Ridge(random_state=seed, **p)
    raise ValueError(f"Unknown estimator {name!r}")


def sample_params(trial: optuna.Trial, space: dict) -> dict:
    """Sample one parameter set from a configs/models.yaml search space."""
    params: dict = {}
    for pname, spec in space.items():
        t = spec["type"]
        if t == "int":
            params[pname] = trial.suggest_int(pname, spec["low"], spec["high"])
        elif t == "float":
            params[pname] = trial.suggest_float(
                pname, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif t == "categorical":
            params[pname] = trial.suggest_categorical(pname, spec["choices"])
    return params


# ----------------------------------------------------------------- CV
def make_cv(task_kind: str, is_time_series: bool, n_splits: int, seed: int):
    if is_time_series:
        return TimeSeriesSplit(n_splits=n_splits)
    if task_kind == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _predict(est, X: pd.DataFrame, task_kind: str) -> np.ndarray:
    if task_kind == "regression":
        return np.asarray(est.predict(X))
    proba = np.asarray(est.predict_proba(X))
    return proba[:, 1] if proba.shape[1] == 2 else proba


def _augment(
    base: pd.DataFrame, raw: pd.DataFrame, fit_raw: pd.DataFrame,
    fit_y: pd.Series, deferred: list[FEOperation],
) -> pd.DataFrame:
    """Append in-fold deferred encodings to a base feature frame."""
    if not deferred:
        return base
    out = base.copy()
    for op in deferred:
        out[op.output_name] = apply_deferred_op(op, fit_raw, fit_y, raw).to_numpy()
    return out


# ----------------------------------------------------------------- training
@dataclass
class TrainArtifacts:
    cv_result: CVResult
    oof: np.ndarray
    test_pred: np.ndarray | None
    fold_test_preds: list[np.ndarray] = field(default_factory=list)


def train_model(
    *,
    name: str,
    space: dict,
    X: pd.DataFrame,
    y: pd.Series,
    raw_train: pd.DataFrame,
    X_test: pd.DataFrame | None,
    raw_test: pd.DataFrame | None,
    deferred_ops: list[FEOperation],
    task_kind: str,
    metric: Metric,
    cv,
    models_dir: Path,
    n_trials: int = 50,
    timeout: int | None = 3600,
    best_prior_score: float | None = None,
) -> TrainArtifacts:
    """Optuna-tune `name`, refit on full train, persist artifacts, return results."""
    models_dir.mkdir(parents=True, exist_ok=True)
    direction = "maximize" if metric.higher_is_better else "minimize"

    def run_cv(params: dict) -> tuple[np.ndarray, list]:
        oof = (np.zeros(len(y)) if task_kind != "classification"
               else _alloc_oof(y, X, task_kind))
        fold_models = []
        for tr_idx, va_idx in cv.split(X, y):
            Xtr = _augment(X.iloc[tr_idx], raw_train.iloc[tr_idx],
                           raw_train.iloc[tr_idx], y.iloc[tr_idx], deferred_ops)
            Xva = _augment(X.iloc[va_idx], raw_train.iloc[va_idx],
                           raw_train.iloc[tr_idx], y.iloc[tr_idx], deferred_ops)
            est = build_estimator(name, params)
            est.fit(Xtr, y.iloc[tr_idx])
            oof[va_idx] = _predict(est, Xva, task_kind)
            fold_models.append(est)
        return oof, fold_models

    # --- Optuna study with early-stopping callback.
    trial_scores: list[float] = []
    pruned_early = {"flag": False}

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial, space)
        oof, _ = run_cv(params)
        s = metric.score(y.to_numpy(), oof)
        trial_scores.append(s)
        return s

    def early_stop(study, trial) -> None:
        if best_prior_score is None or len(trial_scores) < EARLY_STOP_MIN_TRIALS:
            return
        median = float(np.median(trial_scores[:EARLY_STOP_MIN_TRIALS]))
        if metric.higher_is_better:
            worse = median < best_prior_score * (1 - EARLY_STOP_REL)
        else:
            worse = median > best_prior_score * (1 + EARLY_STOP_REL)
        if worse:
            pruned_early["flag"] = True
            study.stop()

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[early_stop])

    best_params = study.best_params
    # --- Refit CV with best params to produce OOF + per-fold models + fold test preds.
    oof, fold_models = run_cv(best_params)
    fold_test_preds: list[np.ndarray] = []
    for k, est in enumerate(fold_models):
        joblib.dump(est, models_dir / f"{name}_fold{k}.joblib")
        if X_test is not None:
            Xte = _augment(X_test, raw_test, raw_train, y, deferred_ops)
            fold_test_preds.append(_predict(est, Xte, task_kind))
    if fold_test_preds:
        # Fold-averaged test predictions — the correct stacking inference input.
        np.save(models_dir / f"{name}_foldtest.npy", np.mean(fold_test_preds, axis=0))

    # --- Full-train refit (deferred ops fit on full train).
    Xfull = _augment(X, raw_train, raw_train, y, deferred_ops)
    full_est = build_estimator(name, best_params)
    full_est.fit(Xfull, y)
    joblib.dump(full_est, models_dir / f"{name}_fulltrain.joblib")

    test_pred = None
    if X_test is not None:
        Xte = _augment(X_test, raw_test, raw_train, y, deferred_ops)
        test_pred = _predict(full_est, Xte, task_kind)
        np.save(models_dir / f"{name}_test.npy", test_pred)
    np.save(models_dir / f"{name}_oof.npy", oof)

    final_score = metric.score(y.to_numpy(), oof)
    result = CVResult(
        model=name,
        oof_score=final_score,
        status="PRUNED_EARLY" if pruned_early["flag"] else "COMPLETE",
        best_params=best_params,
        n_trials=len(study.trials),
    )
    return TrainArtifacts(result, oof, test_pred, fold_test_preds)


def _alloc_oof(y: pd.Series, X: pd.DataFrame, task_kind: str) -> np.ndarray:
    n_classes = y.nunique()
    if task_kind == "classification" and n_classes > 2:
        return np.zeros((len(y), n_classes))
    return np.zeros(len(y))


# ----------------------------------------------------------------- Phase 5a probe
def probe_feature_importance(
    name: str, X: pd.DataFrame, y: pd.Series, *, task_kind: str, seed: int = 42
) -> pd.Series:
    """Fast importance probe: fit the model with default params on a 3-fold split
    and return mean importances per feature (descending).

    Uses native feature_importances_ when available, else permutation importance.
    """
    from sklearn.inspection import permutation_importance  # noqa: PLC0415

    cv = make_cv(task_kind, is_time_series=False, n_splits=3, seed=seed)
    totals = np.zeros(X.shape[1])
    n = 0
    for tr_idx, va_idx in cv.split(X, y):
        est = build_estimator(name, {})
        est.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        if hasattr(est, "feature_importances_"):
            totals += np.asarray(est.feature_importances_, dtype=float)
        else:
            r = permutation_importance(
                est, X.iloc[va_idx], y.iloc[va_idx], n_repeats=3, random_state=seed
            )
            totals += r.importances_mean
        n += 1
    importances = pd.Series(totals / max(n, 1), index=X.columns)
    return importances.sort_values(ascending=False)
