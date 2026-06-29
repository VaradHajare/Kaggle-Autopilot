"""Phase 4 — Model Selection.

Pure selection logic: infer task kind, filter LLM-proposed models against the
catalog + available search spaces, and enforce the spec's hard rules (always
include a gradient-boosting model for tabular; gate neural nets).

Estimator construction lives in the trainer (Phase 5), not here.
"""

from __future__ import annotations

from agent.memory import ModelCandidate

# name -> (task kind, family). Used by selection and by the trainer's factory.
MODEL_CATALOG: dict[str, tuple[str, str]] = {
    "LGBMClassifier": ("classification", "gbm"),
    "LGBMRegressor": ("regression", "gbm"),
    "XGBClassifier": ("classification", "gbm"),
    "XGBRegressor": ("regression", "gbm"),
    "CatBoostClassifier": ("classification", "gbm"),
    "CatBoostRegressor": ("regression", "gbm"),
    "RandomForestClassifier": ("classification", "tree"),
    "RandomForestRegressor": ("regression", "tree"),
    "LogisticRegression": ("classification", "linear"),
    "Ridge": ("regression", "linear"),
}

_REGRESSION_METRICS = {"rmse", "mae", "rmsle", "r2", "mse", "smape", "mape"}
_CLASSIFICATION_METRICS = {"auc", "logloss", "accuracy", "f1", "logarithmic loss", "map"}

NEURAL_MIN_ROWS = 5000


def infer_task_kind(problem_type: str, eval_metric: str, n_target_classes: int | None = None) -> str:
    """Return 'classification' or 'regression'."""
    pt = (problem_type or "").lower()
    em = (eval_metric or "").lower()
    if any(k in pt for k in ("regress",)) or em in _REGRESSION_METRICS:
        return "regression"
    if any(k in pt for k in ("class", "binary", "multiclass")) or em in _CLASSIFICATION_METRICS:
        return "classification"
    if n_target_classes is not None:
        return "classification" if n_target_classes <= 50 else "regression"
    return "classification"


def _default_gbm(task_kind: str) -> str:
    return "LGBMClassifier" if task_kind == "classification" else "LGBMRegressor"


def select_models(
    candidates: list[ModelCandidate],
    *,
    task_kind: str,
    search_spaces: dict,
    allow_neural: bool = False,
    n_rows: int = 0,
) -> list[ModelCandidate]:
    """Filter candidates to known, task-appropriate models with a search space,
    then enforce hard rules. Returns a priority-ordered list (1 = highest)."""
    valid: list[ModelCandidate] = []
    seen: set[str] = set()
    for c in sorted(candidates, key=lambda m: m.priority):
        kind_family = MODEL_CATALOG.get(c.model)
        if kind_family is None or c.model in seen:
            continue
        kind, family = kind_family
        if kind != task_kind:
            continue
        if family == "neural" and not (allow_neural and n_rows >= NEURAL_MIN_ROWS):
            continue
        if c.model not in search_spaces:
            continue
        valid.append(c)
        seen.add(c.model)

    # Hard rule: always include a gradient-boosting model for tabular data.
    if not any(MODEL_CATALOG[c.model][1] == "gbm" for c in valid):
        gbm = _default_gbm(task_kind)
        if gbm in search_spaces:
            valid.append(ModelCandidate(model=gbm, priority=0,
                                        rationale="Hard rule: ensure a GBM baseline."))

    # Re-number priorities 1..n in final order.
    valid.sort(key=lambda m: m.priority)
    for i, c in enumerate(valid, start=1):
        c.priority = i
    return valid


def remaining_submission_budget(effective_daily_limit: int, submissions_today: int) -> int:
    return max(0, effective_daily_limit - submissions_today)
