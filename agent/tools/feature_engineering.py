"""Phase 3 — Feature Engineering registry and executor.

Two execution paths:
  - IMMEDIATE ops run now, fit on train and applied to test (strict no-leakage).
  - DEFERRED ops (target / group encoding) are NEVER fit on full train. They are
    validated and stored in RunState.deferred_fe_ops, then applied inside the
    Phase 5b CV fold loop via `apply_deferred_op` (fit on fold-train only).

Invoking a deferred op through the immediate executor raises DeferredOpError.
Unrecognized operation names are rejected (caller logs and skips them).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agent.errors import DeferredOpError
from agent.memory import FEOperation

DEFERRED_OPS = {"target_encode", "group_mean_encode", "group_std_encode"}
TIME_SERIES_OPS = {"lag", "rolling_mean"}
# Ops that overfit when pointed at an identity-like high-cardinality column
# (e.g. TF-IDF/SVD on Name or Ticket): tokens are near-unique, so the model
# memorizes rows that don't recur in a disjoint test set. Skipped in Phase 3.
IDENTITY_UNSAFE_OPS = {"tfidf_svd"}


def is_identity_unsafe(op: FEOperation, identity_cols: set[str]) -> bool:
    """True when `op` would overfit by operating on an identity-like column."""
    return op.operation in IDENTITY_UNSAFE_OPS and any(
        c in identity_cols for c in op.columns
    )

# operation -> required number of input columns.
REGISTRY: dict[str, int] = {
    "log_transform": 1,
    "sqrt_transform": 1,
    "interaction_multiply": 2,
    "interaction_ratio": 2,
    "polynomial_degree2": 1,
    "bin_equal_width": 1,
    "bin_equal_freq": 1,
    "group_mean_encode": 2,
    "group_std_encode": 2,
    "target_encode": 1,
    "tfidf_svd": 1,
    "datetime_parts": 1,
    "lag": 1,
    "rolling_mean": 1,
    "count_encoding": 1,
    # Semantic extractors — low-cardinality, generalizing features pulled from
    # identity-like columns instead of TF-IDF/label-encoding the raw values.
    "extract_title": 1,      # honorific from a name column (Mr/Mrs/Miss/Rare)
    "family_size": 2,        # sib/spouse + parent/child counts + 1
    "is_alone": 2,           # 1 when family_size == 1
    "cabin_deck": 1,         # leading deck letter of a cabin code
}


class FEValidationError(Exception):
    """An operation is malformed or not in the registry."""


def validate_operation(op: FEOperation, *, is_time_series: bool) -> None:
    """Raise FEValidationError if op is unknown, has the wrong column count, or
    requires time-series data that isn't present."""
    if op.operation not in REGISTRY:
        raise FEValidationError(f"Unknown operation {op.operation!r}")
    need = REGISTRY[op.operation]
    if len(op.columns) != need:
        raise FEValidationError(
            f"{op.operation!r} needs {need} column(s), got {len(op.columns)}"
        )
    if op.operation in TIME_SERIES_OPS and not is_time_series:
        raise FEValidationError(f"{op.operation!r} requires is_time_series=True")


# ----------------------------------------------------------------- immediate ops
def _has_neg(*series: pd.Series) -> bool:
    return any((s.dropna() < 0).any() for s in series)


def execute_immediate(
    train: pd.DataFrame, test: pd.DataFrame | None, op: FEOperation
) -> list[str]:
    """Apply an immediate op, mutating train/test in place. Returns the list of
    newly created column names. Raises DeferredOpError for deferred ops."""
    if op.operation in DEFERRED_OPS:
        raise DeferredOpError(
            f"{op.operation!r} is in-fold only and must run in the Phase 5b CV loop, "
            "never on full train."
        )

    name = op.output_name
    cols = op.columns
    created: list[str] = []

    def put(frame: pd.DataFrame | None, col: str, values) -> None:
        if frame is not None:
            frame[col] = values

    if op.operation == "log_transform":
        if _has_neg(train[cols[0]]):
            return []
        put(train, name, np.log1p(train[cols[0]]))
        if test is not None and cols[0] in test:
            put(test, name, np.log1p(test[cols[0]]))
        created = [name]

    elif op.operation == "sqrt_transform":
        if _has_neg(train[cols[0]]):
            return []
        put(train, name, np.sqrt(train[cols[0]]))
        if test is not None and cols[0] in test:
            put(test, name, np.sqrt(test[cols[0]]))
        created = [name]

    elif op.operation == "interaction_multiply":
        put(train, name, train[cols[0]] * train[cols[1]])
        if test is not None and all(c in test for c in cols):
            put(test, name, test[cols[0]] * test[cols[1]])
        created = [name]

    elif op.operation == "interaction_ratio":
        put(train, name, train[cols[0]] / (train[cols[1]] + 1e-8))
        if test is not None and all(c in test for c in cols):
            put(test, name, test[cols[0]] / (test[cols[1]] + 1e-8))
        created = [name]

    elif op.operation == "polynomial_degree2":
        put(train, name, train[cols[0]] ** 2)
        if test is not None and cols[0] in test:
            put(test, name, test[cols[0]] ** 2)
        created = [name]

    elif op.operation in ("bin_equal_width", "bin_equal_freq"):
        n_bins = int(op.params.get("bins", 10))
        if op.operation == "bin_equal_width":
            edges = np.linspace(train[cols[0]].min(), train[cols[0]].max(), n_bins + 1)
        else:
            edges = np.unique(
                np.quantile(train[cols[0]].dropna(), np.linspace(0, 1, n_bins + 1))
            )
        if len(edges) < 2:
            return []
        edges[0], edges[-1] = -np.inf, np.inf
        put(train, name, pd.cut(train[cols[0]], bins=edges, labels=False))
        if test is not None and cols[0] in test:
            put(test, name, pd.cut(test[cols[0]], bins=edges, labels=False))
        created = [name]

    elif op.operation == "count_encoding":
        freq = train[cols[0]].value_counts()
        put(train, name, train[cols[0]].map(freq).fillna(0))
        if test is not None and cols[0] in test:
            put(test, name, test[cols[0]].map(freq).fillna(0))
        created = [name]

    elif op.operation == "datetime_parts":
        created = _datetime_parts(train, test, cols[0], name)

    elif op.operation == "extract_title":
        created = _extract_title(train, test, cols[0], name, op.params)

    elif op.operation == "family_size":
        put(train, name, _family_size(train, cols))
        if test is not None and all(c in test for c in cols):
            put(test, name, _family_size(test, cols))
        created = [name]

    elif op.operation == "is_alone":
        put(train, name, (_family_size(train, cols) == 1).astype(int))
        if test is not None and all(c in test for c in cols):
            put(test, name, (_family_size(test, cols) == 1).astype(int))
        created = [name]

    elif op.operation == "cabin_deck":
        put(train, name, _cabin_deck(train[cols[0]]))
        if test is not None and cols[0] in test:
            put(test, name, _cabin_deck(test[cols[0]]))
        created = [name]

    elif op.operation == "tfidf_svd":
        created = _tfidf_svd(train, test, cols[0], name, op.params)

    elif op.operation == "lag":
        periods = int(op.params.get("periods", 1))
        put(train, name, train[cols[0]].shift(periods))
        if test is not None and cols[0] in test:
            put(test, name, test[cols[0]].shift(periods))
        created = [name]

    elif op.operation == "rolling_mean":
        window = int(op.params.get("window", 3))
        put(train, name, train[cols[0]].rolling(window, min_periods=1).mean())
        if test is not None and cols[0] in test:
            put(test, name, test[cols[0]].rolling(window, min_periods=1).mean())
        created = [name]

    return created


def _datetime_parts(
    train: pd.DataFrame, test: pd.DataFrame | None, col: str, prefix: str
) -> list[str]:
    created: list[str] = []
    tr_dt = pd.to_datetime(train[col], errors="coerce")
    min_dt = tr_dt.min()
    for frame in (train, test):
        if frame is None or col not in frame:
            continue
        dt = pd.to_datetime(frame[col], errors="coerce")
        frame[f"{prefix}_year"] = dt.dt.year
        frame[f"{prefix}_month"] = dt.dt.month
        frame[f"{prefix}_dow"] = dt.dt.dayofweek
        frame[f"{prefix}_hour"] = dt.dt.hour
        frame[f"{prefix}_days_since_min"] = (dt - min_dt).dt.days
    return [f"{prefix}_{p}" for p in ("year", "month", "dow", "hour", "days_since_min")]


def _tfidf_svd(
    train: pd.DataFrame, test: pd.DataFrame | None, col: str, prefix: str, params: dict
) -> list[str]:
    from sklearn.decomposition import TruncatedSVD  # noqa: PLC0415
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415

    n_comp = int(params.get("components", 50))
    max_feat = int(params.get("max_features", 500))
    vec = TfidfVectorizer(max_features=max_feat)
    tr_text = train[col].fillna("").astype(str)
    tfidf = vec.fit_transform(tr_text)
    n_comp = min(n_comp, max(1, tfidf.shape[1] - 1)) if tfidf.shape[1] > 1 else 1
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    tr_svd = svd.fit_transform(tfidf)
    names = [f"{prefix}_{i}" for i in range(n_comp)]
    # Assign all SVD columns in one block — column-by-column insertion fragments
    # the frame and triggers pandas PerformanceWarning.
    train[names] = tr_svd
    if test is not None and col in test:
        te_svd = svd.transform(vec.transform(test[col].fillna("").astype(str)))
        test[names] = te_svd
    return names


def _extract_title(
    train: pd.DataFrame, test: pd.DataFrame | None, col: str, prefix: str, params: dict
) -> list[str]:
    """Pull the honorific (e.g. 'Mr', 'Miss') from a name column. Titles seen
    fewer than `min_count` times in train collapse to 'Rare'; unseen test titles
    map to 'Rare' too. Fit on train only."""
    min_count = int(params.get("min_count", 10))
    pat = r" ([A-Za-z]+)\."

    def titles(frame: pd.DataFrame) -> pd.Series:
        return frame[col].astype(str).str.extract(pat, expand=False)

    tr_titles = titles(train)
    counts = tr_titles.value_counts()
    common = set(counts[counts >= min_count].index)
    train[prefix] = tr_titles.where(tr_titles.isin(common), "Rare").fillna("Rare")
    if test is not None and col in test:
        te_titles = titles(test)
        test[prefix] = te_titles.where(te_titles.isin(common), "Rare").fillna("Rare")
    return [prefix]


def _family_size(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Total household size: sib/spouse + parent/child counts + the passenger."""
    return frame[cols[0]].fillna(0) + frame[cols[1]].fillna(0) + 1


def _cabin_deck(series: pd.Series) -> pd.Series:
    """Leading deck letter of a cabin code; missing cabins map to 'U'."""
    return series.map(lambda v: str(v)[0] if pd.notna(v) and str(v) else "U")


# ----------------------------------------------------------------- deferred ops
def apply_deferred_op(
    op: FEOperation,
    fit_df: pd.DataFrame,
    fit_target: pd.Series,
    transform_df: pd.DataFrame,
    *,
    smoothing: float = 1.0,
) -> pd.Series:
    """Compute one deferred encoding fit on fit_df (+fit_target) and applied to
    transform_df. Used inside the CV fold loop and for the final full-train refit.

    Returns a Series aligned to transform_df.index.
    """
    if op.operation not in DEFERRED_OPS:
        raise DeferredOpError(f"{op.operation!r} is not a deferred operation.")

    if op.operation == "target_encode":
        col = op.columns[0]
        global_mean = fit_target.mean()
        stats = fit_target.groupby(fit_df[col]).agg(["mean", "count"])
        # Smoothed target mean.
        smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (
            stats["count"] + smoothing
        )
        return transform_df[col].map(smooth).fillna(global_mean)

    group_col, value_col = op.columns
    agg = "mean" if op.operation == "group_mean_encode" else "std"
    stats = fit_df.groupby(group_col)[value_col].agg(agg)
    fallback = getattr(fit_df[value_col], agg)()
    return transform_df[group_col].map(stats).fillna(fallback if pd.notna(fallback) else 0.0)


# ----------------------------------------------------------------- standard prep
def standard_preprocess(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    *,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Median-impute numerics and ordinal-encode (tree-friendly) categoricals.

    Fit on train, applied to test. Unknown test categories map to -1. Returns
    new frames restricted to feature_columns.
    """
    tr = train[feature_columns].copy()
    te = test[feature_columns].copy() if test is not None else None

    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(tr[col]) or pd.api.types.is_bool_dtype(tr[col]):
            median = pd.to_numeric(tr[col], errors="coerce").median()
            tr[col] = pd.to_numeric(tr[col], errors="coerce").fillna(median)
            if te is not None:
                te[col] = pd.to_numeric(te[col], errors="coerce").fillna(median)
        else:
            # Ordinal encode: map each train category to an int code.
            cats = {v: i for i, v in enumerate(tr[col].astype("object").fillna("__na__").unique())}
            tr[col] = tr[col].astype("object").fillna("__na__").map(cats).astype(int)
            if te is not None:
                te[col] = (
                    te[col].astype("object").fillna("__na__").map(cats).fillna(-1).astype(int)
                )
    return tr, te
