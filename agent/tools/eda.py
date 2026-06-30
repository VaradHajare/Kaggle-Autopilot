"""Phase 2 — Exploratory Data Analysis.

Pure analysis functions over a pandas DataFrame: column typing, target-column
identification, time-series detection, leakage heuristics, and summary stats.

Note: the spec lists Polars as the primary engine with a pandas fallback. For
correctness and broad operator coverage this implementation works in pandas;
`load_table` reads csv/parquet/feather/json by extension.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from agent.memory import EDASummary

# Leakage term list (exact word-boundary match, case-insensitive).
_LEAK_TERMS = (
    "target", "label", "answer", "result", "pred",
    "score", "probability", "output", "leak",
)
_LEAK_TERM_SET = set(_LEAK_TERMS)
# `id` only as a standalone _id suffix or the whole name.
_ID_RE = re.compile(r"(_id$|^id$)", re.IGNORECASE)
# Time-keyword column names (word-boundary).
_TIME_NAME_RE = re.compile(r"\b(date|timestamp|week|month|year)\b", re.IGNORECASE)

HIGH_MISSING_THRESHOLD = 0.30
HIGH_CORR_LEAK_THRESHOLD = 0.99
TEXT_AVG_LEN = 30  # avg string length above which an object column is "text"
# An object column with many near-unique short values (e.g. Name, Ticket, Cabin)
# is identity-like: label-encoding or TF-IDF on it overfits to specific rows and
# does not generalize to a disjoint test set. Bucketed separately and kept out of
# the raw feature matrix; semantic ops (titles, group size) extract real signal.
HIGH_CARD_MIN_UNIQUE = 20    # at least this many distinct values
HIGH_CARD_UNIQUE_RATIO = 0.5  # and unique/non-null ratio above this


def load_table(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    """Read a tabular file by extension."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=nrows)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".feather":
        return pd.read_feather(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported data format: {suffix}")


def read_header(path: Path) -> list[str]:
    """Column names only (cheap header read for csv; full read otherwise)."""
    if path.suffix.lower() == ".csv":
        return list(pd.read_csv(path, nrows=0).columns)
    return list(load_table(path).columns)


def identify_target_columns(
    sample_cols: list[str], test_cols: list[str]
) -> tuple[str | None, list[str]]:
    """ID column = the column present in BOTH test and sample_submission.
    Targets = the remaining sample_submission columns.

    Falls back to (first sample col as ID, rest as targets) if no overlap.
    """
    overlap = [c for c in sample_cols if c in set(test_cols)]
    if overlap:
        id_col = overlap[0]
        targets = [c for c in sample_cols if c != id_col]
        return id_col, targets
    # Fallback.
    if not sample_cols:
        return None, []
    return sample_cols[0], sample_cols[1:]


def _is_high_cardinality(s: pd.Series) -> bool:
    """True for identity-like object columns: many distinct, near-unique values."""
    non_null = s.dropna()
    if non_null.empty:
        return False
    nun = non_null.nunique()
    return nun >= HIGH_CARD_MIN_UNIQUE and (nun / len(non_null)) > HIGH_CARD_UNIQUE_RATIO


def classify_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Bucket columns into numeric / categorical / datetime / text / boolean /
    high_cardinality. High-cardinality identity-like object columns are split out
    so the pipeline keeps them out of the raw (label-encoded) feature matrix."""
    numeric, categorical, datetime_c, text, boolean, high_card = [], [], [], [], [], []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            boolean.append(col)
        elif pd.api.types.is_datetime64_any_dtype(s):
            datetime_c.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            nun = s.dropna().nunique()
            if nun <= 2 and set(s.dropna().unique()).issubset({0, 1}):
                boolean.append(col)
            else:
                numeric.append(col)
        else:
            # object/category: text if long strings; identity-like if near-unique;
            # otherwise an ordinary low-cardinality categorical.
            avg_len = s.dropna().astype(str).str.len().mean() if s.notna().any() else 0
            if avg_len and avg_len > TEXT_AVG_LEN:
                text.append(col)
            elif _is_high_cardinality(s):
                high_card.append(col)
            else:
                categorical.append(col)
    return {
        "numeric": numeric,
        "categorical": categorical,
        "datetime": datetime_c,
        "text": text,
        "boolean": boolean,
        "high_cardinality": high_card,
    }


def _is_monotonic_increasing(series: pd.Series, frac: float = 0.95) -> bool:
    s = series.dropna()
    if len(s) < 3:
        return False
    if pd.api.types.is_datetime64_any_dtype(s):
        s = s.astype("int64")  # nanoseconds — numeric diff
    diffs = s.reset_index(drop=True).diff().dropna()
    if diffs.empty:
        return False
    non_decreasing = (diffs >= 0).mean()
    return bool(non_decreasing >= frac)


def detect_time_series(df: pd.DataFrame, problem_type_hint: str = "") -> bool:
    """True if ANY of the three spec criteria hold."""
    hint = (problem_type_hint or "").lower()
    if any(k in hint for k in ("time", "series", "forecast")):
        return True
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s) and _is_monotonic_increasing(s):
            return True
        if _TIME_NAME_RE.search(col):
            coerced = pd.to_numeric(s, errors="coerce")
            if coerced.notna().any() and _is_monotonic_increasing(coerced):
                return True
            dt = pd.to_datetime(s, errors="coerce")
            if dt.notna().any() and _is_monotonic_increasing(dt):
                return True
    return False


def _name_tokens(col: str) -> set[str]:
    """Lowercase tokens split on non-alphanumerics (treats `_` as a separator)."""
    return {t for t in re.split(r"[^a-z0-9]+", col.lower()) if t}


def _name_flags_leak(col: str) -> bool:
    if _name_tokens(col) & _LEAK_TERM_SET:
        return True
    return bool(_ID_RE.search(col))


def detect_leakage(
    df: pd.DataFrame,
    target_columns: list[str],
    *,
    id_column: str | None = None,
    known_features: set[str] | None = None,
) -> list[str]:
    """Flag suspected leakage columns by name heuristic and >0.99 correlation
    with any target. Target and ID columns are never self-flagged."""
    known_features = known_features or set()
    protected = set(target_columns) | ({id_column} if id_column else set())
    flagged: set[str] = set()

    for col in df.columns:
        if col in protected:
            continue
        if _name_flags_leak(col):
            flagged.add(col)

    # Correlation-based: only meaningful for numeric features vs numeric targets.
    num_targets = [t for t in target_columns if t in df.columns
                   and pd.api.types.is_numeric_dtype(df[t])]
    for col in df.columns:
        if col in protected or col in flagged or col in known_features:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        for t in num_targets:
            pair = df[[col, t]].dropna()
            if len(pair) < 3 or pair[col].nunique() < 2:
                continue
            r = abs(pair[col].corr(pair[t]))
            if pd.notna(r) and r > HIGH_CORR_LEAK_THRESHOLD:
                flagged.add(col)
                break
    return sorted(flagged)


def top_correlations(df: pd.DataFrame, top_n: int = 20) -> list[tuple[str, str, float]]:
    """Top-N numeric column pairs by absolute Pearson r."""
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return []
    corr = num.corr(numeric_only=True).abs()
    pairs: list[tuple[str, str, float]] = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if pd.notna(v):
                pairs.append((cols[i], cols[j], float(v)))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]


def build_eda_summary(
    df: pd.DataFrame,
    target_columns: list[str],
    *,
    id_column: str | None = None,
) -> EDASummary:
    types = classify_columns(df)
    missing = df.isna().mean()
    high_missing = sorted(missing[missing > HIGH_MISSING_THRESHOLD].index.tolist())
    leakage = detect_leakage(df, target_columns, id_column=id_column)
    return EDASummary(
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        numeric_cols=types["numeric"],
        categorical_cols=types["categorical"],
        datetime_cols=types["datetime"],
        text_cols=types["text"],
        boolean_cols=types["boolean"],
        high_cardinality_cols=types["high_cardinality"],
        high_missing_cols=high_missing,
        duplicate_rows=int(df.duplicated().sum()),
        leakage_flags=leakage,
        memory_mb=float(df.memory_usage(deep=True).sum() / 1e6),
    )


def maybe_profile_report(df: pd.DataFrame, out_path: Path) -> bool:
    """Best-effort ydata-profiling HTML report. Returns True if generated.

    ydata-profiling is a heavy optional dependency; if absent we skip rather
    than fail the pipeline.
    """
    try:
        from ydata_profiling import ProfileReport  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return False
    try:
        ProfileReport(df, minimal=True, progress_bar=False).to_file(out_path)
        return True
    except Exception:  # noqa: BLE001
        return False
