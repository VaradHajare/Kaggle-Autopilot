"""Phase 2 EDA: typing, target ID, time-series, leakage, summary, orchestration."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from agent.llm import LLMClient
from agent.orchestrator import Orchestrator
from agent.tools import eda as eda_tools
from agent.tools.kaggle_api import KaggleClient
from tests.conftest import FakeAnthropic, FakeKaggleApi


# ----------------------------------------------------------------- target ID
def test_identify_target_single():
    id_col, targets = eda_tools.identify_target_columns(["id", "y"], ["id", "f1"])
    assert id_col == "id"
    assert targets == ["y"]


def test_identify_target_multi():
    id_col, targets = eda_tools.identify_target_columns(
        ["id", "a", "b", "c"], ["id", "f1"]
    )
    assert id_col == "id"
    assert targets == ["a", "b", "c"]


def test_identify_target_fallback_no_overlap():
    id_col, targets = eda_tools.identify_target_columns(["row", "y"], ["x1", "x2"])
    assert id_col == "row"
    assert targets == ["y"]


# ----------------------------------------------------------------- typing
def test_classify_columns():
    df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0],
        "cat": ["a", "b", "a"],
        "flag": [True, False, True],
        "binint": [0, 1, 0],
        "txt": ["a long sentence here" * 3, "another long body of text" * 2, "more text" * 5],
        "dt": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
    })
    types = eda_tools.classify_columns(df)
    assert "num" in types["numeric"]
    assert "cat" in types["categorical"]
    assert "flag" in types["boolean"]
    assert "binint" in types["boolean"]
    assert "txt" in types["text"]
    assert "dt" in types["datetime"]


def test_classify_columns_flags_high_cardinality_identity():
    n = 50
    df = pd.DataFrame({
        "ticket": [f"T{i}" for i in range(n)],          # all unique -> identity
        "name": [f"Person {i}, Mr. X" for i in range(n)],  # all unique -> identity
        "sex": (["m", "f"] * n)[:n],                     # low-cardinality categorical
    })
    types = eda_tools.classify_columns(df)
    assert "ticket" in types["high_cardinality"]
    assert "name" in types["high_cardinality"]
    assert "sex" in types["categorical"]
    # Identity columns must NOT leak into the ordinary categorical bucket.
    assert "ticket" not in types["categorical"]
    assert "name" not in types["categorical"]


# ----------------------------------------------------------------- time series
def test_time_series_by_hint():
    df = pd.DataFrame({"x": [1, 2, 3]})
    assert eda_tools.detect_time_series(df, "time series forecasting") is True


def test_time_series_by_monotonic_datetime():
    df = pd.DataFrame({
        "date": pd.to_datetime(pd.date_range("2021-01-01", periods=50)),
        "v": range(50),
    })
    assert eda_tools.detect_time_series(df) is True


def test_time_series_false_for_shuffled():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.permutation(100), "b": rng.normal(size=100)})
    assert eda_tools.detect_time_series(df, "binary classification") is False


# ----------------------------------------------------------------- leakage
def test_leakage_name_heuristic():
    df = pd.DataFrame({
        "f1": [1, 2, 3],
        "leak_col": [1, 2, 3],
        "prediction_score": [1, 2, 3],
        "customer_id": [1, 2, 3],
        "y": [0, 1, 0],
    })
    flags = eda_tools.detect_leakage(df, ["y"], id_column=None)
    assert "leak_col" in flags
    assert "prediction_score" in flags
    assert "customer_id" in flags
    assert "f1" not in flags
    assert "y" not in flags


def test_leakage_id_word_boundary_no_false_positive():
    # 'valid', 'grid' must NOT be flagged by the id rule.
    df = pd.DataFrame({"valid": [1, 2], "grid": [3, 4], "y": [0, 1]})
    flags = eda_tools.detect_leakage(df, ["y"])
    assert flags == []


def test_leakage_high_correlation():
    y = np.arange(100, dtype=float)
    df = pd.DataFrame({"f": y * 2.0, "noise": np.random.default_rng(1).normal(size=100), "y": y})
    flags = eda_tools.detect_leakage(df, ["y"])
    assert "f" in flags
    assert "noise" not in flags


# ----------------------------------------------------------------- summary
def test_build_summary(tmp_path):
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [None, None, None, 1],  # 75% missing
        "y": [0, 1, 0, 1],
    })
    summary = eda_tools.build_eda_summary(df, ["y"])
    assert summary.n_rows == 4
    assert "b" in summary.high_missing_cols


def test_load_table_csv(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    df = eda_tools.load_table(p)
    assert list(df.columns) == ["a", "b"]


# ----------------------------------------------------------------- orchestration
def _orch(tmp_path, settings, config, anthropic_text):
    kaggle = KaggleClient(api=FakeKaggleApi())
    llm = LLMClient(api_key="t", client=FakeAnthropic(anthropic_text))
    return Orchestrator(settings=settings, config=config, kaggle=kaggle, llm=llm,
                        runs_root=tmp_path / "runs")


def test_eda_phase_end_to_end(tmp_path, settings, config):
    config.save_plots = False
    analysis = json.dumps({
        "confirmed_problem_type": "binary",
        "high_risk_columns": [],
        "imputation_strategies": {"f1": "median"},
        "anomaly_flags": [],
        "fe_directions": ["interactions"],
        "fe_followups": [],
    })
    orch = _orch(tmp_path, settings, config, analysis)
    state = orch.bootstrap("demo-comp")
    raw = state.run_dir / "raw"
    (raw).mkdir(parents=True, exist_ok=True)
    (raw / "train.csv").write_text("id,f1,y\n1,0.5,0\n2,0.9,1\n3,0.2,0\n", encoding="utf-8")
    (raw / "test.csv").write_text("id,f1\n4,0.4\n", encoding="utf-8")
    (raw / "sample_submission.csv").write_text("id,y\n4,0\n", encoding="utf-8")
    state = orch.ingest(state)
    state = orch.eda(state)

    assert state.last_completed_phase == "2"
    assert state.target_columns == ["y"]
    assert state.id_column == "id"
    assert state.eda_analysis.confirmed_problem_type == "binary"
    assert (state.run_dir / "eda_analysis.json").exists()


def test_eda_falls_back_on_bad_llm(tmp_path, settings, config):
    config.save_plots = False
    orch = _orch(tmp_path, settings, config, "not json")
    state = orch.bootstrap("demo-comp")
    raw = state.run_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "train.csv").write_text("id,f1,y\n1,0.5,0\n2,0.9,1\n3,0.2,0\n", encoding="utf-8")
    (raw / "test.csv").write_text("id,f1\n4,0.4\n", encoding="utf-8")
    (raw / "sample_submission.csv").write_text("id,y\n4,0\n", encoding="utf-8")
    state = orch.ingest(state)
    state = orch.eda(state)
    # Falls back to competition_meta.problem_type.
    assert state.eda_analysis.confirmed_problem_type == state.competition_meta.problem_type
