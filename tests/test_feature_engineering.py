"""Phase 3 FE: registry validation, immediate ops, deferred-op leakage guard,
no test fitting, and orchestration."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from agent.errors import DeferredOpError
from agent.llm import LLMClient
from agent.memory import FEOperation
from agent.orchestrator import Orchestrator
from agent.tools import feature_engineering as fe
from agent.tools.kaggle_api import KaggleClient
from tests.conftest import FakeAnthropic, FakeKaggleApi


# ----------------------------------------------------------------- validation
def test_all_registry_ops_have_arity():
    # Every registry op must declare a positive required-column count.
    assert all(n >= 1 for n in fe.REGISTRY.values())
    assert len(fe.REGISTRY) == 15


def test_validate_unknown_operation():
    op = FEOperation(operation="black_magic", columns=["a"], output_name="x")
    with pytest.raises(fe.FEValidationError, match="Unknown"):
        fe.validate_operation(op, is_time_series=False)


def test_validate_wrong_column_count():
    op = FEOperation(operation="interaction_multiply", columns=["a"], output_name="x")
    with pytest.raises(fe.FEValidationError, match="needs 2"):
        fe.validate_operation(op, is_time_series=False)


def test_validate_timeseries_op_requires_flag():
    op = FEOperation(operation="lag", columns=["a"], output_name="a_lag")
    with pytest.raises(fe.FEValidationError, match="time_series"):
        fe.validate_operation(op, is_time_series=False)
    fe.validate_operation(op, is_time_series=True)  # ok


# ----------------------------------------------------------------- deferred guard
@pytest.mark.parametrize("opname", ["target_encode", "group_mean_encode", "group_std_encode"])
def test_deferred_ops_reject_immediate_execution(opname):
    cols = ["g", "v"] if "group" in opname else ["c"]
    op = FEOperation(operation=opname, columns=cols, output_name="enc")
    df = pd.DataFrame({"g": [1, 2], "v": [3.0, 4.0], "c": ["a", "b"]})
    with pytest.raises(DeferredOpError):
        fe.execute_immediate(df, None, op)


# ----------------------------------------------------------------- immediate ops
def test_log_transform_skips_negatives():
    df = pd.DataFrame({"x": [-1.0, 2.0]})
    created = fe.execute_immediate(df, None, FEOperation(
        operation="log_transform", columns=["x"], output_name="lx"))
    assert created == []
    assert "lx" not in df


def test_interaction_and_ratio():
    train = pd.DataFrame({"a": [2.0, 4.0], "b": [1.0, 2.0]})
    test = pd.DataFrame({"a": [6.0], "b": [3.0]})
    fe.execute_immediate(train, test, FEOperation(
        operation="interaction_multiply", columns=["a", "b"], output_name="ab"))
    fe.execute_immediate(train, test, FEOperation(
        operation="interaction_ratio", columns=["a", "b"], output_name="r"))
    assert train["ab"].tolist() == [2.0, 8.0]
    assert test["ab"].tolist() == [18.0]
    assert pytest.approx(train["r"].tolist(), rel=1e-3) == [2.0, 2.0]


def test_count_encoding_fit_on_train_only():
    # The frequency map is learned from train; test must use train frequencies.
    train = pd.DataFrame({"c": ["a", "a", "b"]})
    test = pd.DataFrame({"c": ["a", "z"]})  # 'z' unseen in train
    fe.execute_immediate(train, test, FEOperation(
        operation="count_encoding", columns=["c"], output_name="c_cnt"))
    assert train["c_cnt"].tolist() == [2, 2, 1]
    assert test["c_cnt"].tolist() == [2, 0]  # 'a'->2 from train, unseen->0


def test_all_immediate_registry_ops_callable():
    """Spec requirement: every non-deferred registry op is implemented and runnable."""
    n = 40
    rng = np.random.default_rng(0)
    train = pd.DataFrame({
        "a": rng.uniform(1, 10, n),
        "b": rng.uniform(1, 10, n),
        "cat": rng.choice(["x", "y", "z"], n),
        "txt": ["hello world foo bar " * 2] * n,
        "d": pd.date_range("2021-01-01", periods=n, freq="D"),
    })
    specs = {
        "log_transform": (["a"], {}),
        "sqrt_transform": (["a"], {}),
        "interaction_multiply": (["a", "b"], {}),
        "interaction_ratio": (["a", "b"], {}),
        "polynomial_degree2": (["a"], {}),
        "bin_equal_width": (["a"], {}),
        "bin_equal_freq": (["a"], {}),
        "count_encoding": (["cat"], {}),
        "tfidf_svd": (["txt"], {"components": 3, "max_features": 20}),
        "datetime_parts": (["d"], {}),
        "lag": (["a"], {"periods": 1}),
        "rolling_mean": (["a"], {"window": 3}),
    }
    for opname, (cols, params) in specs.items():
        op = FEOperation(operation=opname, columns=cols, output_name=f"{opname}_out",
                         params=params)
        created = fe.execute_immediate(train.copy(), None, op)
        assert created, f"{opname} produced no columns"


def test_tfidf_svd_adds_matching_train_and_test_columns():
    train = pd.DataFrame({"txt": ["hello world foo", "bar baz qux", "foo bar hello"]})
    test = pd.DataFrame({"txt": ["hello bar", "qux foo"]})
    op = FEOperation(
        operation="tfidf_svd", columns=["txt"], output_name="txt_svd",
        params={"components": 2, "max_features": 10},
    )
    created = fe.execute_immediate(train, test, op)

    assert created  # at least one SVD component column
    # Both frames gained exactly the created columns, with the right row counts.
    for name in created:
        assert name in train.columns and name in test.columns
    assert len(train) == 3 and len(test) == 2
    assert train[created].notna().all().all()
    assert test[created].notna().all().all()


def test_group_std_encode_infold():
    fit_X = pd.DataFrame({"g": ["x", "x", "x", "y"], "v": [10.0, 20.0, 30.0, 5.0]})
    fit_y = pd.Series([0, 0, 0, 0])
    transform_X = pd.DataFrame({"g": ["x", "y"]})
    op = FEOperation(operation="group_std_encode", columns=["g", "v"], output_name="gs")
    out = fe.apply_deferred_op(op, fit_X, fit_y, transform_X)
    assert out.iloc[0] == pytest.approx(pd.Series([10.0, 20.0, 30.0]).std())


def test_datetime_parts():
    train = pd.DataFrame({"d": pd.to_datetime(["2021-01-01", "2021-06-15"])})
    created = fe.execute_immediate(train, None, FEOperation(
        operation="datetime_parts", columns=["d"], output_name="d"))
    assert "d_year" in created
    assert train["d_year"].tolist() == [2021, 2021]
    assert train["d_month"].tolist() == [1, 6]


# ----------------------------------------------------------------- deferred in-fold
def test_target_encode_infold_no_leakage():
    # Fit on fold-train only; the transform set never contributes to its own stats.
    fit_X = pd.DataFrame({"c": ["a", "a", "b", "b"]})
    fit_y = pd.Series([1.0, 1.0, 0.0, 0.0])
    transform_X = pd.DataFrame({"c": ["a", "b", "c"]})  # 'c' unseen -> global mean
    op = FEOperation(operation="target_encode", columns=["c"], output_name="c_te")
    out = fe.apply_deferred_op(op, fit_X, fit_y, transform_X, smoothing=0.0)
    assert out.iloc[0] > out.iloc[1]          # 'a' (high target) > 'b'
    assert out.iloc[2] == pytest.approx(0.5)  # unseen -> global mean


def test_group_mean_encode_infold():
    fit_X = pd.DataFrame({"g": ["x", "x", "y"], "v": [10.0, 20.0, 100.0]})
    fit_y = pd.Series([0, 0, 0])
    transform_X = pd.DataFrame({"g": ["x", "y", "z"]})
    op = FEOperation(operation="group_mean_encode", columns=["g", "v"], output_name="ge")
    out = fe.apply_deferred_op(op, fit_X, fit_y, transform_X)
    assert out.iloc[0] == pytest.approx(15.0)
    assert out.iloc[1] == pytest.approx(100.0)


# ----------------------------------------------------------------- standard prep
def test_standard_preprocess_no_test_fit():
    train = pd.DataFrame({"n": [1.0, np.nan, 3.0], "c": ["a", "b", "a"]})
    test = pd.DataFrame({"n": [np.nan, 5.0], "c": ["a", "zzz"]})  # zzz unseen
    tr, te = fe.standard_preprocess(train, test, feature_columns=["n", "c"])
    assert tr["n"].tolist() == [1.0, 2.0, 3.0]  # median(1,3)=2 imputed
    assert te["n"].tolist() == [2.0, 5.0]       # test uses TRAIN median
    assert te["c"].tolist()[1] == -1            # unseen category -> -1


# ----------------------------------------------------------------- orchestration
def _orch(tmp_path, settings, config, text):
    kaggle = KaggleClient(api=FakeKaggleApi())
    llm = LLMClient(api_key="t", client=FakeAnthropic(text))
    return Orchestrator(settings=settings, config=config, kaggle=kaggle, llm=llm,
                        runs_root=tmp_path / "runs")


def _seed_through_eda(orch, slug="demo-comp"):
    state = orch.bootstrap(slug)
    raw = state.run_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "train.csv").write_text(
        "id,f1,f2,cat,y\n1,0.5,10,a,0\n2,0.9,20,b,1\n3,0.2,30,a,0\n4,0.7,40,b,1\n",
        encoding="utf-8")
    (raw / "test.csv").write_text("id,f1,f2,cat\n5,0.4,50,a\n", encoding="utf-8")
    (raw / "sample_submission.csv").write_text("id,y\n5,0\n", encoding="utf-8")
    orch.ingest(state)
    return orch.eda(state)


def test_fe_phase_immediate_and_deferred(tmp_path, settings, config):
    config.save_plots = False
    strategy = json.dumps([
        {"operation": "interaction_multiply", "columns": ["f1", "f2"],
         "output_name": "f1f2", "rationale": "x"},
        {"operation": "target_encode", "columns": ["cat"],
         "output_name": "cat_te", "rationale": "x"},
        {"operation": "bogus_op", "columns": ["f1"], "output_name": "z", "rationale": "x"},
    ])
    orch = _orch(tmp_path, settings, config, strategy)
    # Bootstrap/ingest/eda use the EDA fallback (strategy text isn't valid EDA JSON,
    # but EDA falls back gracefully).
    state = _seed_through_eda(orch)
    state = orch.feature_engineering(state)

    assert state.last_completed_phase == "3"
    # Immediate op applied; deferred op stored, not executed; bogus skipped.
    assert any(o.operation == "interaction_multiply" for o in state.feature_engineering_ops)
    assert [o.operation for o in state.deferred_fe_ops] == ["target_encode"]
    assert "f1f2" in state.active_features
    assert "cat_te" not in state.active_features  # deferred, not yet materialized
    assert (state.run_dir / "processed" / "train_fe_base.parquet").exists()
