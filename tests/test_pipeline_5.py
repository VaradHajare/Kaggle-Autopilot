"""End-to-end sequencing through Phase 5b with a generic LLM stub.

With a generic '{}' LLM response every phase uses its documented fallback
(EDA default, no FE ops, default GBM model), so this exercises phase sequencing
and state serialization rather than LLM-driven branching.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent.llm import LLMClient
from agent.memory import RunState
from agent.orchestrator import Orchestrator
from agent.tools.kaggle_api import KaggleClient
from tests.conftest import FakeAnthropic, FakeKaggleApi


def _seed_data(run_dir):
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({
        "id": range(n),
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "cat": rng.choice(["a", "b", "c"], size=n),
    })
    df["y"] = ((df["f1"] + df["f2"]) > 0).astype(int)
    raw = run_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw / "train.csv", index=False)
    df.drop(columns=["y"]).head(10).to_csv(raw / "test.csv", index=False)
    df[["id", "y"]].head(10).to_csv(raw / "sample_submission.csv", index=False)


@pytest.fixture
def fast_orch(tmp_path, settings, config):
    config.save_plots = False
    config.optuna_n_trials = 1
    config.cv_folds = 3
    config.max_features = 2000
    kaggle = KaggleClient(api=FakeKaggleApi(eval_metric="AUC"))
    llm = LLMClient(api_key="t", client=FakeAnthropic("{}"))
    return Orchestrator(settings=settings, config=config, kaggle=kaggle, llm=llm,
                        runs_root=tmp_path / "runs")


def test_pipeline_through_training(fast_orch):
    state = fast_orch.bootstrap("demo-comp")
    _seed_data(state.run_dir)
    fast_orch.ingest(state)
    fast_orch.eda(state)
    fast_orch.feature_engineering(state)
    fast_orch.model_selection(state)
    fast_orch.feature_pruning(state)
    state = fast_orch.train(state)

    assert state.last_completed_phase == "5b"
    assert state.cv_results, "expected at least one trained model"
    # Default GBM selected and trained; artifacts exist.
    best = state.cv_results[0].model
    assert (state.run_dir / "models" / f"{best}_oof.npy").exists()
    assert (state.run_dir / "models" / f"{best}_test.npy").exists()
    # State persisted and reloadable.
    assert RunState.load(state.run_dir).last_completed_phase == "5b"


def test_pruning_triggers_when_over_cap(fast_orch):
    fast_orch.config.max_features = 1  # force pruning
    state = fast_orch.bootstrap("demo-comp")
    _seed_data(state.run_dir)
    fast_orch.ingest(state)
    fast_orch.eda(state)
    fast_orch.feature_engineering(state)
    fast_orch.model_selection(state)
    state = fast_orch.feature_pruning(state)

    assert state.last_completed_phase == "5a"
    assert len(state.active_features) == 1
