"""Full-pipeline integration (FIX B6): phases 0 -> 8 with mocked APIs, plus the
Phase 9 iteration re-entry mapping. Verifies sequencing, state serialization,
and resume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent.llm import LLMClient
from agent.memory import (
    CVResult,
    EnsembleStrategy,
    LeaderboardEntry,
    LLMEDAAnalysis,
    RunState,
)
from agent.orchestrator import Orchestrator
from agent.tools.kaggle_api import KaggleClient
from tests.conftest import FakeAnthropic, FakeKaggleApi


def _seed_data(run_dir):
    rng = np.random.default_rng(0)
    n = 90
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
    df.drop(columns=["y"]).head(12).to_csv(raw / "test.csv", index=False)
    df[["id", "y"]].head(12).to_csv(raw / "sample_submission.csv", index=False)


@pytest.fixture
def orch(tmp_path, settings, config):
    config.save_plots = False
    config.optuna_n_trials = 1
    config.cv_folds = 3
    config.auto_submit = True
    kaggle = KaggleClient(api=FakeKaggleApi(eval_metric="AUC", public_score="0.84"))
    llm = LLMClient(api_key="t", client=FakeAnthropic("{}"))
    return Orchestrator(settings=settings, config=config, kaggle=kaggle, llm=llm,
                        runs_root=tmp_path / "runs")


def test_full_pipeline_0_to_8(orch):
    state = orch.bootstrap("demo-comp")
    _seed_data(state.run_dir)
    orch.ingest(state)
    orch.eda(state)
    orch.feature_engineering(state)
    orch.model_selection(state)
    orch.feature_pruning(state)
    orch.train(state)
    orch.ensemble(state)
    orch.generate_submission(state)
    state = orch.leaderboard(state, sleep=lambda s: None)

    assert state.last_completed_phase == "8"
    # Submission produced and selected as best.
    assert state.best_submission is not None
    assert state.best_submission.path.exists()
    # Ensemble ran and produced a test prediction file.
    assert (state.run_dir / "models" / "ensemble_test.npy").exists()
    # Leaderboard recorded and persisted.
    assert state.leaderboard_entries and state.leaderboard_entries[0].public_lb_score == 0.84
    assert (state.run_dir / "leaderboard.json").exists()
    # State reloadable at the final phase.
    assert RunState.load(state.run_dir).last_completed_phase == "8"


def test_run_pipeline_drives_all_phases(orch):
    state = orch.bootstrap("demo-comp")
    _seed_data(state.run_dir)
    # run_pipeline re-bootstraps; reuse the same slug via resume to keep seeded data.
    state = orch.run_pipeline("demo-comp", resume=True)
    assert state.last_completed_phase == "8"
    assert state.best_submission is not None


def test_leaderboard_skipped_without_auto_submit(orch):
    orch.config.auto_submit = False
    state = orch.bootstrap("demo-comp")
    _seed_data(state.run_dir)
    for step in (orch.ingest, orch.eda, orch.feature_engineering, orch.model_selection,
                 orch.feature_pruning, orch.train, orch.ensemble, orch.generate_submission):
        step(state)
    state = orch.leaderboard(state, sleep=lambda s: None)
    assert state.last_completed_phase == "8"
    # Nothing submitted.
    assert not state.leaderboard_entries


# ----------------------------------------------------------------- Phase 9
def _min_state(tmp_path):
    from agent.memory import CompetitionMeta
    return RunState(slug="d", run_dir=tmp_path / "d",
                    competition_meta=CompetitionMeta(slug="d", eval_metric="AUC",
                                                     problem_type="binary"))


def test_iteration_stops_at_budget(tmp_path, settings, config):
    config.max_iterations = 1
    o = Orchestrator(settings=settings, config=config,
                     kaggle=KaggleClient(api=FakeKaggleApi()),
                     llm=LLMClient(api_key="t", client=FakeAnthropic("{}")),
                     runs_root=tmp_path / "runs")
    state = _min_state(tmp_path)
    assert o.plan_iteration(state) is None


def test_iteration_lowest_phase_wins(tmp_path, settings, config):
    config.max_iterations = 3
    o = Orchestrator(settings=settings, config=config,
                     kaggle=KaggleClient(api=FakeKaggleApi()),
                     llm=LLMClient(api_key="t", client=FakeAnthropic("{}")),
                     runs_root=tmp_path / "runs")
    state = _min_state(tmp_path)
    # Trigger both: CV-LB gap (5b) AND FE followups (3). Lowest -> '3'.
    state.leaderboard_entries = [LeaderboardEntry(
        timestamp=__import__("datetime").datetime.now(),
        submission_file="x", cv_score=0.9, public_lb_score=0.85, delta=-0.05)]
    state.eda_analysis = LLMEDAAnalysis(confirmed_problem_type="binary",
                                        fe_followups=["add interactions"])
    assert o.plan_iteration(state) == "3"


def test_iteration_cv_lb_gap_triggers_5b(tmp_path, settings, config):
    config.max_iterations = 3
    o = Orchestrator(settings=settings, config=config,
                     kaggle=KaggleClient(api=FakeKaggleApi()),
                     llm=LLMClient(api_key="t", client=FakeAnthropic("{}")),
                     runs_root=tmp_path / "runs")
    state = _min_state(tmp_path)
    state.leaderboard_entries = [LeaderboardEntry(
        timestamp=__import__("datetime").datetime.now(),
        submission_file="x", cv_score=0.9, public_lb_score=0.85, delta=-0.05)]
    assert o.plan_iteration(state) == "5b"
