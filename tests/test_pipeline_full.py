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
    ModelCandidate,
    RunState,
)
from agent.orchestrator import Orchestrator
from agent.tools import trainer as train_tools
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


def test_leaderboard_pending_score_logs_pending_status(tmp_path, settings, config):
    """Submitted, but the public LB score has not posted before the poll window
    closes: the entry keeps public_lb_score=None and the run log records PENDING
    (not COMPLETE) so the audit trail never claims a score we didn't observe."""
    config.save_plots = False
    config.optuna_n_trials = 1
    config.cv_folds = 3
    config.auto_submit = True
    config.lb_poll_timeout_minutes = 0  # poll window elapses immediately -> None
    kaggle = KaggleClient(api=FakeKaggleApi(eval_metric="AUC", public_score="pending"))
    llm = LLMClient(api_key="t", client=FakeAnthropic("{}"))
    o = Orchestrator(settings=settings, config=config, kaggle=kaggle, llm=llm,
                     runs_root=tmp_path / "runs")
    state = o.bootstrap("demo-comp")
    _seed_data(state.run_dir)
    for step in (o.ingest, o.eda, o.feature_engineering, o.model_selection,
                 o.feature_pruning, o.train, o.ensemble, o.generate_submission):
        step(state)
    state = o.leaderboard(state, sleep=lambda s: None)

    assert state.last_completed_phase == "8"
    assert state.leaderboard_entries  # a submission was uploaded
    assert state.leaderboard_entries[-1].public_lb_score is None
    log_text = (state.run_dir / "run_log.md").read_text(encoding="utf-8")
    assert "**Status:** PENDING" in log_text


def test_train_recovers_from_single_model_failure(orch, monkeypatch):
    """A single model raising must not abort Phase 5b: it is recorded as an ERROR
    CVResult and logged to RunState.errors, and the surviving model still trains
    and ensembles (spec: recoverable errors skip and continue)."""
    state = orch.bootstrap("demo-comp")
    _seed_data(state.run_dir)
    for step in (orch.ingest, orch.eda, orch.feature_engineering,
                 orch.model_selection, orch.feature_pruning):
        step(state)
    state.selected_models = [
        ModelCandidate(model="LGBMClassifier", priority=1),
        ModelCandidate(model="LogisticRegression", priority=2),
    ]
    real = train_tools.train_model

    def flaky(*, name, **kw):
        if name == "LogisticRegression":
            raise RuntimeError("boom: estimator blew up")
        return real(name=name, **kw)

    monkeypatch.setattr(train_tools, "train_model", flaky)
    orch.train(state)

    statuses = {r.model: r.status for r in state.cv_results}
    assert statuses["LGBMClassifier"] == "COMPLETE"
    assert statuses["LogisticRegression"] == "ERROR"
    assert any(e.phase == "5b" and "LogisticRegression" in e.recovery_action
               for e in state.errors)
    # Downstream still works with the surviving model.
    orch.ensemble(state)
    assert state.ensemble_strategy is not None


def test_fe_followups_consumed_after_reentry(tmp_path, settings, config):
    """The FE-followup re-entry trigger is one-shot: after re-entering Phase 3 it
    is cleared, so a deterministic FE pass can't loop forever and starve the
    CV-LB (5b) / ensemble (6) triggers."""
    config.save_plots = False
    config.optuna_n_trials = 1
    config.cv_folds = 3
    config.auto_submit = False
    config.max_iterations = 2
    kaggle = KaggleClient(api=FakeKaggleApi(eval_metric="AUC"))
    llm = LLMClient(api_key="t", client=FakeAnthropic(
        '{"confirmed_problem_type": "binary", "fe_followups": ["add interactions"]}'))
    o = Orchestrator(settings=settings, config=config, kaggle=kaggle, llm=llm,
                     runs_root=tmp_path / "runs")
    state = o.bootstrap("demo-comp")
    _seed_data(state.run_dir)
    state = o.run_pipeline("demo-comp", resume=True)

    assert state.iteration == 1  # re-entered exactly once
    assert state.eda_analysis is not None
    assert state.eda_analysis.fe_followups == []  # trigger consumed


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
