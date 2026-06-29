"""Phase 4 model selection: task inference, filtering, hard rules, budget."""

from __future__ import annotations

import json

import pytest

from agent.config import load_model_search_spaces
from agent.llm import LLMClient
from agent.memory import ModelCandidate
from agent.orchestrator import Orchestrator
from agent.tools import model_selector as ms
from agent.tools.kaggle_api import KaggleClient
from tests.conftest import FakeAnthropic, FakeKaggleApi

SPACES = load_model_search_spaces()


@pytest.mark.parametrize("pt,em,expected", [
    ("binary classification", "AUC", "classification"),
    ("regression", "RMSE", "regression"),
    ("", "LogLoss", "classification"),
    ("", "MAE", "regression"),
])
def test_infer_task_kind(pt, em, expected):
    assert ms.infer_task_kind(pt, em) == expected


def test_select_filters_wrong_task_and_unknown():
    cands = [
        ModelCandidate(model="LGBMClassifier", priority=1, rationale=""),
        ModelCandidate(model="Ridge", priority=2, rationale=""),          # regression -> dropped
        ModelCandidate(model="MadeUpModel", priority=3, rationale=""),    # unknown -> dropped
        ModelCandidate(model="LogisticRegression", priority=4, rationale=""),
    ]
    out = ms.select_models(cands, task_kind="classification", search_spaces=SPACES)
    names = [m.model for m in out]
    assert "LGBMClassifier" in names
    assert "LogisticRegression" in names
    assert "Ridge" not in names
    assert "MadeUpModel" not in names
    assert [m.priority for m in out] == list(range(1, len(out) + 1))


def test_hard_rule_injects_gbm_when_missing():
    cands = [ModelCandidate(model="LogisticRegression", priority=1, rationale="")]
    out = ms.select_models(cands, task_kind="classification", search_spaces=SPACES)
    assert any(m.model == "LGBMClassifier" for m in out)


def test_neural_gated(monkeypatch):
    # No neural models in the default catalog, so this just confirms no crash and
    # that the GBM hard rule still fires for an empty candidate list.
    out = ms.select_models([], task_kind="regression", search_spaces=SPACES,
                           allow_neural=True, n_rows=10)
    assert out and out[0].model == "LGBMRegressor"


def test_budget():
    assert ms.remaining_submission_budget(5, 2) == 3
    assert ms.remaining_submission_budget(5, 9) == 0


# ----------------------------------------------------------------- orchestration
def test_model_selection_phase(tmp_path, settings, config):
    config.save_plots = False
    ranked = json.dumps([
        {"model": "LGBMClassifier", "priority": 1, "rationale": "gbm"},
        {"model": "LogisticRegression", "priority": 2, "rationale": "linear"},
    ])
    kaggle = KaggleClient(api=FakeKaggleApi())
    llm = LLMClient(api_key="t", client=FakeAnthropic(ranked))
    orch = Orchestrator(settings=settings, config=config, kaggle=kaggle, llm=llm,
                        runs_root=tmp_path / "runs")
    state = orch.bootstrap("demo-comp")
    raw = state.run_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "train.csv").write_text("id,f1,cat,y\n1,0.5,a,0\n2,0.9,b,1\n3,0.2,a,0\n", encoding="utf-8")
    (raw / "test.csv").write_text("id,f1,cat\n4,0.4,a\n", encoding="utf-8")
    (raw / "sample_submission.csv").write_text("id,y\n4,0\n", encoding="utf-8")
    orch.ingest(state)
    orch.eda(state)
    orch.feature_engineering(state)
    state = orch.model_selection(state)

    assert state.last_completed_phase == "4"
    assert [m.model for m in state.selected_models][:1] == ["LGBMClassifier"]
