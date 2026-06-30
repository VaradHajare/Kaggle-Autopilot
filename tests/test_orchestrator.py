"""Phase 0 bootstrap orchestration with mocked Kaggle + Anthropic."""

from __future__ import annotations

import pytest

from agent.errors import (
    AgentFatalError,
    BootstrapError,
    LLMQuotaError,
    StateVersionMismatchError,
)
from agent.llm import LLMClient
from agent.memory import RunState
from agent.orchestrator import Orchestrator
from agent.tools.kaggle_api import KaggleClient
from tests.conftest import (
    ApiStatusError,
    FakeAnthropic,
    FakeAnthropicRaising,
    FakeKaggleApi,
)


def _orch(tmp_path, settings, config, *, kaggle_api=None, anthropic_text='{"ok": true}'):
    kaggle = KaggleClient(api=kaggle_api or FakeKaggleApi())
    llm = LLMClient(api_key="test", client=FakeAnthropic(anthropic_text))
    return Orchestrator(
        settings=settings, config=config, kaggle=kaggle, llm=llm,
        runs_root=tmp_path / "runs",
    )


def test_optuna_trials_env_override_wins_over_yaml(tmp_path, settings, config):
    config.optuna_n_trials = 50  # the agent.yaml value
    settings = settings.model_copy(update={"optuna_n_trials": 7})  # OPTUNA_N_TRIALS=7
    orch = _orch(tmp_path, settings, config)
    assert orch.config.optuna_n_trials == 7


def test_optuna_trials_yaml_kept_when_env_unset(tmp_path, settings, config):
    config.optuna_n_trials = 30
    settings = settings.model_copy(update={"optuna_n_trials": None})  # OPTUNA_N_TRIALS unset
    orch = _orch(tmp_path, settings, config)
    assert orch.config.optuna_n_trials == 30


def test_bootstrap_happy_path(tmp_path, settings, config):
    orch = _orch(tmp_path, settings, config)
    state = orch.bootstrap("https://www.kaggle.com/competitions/demo-comp")

    assert state.slug == "demo-comp"
    assert state.last_completed_phase == "0"
    assert state.state_path().exists()
    for sub in ("raw", "processed", "models", "submissions"):
        assert (state.run_dir / sub).is_dir()
    assert (state.run_dir / "run_log.md").exists()


def test_missing_kaggle_creds_is_fatal(tmp_path, settings, config):
    settings.kaggle_key = None
    orch = _orch(tmp_path, settings, config)
    with pytest.raises(AgentFatalError):
        orch.bootstrap("demo-comp")


def test_missing_llm_key_is_fatal(tmp_path, settings, config):
    settings.gemini_api_key = None  # active provider key absent
    orch = _orch(tmp_path, settings, config)
    with pytest.raises(AgentFatalError):
        orch.bootstrap("demo-comp")


def test_rules_not_accepted_is_fatal(tmp_path, settings, config):
    orch = _orch(tmp_path, settings, config, kaggle_api=FakeKaggleApi(rules_accepted=False))
    with pytest.raises(AgentFatalError, match="rules"):
        orch.bootstrap("demo-comp")


def test_insufficient_disk_is_fatal(tmp_path, settings, config):
    huge = FakeKaggleApi(total_bytes=10**18)  # ~1 EB, exceeds any real free space
    orch = _orch(tmp_path, settings, config, kaggle_api=huge)
    with pytest.raises(AgentFatalError, match="disk"):
        orch.bootstrap("demo-comp")


def test_high_stakes_requires_confirmation(tmp_path, settings, config):
    featured = FakeKaggleApi(category="featured", reward="$50,000")
    orch = _orch(tmp_path, settings, config, kaggle_api=featured)
    with pytest.raises(AgentFatalError, match="high-stakes"):
        orch.bootstrap("demo-comp")
    # With confirmation it proceeds.
    state = orch.bootstrap("demo-comp", confirm_high_stakes=True)
    assert state.last_completed_phase == "0"


def test_existing_state_without_flag_is_ambiguous(tmp_path, settings, config):
    orch = _orch(tmp_path, settings, config)
    orch.bootstrap("demo-comp")
    with pytest.raises(BootstrapError, match="Existing run state"):
        orch.bootstrap("demo-comp")


def test_resume_returns_saved_state(tmp_path, settings, config):
    orch = _orch(tmp_path, settings, config)
    orch.bootstrap("demo-comp")
    resumed = orch.bootstrap("demo-comp", resume=True)
    assert resumed.last_completed_phase == "0"


def test_force_restart_clears_state(tmp_path, settings, config):
    orch = _orch(tmp_path, settings, config)
    first = orch.bootstrap("demo-comp")
    (first.run_dir / "marker.txt").write_text("stale", encoding="utf-8")
    second = orch.bootstrap("demo-comp", force_restart=True)
    assert not (second.run_dir / "marker.txt").exists()


def test_resume_version_mismatch_raises(tmp_path, settings, config):
    orch = _orch(tmp_path, settings, config)
    state = orch.bootstrap("demo-comp")
    p = state.state_path()
    p.write_text(p.read_text().replace('"1.1"', '"0.9"'), encoding="utf-8")
    with pytest.raises(StateVersionMismatchError):
        orch.bootstrap("demo-comp", resume=True)


def test_resume_validates_llm_credentials(tmp_path, settings, config):
    # A run exists on disk, but the LLM key is now invalid. Resume must fail
    # loud at bootstrap rather than deep inside a later phase.
    _orch(tmp_path, settings, config).bootstrap("demo-comp")

    bad_llm = LLMClient(
        api_key="t", client=FakeAnthropicRaising(ApiStatusError("invalid x-api-key", 401))
    )
    broken = Orchestrator(
        settings=settings, config=config,
        kaggle=KaggleClient(api=FakeKaggleApi()), llm=bad_llm,
        runs_root=tmp_path / "runs",
    )
    with pytest.raises(AgentFatalError):
        broken.bootstrap("demo-comp", resume=True)


def test_record_fatal_persists_to_state_and_log(tmp_path, settings, config):
    orch = _orch(tmp_path, settings, config)
    state = orch.bootstrap("demo-comp")

    orch._record_fatal(state, "2", LLMQuotaError("quota gone", remediation="wait"))

    assert len(state.errors) == 1
    rec = state.errors[0]
    assert rec.phase == "2"
    assert rec.error_type == "LLMQuotaError"
    assert rec.recovery_action == "wait"

    # Persisted to disk and surfaced in run_log.md.
    reloaded = RunState.load(state.run_dir)
    assert reloaded.errors[0].message == "quota gone"
    log_text = (state.run_dir / "run_log.md").read_text(encoding="utf-8")
    assert "FATAL" in log_text and "wait" in log_text
