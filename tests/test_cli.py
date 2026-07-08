"""CLI smoke tests via Typer's CliRunner with mocked orchestrator internals."""

from __future__ import annotations

from datetime import datetime, timezone

from typer.testing import CliRunner

from agent import cli
from agent.errors import AgentFatalError
from agent.memory import (
    CompetitionMeta,
    CVResult,
    EnsembleStrategy,
    LeaderboardEntry,
    RunState,
    SubmissionRecord,
)

runner = CliRunner()


def test_run_reports_fatal_error(monkeypatch):
    class Boom:
        config = type("C", (), {"auto_submit": False, "max_iterations": 3})()

        def run_pipeline(self, *a, **k):
            raise AgentFatalError("nope", remediation="do X")

    monkeypatch.setattr(cli, "_build", lambda *a, **k: Boom())
    monkeypatch.setattr(cli, "Settings", lambda: type("S", (), {"log_level": "INFO"})())
    result = runner.invoke(cli.app, ["run", "demo-comp"])
    assert result.exit_code == 1
    assert "FATAL" in result.output
    assert "do X" in result.output


def test_run_success(monkeypatch):
    class OK:
        config = type("C", (), {"auto_submit": False, "max_iterations": 3})()

        def run_pipeline(self, *a, **k):
            from types import SimpleNamespace
            return SimpleNamespace(slug="demo-comp", last_completed_phase="8",
                                   iteration=0, best_submission=None)

    monkeypatch.setattr(cli, "_build", lambda *a, **k: OK())
    monkeypatch.setattr(cli, "Settings", lambda: type("S", (), {"log_level": "INFO"})())
    result = runner.invoke(cli.app, ["run", "demo-comp"])
    assert result.exit_code == 0
    assert "Pipeline complete" in result.output


# ----------------------------------------------------------------- entrypoint
def test_module_entrypoint_exposes_app():
    """`python -m agent` must resolve to the same Typer app as `agent.cli`."""
    import agent.__main__ as entry

    assert entry.app is cli.app


# ----------------------------------------------------------------- history
def _seed_run(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "submissions").mkdir(exist_ok=True)
    sub = run_dir / "submissions" / "s.csv"
    sub.write_text("id,y\n1,0\n", encoding="utf-8")
    state = RunState(
        slug="demo-comp",
        run_dir=run_dir,
        competition_meta=CompetitionMeta(
            slug="demo-comp", eval_metric="AUC", problem_type="binary"
        ),
        cv_results=[CVResult(model="LGBMClassifier", oof_score=0.87, status="COMPLETE")],
        ensemble_strategy=EnsembleStrategy(
            method="weighted_average", blended_oof_score=0.88
        ),
        leaderboard_entries=[
            LeaderboardEntry(
                timestamp=datetime.now(timezone.utc), submission_file="s.csv",
                cv_score=0.88, public_lb_score=0.85, iteration=0,
            )
        ],
        submission_paths=[
            SubmissionRecord(
                path=sub, cv_score=0.88, timestamp=datetime.now(timezone.utc)
            )
        ],
        last_completed_phase="8",
    )
    state.save()
    return state


def test_history_prints_run_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RUNS_ROOT", tmp_path)
    _seed_run(tmp_path / "demo-comp")
    result = runner.invoke(cli.app, ["history", "demo-comp"])
    assert result.exit_code == 0
    assert "LGBMClassifier" in result.output
    assert "0.87000" in result.output
    assert "weighted_average" in result.output
    assert "Best submission" in result.output


def test_history_missing_run_is_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RUNS_ROOT", tmp_path)
    result = runner.invoke(cli.app, ["history", "nope"])
    assert result.exit_code == 1
    assert "No run found" in result.output


# ----------------------------------------------------------------- clean
def test_clean_removes_run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RUNS_ROOT", tmp_path)
    run_dir = tmp_path / "demo-comp"
    (run_dir / "models").mkdir(parents=True)
    (run_dir / "models" / "m.joblib").write_text("x", encoding="utf-8")
    (run_dir / "state.json").write_text("{}", encoding="utf-8")
    result = runner.invoke(cli.app, ["clean", "demo-comp"])
    assert result.exit_code == 0
    assert not run_dir.exists()


def test_clean_keep_models_preserves_models_only(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RUNS_ROOT", tmp_path)
    run_dir = tmp_path / "demo-comp"
    (run_dir / "models").mkdir(parents=True)
    (run_dir / "models" / "m.joblib").write_text("x", encoding="utf-8")
    (run_dir / "processed").mkdir(parents=True)
    (run_dir / "state.json").write_text("{}", encoding="utf-8")
    result = runner.invoke(cli.app, ["clean", "demo-comp", "--keep-models"])
    assert result.exit_code == 0
    assert (run_dir / "models" / "m.joblib").exists()
    assert not (run_dir / "processed").exists()
    assert not (run_dir / "state.json").exists()


def test_clean_missing_dir_is_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RUNS_ROOT", tmp_path)
    result = runner.invoke(cli.app, ["clean", "nope"])
    assert result.exit_code == 1
    assert "No run directory" in result.output
