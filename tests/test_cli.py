"""CLI smoke tests via Typer's CliRunner with mocked orchestrator internals."""

from __future__ import annotations

from typer.testing import CliRunner

from agent import cli
from agent.errors import AgentFatalError

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
