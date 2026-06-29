"""Typer CLI. Phases are built incrementally; commands wire to what exists.

Currently: `run` executes Phase 0 (Bootstrap & Validation). Later phases are
appended to the orchestrator and surfaced here as they land.
"""

from __future__ import annotations

import sys

import typer
from loguru import logger

from agent.config import AgentConfig, Settings
from agent.errors import AgentFatalError
from agent.orchestrator import Orchestrator

app = typer.Typer(add_completion=False, help="Kaggle Auto Competitor")


def _build(settings: Settings | None = None) -> Orchestrator:
    settings = settings or Settings()
    config = AgentConfig.load()
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level)
    return Orchestrator(settings=settings, config=config)


def _run_or_die(fn) -> None:
    try:
        fn()
    except AgentFatalError as exc:
        typer.secho(f"FATAL: {exc}", fg=typer.colors.RED, err=True)
        if exc.remediation:
            typer.secho(f"  -> {exc.remediation}", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=1)


@app.command()
def run(
    url: str = typer.Argument(..., help="Kaggle competition URL or slug"),
    submit: bool = typer.Option(False, "--submit", help="Enable auto-submit to Kaggle"),
    force_restart: bool = typer.Option(False, "--force-restart"),
    resume: bool = typer.Option(False, "--resume"),
    confirm_high_stakes: bool = typer.Option(False, "--confirm-high-stakes"),
    max_iterations: int = typer.Option(None, "--max-iterations"),
    log_level: str = typer.Option(None, "--log-level"),
) -> None:
    """Run the pipeline (Phase 0 implemented; later phases follow)."""

    def _go() -> None:
        settings = Settings()
        if log_level:
            settings.log_level = log_level
        orch = _build(settings)
        if submit:
            orch.config.auto_submit = True
        if max_iterations is not None:
            orch.config.max_iterations = max_iterations
        state = orch.run_pipeline(
            url, resume=resume, force_restart=force_restart,
            confirm_high_stakes=confirm_high_stakes,
        )
        best = state.best_submission
        typer.secho(f"Pipeline complete for {state.slug} "
                    f"(phase {state.last_completed_phase}, iter {state.iteration}).",
                    fg=typer.colors.GREEN)
        if best is not None:
            typer.echo(f"Best submission: {best.path} (cv={best.cv_score:.5f})")

    _run_or_die(_go)


@app.command()
def resume(slug: str = typer.Argument(...)) -> None:
    """Resume a run from its last checkpoint."""

    def _go() -> None:
        orch = _build()
        state = orch.run_pipeline(slug, resume=True)
        typer.secho(f"Resumed and completed {state.slug} at phase "
                    f"{state.last_completed_phase}.", fg=typer.colors.GREEN)

    _run_or_die(_go)


if __name__ == "__main__":
    app()
