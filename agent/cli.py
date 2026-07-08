"""Typer CLI — the operator interface to the full 9-phase pipeline.

`run` / `resume` drive the orchestrator (phases 0-8 with the Phase 9 iteration
loop). `history` and `clean` are local, credential-free inspection/cleanup
commands that read or remove a run's `runs/<slug>/` directory.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import typer
from loguru import logger

from agent.config import AgentConfig, Settings
from agent.errors import AgentFatalError
from agent.memory import RunState
from agent.orchestrator import Orchestrator

# Runs live under ./runs/<slug> — the orchestrator's default runs_root.
RUNS_ROOT = Path("runs")

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
    """Run the full pipeline (phases 0-8, with the Phase 9 iteration loop)."""

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


@app.command()
def history(slug: str = typer.Argument(..., help="Competition slug")) -> None:
    """Print CV results, ensemble, submissions, and leaderboard for a saved run."""

    def _go() -> None:
        run_dir = RUNS_ROOT / slug
        if not (run_dir / "state.json").exists():
            raise AgentFatalError(
                f"No run found for {slug!r} at {run_dir}.",
                remediation="Run the pipeline first with `agent run <url>`.",
            )
        state = RunState.load(run_dir)  # raises on state-version mismatch (fatal)
        typer.secho(
            f"Run history - {state.slug} "
            f"(last phase {state.last_completed_phase}, iter {state.iteration})",
            fg=typer.colors.CYAN,
        )
        for r in state.cv_results:
            typer.echo(f"  CV {r.model}: {r.oof_score:.5f} ({r.status})")
        if state.ensemble_strategy is not None:
            es = state.ensemble_strategy
            suffix = "" if es.blended_oof_score is None else f" (oof={es.blended_oof_score:.5f})"
            typer.echo(f"  Ensemble: {es.method}{suffix}")
        for e in state.leaderboard_entries:
            lb = e.public_lb_score
            typer.echo(
                f"  LB iter {e.iteration}: cv={e.cv_score:.5f} "
                f"lb={lb if lb is not None else 'pending'}"
            )
        best = state.best_submission
        if best is not None:
            typer.secho(
                f"Best submission: {best.path} (cv={best.cv_score:.5f})",
                fg=typer.colors.GREEN,
            )

    _run_or_die(_go)


@app.command()
def clean(
    slug: str = typer.Argument(..., help="Competition slug"),
    keep_models: bool = typer.Option(
        False, "--keep-models", help="Preserve the models/ directory"
    ),
) -> None:
    """Delete a run's working directory (optionally keeping trained models)."""

    def _go() -> None:
        run_dir = RUNS_ROOT / slug
        if not run_dir.exists():
            raise AgentFatalError(f"No run directory to clean at {run_dir}.")
        if keep_models:
            removed = 0
            for child in sorted(run_dir.iterdir()):
                if child.name == "models":
                    continue
                shutil.rmtree(child) if child.is_dir() else child.unlink()
                removed += 1
            typer.secho(
                f"Cleaned {run_dir} - kept models/, removed {removed} entrie(s).",
                fg=typer.colors.GREEN,
            )
        else:
            shutil.rmtree(run_dir)
            typer.secho(f"Removed {run_dir}.", fg=typer.colors.GREEN)

    _run_or_die(_go)


if __name__ == "__main__":
    app()
