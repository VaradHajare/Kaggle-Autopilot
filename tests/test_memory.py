"""RunState serialization, versioning, and computed properties."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent.errors import StateVersionMismatchError
from agent.memory import (
    STATE_VERSION,
    CompetitionMeta,
    RunState,
    SubmissionRecord,
)


def _state(tmp_path: Path) -> RunState:
    return RunState(
        slug="demo",
        run_dir=tmp_path / "demo",
        competition_meta=CompetitionMeta(
            slug="demo", eval_metric="AUC", problem_type="tabular"
        ),
    )


def test_save_and_load_roundtrip(tmp_path):
    state = _state(tmp_path)
    state.is_time_series = True
    state.target_columns = ["y"]
    state.save()

    loaded = RunState.load(state.run_dir)
    assert loaded.slug == "demo"
    assert loaded.is_time_series is True
    assert loaded.target_columns == ["y"]
    assert loaded.state_version == STATE_VERSION


def test_version_mismatch_raises(tmp_path):
    state = _state(tmp_path)
    state.save()
    # Corrupt the version on disk.
    p = state.state_path()
    p.write_text(p.read_text().replace(STATE_VERSION, "0.9"), encoding="utf-8")

    with pytest.raises(StateVersionMismatchError):
        RunState.load(state.run_dir)


def test_best_submission_picks_highest_cv(tmp_path):
    state = _state(tmp_path)
    now = datetime.now(timezone.utc)
    state.submission_paths = [
        SubmissionRecord(path=Path("a.csv"), cv_score=0.80, timestamp=now),
        SubmissionRecord(path=Path("b.csv"), cv_score=0.91, timestamp=now),
        SubmissionRecord(path=Path("c.csv"), cv_score=0.85, timestamp=now),
    ]
    assert state.best_submission.cv_score == 0.91
    assert state.best_submission.path == Path("b.csv")


def test_best_submission_none_when_empty(tmp_path):
    assert _state(tmp_path).best_submission is None
