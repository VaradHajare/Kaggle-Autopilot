"""Phase 1 ingestion: unzip, file detection, format priority, orchestration."""

from __future__ import annotations

import zipfile

import pytest

from agent.llm import LLMClient
from agent.orchestrator import Orchestrator
from agent.tools.ingest import IngestError, detect_files, unzip_all
from agent.tools.kaggle_api import KaggleClient
from tests.conftest import FakeAnthropic, FakeKaggleApi


def _write(p, text="x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_detect_files_basic(tmp_path):
    _write(tmp_path / "train.csv")
    _write(tmp_path / "test.csv")
    _write(tmp_path / "sample_submission.csv")
    files = detect_files(tmp_path)
    assert files["train"].name == "train.csv"
    assert files["test"].name == "test.csv"
    assert files["sample_submission"].name == "sample_submission.csv"


def test_format_priority_prefers_csv(tmp_path):
    _write(tmp_path / "train.parquet")
    _write(tmp_path / "train.csv")
    _write(tmp_path / "sample_submission.csv")
    files = detect_files(tmp_path)
    assert files["train"].suffix == ".csv"


def test_missing_train_raises(tmp_path):
    _write(tmp_path / "sample_submission.csv")
    with pytest.raises(IngestError, match="train"):
        detect_files(tmp_path)


def test_missing_sample_submission_raises(tmp_path):
    _write(tmp_path / "train.csv")
    with pytest.raises(IngestError, match="sample_submission"):
        detect_files(tmp_path)


def test_unzip_all_extracts(tmp_path):
    inner = tmp_path / "train.csv"
    _write(inner, "a,b\n1,2\n")
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(inner, "train.csv")
    inner.unlink()
    extracted = unzip_all(tmp_path)
    assert archive in extracted
    assert (tmp_path / "train.csv").exists()


def _orch(tmp_path, settings, config):
    kaggle = KaggleClient(api=FakeKaggleApi())
    llm = LLMClient(api_key="t", client=FakeAnthropic())
    return Orchestrator(settings=settings, config=config, kaggle=kaggle, llm=llm,
                        runs_root=tmp_path / "runs")


def test_ingest_phase_end_to_end(tmp_path, settings, config):
    orch = _orch(tmp_path, settings, config)
    state = orch.bootstrap("demo-comp")
    # Fake download is a no-op; stage files in raw/ as if downloaded.
    raw = state.run_dir / "raw"
    _write(raw / "train.csv", "id,y\n1,0\n")
    _write(raw / "test.csv", "id\n1\n")
    _write(raw / "sample_submission.csv", "id,y\n1,0\n")

    state = orch.ingest(state)
    assert state.last_completed_phase == "1"
    assert state.train_path.name == "train.csv"
    assert state.sample_submission_path.name == "sample_submission.csv"
    # Persisted.
    from agent.memory import RunState
    reloaded = RunState.load(state.run_dir)
    assert reloaded.train_path is not None
