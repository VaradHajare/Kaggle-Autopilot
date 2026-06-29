"""RunState — the single source of truth passed between all phases.

RunState is serialized to runs/<slug>/state.json after every phase completes.
On resume, STATE_VERSION is checked first; a mismatch raises
StateVersionMismatchError rather than silently loading incompatible state.

Repository rule: never bypass RunState. All cross-phase data flows through it.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, computed_field

from agent.errors import StateVersionMismatchError

# Bump on any breaking schema change. Checked on resume.
STATE_VERSION = "1.1"


class CompetitionMeta(BaseModel):
    slug: str
    eval_metric: str
    problem_type: str
    deadline: datetime | None = None
    team_size_limit: int | None = None
    # Authoritative per-day submission limit, fetched from Kaggle (never from env).
    daily_submission_limit: int = 5
    submissions_today: int = 0
    is_featured: bool = False
    prize_usd: float = 0.0


class EDASummary(BaseModel):
    n_rows: int
    n_cols: int
    numeric_cols: list[str] = Field(default_factory=list)
    categorical_cols: list[str] = Field(default_factory=list)
    datetime_cols: list[str] = Field(default_factory=list)
    text_cols: list[str] = Field(default_factory=list)
    boolean_cols: list[str] = Field(default_factory=list)
    high_missing_cols: list[str] = Field(default_factory=list)
    duplicate_rows: int = 0
    leakage_flags: list[str] = Field(default_factory=list)
    memory_mb: float = 0.0


class LLMEDAAnalysis(BaseModel):
    confirmed_problem_type: str
    high_risk_columns: list[str] = Field(default_factory=list)
    imputation_strategies: dict[str, str] = Field(default_factory=dict)
    anomaly_flags: list[str] = Field(default_factory=list)
    fe_directions: list[str] = Field(default_factory=list)
    fe_followups: list[str] = Field(default_factory=list)


class FEOperation(BaseModel):
    operation: str
    columns: list[str]
    output_name: str
    rationale: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


class ModelCandidate(BaseModel):
    model: str
    priority: int
    rationale: str = ""


class CVResult(BaseModel):
    model: str
    oof_score: float
    status: str = "COMPLETE"  # COMPLETE | PRUNED_EARLY | SKIPPED | ERROR
    best_params: dict[str, Any] = Field(default_factory=dict)
    n_trials: int = 0


class EnsembleStrategy(BaseModel):
    method: str  # weighted_average | rank_average | stacking | none
    rationale: str = ""
    blended_oof_score: float | None = None


class SubmissionRecord(BaseModel):
    path: Path
    cv_score: float
    timestamp: datetime
    public_lb_score: float | None = None


class LeaderboardEntry(BaseModel):
    timestamp: datetime
    submission_file: str
    cv_score: float
    public_lb_score: float | None = None
    delta: float | None = None
    rank: int | None = None
    iteration: int = 0


class AgentErrorRecord(BaseModel):
    timestamp: datetime
    phase: str
    error_type: str
    message: str
    traceback: str = ""
    recovery_action: str = ""


class RunState(BaseModel):
    state_version: str = STATE_VERSION

    slug: str
    run_dir: Path
    competition_meta: CompetitionMeta
    is_time_series: bool = False
    target_columns: list[str] = Field(default_factory=list)
    id_column: str | None = None

    # Phase 1 — ingested file paths (relative to run_dir/raw).
    train_path: Path | None = None
    test_path: Path | None = None
    sample_submission_path: Path | None = None

    eda_summary: EDASummary | None = None
    eda_analysis: LLMEDAAnalysis | None = None
    feature_engineering_ops: list[FEOperation] = Field(default_factory=list)
    deferred_fe_ops: list[FEOperation] = Field(default_factory=list)
    active_features: list[str] = Field(default_factory=list)

    selected_models: list[ModelCandidate] = Field(default_factory=list)
    cv_results: list[CVResult] = Field(default_factory=list)
    ensemble_strategy: EnsembleStrategy | None = None

    submission_paths: list[SubmissionRecord] = Field(default_factory=list)

    leaderboard_entries: list[LeaderboardEntry] = Field(default_factory=list)
    iteration: int = 0
    last_completed_phase: str | None = None
    total_tokens_used: int = 0
    errors: list[AgentErrorRecord] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def best_submission(self) -> SubmissionRecord | None:
        """The submission record with the highest CV score, or None."""
        if not self.submission_paths:
            return None
        return max(self.submission_paths, key=lambda r: r.cv_score)

    @property
    def effective_daily_limit(self) -> int:
        """min(personal cap, competition's authoritative limit) — set by caller
        via competition_meta. The personal cap is applied in submitter/orchestrator
        where the env value is available."""
        return self.competition_meta.daily_submission_limit

    def state_path(self) -> Path:
        return self.run_dir / "state.json"

    def save(self) -> None:
        """Serialize to runs/<slug>/state.json after a phase completes."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        self.state_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, run_dir: Path) -> "RunState":
        """Deserialize state.json. Raises StateVersionMismatchError on version
        mismatch — never silently migrate."""
        path = Path(run_dir) / "state.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        found = raw.get("state_version")
        if found != STATE_VERSION:
            raise StateVersionMismatchError(
                f"state.json version {found!r} != current schema {STATE_VERSION!r}.",
                remediation=(
                    "The state schema changed in a breaking way. Re-run with "
                    "--force-restart to discard the stale state and start fresh."
                ),
            )
        return cls.model_validate(raw)
