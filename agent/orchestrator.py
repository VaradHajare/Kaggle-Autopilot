"""Top-level agent loop — drives phase sequencing.

Dependency direction: orchestrator -> tools -> memory. The orchestrator owns
RunState, persists it after each phase, and decides recoverable vs fatal failures.

Currently implemented: Phase 0 (Bootstrap & Validation).
"""

from __future__ import annotations

import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from agent.config import AgentConfig, Settings, load_model_search_spaces
from agent.errors import AgentFatalError, BootstrapError
from agent.llm import BaseLLM, build_llm
from agent.memory import (
    AgentErrorRecord,
    CompetitionMeta,
    EnsembleStrategy,
    FEOperation,
    LeaderboardEntry,
    LLMEDAAnalysis,
    ModelCandidate,
    RunState,
    SubmissionRecord,
)
from agent.run_log import RunLog
from agent.tools import eda as eda_tools
from agent.tools import ensembler as ens_tools
from agent.tools import feature_engineering as fe_tools
from agent.tools import model_selector as ms_tools
from agent.tools import submitter as sub_tools
from agent.tools import trainer as train_tools
from agent.tools.ingest import detect_files, unzip_all
from agent.tools.kaggle_api import KaggleClient, parse_competition_slug

# Require 1.5x the competition's total file size in free disk before download.
DISK_HEADROOM_FACTOR = 1.5
SUBDIRS = ("raw", "processed", "models", "submissions")


def _state_context(state: RunState) -> str:
    """Compact RunState JSON for LLM system prompts (kept small on purpose)."""
    return state.model_dump_json(
        include={
            "slug", "competition_meta", "is_time_series",
            "target_columns", "id_column", "iteration",
        }
    )


def _rel(path: Path | None, base: Path) -> str:
    """Render a path relative to the run dir for logging, falling back to name."""
    if path is None:
        return "<none>"
    try:
        return str(path.relative_to(base))
    except ValueError:
        return path.name


class Orchestrator:
    def __init__(
        self,
        *,
        settings: Settings,
        config: AgentConfig,
        kaggle: KaggleClient | None = None,
        llm: BaseLLM | None = None,
        runs_root: Path | str = "runs",
    ) -> None:
        self.settings = settings
        self.config = config
        # Documented env override: OPTUNA_N_TRIALS wins over agent.yaml when set.
        # Resolved here, once, at bootstrap — config is not re-read mid-run.
        if settings.optuna_n_trials is not None:
            self.config.optuna_n_trials = settings.optuna_n_trials
        self.kaggle = kaggle or KaggleClient()
        self.llm = llm or build_llm(settings)
        self.runs_root = Path(runs_root)

    def _validate_runtime_credentials(self) -> None:
        """Kaggle + active-LLM auth checks. Run on both fresh start and resume so
        a missing/invalid key fails loud at bootstrap, not mid-pipeline."""
        if not (self.settings.kaggle_username and self.settings.kaggle_key):
            raise AgentFatalError(
                "Kaggle credentials missing.",
                remediation="Set KAGGLE_USERNAME and KAGGLE_KEY in .env.",
            )
        if not self.kaggle.check_credentials():
            raise AgentFatalError(
                "Kaggle API authentication failed.",
                remediation="Verify KAGGLE_USERNAME / KAGGLE_KEY are correct.",
            )

        provider = self.settings.llm_provider
        if not self.settings.llm_api_key:
            raise AgentFatalError(
                f"{provider} API key missing.",
                remediation=f"Set {self.settings.llm_key_env} in .env.",
            )
        if not self.llm.check_credentials():
            raise AgentFatalError(
                f"{provider} API authentication failed.",
                remediation=f"Verify {self.settings.llm_key_env} is valid.",
            )

    # ------------------------------------------------------------------ Phase 0
    def bootstrap(
        self,
        url_or_slug: str,
        *,
        resume: bool = False,
        force_restart: bool = False,
        confirm_high_stakes: bool = False,
    ) -> RunState:
        """Phase 0 — validate everything before any data operation."""
        t0 = time.time()

        # 1. Parse + validate URL.
        try:
            slug = parse_competition_slug(url_or_slug)
        except ValueError as exc:
            raise BootstrapError(str(exc), remediation="Pass a valid competition URL or slug.")

        run_dir = self.runs_root / slug
        state_file = run_dir / "state.json"

        # 2. Resume / force-restart / ambiguous-state handling.
        if state_file.exists():
            if force_restart:
                logger.warning("Force-restart: clearing existing run dir {}", run_dir)
                shutil.rmtree(run_dir)
            elif resume:
                state = RunState.load(run_dir)  # raises on version mismatch
                # Validate credentials on resume too — otherwise a bad/expired
                # LLM or Kaggle key only surfaces deep in a later phase.
                self._validate_runtime_credentials()
                logger.info("Resuming {} from phase {}", slug, state.last_completed_phase)
                return state
            else:
                raise BootstrapError(
                    f"Existing run state found at {state_file}. "
                    "Choose --resume to continue or --force-restart to discard.",
                    remediation="Re-run with --resume or --force-restart.",
                )

        # 3 + 4. Kaggle + LLM credentials (shared with the resume path).
        self._validate_runtime_credentials()

        # 5. Rules acceptance — mandatory before any data operation.
        if not self.kaggle.check_rules_accepted(slug):
            rules_url = f"https://www.kaggle.com/competitions/{slug}/rules"
            raise AgentFatalError(
                f"Competition rules for {slug!r} have not been accepted.",
                remediation=f"Accept the rules at {rules_url} then re-run.",
            )

        # 6. Disk space check before download.
        total_bytes = self.kaggle.get_total_file_size(slug)
        required = int(total_bytes * DISK_HEADROOM_FACTOR)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.runs_root).free
        if free < required:
            shortfall = required - free
            raise AgentFatalError(
                f"Insufficient disk space for {slug}: need {required:,} bytes "
                f"(1.5x data), have {free:,} free — short by {shortfall:,} bytes.",
                remediation=f"Free at least {shortfall:,} bytes near {self.runs_root.resolve()}.",
            )

        # 7. Create run dir tree.
        for sub in SUBDIRS:
            (run_dir / sub).mkdir(parents=True, exist_ok=True)

        # 8. Metadata + high-stakes guard.
        meta = self.kaggle.get_competition_metadata(slug)
        self._guard_high_stakes(meta, confirm_high_stakes)

        # 9. Build RunState, write run_log header, persist.
        state = RunState(slug=slug, run_dir=run_dir, competition_meta=meta)
        run_log = RunLog(run_dir)
        run_log.write_header(slug, self._env_fingerprint())
        run_log.phase(
            "0", "Bootstrap & Validation",
            status="COMPLETE",
            duration_s=time.time() - t0,
            summary=[
                f"slug={slug}",
                f"eval_metric={meta.eval_metric}",
                f"daily_submission_limit={meta.daily_submission_limit}",
                f"data_size_bytes={total_bytes:,}",
            ],
        )
        state.last_completed_phase = "0"
        state.save()
        return state

    # ------------------------------------------------------------- full pipeline
    def _sequence(self) -> list[tuple[str, object]]:
        return [
            ("1", self.ingest),
            ("2", self.eda),
            ("3", self.feature_engineering),
            ("4", self.model_selection),
            ("5a", self.feature_pruning),
            ("5b", self.train),
            ("6", self.ensemble),
            ("7", self.generate_submission),
            ("8", self.leaderboard),
        ]

    def run_pipeline(
        self,
        url_or_slug: str,
        *,
        resume: bool = False,
        force_restart: bool = False,
        confirm_high_stakes: bool = False,
    ) -> RunState:
        """Bootstrap then drive phases 1->8, looping per the Phase 9 mapping."""
        state = self.bootstrap(
            url_or_slug, resume=resume, force_restart=force_restart,
            confirm_high_stakes=confirm_high_stakes,
        )
        sequence = self._sequence()
        start = self._resume_index(sequence, state) if resume else 0

        while True:
            for _id, fn in sequence[start:]:
                try:
                    fn(state)
                except AgentFatalError as exc:
                    self._record_fatal(state, _id, exc)
                    raise
            reentry = self.plan_iteration(state)
            if reentry is None:
                break
            state.iteration += 1
            logger.info("Iteration {} — re-entering at phase {}", state.iteration, reentry)
            start = next(i for i, (pid, _) in enumerate(sequence) if pid == reentry)
        return state

    @staticmethod
    def _resume_index(sequence: list[tuple[str, object]], state: RunState) -> int:
        """Index of the first phase to run after the last completed one."""
        order = ["0", "1", "2", "3", "4", "5a", "5b", "6", "7", "8"]
        last = state.last_completed_phase or "0"
        try:
            next_pos = order.index(last) + 1
        except ValueError:
            return 0
        for i, (pid, _) in enumerate(sequence):
            if order.index(pid) >= next_pos:
                return i
        return len(sequence)  # everything done

    def _record_fatal(self, state: RunState, phase_id: str, exc: AgentFatalError) -> None:
        """Append a fatal phase failure to RunState.errors, persist, and log it
        before the exception propagates to the CLI. Keeps fatal runs auditable
        from state.json / run_log.md instead of a bare traceback."""
        state.errors.append(
            AgentErrorRecord(
                timestamp=datetime.now(timezone.utc),
                phase=phase_id,
                error_type=type(exc).__name__,
                message=str(exc),
                recovery_action=exc.remediation or "",
            )
        )
        try:
            state.save()
        except Exception:  # noqa: BLE001 — never mask the original failure
            logger.error("Failed to persist state after fatal error in phase {}", phase_id)
        entry = [str(exc)]
        if exc.remediation:
            entry.append(f"remediation: {exc.remediation}")
        RunLog(state.run_dir).phase(
            phase_id, f"Phase {phase_id}", status="FATAL", errors=entry,
        )

    # ------------------------------------------------------------------ Phase 1
    def ingest(self, state: RunState) -> RunState:
        """Phase 1 — download competition data, unzip, detect canonical files."""
        t0 = time.time()
        raw_dir = state.run_dir / "raw"
        run_log = RunLog(state.run_dir)

        # Download only if raw/ has no data yet (idempotent on resume).
        if not any(raw_dir.rglob("*")):
            self.kaggle.download(state.slug, raw_dir)
        unzip_all(raw_dir)

        files = detect_files(raw_dir)
        state.train_path = files["train"]
        state.test_path = files["test"]
        state.sample_submission_path = files["sample_submission"]

        run_log.phase(
            "1", "Competition Ingestion",
            status="COMPLETE",
            duration_s=time.time() - t0,
            summary=[
                f"train={_rel(state.train_path, state.run_dir)}",
                f"test={_rel(state.test_path, state.run_dir)}",
                f"sample_submission={_rel(state.sample_submission_path, state.run_dir)}",
            ],
            metrics={
                "eval_metric": state.competition_meta.eval_metric,
                "problem_type": state.competition_meta.problem_type,
            },
        )
        state.last_completed_phase = "1"
        state.save()
        return state

    # ------------------------------------------------------------------ Phase 2
    def eda(self, state: RunState) -> RunState:
        """Phase 2 — EDA: typing, target ID, time-series + leakage detection,
        LLM analysis."""
        t0 = time.time()
        run_log = RunLog(state.run_dir)

        train = eda_tools.load_table(state.train_path)
        sample_cols = eda_tools.read_header(state.sample_submission_path)
        test_cols = (
            eda_tools.read_header(state.test_path) if state.test_path else []
        )

        id_col, targets = eda_tools.identify_target_columns(sample_cols, test_cols)
        if not targets:
            run_log.section("WARNING", "No target columns identified from sample_submission.")
        state.id_column = id_col
        state.target_columns = targets

        state.is_time_series = eda_tools.detect_time_series(
            train, state.competition_meta.problem_type
        )
        summary = eda_tools.build_eda_summary(train, targets, id_column=id_col)
        state.eda_summary = summary

        if self.config.save_plots:
            eda_tools.maybe_profile_report(train, state.run_dir / "eda_report.html")

        if summary.leakage_flags:
            run_log.section(
                "LEAKAGE FLAGS",
                "\n".join(f"- `{c}` excluded from feature engineering"
                          for c in summary.leakage_flags),
            )

        # LLM analysis (falls back to heuristic defaults on failure).
        analysis = self._llm_eda_analysis(state, summary)
        state.eda_analysis = analysis
        (state.run_dir / "eda_analysis.json").write_text(
            analysis.model_dump_json(indent=2), encoding="utf-8"
        )

        run_log.phase(
            "2", "Exploratory Data Analysis",
            status="COMPLETE",
            duration_s=time.time() - t0,
            tokens_in=self.llm.total_tokens_in,
            tokens_out=self.llm.total_tokens_out,
            summary=[
                f"shape={summary.n_rows}x{summary.n_cols}",
                f"targets={targets}",
                f"id_column={id_col}",
                f"is_time_series={state.is_time_series}",
            ],
            decisions=[f"confirmed_problem_type={analysis.confirmed_problem_type}"],
            metrics={
                "high_missing_cols": len(summary.high_missing_cols),
                "leakage_flags": len(summary.leakage_flags),
                "duplicate_rows": summary.duplicate_rows,
            },
        )
        state.total_tokens_used = self.llm.total_tokens_in + self.llm.total_tokens_out
        state.last_completed_phase = "2"
        state.save()
        return state

    def _llm_eda_analysis(self, state: RunState, summary) -> LLMEDAAnalysis:
        default = LLMEDAAnalysis(
            confirmed_problem_type=state.competition_meta.problem_type,
            high_risk_columns=summary.leakage_flags,
        )
        system = (
            "You are a Kaggle EDA analyst. Given an EDA summary, return JSON with keys: "
            "confirmed_problem_type (str), high_risk_columns (list[str]), "
            "imputation_strategies (obj col->strategy), anomaly_flags (list[str]), "
            "fe_directions (list[str]), fe_followups (list[str])."
        )
        res = self.llm.call_json(
            system=system,
            user=f"EDA summary:\n{summary.model_dump_json()}",
            run_state_json=_state_context(state),
            default=default.model_dump(),
        )
        try:
            return LLMEDAAnalysis.model_validate(res.data)
        except Exception:  # noqa: BLE001
            return default

    # ------------------------------------------------------------------ Phase 3
    def feature_engineering(self, state: RunState) -> RunState:
        """Phase 3 — LLM-guided FE. Immediate ops run now (fit on train);
        deferred target/group encodings are stored for the in-fold CV path."""
        t0 = time.time()
        run_log = RunLog(state.run_dir)

        train = eda_tools.load_table(state.train_path)
        test = eda_tools.load_table(state.test_path) if state.test_path else None

        drop = (
            set(state.target_columns)
            | ({state.id_column} if state.id_column else set())
            | set(state.eda_summary.leakage_flags)
        )
        base_feature_cols = [c for c in train.columns if c not in drop]

        ops = self._llm_feature_strategy(state, base_feature_cols)

        immediate_created: list[str] = []
        executed: list[FEOperation] = []
        deferred: list[FEOperation] = []
        errors: list[str] = []
        identity_cols = set(state.eda_summary.high_cardinality_cols)

        for op in ops:
            try:
                fe_tools.validate_operation(op, is_time_series=state.is_time_series)
            except fe_tools.FEValidationError as exc:
                errors.append(f"{op.operation}: {exc}")
                continue
            if fe_tools.is_identity_unsafe(op, identity_cols):
                errors.append(
                    f"{op.operation}: skipped on identity-like column(s) {op.columns} "
                    "— overfits, does not generalize"
                )
                continue
            if op.operation in fe_tools.DEFERRED_OPS:
                deferred.append(op)
                continue
            try:
                created = fe_tools.execute_immediate(train, test, op)
                immediate_created.extend(created)
                executed.append(op)
            except Exception as exc:  # noqa: BLE001 — skip and continue
                errors.append(f"{op.operation}: {exc}")

        # Build the model-ready base matrix: raw features minus text/datetime and
        # identity-like high-cardinality columns (kept out so they aren't label-
        # encoded into per-row identifiers), plus immediate-created columns.
        # Deferred ops are applied in Phase 5b.
        exclude_raw = (
            set(state.eda_summary.text_cols)
            | set(state.eda_summary.datetime_cols)
            | identity_cols
        )
        model_cols = [c for c in base_feature_cols if c not in exclude_raw]
        model_cols += [c for c in immediate_created if c not in model_cols]
        model_cols = list(dict.fromkeys(model_cols))

        tr_base, te_base = fe_tools.standard_preprocess(
            train, test, feature_columns=model_cols
        )
        proc = state.run_dir / "processed"
        proc.mkdir(parents=True, exist_ok=True)
        tr_base.to_parquet(proc / "train_fe_base.parquet")
        if te_base is not None:
            te_base.to_parquet(proc / "test_fe_base.parquet")

        state.feature_engineering_ops = executed
        state.deferred_fe_ops = deferred
        state.active_features = list(tr_base.columns)

        run_log.phase(
            "3", "Feature Engineering",
            status="COMPLETE",
            duration_s=time.time() - t0,
            tokens_in=self.llm.total_tokens_in,
            tokens_out=self.llm.total_tokens_out,
            summary=[
                f"features_before={len(base_feature_cols)}",
                f"features_after={len(state.active_features)}",
                f"immediate_ops={len(executed)}",
                f"deferred_ops={len(deferred)}",
            ],
            metrics={"skipped_ops": len(errors)},
            errors=errors or None,
        )
        state.total_tokens_used = self.llm.total_tokens_in + self.llm.total_tokens_out
        state.last_completed_phase = "3"
        state.save()
        return state

    def _llm_feature_strategy(
        self, state: RunState, feature_cols: list[str]
    ) -> list[FEOperation]:
        system = (
            "You are a Kaggle feature engineer. Given the column schema and the "
            "allowed operation registry, return a JSON array of operations. Each "
            "operation has keys: operation, columns (list), output_name, rationale, "
            "and optional params. Only use operations from the registry.\n"
            "IMPORTANT — identity-like high-cardinality columns (names, ticket ids, "
            "cabin/serial codes; listed under identity_columns) have near-unique "
            "values that DO NOT generalize to the disjoint test set. Never apply "
            "tfidf_svd to them (it is rejected). Instead extract generalizing signal: "
            "extract_title for a name column; family_size and is_alone from "
            "sibling/spouse + parent/child counts; cabin_deck for a cabin code; "
            "count_encoding to turn a shared id into a group size."
        )
        schema = {
            "columns": feature_cols,
            "identity_columns": state.eda_summary.high_cardinality_cols
            if state.eda_summary else [],
            "registry": list(fe_tools.REGISTRY.keys()),
            "is_time_series": state.is_time_series,
            "eda_directions": state.eda_analysis.fe_directions if state.eda_analysis else [],
        }
        res = self.llm.call_json(
            system=system,
            user=f"Schema:\n{schema}",
            run_state_json=_state_context(state),
            default=[],
            max_tokens=8192,
        )
        data = res.data
        if isinstance(data, dict):
            data = data.get("operations", [])
        out: list[FEOperation] = []
        for item in data or []:
            try:
                out.append(FEOperation.model_validate(item))
            except Exception:  # noqa: BLE001 — malformed entries are skipped
                continue
        return out

    # ------------------------------------------------------------------ Phase 4
    def model_selection(self, state: RunState) -> RunState:
        """Phase 4 — LLM-ranked model list, filtered + hard-rule-enforced."""
        t0 = time.time()
        run_log = RunLog(state.run_dir)
        spaces = load_model_search_spaces()

        task_kind = ms_tools.infer_task_kind(
            state.eda_analysis.confirmed_problem_type if state.eda_analysis else "",
            state.competition_meta.eval_metric,
        )
        candidates = self._llm_model_selection(state, task_kind)
        n_rows = state.eda_summary.n_rows if state.eda_summary else 0
        selected = ms_tools.select_models(
            candidates,
            task_kind=task_kind,
            search_spaces=spaces,
            allow_neural=self.config.allow_neural,
            n_rows=n_rows,
        )
        state.selected_models = selected

        effective_limit = min(
            self.settings.submission_daily_limit,
            state.competition_meta.daily_submission_limit,
        )
        remaining = ms_tools.remaining_submission_budget(
            effective_limit, state.competition_meta.submissions_today
        )
        warnings = []
        if remaining <= 1:
            warnings.append(f"Only {remaining} submission slot(s) left today.")

        run_log.phase(
            "4", "Model Selection",
            status="COMPLETE",
            duration_s=time.time() - t0,
            tokens_in=self.llm.total_tokens_in,
            tokens_out=self.llm.total_tokens_out,
            summary=[f"task_kind={task_kind}"]
            + [f"#{m.priority} {m.model}" for m in selected],
            metrics={"effective_daily_limit": effective_limit, "remaining_today": remaining},
            errors=warnings or None,
        )
        state.total_tokens_used = self.llm.total_tokens_in + self.llm.total_tokens_out
        state.last_completed_phase = "4"
        state.save()
        return state

    def _llm_model_selection(self, state: RunState, task_kind: str) -> list[ModelCandidate]:
        default = [
            {"model": ms_tools._default_gbm(task_kind), "priority": 1,
             "rationale": "Default GBM baseline."}
        ]
        system = (
            "You are a Kaggle model selector. Given the task kind, eval metric, "
            "dataset size and feature count, return a JSON array of models ranked by "
            "priority (1=best). Each: {model, priority, rationale}. Choose from common "
            "sklearn/boosting estimator class names."
        )
        ctx = {
            "task_kind": task_kind,
            "eval_metric": state.competition_meta.eval_metric,
            "n_rows": state.eda_summary.n_rows if state.eda_summary else 0,
            "n_features": len(state.active_features),
            "catalog": list(ms_tools.MODEL_CATALOG.keys()),
        }
        res = self.llm.call_json(
            system=system, user=f"Context:\n{ctx}",
            run_state_json=_state_context(state), default=default,
        )
        data = res.data
        if isinstance(data, dict):
            data = data.get("models", [])
        out: list[ModelCandidate] = []
        for item in data or []:
            try:
                out.append(ModelCandidate.model_validate(item))
            except Exception:  # noqa: BLE001
                continue
        return out or [ModelCandidate.model_validate(default[0])]

    # ----------------------------------------------------------------- Phase 5a
    def feature_pruning(self, state: RunState) -> RunState:
        """Phase 5a — probe importances and prune to max_features (if needed)."""
        t0 = time.time()
        run_log = RunLog(state.run_dir)
        proc = state.run_dir / "processed"
        X = pd.read_parquet(proc / "train_fe_base.parquet")

        if len(state.active_features) <= self.config.max_features:
            run_log.phase("5a", "Feature Importance Probe & Pruning", status="SKIPPED",
                          duration_s=time.time() - t0,
                          summary=["Feature count within limit — pruning skipped."])
            state.last_completed_phase = "5a"
            state.save()
            return state

        y = self._load_target(state)
        task_kind = self._task_kind(state)
        probe_model = state.selected_models[0].model
        importances = train_tools.probe_feature_importance(
            probe_model, X, y, task_kind=task_kind, seed=self.config.cv_seed
        )
        keep = list(importances.head(self.config.max_features).index)
        dropped = [c for c in state.active_features if c not in set(keep)]

        X[keep].to_parquet(proc / "train_fe_base.parquet")
        test_path = proc / "test_fe_base.parquet"
        if test_path.exists():
            te = pd.read_parquet(test_path)
            te[[c for c in keep if c in te.columns]].to_parquet(test_path)
        state.active_features = keep

        run_log.phase("5a", "Feature Importance Probe & Pruning", status="COMPLETE",
                      duration_s=time.time() - t0,
                      summary=[f"kept={len(keep)}", f"dropped={len(dropped)}"],
                      metrics={"probe_model": probe_model})
        state.last_completed_phase = "5a"
        state.save()
        return state

    # ----------------------------------------------------------------- Phase 5b
    def train(self, state: RunState) -> RunState:
        """Phase 5b — Optuna search per model with in-fold deferred encoding."""
        t0 = time.time()
        run_log = RunLog(state.run_dir)
        proc = state.run_dir / "processed"
        models_dir = state.run_dir / "models"

        X = pd.read_parquet(proc / "train_fe_base.parquet")[state.active_features]
        test_path = proc / "test_fe_base.parquet"
        X_test = pd.read_parquet(test_path) if test_path.exists() else None
        if X_test is not None:
            X_test = X_test[[c for c in state.active_features if c in X_test.columns]]

        y = self._load_target(state)
        raw_train = eda_tools.load_table(state.train_path)
        raw_test = eda_tools.load_table(state.test_path) if state.test_path else None

        task_kind = self._task_kind(state)
        metric = train_tools.resolve_metric(state.competition_meta.eval_metric, task_kind)
        cv = train_tools.make_cv(task_kind, state.is_time_series,
                                 self.config.cv_folds, self.config.cv_seed)
        spaces = load_model_search_spaces()

        results: list = []
        best_prior: float | None = None
        deadline = t0 + self.config.max_training_hours * 3600
        for cand in state.selected_models:
            if time.time() > deadline:
                run_log.section("TRAINING BUDGET",
                                f"Time budget reached; skipping {cand.model} and lower.")
                break
            space = spaces.get(cand.model, {})
            artifacts = train_tools.train_model(
                name=cand.model, space=space, X=X, y=y, raw_train=raw_train,
                X_test=X_test, raw_test=raw_test, deferred_ops=state.deferred_fe_ops,
                task_kind=task_kind, metric=metric, cv=cv, models_dir=models_dir,
                n_trials=self.config.optuna_n_trials, timeout=self.config.optuna_timeout,
                best_prior_score=best_prior,
            )
            results.append(artifacts.cv_result)
            score = artifacts.cv_result.oof_score
            if best_prior is None or (
                (score > best_prior) if metric.higher_is_better else (score < best_prior)
            ):
                best_prior = score

        state.cv_results = results
        ordered = sorted(results, key=lambda r: r.oof_score,
                         reverse=metric.higher_is_better)
        run_log.phase("5b", "Training & Cross-Validation", status="COMPLETE",
                      duration_s=time.time() - t0,
                      summary=[f"{r.model}: {r.oof_score:.5f} ({r.status})" for r in ordered],
                      metrics={"eval_metric": metric.name,
                               "best": ordered[0].model if ordered else "none"})
        state.last_completed_phase = "5b"
        state.save()
        return state

    # ----------------------------------------------------------------- Phase 6
    def ensemble(self, state: RunState) -> RunState:
        """Phase 6 — blend models via weighted avg / rank avg / stacking / none."""
        t0 = time.time()
        run_log = RunLog(state.run_dir)
        models_dir = state.run_dir / "models"

        completed = [r for r in state.cv_results if r.status != "ERROR"]
        if not completed:
            raise AgentFatalError("No trained models available to ensemble.")

        y = self._load_target(state).to_numpy()
        task_kind = self._task_kind(state)
        metric = train_tools.resolve_metric(state.competition_meta.eval_metric, task_kind)

        names = [r.model for r in completed]
        scores = [r.oof_score for r in completed]
        oof_list = [np.load(models_dir / f"{n}_oof.npy") for n in names]
        test_list, foldtest_list = [], []
        for n in names:
            tp = models_dir / f"{n}_test.npy"
            ftp = models_dir / f"{n}_foldtest.npy"
            test_list.append(np.load(tp) if tp.exists() else None)
            foldtest_list.append(np.load(ftp) if ftp.exists() else None)

        best_idx = ens_tools._best_index(scores, metric.higher_is_better)
        method = self._llm_ensemble_method(state, names, scores)

        # Single dominant model -> skip ensembling.
        if len(completed) < self.config.min_models_for_ensemble:
            method = "none"

        used, blended_oof, test_pred = ens_tools.run_ensemble(
            method, oof_list=oof_list, test_list=test_list, foldtest_list=foldtest_list,
            scores=scores, y=y, task_kind=task_kind,
            higher_is_better=metric.higher_is_better,
            seed=self.config.cv_seed, n_splits=self.config.cv_folds,
        )
        blended_score = metric.score(y, np.asarray(blended_oof))
        if test_pred is None:
            test_pred = test_list[best_idx]
        np.save(models_dir / "ensemble_test.npy", np.asarray(test_pred))

        state.ensemble_strategy = EnsembleStrategy(
            method=used, blended_oof_score=blended_score,
            rationale=f"best_solo={max(scores) if metric.higher_is_better else min(scores):.5f}",
        )
        run_log.phase("6", "Ensembling", status="COMPLETE",
                      duration_s=time.time() - t0,
                      summary=[f"method={used}", f"models={names}"],
                      metrics={"blended_oof": round(blended_score, 6),
                               "best_solo": round(scores[best_idx], 6)})
        state.last_completed_phase = "6"
        state.save()
        return state

    def _llm_ensemble_method(self, state: RunState, names: list[str], scores: list[float]) -> str:
        system = (
            "You are a Kaggle ensembler. Given model OOF scores, choose a blending "
            'method. Return JSON: {"method": one of '
            '"weighted_average"|"rank_average"|"stacking"|"none", "rationale": str}.'
        )
        res = self.llm.call_json(
            system=system, user=f"models={names}\nscores={scores}",
            run_state_json=_state_context(state),
            default={"method": self.config.ensemble_fallback},
        )
        method = (res.data or {}).get("method", self.config.ensemble_fallback)
        valid = {"weighted_average", "rank_average", "stacking", "none"}
        return method if method in valid else self.config.ensemble_fallback

    # ----------------------------------------------------------------- Phase 7
    def generate_submission(self, state: RunState) -> RunState:
        """Phase 7 — map ensemble predictions into sample_submission format."""
        t0 = time.time()
        run_log = RunLog(state.run_dir)
        models_dir = state.run_dir / "models"

        sample = eda_tools.load_table(state.sample_submission_path)
        preds = np.load(models_dir / "ensemble_test.npy")
        task_kind = self._task_kind(state)
        metric = train_tools.resolve_metric(state.competition_meta.eval_metric, task_kind)

        label_domain = None
        if metric.kind == "label":
            label_domain = sorted(self._load_target(state).unique().tolist())
        processed = sub_tools.postprocess(preds, metric_kind=metric.kind,
                                          label_domain=label_domain)

        submission = sub_tools.build_submission(
            sample, id_column=state.id_column or sample.columns[0],
            target_columns=state.target_columns, preds=processed,
        )
        sub_tools.validate_submission(submission, sample)

        cv_score = (state.ensemble_strategy.blended_oof_score
                    if state.ensemble_strategy else 0.0) or 0.0
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        fname = sub_tools.submission_filename(ts, cv_score)
        path = state.run_dir / "submissions" / fname
        submission.to_csv(path, index=False)

        state.submission_paths.append(SubmissionRecord(
            path=path, cv_score=cv_score, timestamp=datetime.now(timezone.utc)))

        run_log.phase("7", "Submission Generation", status="COMPLETE",
                      duration_s=time.time() - t0,
                      summary=[f"file={fname}", f"rows={len(submission)}"],
                      metrics={"cv_score": round(cv_score, 6)})
        state.last_completed_phase = "7"
        state.save()
        return state

    # ----------------------------------------------------------------- Phase 8
    def leaderboard(self, state: RunState, *, sleep=time.sleep) -> RunState:
        """Phase 8 — submit the best file (if auto_submit) and poll for LB score."""
        t0 = time.time()
        run_log = RunLog(state.run_dir)
        best = state.best_submission
        if best is None:
            raise AgentFatalError("No submission available for leaderboard tracking.")

        effective_limit = min(self.settings.submission_daily_limit,
                              state.competition_meta.daily_submission_limit)
        remaining = ms_tools.remaining_submission_budget(
            effective_limit, state.competition_meta.submissions_today)

        if not self.config.auto_submit:
            run_log.phase("8", "Leaderboard Tracking", status="SKIPPED",
                          duration_s=time.time() - t0,
                          summary=[f"auto_submit disabled — best: {best.path}",
                                   "Submit manually or re-run with --submit."])
            state.last_completed_phase = "8"
            state.save()
            return state

        if remaining < 1:
            run_log.phase("8", "Leaderboard Tracking", status="SKIPPED",
                          duration_s=time.time() - t0,
                          summary=["No submission quota remaining today."])
            state.last_completed_phase = "8"
            state.save()
            return state

        self.kaggle.submit(state.slug, best.path,
                           f"cv={best.cv_score:.5f} iter={state.iteration}")
        lb_score = self._poll_lb(state.slug, sleep=sleep)
        delta = (lb_score - best.cv_score) if lb_score is not None else None
        state.competition_meta.submissions_today += 1
        state.leaderboard_entries.append(LeaderboardEntry(
            timestamp=datetime.now(timezone.utc), submission_file=str(best.path),
            cv_score=best.cv_score, public_lb_score=lb_score, delta=delta,
            iteration=state.iteration))
        self._write_leaderboard_json(state)

        diagnostics = []
        if lb_score is not None and abs(delta) > 0.01:
            diagnostics.append(
                f"CV-LB gap {delta:+.4f} > 0.01 — possible overfitting / CV mismatch.")
        run_log.phase("8", "Leaderboard Tracking",
                      status="COMPLETE" if lb_score is not None else "COMPLETE",
                      duration_s=time.time() - t0,
                      summary=[f"cv={best.cv_score:.5f}",
                               f"lb={lb_score if lb_score is not None else 'pending'}"],
                      metrics={"delta": delta}, errors=diagnostics or None)
        state.last_completed_phase = "8"
        state.save()
        return state

    def _poll_lb(self, slug: str, *, sleep=time.sleep) -> float | None:
        deadline = time.time() + self.config.lb_poll_timeout_minutes * 60
        while time.time() < deadline:
            score = self.kaggle.latest_submission_score(slug)
            if score is not None:
                return score
            sleep(30)
        return None

    def _write_leaderboard_json(self, state: RunState) -> None:
        import json
        path = state.run_dir / "leaderboard.json"
        payload = [e.model_dump(mode="json") for e in state.leaderboard_entries]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ----------------------------------------------------------------- Phase 9
    def plan_iteration(self, state: RunState) -> str | None:
        """Phase 9 — decide the lowest-numbered re-entry phase, or None to stop.

        Returns the re-entry phase id ('3','5a','5b','6') or None when no trigger
        fires or the iteration budget is exhausted.
        """
        if state.iteration + 1 >= self.config.max_iterations:
            return None

        metric = train_tools.resolve_metric(
            state.competition_meta.eval_metric, self._task_kind(state))
        triggers: list[tuple[int, str]] = []

        # CV-to-LB gap > 0.01 -> Phase 5b.
        last = state.leaderboard_entries[-1] if state.leaderboard_entries else None
        if last and last.public_lb_score is not None and abs(last.delta or 0) > 0.01:
            triggers.append((5, "5b"))

        # Ensemble barely better than best solo (< 0.002) -> Phase 6.
        if state.ensemble_strategy and state.ensemble_strategy.blended_oof_score is not None:
            solos = [r.oof_score for r in state.cv_results]
            if solos:
                best_solo = max(solos) if metric.higher_is_better else min(solos)
                gain = abs(state.ensemble_strategy.blended_oof_score - best_solo)
                if gain < 0.002:
                    triggers.append((6, "6"))

        # LLM-suggested FE follow-ups -> Phase 3.
        if state.eda_analysis and state.eda_analysis.fe_followups:
            triggers.append((3, "3"))

        if not triggers:
            return None
        # Lowest-numbered re-entry phase wins.
        triggers.sort(key=lambda t: t[0])
        return triggers[0][1]

    def _load_target(self, state: RunState) -> "pd.Series":
        raw = eda_tools.load_table(state.train_path)
        return raw[state.target_columns[0]]

    def _task_kind(self, state: RunState) -> str:
        return ms_tools.infer_task_kind(
            state.eda_analysis.confirmed_problem_type if state.eda_analysis else "",
            state.competition_meta.eval_metric,
        )

    # ------------------------------------------------------------------ helpers
    def _guard_high_stakes(self, meta: CompetitionMeta, confirmed: bool) -> None:
        if (meta.is_featured or meta.prize_usd > 10_000) and not confirmed:
            raise AgentFatalError(
                f"{meta.slug!r} is high-stakes (featured={meta.is_featured}, "
                f"prize=${meta.prize_usd:,.0f}).",
                remediation="Re-run with --confirm-high-stakes to proceed.",
            )

    def _env_fingerprint(self) -> dict[str, str]:
        fp = {
            "llm_provider": self.settings.llm_provider,
            "model": self.settings.active_model,
            "auto_submit": str(self.config.auto_submit),
            "cv_folds": str(self.config.cv_folds),
        }
        fp.update(self.settings.masked())
        return fp
