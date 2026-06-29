"""Typed configuration, resolved once at bootstrap.

Two layers:
  - Settings: secrets + env-driven tuning (pydantic-settings, reads .env / environ).
  - AgentConfig: behaviour flags loaded from configs/agent.yaml.

Repository rule: no hidden configuration. Every tunable lives here, in agent.yaml,
or in models.yaml. Config is read once and passed down — never re-read mid-run.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AGENT_CONFIG = REPO_ROOT / "configs" / "agent.yaml"
DEFAULT_MODELS_CONFIG = REPO_ROOT / "configs" / "models.yaml"


class Settings(BaseSettings):
    """Secrets and env-driven tuning. Loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM provider selection. "gemini" (default for now) or "anthropic".
    llm_provider: str = "gemini"

    # Gemini (Google AI Studio)
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"

    # Anthropic (kept for later — flip llm_provider to "anthropic" to use)
    anthropic_api_key: str | None = None
    agent_model: str = "claude-sonnet-4-6"

    kaggle_username: str | None = None
    kaggle_key: str | None = None

    agent_max_tokens: int = 8192
    # None means "not set via env" -> the configs/agent.yaml value wins. When
    # OPTUNA_N_TRIALS is exported it overrides agent.yaml at bootstrap (see
    # Orchestrator.__init__).
    optuna_n_trials: int | None = None
    submission_daily_limit: int = 5
    log_level: str = "INFO"

    @property
    def llm_api_key(self) -> str | None:
        """API key for the active provider."""
        return self.gemini_api_key if self.llm_provider == "gemini" else self.anthropic_api_key

    @property
    def llm_key_env(self) -> str:
        """Env-var name to set for the active provider's key."""
        return "GEMINI_API_KEY" if self.llm_provider == "gemini" else "ANTHROPIC_API_KEY"

    @property
    def active_model(self) -> str:
        """Model id for the active provider."""
        return self.gemini_model if self.llm_provider == "gemini" else self.agent_model

    def masked(self) -> dict[str, str]:
        """Credentials masked for safe logging."""

        def mask(v: str | None) -> str:
            return "***" if v else "<unset>"

        return {
            "gemini_api_key": mask(self.gemini_api_key),
            "anthropic_api_key": mask(self.anthropic_api_key),
            "kaggle_username": mask(self.kaggle_username),
            "kaggle_key": mask(self.kaggle_key),
        }


class AgentConfig(BaseModel):
    """Behaviour flags from configs/agent.yaml."""

    auto_submit: bool = False
    max_iterations: int = 3
    max_training_hours: float = 4
    allow_neural: bool = False
    token_budget: int | None = None

    lb_poll_timeout_minutes: int = 10

    cv_folds: int = 5
    cv_seed: int = 42

    optuna_n_trials: int = 50
    optuna_timeout: int = 3600

    max_features: int = 2000
    text_tfidf_max_features: int = 500
    text_svd_components: int = 50

    min_models_for_ensemble: int = 2
    ensemble_fallback: str = "weighted_average"

    log_level: str = "INFO"
    save_plots: bool = True

    @classmethod
    def load(cls, path: Path | str = DEFAULT_AGENT_CONFIG) -> "AgentConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)


def load_model_search_spaces(path: Path | str = DEFAULT_MODELS_CONFIG) -> dict:
    """Load per-model Optuna search spaces from configs/models.yaml."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
