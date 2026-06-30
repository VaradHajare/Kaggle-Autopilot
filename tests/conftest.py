"""Shared fixtures: fake Kaggle API, fake Anthropic client, settings."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.config import AgentConfig, Settings
from agent.llm import GeminiLLM, LLMClient
from agent.tools.kaggle_api import KaggleClient


class FakeCompetition(SimpleNamespace):
    """Mimics a kaggle Competition object."""


class FakeFile(SimpleNamespace):
    pass


class FakeKaggleApi:
    """In-memory stand-in for kaggle.KaggleApi. No network, no credentials."""

    def __init__(
        self,
        *,
        slug: str = "demo-comp",
        rules_accepted: bool = True,
        total_bytes: int = 1_000,
        eval_metric: str = "AUC",
        category: str = "tabular",
        reward: str = "$0",
        daily_limit: int = 5,
        auth_ok: bool = True,
        public_score: str | None = "0.85",
    ) -> None:
        self.auth_ok = auth_ok
        self.public_score = public_score
        self.rules_accepted = rules_accepted
        self.submitted: list[tuple] = []
        self._comp = FakeCompetition(
            ref=f"https://www.kaggle.com/competitions/{slug}",
            evaluationMetric=eval_metric,
            category=category,
            reward=reward,
            maxDailySubmissions=daily_limit,
            maxTeamSize=5,
            deadline=None,
            userHasEntered=rules_accepted,
        )
        self._files = [FakeFile(name="train.csv", totalBytes=total_bytes)]

    def authenticate(self) -> None:
        if not self.auth_ok:
            raise RuntimeError("bad credentials")

    def competitions_list(self, *args, **kwargs):
        if not self.auth_ok:
            raise RuntimeError("bad credentials")
        return [self._comp]

    def competition_list_files(self, competition):
        if not self.rules_accepted:
            raise RuntimeError("403 Forbidden - You must accept the competition rules")
        return self._files

    def competition_download_files(self, competition, path):
        (path)  # no-op in tests

    def competition_submissions(self, competition):
        if not self.submitted:
            return []
        return [SimpleNamespace(publicScore=self.public_score)]

    def competition_submit(self, file_name, message, competition):
        self.submitted.append((file_name, message, competition))
        return SimpleNamespace(status="complete")


class FakeMessages:
    def __init__(self, text='{"ok": true}'):
        self._text = text

    def create(self, **kwargs):
        return SimpleNamespace(
            content=[SimpleNamespace(text=self._text)],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )


class FakeAnthropic:
    def __init__(self, text='{"ok": true}'):
        self.messages = FakeMessages(text)


class ApiStatusError(Exception):
    """Mimics a provider SDK error carrying an HTTP status code."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _RaisingMessages:
    """messages.create that records every call and always raises `exc`."""

    def __init__(self, exc: Exception, calls: list) -> None:
        self._exc = exc
        self.calls = calls

    def create(self, **kwargs):
        self.calls.append(kwargs)
        raise self._exc


class FakeAnthropicRaising:
    """Anthropic stand-in whose every request raises a given exception."""

    def __init__(self, exc: Exception) -> None:
        self.calls: list = []
        self.messages = _RaisingMessages(exc, self.calls)


class FakeGeminiModels:
    def __init__(self, text='{"ok": true}'):
        self._text = text

    def generate_content(self, model, contents, config=None):
        return SimpleNamespace(
            text=self._text,
            usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5),
        )


class FakeGemini:
    """In-memory stand-in for google.genai.Client."""

    def __init__(self, text='{"ok": true}'):
        self.models = FakeGeminiModels(text)


@pytest.fixture
def fake_kaggle():
    return FakeKaggleApi()


@pytest.fixture
def kaggle_client(fake_kaggle):
    return KaggleClient(api=fake_kaggle)


@pytest.fixture
def llm_client():
    return LLMClient(api_key="test", client=FakeAnthropic())


@pytest.fixture
def gemini_client():
    return GeminiLLM(api_key="test", client=FakeGemini())


@pytest.fixture
def settings():
    # Default provider is Gemini; set its key so bootstrap passes.
    return Settings(
        llm_provider="gemini",
        gemini_api_key="gm-test",
        anthropic_api_key="sk-ant-test",
        kaggle_username="tester",
        kaggle_key="key",
    )


@pytest.fixture
def config():
    return AgentConfig()  # defaults
