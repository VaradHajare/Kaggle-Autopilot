"""LLM client: JSON parsing, retry-once, fallback, token accounting."""

from __future__ import annotations

import pytest

from agent.errors import LLMAuthError, LLMQuotaError
from agent.llm import LLMClient, classify_llm_error
from tests.conftest import ApiStatusError, FakeAnthropic, FakeAnthropicRaising


def test_call_json_parses_and_counts_tokens():
    client = LLMClient(api_key="t", client=FakeAnthropic('{"a": 1}'))
    res = client.call_json(system="s", user="u", run_state_json="{}", default={})
    assert res.data == {"a": 1}
    assert res.fell_back is False
    assert res.tokens_in == 10 and res.tokens_out == 5
    assert client.total_tokens_in == 10


def test_call_json_falls_back_on_bad_json():
    client = LLMClient(api_key="t", client=FakeAnthropic("not json at all"))
    res = client.call_json(system="s", user="u", run_state_json="{}", default={"fallback": True})
    assert res.fell_back is True
    assert res.data == {"fallback": True}
    # Two attempts were made -> tokens counted twice.
    assert res.tokens_out == 10


def test_check_credentials_ok():
    client = LLMClient(api_key="t", client=FakeAnthropic("ok"))
    assert client.check_credentials() is True


# ----------------------------------------------------- error classification
def test_classify_auth_by_status_code():
    err = classify_llm_error(
        ApiStatusError("nope", 401), provider="anthropic", key_env="ANTHROPIC_API_KEY"
    )
    assert isinstance(err, LLMAuthError)
    assert "ANTHROPIC_API_KEY" in (err.remediation or "")


def test_classify_quota_by_status_code():
    err = classify_llm_error(
        ApiStatusError("nope", 429), provider="gemini", key_env="GEMINI_API_KEY"
    )
    assert isinstance(err, LLMQuotaError)


def test_classify_auth_and_quota_by_phrase():
    auth = classify_llm_error(
        Exception("invalid x-api-key"), provider="anthropic", key_env="K"
    )
    quota = classify_llm_error(
        Exception("429 RESOURCE_EXHAUSTED quota"), provider="gemini", key_env="K"
    )
    assert isinstance(auth, LLMAuthError)
    assert isinstance(quota, LLMQuotaError)


def test_classify_unknown_returns_none():
    assert classify_llm_error(Exception("weird"), provider="x", key_env="Y") is None


# ----------------------------------------------------- in-call error handling
def test_call_json_auth_error_is_fatal_and_not_retried():
    sleeps: list = []
    fake = FakeAnthropicRaising(ApiStatusError("invalid x-api-key", 401))
    client = LLMClient(api_key="t", client=fake, sleep=sleeps.append)
    with pytest.raises(LLMAuthError):
        client.call_json(system="s", user="u", run_state_json="{}", default={})
    assert len(fake.calls) == 1  # auth failures are not retried
    assert sleeps == []


def test_call_json_quota_error_retries_then_raises_fatal():
    sleeps: list = []
    fake = FakeAnthropicRaising(ApiStatusError("429 RESOURCE_EXHAUSTED", 429))
    client = LLMClient(api_key="t", client=fake, sleep=sleeps.append)
    with pytest.raises(LLMQuotaError):
        client.call_json(system="s", user="u", run_state_json="{}", default={})
    assert len(fake.calls) == 3  # 3 attempts with backoff
    assert len(sleeps) == 2
