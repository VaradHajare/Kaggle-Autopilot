"""LLM client: JSON parsing, retry-once, fallback, token accounting."""

from __future__ import annotations

from agent.llm import LLMClient
from tests.conftest import FakeAnthropic


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
