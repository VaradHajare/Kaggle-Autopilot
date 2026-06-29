"""Gemini LLM client: JSON parsing, retry-once, fallback, token accounting,
and the provider factory."""

from __future__ import annotations

from agent.config import Settings
from agent.llm import GeminiLLM, LLMClient, build_llm
from tests.conftest import FakeGemini


def test_call_json_parses_and_counts_tokens():
    client = GeminiLLM(api_key="t", client=FakeGemini('{"a": 1}'))
    res = client.call_json(system="s", user="u", run_state_json="{}", default={})
    assert res.data == {"a": 1}
    assert res.fell_back is False
    assert res.tokens_in == 10 and res.tokens_out == 5


def test_call_json_strips_markdown_fences():
    fenced = "```json\n{\"a\": 2}\n```"
    client = GeminiLLM(api_key="t", client=FakeGemini(fenced))
    res = client.call_json(system="s", user="u", run_state_json="{}", default={})
    assert res.data == {"a": 2}
    assert res.fell_back is False


def test_call_json_falls_back_on_bad_json():
    client = GeminiLLM(api_key="t", client=FakeGemini("not json"))
    res = client.call_json(system="s", user="u", run_state_json="{}", default={"fb": True})
    assert res.fell_back is True
    assert res.data == {"fb": True}
    assert res.tokens_out == 10  # two attempts counted


def test_check_credentials_ok():
    assert GeminiLLM(api_key="t", client=FakeGemini("{}")).check_credentials() is True


def test_factory_selects_gemini_by_default():
    s = Settings(llm_provider="gemini", gemini_api_key="k")
    llm = build_llm(s)
    assert isinstance(llm, GeminiLLM)
    assert llm.model == s.gemini_model


def test_factory_selects_anthropic_when_configured():
    s = Settings(llm_provider="anthropic", anthropic_api_key="k")
    llm = build_llm(s)
    assert isinstance(llm, LLMClient)
    assert llm.model == s.agent_model
