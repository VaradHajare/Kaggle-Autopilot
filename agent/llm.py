"""LLM clients — provider-agnostic.

Every call: passes RunState context as JSON in the system prompt, enforces
JSON-only output, retries once on parse failure, then falls back to a
caller-supplied default. Token usage is accumulated and reported.

Two providers are implemented:
  - GeminiLLM   (Google AI Studio; default for now)
  - LLMClient   (Anthropic; flip llm_provider to "anthropic" to use)

Both share BaseLLM.call_json; only `_raw_call` / `check_credentials` differ.
The underlying SDK client is injectable so tests run without a key.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

JSON_SUFFIX = (
    "You must respond ONLY with valid JSON. No preamble, no markdown fences, "
    'no explanation. If you cannot produce valid JSON, respond with '
    '{"error": "<reason>"}.'
)


class LLMResult:
    """Parsed JSON plus token accounting for one call."""

    def __init__(self, data: Any, tokens_in: int, tokens_out: int, *, fell_back: bool):
        self.data = data
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.fell_back = fell_back


def _strip_fences(text: str) -> str:
    """Drop a leading ```json / ``` fence and trailing ``` if present."""
    t = text.strip()
    if t.startswith("```"):
        nl = t.find("\n")
        t = t[nl + 1 :] if nl != -1 else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


class BaseLLM:
    """Shared JSON-call orchestration. Subclasses implement `_raw_call`."""

    def __init__(self, *, model: str, max_tokens: int) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.total_tokens_in = 0
        self.total_tokens_out = 0

    def call_json(
        self,
        *,
        system: str,
        user: str,
        run_state_json: str,
        default: Any,
        max_tokens: int | None = None,
    ) -> LLMResult:
        """Make a JSON-enforced call. Retry once with a stricter nudge on parse
        failure, then fall back to `default` and log."""
        full_system = f"{system}\n\nRUN_STATE:\n{run_state_json}\n\n{JSON_SUFFIX}"
        attempts = [user, f"{user}\n\nReturn STRICTLY valid JSON only."]
        last_text = ""
        t_in = t_out = 0
        for i, prompt in enumerate(attempts):
            text, ti, to = self._raw_call(
                system=full_system, user=prompt, max_tokens=max_tokens or self.max_tokens
            )
            t_in += ti
            t_out += to
            last_text = text
            try:
                return LLMResult(json.loads(_strip_fences(text)), t_in, t_out, fell_back=False)
            except (json.JSONDecodeError, TypeError):
                if i == 0:
                    logger.warning("LLM returned non-JSON; retrying with stricter prompt.")
        logger.error("LLM JSON parse failed twice; using fallback default. Last: {}",
                     last_text[:200])
        return LLMResult(default, t_in, t_out, fell_back=True)

    def _raw_call(self, *, system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
        raise NotImplementedError

    def check_credentials(self) -> bool:
        raise NotImplementedError


# ----------------------------------------------------------------- Gemini
class GeminiLLM(BaseLLM):
    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = "gemini-2.0-flash",
        max_tokens: int = 8192,
        client: Any | None = None,
    ) -> None:
        super().__init__(model=model, max_tokens=max_tokens)
        self._client = client
        self._api_key = api_key

    @property
    def client(self) -> Any:
        if self._client is None:
            from google import genai  # noqa: PLC0415

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def check_credentials(self) -> bool:
        try:
            self._raw_call(system="Reply with a JSON object.", user="ok", max_tokens=16)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Gemini credential check failed: {}", exc)
            return False

    def _raw_call(self, *, system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
        config = {
            "system_instruction": system,
            "max_output_tokens": max_tokens,
            "response_mime_type": "application/json",
            "temperature": 0,
        }
        resp = self.client.models.generate_content(
            model=self.model, contents=user, config=config
        )
        text = getattr(resp, "text", "") or ""
        usage = getattr(resp, "usage_metadata", None)
        ti = int(getattr(usage, "prompt_token_count", 0) or 0)
        to = int(getattr(usage, "candidates_token_count", 0) or 0)
        self.total_tokens_in += ti
        self.total_tokens_out += to
        return text, ti, to


# ----------------------------------------------------------------- Anthropic
class LLMClient(BaseLLM):
    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 8192,
        client: Any | None = None,
    ) -> None:
        super().__init__(model=model, max_tokens=max_tokens)
        self._client = client
        self._api_key = api_key

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic  # noqa: PLC0415

            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def check_credentials(self) -> bool:
        try:
            self._raw_call(system="Reply with the single character: ok",
                           user="ok", max_tokens=8)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Anthropic credential check failed: {}", exc)
            return False

    def _raw_call(self, *, system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            getattr(block, "text", "") for block in getattr(resp, "content", [])
        )
        usage = getattr(resp, "usage", None)
        ti = int(getattr(usage, "input_tokens", 0) or 0)
        to = int(getattr(usage, "output_tokens", 0) or 0)
        self.total_tokens_in += ti
        self.total_tokens_out += to
        return text, ti, to


# ----------------------------------------------------------------- factory
def build_llm(settings: Any) -> BaseLLM:
    """Construct the LLM client for the configured provider."""
    if settings.llm_provider == "gemini":
        return GeminiLLM(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            max_tokens=settings.agent_max_tokens,
        )
    return LLMClient(
        api_key=settings.anthropic_api_key,
        model=settings.agent_model,
        max_tokens=settings.agent_max_tokens,
    )
