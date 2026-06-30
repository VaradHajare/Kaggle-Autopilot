"""with_retries: backoff, attempt cap, and the retry_on predicate."""

from __future__ import annotations

import pytest

from agent.retry import with_retries


def test_retries_until_success():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    sleeps: list = []
    assert with_retries(flaky, sleep=sleeps.append) == "ok"
    assert calls["n"] == 3
    assert sleeps == [2, 4]  # backoff before attempts 2 and 3


def test_non_retryable_breaks_immediately():
    calls = {"n": 0}

    def always_fails():
        calls["n"] += 1
        raise ValueError("fatal")

    sleeps: list = []
    with pytest.raises(ValueError):
        with_retries(always_fails, sleep=sleeps.append, retry_on=lambda exc: False)
    assert calls["n"] == 1  # no retry
    assert sleeps == []
