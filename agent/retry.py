"""Network retry policy: up to 3 attempts with exponential backoff (2s, 4s, 8s)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from loguru import logger

T = TypeVar("T")

BACKOFF_SECONDS = (2, 4, 8)


def with_retries(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    backoff: tuple[int, ...] = BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    label: str = "network call",
    retry_on: Callable[[Exception], bool] | None = None,
) -> T:
    """Run fn, retrying up to `attempts` times.

    `sleep` is injectable so tests run without real delays. `retry_on` decides
    whether a given exception is retryable (default: retry everything);
    non-retryable exceptions are re-raised immediately without backoff.
    """
    should_retry = retry_on or (lambda _exc: True)
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — retry any transient failure
            last_exc = exc
            if i == attempts - 1 or not should_retry(exc):
                break
            delay = backoff[min(i, len(backoff) - 1)]
            logger.warning("{} failed (attempt {}/{}): {} — retrying in {}s",
                           label, i + 1, attempts, exc, delay)
            sleep(delay)
    assert last_exc is not None
    raise last_exc
