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
) -> T:
    """Run fn, retrying on any exception up to `attempts` times.

    `sleep` is injectable so tests run without real delays.
    """
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — retry any transient failure
            last_exc = exc
            if i < attempts - 1:
                delay = backoff[min(i, len(backoff) - 1)]
                logger.warning("{} failed (attempt {}/{}): {} — retrying in {}s",
                               label, i + 1, attempts, exc, delay)
                sleep(delay)
    assert last_exc is not None
    raise last_exc
