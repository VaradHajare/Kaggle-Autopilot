"""Kaggle I/O: credentials, metadata, rules acceptance, download, submit, leaderboard.

Wraps the official `kaggle` package. The underlying api object is injectable so
tests run against mocks/fixtures with no live calls and no credentials.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

from agent.memory import CompetitionMeta
from agent.retry import with_retries

# Substrings Kaggle returns in the 403 body when rules are not yet accepted.
_RULES_NOT_ACCEPTED = (
    "must accept",
    "accept the competition rules",
    "accept this competition",
)


class KaggleApiLike(Protocol):
    """Subset of kaggle.KaggleApi used here — lets tests supply a fake."""

    def authenticate(self) -> None: ...
    def competitions_list(self, *args: Any, **kwargs: Any) -> list[Any]: ...
    def competition_list_files(self, competition: str) -> list[Any]: ...
    def competition_download_files(self, competition: str, path: str) -> None: ...
    def competition_submissions(self, competition: str) -> list[Any]: ...
    def competition_submit(self, file_name: str, message: str, competition: str) -> Any: ...


def _default_api() -> KaggleApiLike:
    """Construct and authenticate the real Kaggle API (lazy import so the
    package isn't required at module import time / in unit tests)."""
    from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415

    api = KaggleApi()
    api.authenticate()
    return api


# The kaggle >=2.0 client returns typed response objects (e.g.
# ApiListCompetitionsResponse) that wrap the payload in an attribute and use
# snake_case fields, whereas 1.x returned bare lists with camelCase fields. The
# two helpers below normalize across both so the rest of this module is
# version-agnostic.
_LIST_PAYLOAD_ATTRS = ("competitions", "files", "submissions", "kernels", "datasets")


def _as_list(resp: Any) -> list[Any]:
    """Unwrap a kaggle list response into a plain list (1.x and 2.x)."""
    if resp is None:
        return []
    if isinstance(resp, list):
        return resp
    for attr in _LIST_PAYLOAD_ATTRS:
        payload = getattr(resp, attr, None)
        if payload is not None:
            return list(payload)
    try:
        return list(resp)
    except TypeError:
        return [resp]


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """First non-None attribute among `names` (handles snake_case/camelCase)."""
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def parse_competition_slug(url_or_slug: str) -> str:
    """Extract the competition slug from a URL or accept a bare slug.

    'https://www.kaggle.com/competitions/titanic' -> 'titanic'
    'titanic' -> 'titanic'
    """
    text = url_or_slug.strip().rstrip("/")
    m = re.search(r"kaggle\.com/(?:c|competitions)/([^/?#]+)", text)
    if m:
        return m.group(1)
    if "/" in text or " " in text:
        raise ValueError(f"Could not parse a competition slug from {url_or_slug!r}")
    return text


class KaggleClient:
    """Thin, retry-wrapped wrapper over the Kaggle API."""

    def __init__(self, api: KaggleApiLike | None = None) -> None:
        self._api = api

    @property
    def api(self) -> KaggleApiLike:
        if self._api is None:
            self._api = _default_api()
        return self._api

    def check_credentials(self) -> bool:
        """Dry-run `competitions_list` to confirm auth works."""
        try:
            with_retries(lambda: self.api.competitions_list(), label="kaggle auth check")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Kaggle credential check failed: {}", exc)
            return False

    def _find_competition(self, slug: str) -> Any | None:
        results = with_retries(
            lambda: self.api.competitions_list(search=slug),
            label="kaggle competitions_list",
        )
        for comp in _as_list(results):
            ref = str(_attr(comp, "ref", default="")).rstrip("/").split("/")[-1]
            if ref == slug:
                return comp
        return None

    def get_competition_metadata(self, slug: str) -> CompetitionMeta:
        comp = self._find_competition(slug)
        if comp is None:
            raise ValueError(f"Competition {slug!r} not found via Kaggle API.")
        category = str(_attr(comp, "category", default="") or "unknown")
        return CompetitionMeta(
            slug=slug,
            eval_metric=str(
                _attr(comp, "evaluation_metric", "evaluationMetric", default="")
                or "unknown"
            ),
            problem_type=category,
            deadline=_attr(comp, "deadline"),
            team_size_limit=_attr(comp, "max_team_size", "maxTeamSize"),
            daily_submission_limit=int(
                _attr(comp, "max_daily_submissions", "maxDailySubmissions", default=5) or 5
            ),
            submissions_today=0,
            is_featured=category.lower() == "featured",
            prize_usd=_parse_reward(_attr(comp, "reward", default="")),
        )

    def check_rules_accepted(self, slug: str) -> bool:
        """True if the user has accepted the competition rules.

        Trusts a truthy `user_has_entered`/`userHasEntered` flag; otherwise
        (the flag is unreliable for Getting-Started competitions, where it can
        read False while the data is fully accessible) falls back to the
        authoritative file-listing probe and inspects any 403 error body.
        """
        comp = self._find_competition(slug)
        if comp is not None and _attr(comp, "user_has_entered", "userHasEntered"):
            return True
        try:
            with_retries(
                lambda: self.api.competition_list_files(slug),
                label="kaggle rules probe",
            )
            return True
        except Exception as exc:  # noqa: BLE001
            if any(s in str(exc).lower() for s in _RULES_NOT_ACCEPTED):
                return False
            raise

    def get_total_file_size(self, slug: str) -> int:
        """Total bytes of all competition files (for the disk-space check)."""
        files = with_retries(
            lambda: self.api.competition_list_files(slug),
            label="kaggle competition_list_files",
        )
        total = 0
        for f in _as_list(files):
            total += int(_attr(f, "total_bytes", "totalBytes", "size", default=0) or 0)
        return total

    def download(self, slug: str, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        with_retries(
            lambda: self.api.competition_download_files(slug, path=str(dest)),
            label="kaggle download",
        )

    def get_submissions(self, slug: str) -> list[Any]:
        return _as_list(
            with_retries(
                lambda: self.api.competition_submissions(slug),
                label="kaggle submissions",
            )
        )

    def submit(self, slug: str, file_path: Path, message: str) -> Any:
        return with_retries(
            lambda: self.api.competition_submit(str(file_path), message, slug),
            label="kaggle submit",
        )

    def latest_submission_score(self, slug: str) -> float | None:
        """Public LB score of the most recent submission, or None if pending."""
        subs = self.get_submissions(slug)
        if not subs:
            return None
        latest = subs[0]
        raw = _attr(latest, "public_score", "publicScore")
        if raw in (None, "", "pending"):
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None


def _parse_reward(reward: Any) -> float:
    """Best-effort parse of a reward string like '$25,000' -> 25000.0."""
    if reward is None:
        return 0.0
    digits = re.sub(r"[^0-9.]", "", str(reward))
    try:
        return float(digits) if digits else 0.0
    except ValueError:
        return 0.0
