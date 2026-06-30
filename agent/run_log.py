"""Structured append-only logging to runs/<slug>/run_log.md.

Repository rule: never remove logging from any phase. Every phase writes a
RunLog entry. This module owns the run_log.md format defined in the spec (Section 13).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from loguru import logger


class RunLog:
    """Append-only Markdown audit trail for a single run."""

    def __init__(self, run_dir: Path) -> None:
        self.path = Path(run_dir) / "run_log.md"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    def write_header(self, slug: str, env_fingerprint: dict[str, str]) -> None:
        lines = [
            f"# Run Log — {slug}",
            "",
            f"**Started:** {self._now()}",
            "",
            "**Environment fingerprint:**",
            "",
        ]
        lines += [f"- `{k}`: {v}" for k, v in env_fingerprint.items()]
        lines.append("")
        self._append("\n".join(lines))

    def phase(
        self,
        number: str,
        name: str,
        *,
        status: str,
        duration_s: float | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        summary: list[str] | None = None,
        decisions: list[str] | None = None,
        metrics: dict[str, object] | None = None,
        errors: list[str] | None = None,
    ) -> None:
        """Write a structured phase entry (spec Section 13 format)."""
        out = [f"\n## [PHASE {number}] {name} — {self._now()}\n"]
        out.append(f"**Status:** {status}")
        if duration_s is not None:
            out.append(f"**Duration:** {duration_s:.1f}s")
        if tokens_in or tokens_out:
            out.append(f"**Tokens used (this phase):** {tokens_in:,} in / {tokens_out:,} out")
        if summary:
            out.append("\n### Summary")
            out += [f"- {s}" for s in summary]
        if decisions:
            out.append("\n### Decisions made")
            out += [f"- {d}" for d in decisions]
        if metrics:
            out.append("\n### Metrics")
            out += [f"- {k}: {v}" for k, v in metrics.items()]
        if errors:
            out.append("\n### Errors")
            out += [f"- {e}" for e in errors]
        self._append("\n".join(out) + "\n")
        logger.log(
            "ERROR" if status in ("ERROR", "FATAL") else "INFO",
            "[PHASE {}] {} — {}",
            number,
            name,
            status,
        )

    def section(self, title: str, body: str) -> None:
        """Free-form section (e.g. ## LEAKAGE FLAGS)."""
        self._append(f"\n## {title}\n\n{body}\n")

    def _append(self, text: str) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(text)
