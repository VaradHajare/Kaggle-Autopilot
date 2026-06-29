"""Phase 1 ingestion helpers: unzip archives and detect the canonical
train / test / sample_submission files.

File-format detection priority: .csv > .parquet > .feather > .json >
image/audio directories.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from loguru import logger

# Lower index = higher priority.
FORMAT_PRIORITY = (".csv", ".parquet", ".feather", ".json")


class IngestError(Exception):
    """Raised when required competition files cannot be located."""


def unzip_all(raw_dir: Path) -> list[Path]:
    """Extract every .zip in raw_dir (recursively) into raw_dir. Returns the
    list of archives extracted."""
    extracted: list[Path] = []
    for archive in sorted(raw_dir.rglob("*.zip")):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw_dir)
        extracted.append(archive)
        logger.debug("Extracted {}", archive.name)
    return extracted


def _format_rank(path: Path) -> int:
    suffix = path.suffix.lower()
    return FORMAT_PRIORITY.index(suffix) if suffix in FORMAT_PRIORITY else len(FORMAT_PRIORITY)


def _best_match(raw_dir: Path, stem_keywords: tuple[str, ...]) -> Path | None:
    """Find the highest-priority data file whose stem matches any keyword."""
    candidates: list[Path] = []
    for p in raw_dir.rglob("*"):
        if not p.is_file():
            continue
        stem = p.stem.lower()
        if any(kw in stem for kw in stem_keywords):
            candidates.append(p)
    if not candidates:
        return None
    # Prefer exact-name matches, then format priority, then shortest name.
    candidates.sort(key=lambda p: (_format_rank(p), len(p.stem), str(p)))
    return candidates[0]


def detect_files(raw_dir: Path) -> dict[str, Path | None]:
    """Locate train, test, and sample_submission files.

    Returns a dict with keys 'train', 'test', 'sample_submission'. Raises
    IngestError if train or sample_submission cannot be found (test may be
    absent for some competitions, but is required by most).
    """
    sample = _best_match(raw_dir, ("sample_submission", "samplesubmission", "submission"))
    # Avoid mistaking sample_submission for test; exclude it explicitly.
    test = _best_match(raw_dir, ("test",))
    train = _best_match(raw_dir, ("train",))

    if train is None:
        raise IngestError(f"No train file found under {raw_dir}.")
    if sample is None:
        raise IngestError(f"No sample_submission file found under {raw_dir}.")
    return {"train": train, "test": test, "sample_submission": sample}
