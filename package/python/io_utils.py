"""Input/output helpers for locating and persisting datasets."""

from __future__ import annotations

from pathlib import Path

from .config import (
    DEFAULT_INPUT_NAME,
    ENV_INPUT_FILE,
    FALLBACK_INPUT_NAME,
    INPUT_DIR,
    VISUAL_DIR,
)


def ensure_directories() -> None:
    """Guarantee that output folders exist."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)


def _expand_candidate(path: Path) -> list[Path]:
    """Return concrete paths for a candidate (supports wildcards)."""
    name = path.name
    if any(symbol in name for symbol in '*?[]'):
        base_dir = path.parent if path.parent != Path('.') else INPUT_DIR
        return sorted(base_dir.glob(name))
    if path.exists():
        return [path]
    return []


def resolve_input_file() -> Path:
    """Return the Excel path considering overrides and fallbacks."""
    candidates: list[Path] = []

    if ENV_INPUT_FILE:
        override = Path(ENV_INPUT_FILE)
        if not override.is_absolute():
            override = INPUT_DIR / override
        candidates.append(override)

    candidates.extend(
        [
            INPUT_DIR / DEFAULT_INPUT_NAME,
            INPUT_DIR / FALLBACK_INPUT_NAME,
        ]
    )

    for candidate in candidates:
        matches = _expand_candidate(candidate)
        if matches:
            return matches[0]

    listed = ', '.join(p.name for p in INPUT_DIR.glob('*.xlsx'))
    raise FileNotFoundError(
        'No input spreadsheet found. '
        f'Available files: {listed or "none"}.'
    )
