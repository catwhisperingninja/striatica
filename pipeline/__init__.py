"""striatica data pipeline."""

from __future__ import annotations

__all__ = ["__version__"]


def _resolve_version() -> str:
    """Single source of truth for the pipeline version: ``pyproject.toml``.

    Every version surface (banner, dataset metadata, validation sidecar, traced-
    circuit provenance) reads this. Do not hardcode a version anywhere else.

    ``pyproject.toml`` wins when it is present — a source checkout or editable
    install can have a bumped pyproject and stale installed metadata, and the
    file is the thing a human edits. Falls back to installed package metadata
    (wheel installs, where no pyproject ships alongside the package).
    """
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        with open(pyproject, "rb") as fh:
            return str(tomllib.load(fh)["project"]["version"])
    except (OSError, KeyError, ValueError, tomllib.TOMLDecodeError):
        pass

    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("striatica")
    except PackageNotFoundError:
        return "0.0.0+unknown"


__version__ = _resolve_version()
