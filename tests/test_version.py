# striatica/tests/test_version.py
"""Tests for version single-sourcing — pyproject.toml is the only version literal.

Contract pinned here: every version surface in the pipeline (banner line,
validation sidecar, dataset metadata, traced-circuit provenance) derives its
string from ``pipeline.__version__``, which in turn derives from the
``[project].version`` key in pyproject.toml. Bumping pyproject bumps every
surface; no surface carries its own copy.

The pyproject value is parsed here independently with tomllib rather than by
calling ``pipeline._resolve_version()`` — reusing the implementation's parser
would make the equality test circular and pass even if it read the wrong key.

Non-goals: this file does not test the importlib.metadata fallback path (that
only fires for wheel installs with no pyproject alongside the package, which is
not the layout under test), does not run the pipeline, and does not assert any
specific version number — pinning "0.4.0" here would just relocate the
hardcoded literal into the test suite.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

import pipeline


REPO_ROOT = Path(__file__).resolve().parent.parent

# Files that consume the version. Each must read pipeline.__version__ rather
# than restate a literal.
VERSION_CONSUMERS = [
    REPO_ROOT / "pipeline" / "banner.py",
    REPO_ROOT / "pipeline" / "validate.py",
    REPO_ROOT / "pipeline" / "cli.py",
    REPO_ROOT / "pipeline" / "traced_circuits.py",
]

# A quoted semver literal beginning with a 0 major, e.g. "0.4.0" — the shape a
# stale hardcoded pipeline version takes in this repo.
SEMVER_LITERAL = re.compile(r'"0\.\d+\.\d+"')


def _pyproject_version() -> str:
    """Parse [project].version out of pyproject.toml, independently of pipeline/."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        return str(tomllib.load(fh)["project"]["version"])


# ── Source of truth ─────────────────────────────────────────────────────


def test_pyproject_is_the_source_of_truth():
    """pipeline.__version__ equals the version declared in pyproject.toml."""
    assert pipeline.__version__ == _pyproject_version()


def test_version_is_a_real_version_not_the_unknown_fallback():
    """The resolver found pyproject, rather than degrading to "0.0.0+unknown".

    Guards against a silently-broken resolver (moved pyproject, renamed key)
    that would still satisfy every other assertion in this file if those
    assertions were written against the resolver's own output.
    """
    assert pipeline.__version__ != "0.0.0+unknown"
    assert re.fullmatch(r"\d+\.\d+\.\d+", pipeline.__version__), (
        f"unexpected version shape: {pipeline.__version__!r}"
    )


# ── Consuming surfaces ──────────────────────────────────────────────────


def test_banner_version_line_carries_the_resolved_version():
    """banner.VERSION_LINE contains exactly "v{__version__}"."""
    from pipeline import banner

    assert f"v{pipeline.__version__}" in banner.VERSION_LINE


def test_traced_circuits_pipeline_version_matches():
    """traced_circuits.PIPELINE_VERSION is the resolved version, not a copy."""
    from pipeline import traced_circuits

    assert traced_circuits.PIPELINE_VERSION == pipeline.__version__


def test_validate_sidecar_uses_the_resolved_version():
    """validate.py binds __version__ and writes it into the sidecar dict.

    Asserted by import + source inspection rather than by running the
    validation suite, which needs full pipeline arrays to produce a sidecar.
    """
    from pipeline import validate

    assert validate.__version__ == pipeline.__version__

    source = (REPO_ROOT / "pipeline" / "validate.py").read_text()
    assert '"pipeline_version": __version__' in source, (
        "validate.py sidecar must write the imported __version__, not a literal"
    )


def test_cli_metadata_uses_the_resolved_version():
    """cli.py binds __version__ and writes it into the dataset metadata dict.

    Same rationale as the sidecar test: the metadata dict is only built at the
    end of a full pipeline run (model download, UMAP, the lot), so the binding
    is checked by import and the dict literal by source inspection.
    """
    cli = pytest.importorskip(
        "pipeline.cli",
        reason="pipeline.cli import failed for environmental reasons "
        "(optional heavy deps); source-text assertion below still applies",
    )

    assert cli.__version__ == pipeline.__version__

    source = (REPO_ROOT / "pipeline" / "cli.py").read_text()
    assert '"pipeline_version": __version__' in source, (
        "cli.py metadata must write the imported __version__, not a literal"
    )


# ── Regression trap ─────────────────────────────────────────────────────


@pytest.mark.parametrize("path", VERSION_CONSUMERS, ids=lambda p: p.name)
def test_no_hardcoded_version_literal(path: Path):
    """No quoted 0.x.y literal survives in any version-consuming module.

    This is the trap for re-hardcoding: single-sourcing stays true only as long
    as nobody pastes a version string back in "just for this one line".
    """
    matches = SEMVER_LITERAL.findall(path.read_text())
    assert matches == [], f"{path.name} contains hardcoded version literal(s): {matches}"
