# striatica/tests/test_cli_traced.py
"""Tests for the traced-circuit CLI (unit: cli_dispatch).

Covers two files owned by the cli_dispatch implementer:

1. scripts/generate_traced_circuits.py — thin argparse orchestrator.
2. pipeline/cli.py — `--traced` routing inside cmd_circuits (legacy path
   byte-untouched) and the new `semantics-merge` subcommand.

All graph data comes from the REAL Neuronpedia fixtures in tests/fixtures/
(gemma-fact-dallas-austin, fetched live 2026-07-28) and the REAL dataset
metadata frontend/public/data/gemma-2-2b-layer12-l0604-metadata.json (copied
into tmp dirs — the real file is never modified). No synthetic mock JSON; no
network (urlopen is patched to fail loudly in orchestrator tests).

API CONTRACT (implementers: build scripts/generate_traced_circuits.py to this):

    build_parser() -> argparse.ArgumentParser
        Dests / defaults:
          slug: repeatable ``--slug`` (append -> list)
          slugs_file, name, description, prompt: default None
          list / generate / force / dry_run / allow_l0_mismatch / redact_roles:
              store_true flags, default False
          model: default "gemma-2-2b"
          layer: int, default 12
          dataset_stem: default "gemma-2-2b-layer12-l06"
          out_dir: default "frontend/public/data" (str or Path)

    main(argv: list[str] | None = None) -> int
        Returns 0 (or None) on success. ALL failure paths exit cleanly:
        SystemExit with a nonzero code, or a nonzero int return — never an
        uncaught traceback. Requires at least one graph source among
        --slug / --slugs-file / --list / --generate ("slug" must appear in
        the error text). --generate requires --prompt and consumes --slug as
        the new graph's slug (so --generate and --slug may appear together).

    Module-level bindings (the monkeypatch seams these tests rely on — bind
    the names INTO the script's namespace):
        from pipeline.graph_fetch import (
            fetch_graph, fetch_graph_record, fetch_source_set,
            load_neuronpedia_api_key,
        )
        from pipeline.traced_circuits import (
            build_traced_circuit, write_traced_outputs,
        )
        from pipeline.banner import ...  # helpers for ALL status output

    Behavior pinned here:
      - Dataset metadata is read from {out_dir}/{dataset_stem}-metadata.json;
        a missing file is an actionable error ("metadata" in the message)
        and nothing is written.
      - Per slug: fetch_graph_record(model, slug), fetch_graph(model, slug,
        ... force=<--force>), fetch_source_set(model, record["sourceSetName"]),
        then build_traced_circuit(raw_graph, record, source_set,
        dataset_metadata, name=..., description=..., layer=...,
        allow_l0_mismatch=..., redact_roles=...).
      - ALL built circuits go to ONE write_traced_outputs(circuits, out_root,
        dataset_stem) call (the writer owns the manifest; per-circuit calls
        would clobber it).
      - --dry-run: fetch + parse + validate + banner summary (slug and the
        word "dry" appear on stderr); write_traced_outputs is NOT called and
        NOTHING is written anywhere.
      - --list / --generate need an API key via the module's
        load_neuronpedia_api_key binding; with no key, fail cleanly with
        "NEURONPEDIA_API_KEY" in the stderr text.

pipeline/cli.py contract pinned here:
      - cmd_circuits: if "--traced" is present in args.circuit_args, route to
        generate_traced_circuits.main(<circuit_args with the "--traced" token
        removed, order and all other tokens preserved>) — the legacy
        generate_circuits path must NOT run. Otherwise the legacy behavior is
        byte-untouched: sys.argv = ["striat circuits"] + circuit_args, then
        generate_circuits.main().
      - semantics-merge subcommand: forwards the remaining argv verbatim to
        pipeline.semantics_merge.main(argv) (U5's module; signature
        main(argv: list[str] | None = None) -> int). A nonzero return becomes
        a nonzero process exit. If the module is absent (U5 not built yet),
        exit nonzero with a clear error mentioning "semantics" — never an
        uncaught ImportError.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import shutil
import sys
import types
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from pipeline import cli

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
FIXTURES = PROJECT_ROOT / "tests" / "fixtures"
REAL_DATA_DIR = PROJECT_ROOT / "frontend" / "public" / "data"
DATASET_METADATA_PATH = REAL_DATA_DIR / "gemma-2-2b-layer12-l0604-metadata.json"

RAW_GRAPH = json.loads((FIXTURES / "neuronpedia_graph_gemma.json").read_text())
RECORD = json.loads((FIXTURES / "neuronpedia_graph_record_gemma.json").read_text())
SOURCE_SET = json.loads((FIXTURES / "neuronpedia_sourceset_gemma.json").read_text())
DATASET_METADATA = json.loads(DATASET_METADATA_PATH.read_text())

# ---------------------------------------------------------------------------
# Expected values computed from the REAL fixture contents (verified 2026-07-28
# by running pipeline.traced_circuits.build_traced_circuit on the fixtures):
# the gemma-fact-dallas-austin graph has 53 nodes; exactly 6 valid layer-12
# cross-layer-transcoder feature nodes survive the layer filter.
# ---------------------------------------------------------------------------
SLUG = "gemma-fact-dallas-austin"
MODEL = "gemma-2-2b"
SOURCE_SET_NAME = "gemmascope-transcoder-16k"  # == RECORD["sourceSetName"]
L12_LOCALS = [2082, 2799, 8580, 10631, 12601, 12910]
EXPECTED_NODE_COUNT = 6
EXPECTED_L0 = 6  # sourceset layer_12/width_16k/average_l0_6
DEFAULT_STEM = "gemma-2-2b-layer12-l06"
L0604_STEM = "gemma-2-2b-layer12-l0604"

assert RECORD["slug"] == SLUG, "record fixture changed — update expected values"
assert RECORD["modelId"] == MODEL
assert RECORD["sourceSetName"] == SOURCE_SET_NAME


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_traced_module():
    """Import scripts/generate_traced_circuits.py (fail, not skip, if absent)."""
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    existing = sys.modules.get("generate_traced_circuits")
    if existing is not None and getattr(existing, "__file__", None):
        return existing
    sys.modules.pop("generate_traced_circuits", None)
    try:
        return importlib.import_module("generate_traced_circuits")
    except ModuleNotFoundError:
        pytest.fail(
            "scripts/generate_traced_circuits.py does not exist yet "
            "(cli_dispatch implementation missing)"
        )


def _run_expect_success(mod, argv):
    try:
        rc = mod.main(argv)
    except SystemExit as e:
        assert e.code in (0, None), f"expected success, got SystemExit({e.code!r})"
        return
    assert rc in (0, None), f"expected success (0/None) from main(), got {rc!r}"


def _run_expect_failure(mod, argv):
    try:
        rc = mod.main(argv)
    except SystemExit as e:
        assert e.code not in (0, None), (
            f"expected a failure exit, got SystemExit({e.code!r})"
        )
        return
    assert isinstance(rc, int) and rc != 0, (
        f"expected nonzero return from main(), got {rc!r}"
    )


def _no_network(*args, **kwargs):
    raise AssertionError(
        "network access attempted during a hermetic CLI test "
        "(urllib.request.urlopen was called)"
    )


def _tree_snapshot(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def _norm_write_args(args, kwargs):
    """Normalize write_traced_outputs(circuits, out_root, dataset_stem) calls."""
    circuits = args[0] if len(args) > 0 else kwargs["circuits"]
    out_root = args[1] if len(args) > 1 else kwargs["out_root"]
    stem = args[2] if len(args) > 2 else kwargs["dataset_stem"]
    return circuits, Path(out_root), stem


def _norm_build_args(args, kwargs):
    """Normalize build_traced_circuit calls into a single {name: value} dict.

    Positional order per pipeline.traced_circuits.build_traced_circuit:
    (raw_graph, record, source_set, dataset_metadata, *, keyword-only flags).
    """
    positional = ["raw_graph", "record", "source_set", "dataset_metadata"]
    merged = dict(kwargs)
    merged.update(dict(zip(positional, args)))
    return merged


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def traced_mod():
    return _import_traced_module()


@pytest.fixture(scope="module")
def real_circuit():
    """A real traced circuit built from the real fixtures.

    The identity validator is temporarily neutralized (its own behavior is
    pinned by the graph_fetch and traced_builder units); everything else is
    the real builder on the real fixture data.
    """
    import pipeline.traced_circuits as tc

    original = tc.validate_graph_identity
    tc.validate_graph_identity = lambda *a, **k: None
    try:
        return tc.build_traced_circuit(
            copy.deepcopy(RAW_GRAPH),
            copy.deepcopy(RECORD),
            copy.deepcopy(SOURCE_SET),
            copy.deepcopy(DATASET_METADATA),
        )
    finally:
        tc.validate_graph_identity = original


@pytest.fixture()
def traced_env(traced_mod, monkeypatch, tmp_path):
    """Hermetic orchestrator environment.

    - fetch_* bindings on the script module return deep copies of the real
      fixtures and record their calls (no network; urlopen fails loudly).
    - pipeline.traced_circuits.validate_graph_identity is a no-op (per the
      traced_builder contract, its return value is never relied upon).
    - tmp out-dir seeded with the REAL l0604 dataset metadata under the
      l0604 stem.
    """
    out_dir = tmp_path / "data"
    out_dir.mkdir()

    def seed(stem: str) -> Path:
        target = out_dir / f"{stem}-metadata.json"
        shutil.copyfile(DATASET_METADATA_PATH, target)
        return target

    seed(L0604_STEM)

    calls = {"record": [], "graph": [], "source_set": []}

    def fake_fetch_graph_record(model_id, slug, **kwargs):
        calls["record"].append((model_id, slug))
        return copy.deepcopy(RECORD)

    def fake_fetch_graph(model_id, slug, **kwargs):
        calls["graph"].append((model_id, slug, dict(kwargs)))
        return copy.deepcopy(RAW_GRAPH)

    def fake_fetch_source_set(model_id, name, **kwargs):
        calls["source_set"].append((model_id, name))
        return copy.deepcopy(SOURCE_SET)

    monkeypatch.setattr(traced_mod, "fetch_graph_record", fake_fetch_graph_record)
    monkeypatch.setattr(traced_mod, "fetch_graph", fake_fetch_graph)
    monkeypatch.setattr(traced_mod, "fetch_source_set", fake_fetch_source_set)
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)

    import pipeline.traced_circuits as tc

    monkeypatch.setattr(tc, "validate_graph_identity", lambda *a, **k: None)

    return SimpleNamespace(
        mod=traced_mod,
        out_dir=out_dir,
        stem=L0604_STEM,
        calls=calls,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# pipeline/cli.py — cmd_circuits dispatch
# ---------------------------------------------------------------------------


class TestCmdCircuitsDispatch:
    """'--traced' routes to generate_traced_circuits; legacy path untouched."""

    def _run_dispatch(self, circuit_args, monkeypatch, traced_main=None,
                      legacy_main=None):
        recorded_traced = []
        recorded_legacy = {}

        fake_traced = types.ModuleType("generate_traced_circuits")
        if traced_main is None:
            def traced_main(*a, **k):  # noqa: F811 - default recorder
                argv = a[0] if a else k.get("argv")
                recorded_traced.append(argv)
                return 0
        fake_traced.main = traced_main

        fake_legacy = types.ModuleType("generate_circuits")
        if legacy_main is None:
            def legacy_main():  # noqa: F811 - default recorder
                recorded_legacy["argv"] = list(sys.argv)
        fake_legacy.main = legacy_main

        monkeypatch.setattr(sys, "argv", list(sys.argv))
        with mock.patch.dict(
            sys.modules,
            {
                "generate_traced_circuits": fake_traced,
                "generate_circuits": fake_legacy,
            },
        ):
            try:
                cli.cmd_circuits(argparse.Namespace(circuit_args=circuit_args))
            except SystemExit as e:
                assert e.code in (0, None), (
                    f"dispatch exited nonzero: SystemExit({e.code!r})"
                )
        return recorded_traced, recorded_legacy

    def test_traced_flag_routes_to_traced_main(self, monkeypatch):
        def legacy_must_not_run():
            raise AssertionError(
                "legacy generate_circuits.main ran in --traced mode"
            )

        traced_calls, _ = self._run_dispatch(
            ["--traced", "--slug", SLUG],
            monkeypatch,
            legacy_main=legacy_must_not_run,
        )
        assert traced_calls == [["--slug", SLUG]]

    def test_traced_flag_removed_from_middle_position(self, monkeypatch):
        """Only the '--traced' token is removed; order and every other token
        survive verbatim."""
        traced_calls, _ = self._run_dispatch(
            ["--slug", SLUG, "--traced", "--dry-run"],
            monkeypatch,
        )
        assert traced_calls == [["--slug", SLUG, "--dry-run"]]

    def test_legacy_args_pass_through_byte_untouched(self, monkeypatch):
        """Non-traced invocations keep the exact legacy mechanism:
        sys.argv = ["striat circuits"] + circuit_args, then
        generate_circuits.main()."""

        def traced_must_not_run(*a, **k):
            raise AssertionError(
                "generate_traced_circuits.main ran without --traced"
            )

        _, legacy = self._run_dispatch(
            ["--batch-defaults"],
            monkeypatch,
            traced_main=traced_must_not_run,
        )
        assert legacy["argv"] == ["striat circuits", "--batch-defaults"]

    def test_legacy_empty_args_still_go_legacy(self, monkeypatch):
        def traced_must_not_run(*a, **k):
            raise AssertionError(
                "generate_traced_circuits.main ran without --traced"
            )

        _, legacy = self._run_dispatch(
            [], monkeypatch, traced_main=traced_must_not_run
        )
        assert legacy["argv"] == ["striat circuits"]


# ---------------------------------------------------------------------------
# pipeline/cli.py — semantics-merge subcommand
# ---------------------------------------------------------------------------


class TestSemanticsMergeSubcommand:
    def test_routes_argv_to_semantics_merge_main(self, monkeypatch):
        recorded = []
        fake = types.ModuleType("pipeline.semantics_merge")

        def fake_main(*a, **k):
            argv = a[0] if a else k.get("argv")
            recorded.append(argv)
            return 0

        fake.main = fake_main
        tail = ["dataset.json", "explanations.jsonl", "--out", "merged.json"]
        monkeypatch.setattr(sys, "argv", ["striat", "semantics-merge", *tail])
        with mock.patch.dict(sys.modules, {"pipeline.semantics_merge": fake}):
            try:
                cli.main()
            except SystemExit as e:
                assert e.code in (0, None), (
                    f"semantics-merge success exited nonzero: {e.code!r}"
                )
        assert recorded == [tail]

    def test_nonzero_return_becomes_nonzero_exit(self, monkeypatch):
        fake = types.ModuleType("pipeline.semantics_merge")
        fake.main = lambda *a, **k: 3
        monkeypatch.setattr(
            sys, "argv", ["striat", "semantics-merge", "d.json", "e.jsonl"]
        )
        with mock.patch.dict(sys.modules, {"pipeline.semantics_merge": fake}):
            with pytest.raises(SystemExit) as exc:
                cli.main()
        assert exc.value.code not in (0, None)

    def test_missing_module_gives_actionable_error(self, monkeypatch, capsys):
        """U5 may not be built yet: sys.modules[...] = None forces ImportError.
        The CLI must exit nonzero with a clear message — no raw traceback."""
        monkeypatch.setattr(
            sys, "argv", ["striat", "semantics-merge", "d.json", "e.jsonl"]
        )
        with mock.patch.dict(sys.modules, {"pipeline.semantics_merge": None}):
            with pytest.raises(SystemExit) as exc:
                cli.main()
        assert exc.value.code not in (0, None)
        err = capsys.readouterr().err
        assert "semantics" in err.lower()


# ---------------------------------------------------------------------------
# scripts/generate_traced_circuits.py — argparse surface
# ---------------------------------------------------------------------------


class TestTracedArgparse:
    def test_build_parser_defaults(self, traced_mod):
        parser = traced_mod.build_parser()
        assert isinstance(parser, argparse.ArgumentParser)
        args = parser.parse_args(["--slug", SLUG])
        assert args.slug == [SLUG]
        assert args.slugs_file is None
        assert getattr(args, "list") is False
        assert args.generate is False
        assert args.prompt is None
        assert args.model == "gemma-2-2b"
        assert args.layer == 12
        assert args.dataset_stem == DEFAULT_STEM
        assert Path(args.out_dir) == Path("frontend/public/data")
        assert args.name is None
        assert args.description is None
        assert args.force is False
        assert args.dry_run is False
        assert args.allow_l0_mismatch is False
        assert args.redact_roles is False

    def test_slug_is_repeatable(self, traced_mod):
        args = traced_mod.build_parser().parse_args(
            ["--slug", "slug-a", "--slug", "slug-b"]
        )
        assert args.slug == ["slug-a", "slug-b"]

    def test_layer_parses_as_int(self, traced_mod):
        args = traced_mod.build_parser().parse_args(
            ["--slug", SLUG, "--layer", "10"]
        )
        assert args.layer == 10

    def test_requires_a_graph_source(self, traced_mod, capsys):
        """No --slug / --slugs-file / --list / --generate -> clean failure."""
        _run_expect_failure(traced_mod, [])
        err = capsys.readouterr().err
        assert "slug" in err.lower()

    def test_generate_requires_prompt(self, traced_mod, monkeypatch, capsys):
        monkeypatch.setattr(
            traced_mod, "load_neuronpedia_api_key", lambda: "test-key-not-real"
        )
        monkeypatch.setattr(urllib.request, "urlopen", _no_network)
        _run_expect_failure(traced_mod, ["--generate", "--slug", "test-gen"])
        err = capsys.readouterr().err
        assert "prompt" in err.lower()


# ---------------------------------------------------------------------------
# scripts/generate_traced_circuits.py — API-key gating (--list / --generate)
# ---------------------------------------------------------------------------


class TestApiKeyGating:
    def test_list_without_key_is_actionable_error(
        self, traced_mod, monkeypatch, capsys
    ):
        monkeypatch.setattr(traced_mod, "load_neuronpedia_api_key", lambda: None)
        monkeypatch.delenv("NEURONPEDIA_API_KEY", raising=False)
        monkeypatch.setattr(urllib.request, "urlopen", _no_network)
        _run_expect_failure(traced_mod, ["--list"])
        err = capsys.readouterr().err
        assert "NEURONPEDIA_API_KEY" in err

    def test_generate_without_key_is_actionable_error(
        self, traced_env, monkeypatch, capsys
    ):
        env = traced_env
        monkeypatch.setattr(env.mod, "load_neuronpedia_api_key", lambda: None)
        monkeypatch.delenv("NEURONPEDIA_API_KEY", raising=False)
        _run_expect_failure(
            env.mod,
            [
                "--generate",
                "--prompt", "The capital of France is",
                "--slug", "test-gen-slug",
                "--dataset-stem", env.stem,
                "--out-dir", str(env.out_dir),
            ],
        )
        err = capsys.readouterr().err
        assert "NEURONPEDIA_API_KEY" in err


# ---------------------------------------------------------------------------
# scripts/generate_traced_circuits.py — dry run
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_writes_nothing_and_prints_summary(
        self, traced_env, monkeypatch, tmp_path, capsys
    ):
        env = traced_env
        monkeypatch.chdir(tmp_path)  # any relative-path write lands in tmp

        real_stem_dir = REAL_DATA_DIR / "circuits" / env.stem
        stem_dir_existed = real_stem_dir.exists()
        meta_path = env.out_dir / f"{env.stem}-metadata.json"
        meta_bytes = meta_path.read_bytes()
        before = _tree_snapshot(tmp_path)

        _run_expect_success(
            env.mod,
            [
                "--slug", SLUG,
                "--dataset-stem", env.stem,
                "--out-dir", str(env.out_dir),
                "--dry-run",
            ],
        )

        # fetch + parse actually happened
        assert env.calls["record"] == [(MODEL, SLUG)]
        assert [(m, s) for (m, s, _kw) in env.calls["graph"]] == [(MODEL, SLUG)]

        # ... but NOTHING was written
        assert _tree_snapshot(tmp_path) == before
        assert meta_path.read_bytes() == meta_bytes
        assert not (env.out_dir / "circuits").exists()
        if not stem_dir_existed:
            assert not real_stem_dir.exists()

        # summary via banner helpers (stderr): names the slug, announces dry run
        err = capsys.readouterr().err
        assert SLUG in err
        assert "dry" in err.lower()

    def test_dry_run_never_calls_writer(self, traced_env, monkeypatch):
        env = traced_env
        writes = []
        monkeypatch.setattr(
            env.mod,
            "write_traced_outputs",
            lambda *a, **k: writes.append(_norm_write_args(a, k)),
        )
        _run_expect_success(
            env.mod,
            [
                "--slug", SLUG,
                "--dataset-stem", env.stem,
                "--out-dir", str(env.out_dir),
                "--dry-run",
            ],
        )
        assert writes == []


# ---------------------------------------------------------------------------
# scripts/generate_traced_circuits.py — wet run orchestration
# ---------------------------------------------------------------------------


class TestWetRun:
    def test_wet_run_builds_real_circuit_and_writes_once(
        self, traced_env, monkeypatch
    ):
        """End-to-end through the REAL builder on the REAL fixtures; only the
        writer is a recorder. Expected values are fixture-derived."""
        env = traced_env
        writes = []

        def fake_write(*a, **k):
            circuits, out_root, stem = _norm_write_args(a, k)
            writes.append((circuits, out_root, stem))
            return out_root / "circuits" / stem

        monkeypatch.setattr(env.mod, "write_traced_outputs", fake_write)
        _run_expect_success(
            env.mod,
            ["--slug", SLUG, "--dataset-stem", env.stem,
             "--out-dir", str(env.out_dir)],
        )

        assert env.calls["record"] == [(MODEL, SLUG)]
        assert env.calls["source_set"] == [(MODEL, SOURCE_SET_NAME)]

        assert len(writes) == 1
        circuits, out_root, stem = writes[0]
        assert out_root == env.out_dir
        assert stem == env.stem
        assert len(circuits) == 1

        circuit = circuits[0]
        assert circuit["name"] == SLUG
        assert circuit["type"] == "traced"
        assert circuit["source"] == "neuronpedia"
        assert circuit["edges"] == []
        assert len(circuit["nodes"]) == EXPECTED_NODE_COUNT
        assert sorted(n["featureIndex"] for n in circuit["nodes"]) == L12_LOCALS
        assert circuit["metadata"]["model"] == MODEL
        assert circuit["metadata"]["l0"] == EXPECTED_L0
        # l0604 dataset vs Neuronpedia average_l0_6 source: never verified
        assert circuit["metadata"]["l0Verified"] is False

    def test_flags_forward_to_builder(
        self, traced_env, monkeypatch, real_circuit
    ):
        """--name/--description/--redact-roles/--allow-l0-mismatch reach
        build_traced_circuit; dataset metadata comes from
        {out_dir}/{dataset_stem}-metadata.json (stem present ONLY in tmp)."""
        env = traced_env
        stem = "l0604-tmp-copy"
        meta_path = env.seed(stem)

        build_calls = []

        def fake_build(*a, **k):
            build_calls.append(_norm_build_args(a, k))
            return copy.deepcopy(real_circuit)

        monkeypatch.setattr(env.mod, "build_traced_circuit", fake_build)
        monkeypatch.setattr(
            env.mod,
            "write_traced_outputs",
            lambda *a, **k: env.out_dir / "circuits" / stem,
        )
        _run_expect_success(
            env.mod,
            [
                "--slug", SLUG,
                "--dataset-stem", stem,
                "--out-dir", str(env.out_dir),
                "--name", "my-circuit",
                "--description", "a test description",
                "--redact-roles",
                "--allow-l0-mismatch",
            ],
        )

        assert len(build_calls) == 1
        call = build_calls[0]
        assert call["raw_graph"] == RAW_GRAPH
        assert call["record"] == RECORD
        assert call["source_set"] == SOURCE_SET
        assert call["dataset_metadata"] == json.loads(meta_path.read_text())
        assert call["name"] == "my-circuit"
        assert call["description"] == "a test description"
        assert call["redact_roles"] is True
        assert call["allow_l0_mismatch"] is True
        assert call.get("layer", 12) == 12  # default layer forwarded

    def test_multiple_slug_flags_build_each_and_write_once(
        self, traced_env, monkeypatch, real_circuit
    ):
        env = traced_env
        build_calls, writes = [], []
        monkeypatch.setattr(
            env.mod,
            "build_traced_circuit",
            lambda *a, **k: (
                build_calls.append(_norm_build_args(a, k)),
                copy.deepcopy(real_circuit),
            )[1],
        )
        monkeypatch.setattr(
            env.mod,
            "write_traced_outputs",
            lambda *a, **k: (
                writes.append(_norm_write_args(a, k)),
                env.out_dir / "circuits" / env.stem,
            )[1],
        )
        _run_expect_success(
            env.mod,
            [
                "--slug", "slug-a", "--slug", "slug-b",
                "--dataset-stem", env.stem,
                "--out-dir", str(env.out_dir),
            ],
        )
        assert [s for (_m, s) in env.calls["record"]] == ["slug-a", "slug-b"]
        assert len(build_calls) == 2
        assert len(writes) == 1
        circuits, _out_root, _stem = writes[0]
        assert len(circuits) == 2

    def test_slugs_file_builds_each_slug_and_writes_once(
        self, traced_env, monkeypatch, tmp_path, real_circuit
    ):
        env = traced_env
        slugs_path = tmp_path / "slugs.txt"
        slugs_path.write_text("slug-a\nslug-b\n")

        build_calls, writes = [], []
        monkeypatch.setattr(
            env.mod,
            "build_traced_circuit",
            lambda *a, **k: (
                build_calls.append(_norm_build_args(a, k)),
                copy.deepcopy(real_circuit),
            )[1],
        )
        monkeypatch.setattr(
            env.mod,
            "write_traced_outputs",
            lambda *a, **k: (
                writes.append(_norm_write_args(a, k)),
                env.out_dir / "circuits" / env.stem,
            )[1],
        )
        _run_expect_success(
            env.mod,
            [
                "--slugs-file", str(slugs_path),
                "--dataset-stem", env.stem,
                "--out-dir", str(env.out_dir),
            ],
        )
        assert [s for (_m, s) in env.calls["record"]] == ["slug-a", "slug-b"]
        assert len(build_calls) == 2
        assert len(writes) == 1
        assert len(writes[0][0]) == 2

    def test_force_is_forwarded_to_fetch_graph(
        self, traced_env, monkeypatch, real_circuit
    ):
        env = traced_env
        monkeypatch.setattr(
            env.mod,
            "build_traced_circuit",
            lambda *a, **k: copy.deepcopy(real_circuit),
        )
        monkeypatch.setattr(
            env.mod,
            "write_traced_outputs",
            lambda *a, **k: env.out_dir / "circuits" / env.stem,
        )
        base = ["--slug", SLUG, "--dataset-stem", env.stem,
                "--out-dir", str(env.out_dir)]
        _run_expect_success(env.mod, base)
        _run_expect_success(env.mod, base + ["--force"])

        force_flags = [kw.get("force") for (_m, _s, kw) in env.calls["graph"]]
        assert len(force_flags) == 2
        assert not force_flags[0]  # default: cache honored
        assert force_flags[1]      # --force: cache bypassed

    def test_missing_dataset_metadata_is_actionable_and_writes_nothing(
        self, traced_env, capsys
    ):
        env = traced_env
        # DEFAULT_STEM metadata was never seeded into the tmp out-dir
        _run_expect_failure(
            env.mod,
            ["--slug", SLUG, "--dataset-stem", DEFAULT_STEM,
             "--out-dir", str(env.out_dir)],
        )
        err = capsys.readouterr().err
        assert "metadata" in err.lower()
        assert not (env.out_dir / "circuits").exists()


# ---------------------------------------------------------------------------
# Banner usage
# ---------------------------------------------------------------------------


class TestBannerUsage:
    def test_module_imports_banner_helpers(self, traced_mod):
        src = Path(traced_mod.__file__).read_text()
        assert (
            "from pipeline.banner import" in src
            or "import pipeline.banner" in src
        ), "generate_traced_circuits must use pipeline/banner.py helpers"
        helper_names = (
            "step_header", "info", "success", "error", "warn", "detail",
        )
        assert any(f"{name}(" in src for name in helper_names), (
            "no pipeline.banner helper appears to be called"
        )
