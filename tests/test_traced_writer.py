# striatica/tests/test_traced_writer.py
"""Tests for pipeline/traced_circuits.py writer functions (unit: traced_writer).

Input circuits are built with the tested ``build_traced_circuit`` against the
REAL Neuronpedia fixtures in tests/fixtures/ (gemma-fact-dallas-austin graph,
fetched live 2026-07-28). The identity validator is stubbed for those builds
because the fixture pairing (graph dictionary average_l0_6 vs local dataset
l0_604) is a CONFIRMED mismatch that the real validator must hard-fail — the
validator is not the unit under test here. Perturbations (poisoned clerp,
hostile names) mutate real built circuits in memory; directory trees under
tmp_path are built by COPYING real repo files (no synthetic mock JSON).

API CONTRACT (implementers: extend pipeline/traced_circuits.py to this):

    write_traced_outputs(
        circuits: list[dict],    # frozen-schema traced circuits
                                 # (build_traced_circuit output)
        out_root: Path,          # public data root, e.g. frontend/public/data
        dataset_stem: str,       # e.g. "gemma-2-2b-layer12-l0604"
    ) -> Path                    # {out_root}/circuits/{dataset_stem}

    generate_datasets_manifest(public_data_dir: Path) -> Path
                                 # {public_data_dir}/datasets.json

Contract details these tests pin — write_traced_outputs:

- Returns {out_root}/circuits/{dataset_stem} (Path), creating it (and any
  missing parents) as needed.
- Writes exactly one {circuit["name"]}.json per circuit plus manifest.json
  into that directory, and NOTHING anywhere else: every pre-existing file
  under out_root (root circuits/manifest.json, gpt2 circuit JSONs, ...) must
  be byte-identical afterwards.
- Each circuit file JSON-round-trips equal to the input circuit dict.
- manifest.json is {"circuits": [...]} in INPUT ORDER; each entry is exactly
  {id, name, type, description, nodeCount, edgeCount, path, model, slug} with
  id == name, type == "traced", nodeCount == len(nodes),
  edgeCount == len(edges) == 0,
  path == "/data/circuits/{dataset_stem}/{name}.json", and model/slug taken
  from the circuit's metadata block.
- Calls ``scrub_check`` through the MODULE GLOBAL
  (pipeline.traced_circuits.scrub_check — monkeypatching it must intercept)
  exactly once per circuit, in input order, BEFORE that circuit's file exists
  on disk. A scrub failure aborts with ValueError; the offending circuit's
  file must not exist afterwards, no file under out_root may contain the
  semantic text, and any manifest.json written must not list the offending
  circuit.
- A circuit whose name would place its file outside the {dataset_stem}
  directory (path traversal like "../x", or an absolute path) raises
  ValueError and writes nothing outside that directory.
- CLI output (pipeline/banner.py helpers) is permitted but not asserted.

Contract details these tests pin — generate_datasets_manifest:

- Scans TOP-LEVEL *.json files of public_data_dir only, skipping the
  circuits/ directory (and directories generally), *-metadata.json,
  *-validation.json, and datasets.json itself.
- Reads each remaining JSON's top-level "model", "layer" and "numFeatures"
  keys and writes {public_data_dir}/datasets.json as a JSON ARRAY of
  {file, model, layer, numFeatures} entries (file == basename), sorted by
  "file" ascending. An existing datasets.json is overwritten, never listed.
- Touches nothing else: every other file byte-identical afterwards.
- Returns the Path of the written datasets.json.
"""

import copy
import inspect
import json
import re
import shutil
from pathlib import Path

import pytest

import pipeline.traced_circuits as traced_circuits
from pipeline.traced_circuits import (
    build_traced_circuit,
    generate_datasets_manifest,
    scrub_check,
    write_traced_outputs,
)

REPO_ROOT = Path(__file__).parent.parent
FIXTURES = Path(__file__).parent / "fixtures"
REAL_DATA = REPO_ROOT / "frontend" / "public" / "data"
REAL_CIRCUITS = REAL_DATA / "circuits"

STEM = "gemma-2-2b-layer12-l0604"
DEFAULT_NAME = "gemma-fact-dallas-austin"  # record slug -> default circuit name
MODEL = "gemma-2-2b"
SLUG = "gemma-fact-dallas-austin"
NODE_COUNT = 6  # layer-12 CLT feature nodes in the fixture graph

# Top-level dataset JSONs currently in the real public data dir (12-20 MB
# each; only their scalar head fields matter to the datasets manifest).
DATASET_FILES = [
    "gemma-2-2b-12-gemmascope-res-16k.json",
    "gemma-2-2b-layer12-l0604.json",
    "gpt2-small-6-res-jb.json",
    "pythia-70m-deduped-4-res-sm.json",
]

# Small real files that generate_datasets_manifest must SKIP (copied verbatim
# into the tmp tree).
SMALL_REAL_SKIP_FILES = [
    "gemma-2-2b-12-gemmascope-res-16k-metadata.json",
    "gemma-2-2b-layer12-l0604-metadata.json",
    "gemma-2-2b-layer12-l0604-validation.json",
]

# Real gpt2-era circuit files used to fake the pre-existing tree that the
# traced writer must never touch.
GPT2_TREE_FILES = [
    "manifest.json",
    "coact-capital-of-france.json",
    "sim-6133.json",
]


def _load_fixture(filename: str) -> dict:
    with open(FIXTURES / filename) as f:
        return json.load(f)


_RAW_GRAPH = _load_fixture("neuronpedia_graph_gemma.json")
_RECORD = _load_fixture("neuronpedia_graph_record_gemma.json")
_SOURCE_SET = _load_fixture("neuronpedia_sourceset_gemma.json")
# Dataset metadata from the COMMITTED fixture (byte copy of the real l0604
# sidecar) so this module collects on clean clones without local data files.
_DATASET_METADATA = _load_fixture("gemma-2-2b-layer12-l0604-metadata.json")


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def build(monkeypatch):
    """Build real traced circuits from the real fixture graph.

    The identity validator is stubbed ONLY because the fixture pairing is a
    confirmed l0 mismatch (average_l0_6 graph vs l0_604 dataset) that the
    real validator hard-fails by design. The writer under test never calls
    the validator.
    """
    monkeypatch.setattr(
        traced_circuits, "validate_graph_identity", lambda *a, **k: None
    )

    def _build(**overrides):
        kwargs = {"layer": 12, "allow_l0_mismatch": True}
        kwargs.update(overrides)
        return build_traced_circuit(
            copy.deepcopy(_RAW_GRAPH),
            copy.deepcopy(_RECORD),
            copy.deepcopy(_SOURCE_SET),
            copy.deepcopy(_DATASET_METADATA),
            **kwargs,
        )

    return _build


def _tree_bytes(root: Path) -> dict:
    """Snapshot {relative posix path: bytes} for every file under root."""
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _make_out_root(tmp_path: Path) -> Path:
    out_root = tmp_path / "data"
    out_root.mkdir()
    return out_root


def _make_fake_gpt2_tree(tmp_path: Path) -> Path:
    """out_root with the real root circuits manifest + real gpt2 circuits."""
    out_root = _make_out_root(tmp_path)
    circuits_dir = out_root / "circuits"
    circuits_dir.mkdir()
    for fname in GPT2_TREE_FILES:
        shutil.copyfile(REAL_CIRCUITS / fname, circuits_dir / fname)
    return out_root


def _head_fields(path: Path) -> dict:
    """Extract the real top-level model/layer/numFeatures from a dataset
    JSON's head without loading the multi-MB file."""
    head = path.open("rb").read(4096).decode("utf-8", errors="replace")
    model = re.search(r'"model"\s*:\s*"([^"]*)"', head)
    layer = re.search(r'"layer"\s*:\s*"([^"]*)"', head)
    num = re.search(r'"numFeatures"\s*:\s*(\d+)', head)
    assert model and layer and num, (
        f"could not extract model/layer/numFeatures from head of {path.name}"
    )
    return {
        "model": model.group(1),
        "layer": layer.group(1),
        "numFeatures": int(num.group(1)),
    }


def _make_datasets_tree(tmp_path: Path) -> Path:
    """tmp mirror of frontend/public/data built from real repo files.

    Dataset stubs carry the REAL model/layer/numFeatures values read from the
    real dataset file heads (the full files are 12-20 MB); the small
    metadata/validation files and a real circuit file are copied verbatim.
    """
    tree = tmp_path / "public-data"
    tree.mkdir()
    for fname in DATASET_FILES:
        (tree / fname).write_text(json.dumps(_head_fields(REAL_DATA / fname)))
    for fname in SMALL_REAL_SKIP_FILES:
        shutil.copyfile(REAL_DATA / fname, tree / fname)
    circuits_dir = tree / "circuits"
    circuits_dir.mkdir()
    for fname in ("manifest.json", "coact-capital-of-france.json"):
        shutil.copyfile(REAL_CIRCUITS / fname, circuits_dir / fname)
    # Stale output from a previous run: must be overwritten, never listed.
    (tree / "datasets.json").write_text("[]")
    return tree


def _expected_manifest_entry(circuit: dict) -> dict:
    return {
        "id": circuit["name"],
        "name": circuit["name"],
        "type": "traced",
        "description": circuit["description"],
        "nodeCount": len(circuit["nodes"]),
        "edgeCount": 0,
        "path": f"/data/circuits/{STEM}/{circuit['name']}.json",
        "model": circuit["metadata"]["model"],
        "slug": circuit["metadata"]["slug"],
    }


# ---------------------------------------------------------------------------
# Preconditions (guard against repo-tree drift — later tests build on these)
# ---------------------------------------------------------------------------


def test_real_tree_preconditions():
    for fname in DATASET_FILES:
        fields = _head_fields(REAL_DATA / fname)
        assert set(fields.keys()) == {"model", "layer", "numFeatures"}
        assert fields["model"] and fields["layer"]
        assert fields["numFeatures"] > 0
    # Anchor against the known real gpt2 dataset head.
    assert _head_fields(REAL_DATA / "gpt2-small-6-res-jb.json") == {
        "model": "gpt2-small",
        "layer": "6-res-jb",
        "numFeatures": 24576,
    }
    for fname in SMALL_REAL_SKIP_FILES:
        assert (REAL_DATA / fname).is_file()
    for fname in GPT2_TREE_FILES:
        assert (REAL_CIRCUITS / fname).is_file()


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_public_api_signatures():
    sig = inspect.signature(write_traced_outputs)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ["circuits", "out_root", "dataset_stem"]
    for p in params:
        assert p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert p.default is inspect.Parameter.empty

    sig = inspect.signature(generate_datasets_manifest)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ["public_data_dir"]
    assert params[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params[0].default is inspect.Parameter.empty


# ---------------------------------------------------------------------------
# write_traced_outputs: files + manifest correctness
# ---------------------------------------------------------------------------


def test_write_returns_stem_dir_and_creates_it(build, tmp_path):
    out_root = _make_out_root(tmp_path)  # no circuits/ subdir yet
    ret = write_traced_outputs([build()], out_root, STEM)
    assert isinstance(ret, Path)
    assert ret.resolve() == (out_root / "circuits" / STEM).resolve()
    assert ret.is_dir()


def test_written_circuit_files_round_trip_exactly(build, tmp_path):
    out_root = _make_out_root(tmp_path)
    c1 = build()
    c2 = build(name="dallas-austin-l12", description="Layer-12 slice")
    stem_dir = write_traced_outputs([c1, c2], out_root, STEM)
    # Exactly one file per circuit plus the manifest — nothing else.
    assert {p.name for p in stem_dir.iterdir()} == {
        f"{DEFAULT_NAME}.json",
        "dallas-austin-l12.json",
        "manifest.json",
    }
    for circuit in (c1, c2):
        on_disk = json.loads((stem_dir / f"{circuit['name']}.json").read_text())
        assert on_disk == circuit
        # What reached disk still passes the leak barrier.
        assert scrub_check(on_disk) is None


def test_manifest_exact_fields_and_input_order(build, tmp_path):
    out_root = _make_out_root(tmp_path)
    c1 = build()
    c2 = build(name="dallas-austin-l12", description="Layer-12 slice")
    stem_dir = write_traced_outputs([c1, c2], out_root, STEM)
    manifest = json.loads((stem_dir / "manifest.json").read_text())
    assert manifest == {
        "circuits": [_expected_manifest_entry(c1), _expected_manifest_entry(c2)]
    }
    # Anchors from the real fixture graph.
    first = manifest["circuits"][0]
    assert first["id"] == DEFAULT_NAME
    assert first["type"] == "traced"
    assert first["nodeCount"] == NODE_COUNT
    assert first["edgeCount"] == 0
    assert first["path"] == f"/data/circuits/{STEM}/{DEFAULT_NAME}.json"
    assert first["model"] == MODEL
    assert first["slug"] == SLUG


def test_write_preserves_existing_tree_byte_identical(build, tmp_path):
    out_root = _make_fake_gpt2_tree(tmp_path)
    before = _tree_bytes(out_root)
    write_traced_outputs([build()], out_root, STEM)
    after = _tree_bytes(out_root)
    for rel, data in before.items():
        assert after[rel] == data, f"pre-existing file modified: {rel}"
    assert set(after) - set(before) == {
        f"circuits/{STEM}/{DEFAULT_NAME}.json",
        f"circuits/{STEM}/manifest.json",
    }


# ---------------------------------------------------------------------------
# write_traced_outputs: scrub_check gate
# ---------------------------------------------------------------------------


def test_scrub_check_called_once_per_circuit_before_write(
    build, tmp_path, monkeypatch
):
    out_root = _make_out_root(tmp_path)
    stem_dir = out_root / "circuits" / STEM
    c1 = build()
    c2 = build(name="dallas-austin-l12", description="Layer-12 slice")

    real_scrub = traced_circuits.scrub_check
    seen = []

    def _recording(circuit):
        seen.append(circuit["name"])
        target = stem_dir / f"{circuit['name']}.json"
        assert not target.exists(), (
            "scrub_check must run BEFORE the circuit file is written"
        )
        return real_scrub(circuit)

    monkeypatch.setattr(traced_circuits, "scrub_check", _recording)
    write_traced_outputs([c1, c2], out_root, STEM)
    assert seen == [DEFAULT_NAME, "dallas-austin-l12"]


def test_semantic_leak_aborts_write_and_never_reaches_disk(build, tmp_path):
    out_root = _make_out_root(tmp_path)
    semantic = "increases probability of Texas city names"
    c1 = build()
    poisoned = build(name="poisoned-l12", description="Layer-12 slice")
    poisoned["nodes"][0]["clerp"] = semantic

    with pytest.raises(ValueError):
        write_traced_outputs([c1, poisoned], out_root, STEM)

    stem_dir = out_root / "circuits" / STEM
    assert not (stem_dir / "poisoned-l12.json").exists()
    # The semantic text must not exist ANYWHERE under out_root.
    for p in out_root.rglob("*"):
        if p.is_file():
            assert semantic.encode() not in p.read_bytes(), str(p)
    # Any manifest written must not list the rejected circuit.
    manifest_path = stem_dir / "manifest.json"
    if manifest_path.exists():
        listed = [c["id"] for c in json.loads(manifest_path.read_text())["circuits"]]
        assert "poisoned-l12" not in listed


# ---------------------------------------------------------------------------
# write_traced_outputs: stays inside its dataset-stem directory
# ---------------------------------------------------------------------------


def test_refuses_relative_traversal_name(build, tmp_path):
    out_root = _make_out_root(tmp_path)
    circuit = build(name="../escape")
    with pytest.raises(ValueError):
        write_traced_outputs([circuit], out_root, STEM)
    # Nothing named after the hostile circuit anywhere under tmp_path.
    assert list(tmp_path.rglob("*escape*")) == []


def test_refuses_absolute_path_name(build, tmp_path):
    out_root = _make_out_root(tmp_path)
    outside = tmp_path / "outside"
    circuit = build(name=str(outside / "evil"))
    with pytest.raises(ValueError):
        write_traced_outputs([circuit], out_root, STEM)
    assert not outside.exists() or list(outside.rglob("*")) == []
    assert list(tmp_path.rglob("*evil*")) == []


# ---------------------------------------------------------------------------
# generate_datasets_manifest
# ---------------------------------------------------------------------------


def test_datasets_manifest_content_exact(tmp_path):
    tree = _make_datasets_tree(tmp_path)
    ret = generate_datasets_manifest(tree)
    assert isinstance(ret, Path)
    assert ret.resolve() == (tree / "datasets.json").resolve()

    entries = json.loads((tree / "datasets.json").read_text())
    expected = sorted(
        (
            {"file": fname, **_head_fields(REAL_DATA / fname)}
            for fname in DATASET_FILES
        ),
        key=lambda e: e["file"],
    )
    # Exact content, exact per-entry key set, sorted by "file" — and thereby:
    # no circuits/ entries, no *-metadata.json, no *-validation.json, and no
    # datasets.json self-entry despite the stale one in the tree.
    assert entries == expected
    gpt2 = next(e for e in entries if e["file"] == "gpt2-small-6-res-jb.json")
    assert gpt2 == {
        "file": "gpt2-small-6-res-jb.json",
        "model": "gpt2-small",
        "layer": "6-res-jb",
        "numFeatures": 24576,
    }


def test_datasets_manifest_touches_nothing_else(tmp_path):
    tree = _make_datasets_tree(tmp_path)
    before = _tree_bytes(tree)
    before.pop("datasets.json")
    generate_datasets_manifest(tree)
    after = _tree_bytes(tree)
    stale_replaced = after.pop("datasets.json")
    assert json.loads(stale_replaced) != []  # stale [] was regenerated
    assert after == before
