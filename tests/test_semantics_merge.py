# striatica/tests/test_semantics_merge.py
"""Tests for pipeline/semantics_merge.py (unit: semantics_merge).

Merges a Neuronpedia explanations JSONL (S3 bulk-export line shape) into an
EXISTING dataset JSON, patching ONLY features[].explanation (and the
semanticsRedacted flag). All tests run against a tmp COPY of the REAL dataset
frontend/public/data/gemma-2-2b-layer12-l0604.json (16,384 features, produced
by the transcoder pipeline 2026-04-27) — no synthetic dataset JSON. The
explanations JSONL is written to tmp in the exact REAL S3 export line shape
(verified against data/gpt2-small_6-res-jb_explanations.jsonl: string "index",
"description", "modelId", "layer", plus extra fields the merge must ignore),
with description text neutralized to the fixture convention ("label-N").

API CONTRACT (implementers: create pipeline/semantics_merge.py to this):

    merge_explanations(
        dataset_json_path: str | Path,   # EXISTING dataset JSON (prepare_json shape)
        explanations_jsonl_path: str | Path,
        out_path: str | Path | None = None,  # None -> overwrite dataset_json_path
    ) -> dict                            # the merged dataset dict (== file content)

    neuronpedia_layer_id(layer: str) -> str
        # dataset "layer" string -> Neuronpedia source/layer id
        # "layer12-l0604" -> "12-gemmascope-transcoder-16k"  (transcoder datasets)
        # "6-res-jb"      -> "6-res-jb"                      (already a source id)

    main(argv: list[str] | None = None) -> int
        # usage: DATASET_JSON EXPLANATIONS_JSONL [--out OUT_JSON] [--download]
        # --download: fetch the explanations from Neuronpedia S3 INTO
        #   EXPLANATIONS_JSONL first, by calling download_explanations(
        #   model_id=<dataset "model">, layer=neuronpedia_layer_id(<dataset
        #   "layer">), output_path=EXPLANATIONS_JSONL). Must be called through
        #   the MODULE GLOBAL pipeline.semantics_merge.download_explanations
        #   (monkeypatching that name must intercept — no network in tests).
        # Returns 0 on success; returns a nonzero int (no uncaught exception)
        # on missing files or a tier-gate failure.
        # CLI output goes through pipeline/banner.py helpers (info/success/
        # error/warn/detail, step_header) — permitted, not asserted.

Contract details these tests pin — merge_explanations:

- SAFETY TIER GATE: raises ValueError (message mentions "public tier") when
  the dataset's "model" is not is_public_tier(); writes NOTHING in that case.
  (gemma-2-2b is public tier as of this unit — see pipeline/config.py.)
- JSONL lines: index via int(line["index"]) (real exports use STRINGS),
  text via line["description"]. First occurrence of an index wins (same rule
  as pipeline/prepare.py). Lines skipped without error: blank lines, empty
  descriptions (merge never erases an existing explanation), and indices
  outside [0, numFeatures). Extra line fields are ignored.
- Patches features[i]["explanation"] for matched local indices ONLY; every
  unmatched feature keeps its prior explanation verbatim.
- Sets top-level "semanticsRedacted" to False.
- IRON RULE: the positions, clusterLabels, localDimensions, growthCurves and
  clusters subtrees are byte-identical after the merge (compared via
  json.dumps(..., sort_keys=True) serialization); model, layer, numFeatures,
  dimMethod and every non-explanation feature field are unchanged.
- out_path=None overwrites dataset_json_path in place; with out_path given,
  the source file's BYTES are untouched.
- Missing explanations JSONL -> FileNotFoundError, nothing written.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest

import pipeline.semantics_merge as semantics_merge
from pipeline.semantics_merge import main, merge_explanations, neuronpedia_layer_id

REPO_ROOT = Path(__file__).parent.parent
REAL_DATA = REPO_ROOT / "frontend" / "public" / "data"
DATASET_FILE = REAL_DATA / "gemma-2-2b-layer12-l0604.json"
# The dataset's metadata sidecar, by the {stem}-metadata.json convention.
METADATA_SIDECAR = REAL_DATA / "gemma-2-2b-layer12-l0604-metadata.json"

MODEL = "gemma-2-2b"
LAYER = "layer12-l0604"
NP_LAYER_ID = "12-gemmascope-transcoder-16k"
NUM_FEATURES = 16384
NONEMPTY_EXPLANATIONS_BEFORE = 7582  # real file state, produced 2026-04-27
# The dataset was built from the average_l0_604 transcoder dictionary; the
# Neuronpedia layer-12 gemmascope-transcoder-16k source is DEPLOYED at
# average_l0_6 — a different dictionary (see CLAUDE.md / graph_fetch identity).
DATASET_L0_VARIANT = 604
DEPLOYED_L0_VARIANT = 6

# Subtrees the merge must NEVER alter (IRON RULE).
IRON_RULE_KEYS = ("positions", "clusterLabels", "localDimensions", "growthCurves", "clusters")

# Local index -> neutralized description the merge must apply.
PATCHED = {
    0: "label-0",          # empty -> filled
    1: "label-1",          # nonempty -> overwritten (first occurrence wins)
    27: "label-27",        # int-typed index line
    12345: "label-12345",
    16383: "label-16383",  # last valid local index
}

pytestmark = pytest.mark.skipif(
    not DATASET_FILE.exists(),
    reason="real dataset gemma-2-2b-layer12-l0604.json not present",
)


def _jsonl_line(index, description: str) -> str:
    """One explanations line in the REAL Neuronpedia S3 export shape."""
    return json.dumps({
        "id": f"test-{index}",
        "modelId": MODEL,
        "layer": NP_LAYER_ID,
        "index": index if isinstance(index, int) else str(index),
        "authorId": "test-author",
        "description": description,
        "typeName": "oai_token-act-pair",
        "explanationModelName": "gpt-4o-mini",
    })


def _write_explanations_jsonl(path: Path) -> Path:
    lines = [
        _jsonl_line("0", "label-0"),
        _jsonl_line("1", "label-1"),
        _jsonl_line("1", "label-1-duplicate"),   # duplicate index — must NOT win
        _jsonl_line("2", ""),                    # empty description — skipped
        "",                                      # blank line — tolerated
        _jsonl_line(27, "label-27"),             # int index — coerced like str
        _jsonl_line("12345", "label-12345"),
        _jsonl_line("16383", "label-16383"),
        _jsonl_line("16384", "label-99001"),     # == numFeatures — out of range
        _jsonl_line("-1", "label-99002"),        # negative — out of range
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(obj) -> str:
    return json.dumps(obj, sort_keys=True)


# ---------------------------------------------------------------------------
# Module-scoped fixtures (the 12 MB real dataset is loaded/merged once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pristine():
    """The real dataset, loaded once, with pre-merge snapshots."""
    with open(DATASET_FILE) as f:
        data = json.load(f)
    return {
        "data": data,
        "subtrees": {k: _dump(data[k]) for k in IRON_RULE_KEYS},
        "explanations": [f["explanation"] for f in data["features"]],
        "features_sans_explanation": _dump(
            [{k: v for k, v in f.items() if k != "explanation"} for f in data["features"]]
        ),
    }


@pytest.fixture(scope="module")
def explanations_jsonl(tmp_path_factory) -> Path:
    return _write_explanations_jsonl(
        tmp_path_factory.mktemp("jsonl") / f"{MODEL}_{NP_LAYER_ID}_explanations.jsonl"
    )


@pytest.fixture(scope="module")
def merged(pristine, explanations_jsonl, tmp_path_factory):
    """Copy the real dataset to tmp, merge IN PLACE (out_path=None), reload."""
    work = tmp_path_factory.mktemp("merge") / DATASET_FILE.name
    shutil.copyfile(DATASET_FILE, work)
    result = merge_explanations(work, explanations_jsonl)
    with open(work) as f:
        on_disk = json.load(f)
    return {"result": result, "on_disk": on_disk, "work": work}


@pytest.fixture(scope="module")
def restricted_copy(pristine, tmp_path_factory) -> Path:
    """Real dataset with model swapped to a NON-public-tier id (gemma-2b)."""
    d = dict(pristine["data"])
    d["model"] = "gemma-2b"  # SAE-era Gemma id — restricted tier
    p = tmp_path_factory.mktemp("restricted") / "gemma-2b-layer12.json"
    with open(p, "w") as f:
        json.dump(d, f)
    return p


# ---------------------------------------------------------------------------
# Preconditions: pin the REAL fixture state these tests compute against
# ---------------------------------------------------------------------------


class TestRealDatasetPreconditions:
    def test_dataset_shape(self, pristine):
        d = pristine["data"]
        assert d["model"] == MODEL
        assert d["layer"] == LAYER
        assert d["numFeatures"] == NUM_FEATURES
        assert d["dimMethod"] == "pr"
        assert len(d["features"]) == NUM_FEATURES
        assert len(d["positions"]) == NUM_FEATURES * 3
        assert len(d["clusterLabels"]) == NUM_FEATURES
        assert len(d["localDimensions"]) == NUM_FEATURES
        assert len(d["growthCurves"]) == NUM_FEATURES
        assert len(d["clusters"]) == 22
        # features[] is positionally indexed by local index
        assert all(f["index"] == i for i, f in enumerate(d["features"]))

    def test_premerge_explanation_state(self, pristine):
        expl = pristine["explanations"]
        assert sum(1 for e in expl if e) == NONEMPTY_EXPLANATIONS_BEFORE
        # The specific indices the JSONL targets, in their real pre-merge state:
        assert expl[0] == ""       # gets filled
        assert expl[1] != ""       # gets overwritten
        assert expl[2] != ""       # empty-description line must NOT erase this
        assert expl[3] != ""       # untouched — must be preserved verbatim
        assert expl[27] == ""      # gets filled via int-typed index line


# ---------------------------------------------------------------------------
# merge_explanations — patch semantics
# ---------------------------------------------------------------------------


class TestMergeExplanations:
    def test_patches_expected_indices(self, merged):
        feats = merged["on_disk"]["features"]
        for idx, label in PATCHED.items():
            assert feats[idx]["explanation"] == label, f"feature {idx}"

    def test_first_occurrence_wins_on_duplicate_index(self, merged):
        assert merged["on_disk"]["features"][1]["explanation"] == "label-1"

    def test_empty_description_does_not_erase(self, merged, pristine):
        # Line for index 2 has description "" — original explanation survives.
        assert merged["on_disk"]["features"][2]["explanation"] == pristine["explanations"][2]

    def test_unmatched_features_keep_prior_explanation(self, merged, pristine):
        feats = merged["on_disk"]["features"]
        before = pristine["explanations"]
        for i in range(NUM_FEATURES):
            if i not in PATCHED:
                assert feats[i]["explanation"] == before[i], f"feature {i} changed"

    def test_out_of_range_indices_skipped(self, merged):
        d = merged["on_disk"]
        assert len(d["features"]) == NUM_FEATURES
        assert d["numFeatures"] == NUM_FEATURES
        # The out-of-range descriptions appear nowhere.
        blob = _dump(d["features"])
        assert "label-99001" not in blob
        assert "label-99002" not in blob

    def test_nonempty_count_after_merge(self, merged, pristine):
        before = pristine["explanations"]
        newly_filled = sum(1 for i in PATCHED if before[i] == "")
        after = sum(1 for f in merged["on_disk"]["features"] if f["explanation"])
        assert after == NONEMPTY_EXPLANATIONS_BEFORE + newly_filled

    def test_semantics_redacted_false(self, merged):
        assert merged["on_disk"]["semanticsRedacted"] is False
        assert merged["result"]["semanticsRedacted"] is False

    def test_return_value_matches_disk(self, merged):
        assert _dump(merged["result"]) == _dump(merged["on_disk"])


# ---------------------------------------------------------------------------
# merge_explanations — IRON RULE
# ---------------------------------------------------------------------------


class TestIronRule:
    @pytest.mark.parametrize("key", IRON_RULE_KEYS)
    def test_subtree_byte_identical(self, merged, pristine, key):
        assert _dump(merged["on_disk"][key]) == pristine["subtrees"][key]
        assert _dump(merged["result"][key]) == pristine["subtrees"][key]

    def test_non_explanation_feature_fields_unchanged(self, merged, pristine):
        after = _dump(
            [{k: v for k, v in f.items() if k != "explanation"}
             for f in merged["on_disk"]["features"]]
        )
        assert after == pristine["features_sans_explanation"]

    def test_top_level_scalars_unchanged(self, merged, pristine):
        d = pristine["data"]
        for key in ("model", "layer", "numFeatures", "dimMethod"):
            assert merged["on_disk"][key] == d[key]


# ---------------------------------------------------------------------------
# merge_explanations — out_path, tier gate, missing input
# ---------------------------------------------------------------------------


class TestOutPathAndGuards:
    def test_out_path_leaves_source_untouched(self, pristine, explanations_jsonl, tmp_path):
        src = tmp_path / DATASET_FILE.name
        shutil.copyfile(DATASET_FILE, src)
        src_hash = _sha256(src)
        out = tmp_path / "merged.json"

        result = merge_explanations(src, explanations_jsonl, out_path=out)

        assert _sha256(src) == src_hash, "source file bytes must be untouched"
        assert out.exists()
        with open(out) as f:
            on_disk = json.load(f)
        assert on_disk["features"][0]["explanation"] == "label-0"
        assert _dump(on_disk["positions"]) == pristine["subtrees"]["positions"]
        assert result["features"][0]["explanation"] == "label-0"

    def test_restricted_model_hard_fails(self, restricted_copy, explanations_jsonl, tmp_path):
        src_hash = _sha256(restricted_copy)
        out = tmp_path / "should-not-exist.json"
        with pytest.raises(ValueError, match="public tier"):
            merge_explanations(restricted_copy, explanations_jsonl, out_path=out)
        assert not out.exists(), "tier-gate failure must write nothing"
        assert _sha256(restricted_copy) == src_hash

    def test_missing_jsonl_raises_and_writes_nothing(self, tmp_path):
        src = tmp_path / DATASET_FILE.name
        shutil.copyfile(DATASET_FILE, src)
        src_hash = _sha256(src)
        out = tmp_path / "should-not-exist.json"
        with pytest.raises(FileNotFoundError):
            merge_explanations(src, tmp_path / "nope.jsonl", out_path=out)
        assert not out.exists()
        assert _sha256(src) == src_hash


# ---------------------------------------------------------------------------
# neuronpedia_layer_id
# ---------------------------------------------------------------------------


class TestNeuronpediaLayerId:
    def test_transcoder_layer_string(self):
        # SPIKE-verified: Neuronpedia's layer-12 Gemma Scope transcoder source id.
        assert neuronpedia_layer_id("layer12-l0604") == "12-gemmascope-transcoder-16k"

    def test_transcoder_other_layer(self):
        assert neuronpedia_layer_id("layer5-l088") == "5-gemmascope-transcoder-16k"

    def test_saelens_style_id_passes_through(self):
        assert neuronpedia_layer_id("6-res-jb") == "6-res-jb"


# ---------------------------------------------------------------------------
# main(argv)
# ---------------------------------------------------------------------------


class TestMainCLI:
    def test_success_with_out_flips_redaction_flag(self, pristine, explanations_jsonl, tmp_path):
        # A redacted transcoder output (real shape, flag flipped) gets its
        # explanations merged in and semanticsRedacted set back to False.
        d = dict(pristine["data"])
        d["semanticsRedacted"] = True
        src = tmp_path / DATASET_FILE.name
        with open(src, "w") as f:
            json.dump(d, f)
        out = tmp_path / "merged.json"

        rv = main([str(src), str(explanations_jsonl), "--out", str(out)])

        assert rv == 0
        with open(out) as f:
            on_disk = json.load(f)
        assert on_disk["semanticsRedacted"] is False
        assert on_disk["features"][0]["explanation"] == "label-0"
        assert _dump(on_disk["positions"]) == pristine["subtrees"]["positions"]

    def test_default_is_in_place(self, explanations_jsonl, tmp_path):
        src = tmp_path / DATASET_FILE.name
        shutil.copyfile(DATASET_FILE, src)
        rv = main([str(src), str(explanations_jsonl)])
        assert rv == 0
        with open(src) as f:
            on_disk = json.load(f)
        assert on_disk["features"][0]["explanation"] == "label-0"

    def test_missing_dataset_returns_nonzero(self, explanations_jsonl, tmp_path):
        rv = main([str(tmp_path / "nope.json"), str(explanations_jsonl)])
        assert isinstance(rv, int) and rv != 0

    def test_missing_jsonl_returns_nonzero_and_leaves_dataset(self, tmp_path):
        src = tmp_path / DATASET_FILE.name
        shutil.copyfile(DATASET_FILE, src)
        src_hash = _sha256(src)
        rv = main([str(src), str(tmp_path / "nope.jsonl")])
        assert isinstance(rv, int) and rv != 0
        assert _sha256(src) == src_hash

    def test_restricted_model_returns_nonzero(self, restricted_copy, explanations_jsonl):
        src_hash = _sha256(restricted_copy)
        rv = main([str(restricted_copy), str(explanations_jsonl)])
        assert isinstance(rv, int) and rv != 0
        assert _sha256(restricted_copy) == src_hash

    def test_download_flag_uses_download_helper(self, tmp_path, monkeypatch):
        """--download fetches into EXPLANATIONS_JSONL via the module-global
        download_explanations (S3 helper from pipeline/download.py), with the
        Neuronpedia layer id derived from the dataset — then merges."""
        src = tmp_path / DATASET_FILE.name
        shutil.copyfile(DATASET_FILE, src)
        jsonl_dest = tmp_path / "downloaded_explanations.jsonl"
        out = tmp_path / "merged.json"
        calls = []

        def fake_download(model_id, layer, batch_indices=None, output_path=None):
            calls.append({"model_id": model_id, "layer": layer, "output_path": output_path})
            _write_explanations_jsonl(Path(output_path))
            return 8

        monkeypatch.setattr(semantics_merge, "download_explanations", fake_download)

        rv = main([str(src), str(jsonl_dest), "--download", "--out", str(out)])

        assert rv == 0
        assert len(calls) == 1
        assert calls[0]["model_id"] == MODEL
        assert calls[0]["layer"] == NP_LAYER_ID
        assert Path(calls[0]["output_path"]) == jsonl_dest
        with open(out) as f:
            on_disk = json.load(f)
        assert on_disk["features"][0]["explanation"] == "label-0"


# ---------------------------------------------------------------------------
# L0-variant guard (RED until implemented)
#
# The real dataset was built from the average_l0_604 transcoder dictionary
# (metadata sidecar transcoder.l0_variant == 604), but Neuronpedia's DEPLOYED
# layer-12 gemmascope-transcoder-16k source is average_l0_6. Those are DIFFERENT
# dictionaries whose feature indices are not comparable — merging l0_6-indexed
# Neuronpedia explanations positionally onto l0_604 features silently mislabels
# them. The same hazard graph_fetch.validate_graph_identity hard-fails on.
#
# INTENDED CONTRACT (behavior, not plumbing):
#   - The guard engages ONLY when a metadata sidecar is discoverable next to the
#     dataset (the {stem}-metadata.json convention) AND its transcoder.l0_variant
#     mismatches the deployed variant. When no sidecar is present the merge must
#     proceed unchanged — the existing TestMergeExplanations / TestMainCLI cases
#     seed only the dataset and MUST stay green.
#   - The deployed variant (6 for gemma-2-2b layer-12 gemmascope-transcoder-16k)
#     is the documented reference the sidecar's 604 is checked against; how the
#     implementation sources it is its choice.
#   - On mismatch: refuse with an actionable ValueError naming BOTH variants
#     (604 and 6) and write nothing.
#   - An explicit override lets a caller proceed anyway. Anticipated flag name
#     `allow_l0_mismatch` (mirrors build_traced_circuit /
#     validate_graph_identity / the CLI's --allow-l0-mismatch vocabulary); the
#     override test below pins that name and will need reconciling if the
#     implementer chooses a different one.
# ---------------------------------------------------------------------------


class TestL0VariantGuard:
    pytestmark = pytest.mark.skipif(
        not METADATA_SIDECAR.exists(),
        reason="real metadata sidecar gemma-2-2b-layer12-l0604-metadata.json not present",
    )

    def _seed_dataset_with_sidecar(self, tmp_path: Path):
        dataset = tmp_path / DATASET_FILE.name
        shutil.copyfile(DATASET_FILE, dataset)
        sidecar = tmp_path / f"{DATASET_FILE.stem}-metadata.json"
        shutil.copyfile(METADATA_SIDECAR, sidecar)
        return dataset, sidecar

    def test_sidecar_precondition(self, tmp_path):
        _dataset, sidecar = self._seed_dataset_with_sidecar(tmp_path)
        meta = json.loads(sidecar.read_text())
        assert meta["transcoder"]["l0_variant"] == DATASET_L0_VARIANT

    def test_l0_mismatch_refuses_naming_both_variants(
        self, explanations_jsonl, tmp_path
    ):
        dataset, _sidecar = self._seed_dataset_with_sidecar(tmp_path)
        src_hash = _sha256(dataset)
        with pytest.raises(ValueError) as excinfo:
            merge_explanations(dataset, explanations_jsonl)
        msg = str(excinfo.value)
        # Must quote both sides: the dataset's 604 ...
        assert str(DATASET_L0_VARIANT) in msg
        # ... and the deployed 6 as a standalone number (not the 6 inside 604,
        # 16k, or 16384).
        assert re.search(r"(?<!\d)6(?!\d)", msg), msg
        # A refusal writes nothing — the in-place source is untouched.
        assert _sha256(dataset) == src_hash

    def test_l0_mismatch_allowed_with_override(
        self, explanations_jsonl, tmp_path
    ):
        dataset, _sidecar = self._seed_dataset_with_sidecar(tmp_path)
        out = tmp_path / "merged.json"
        result = merge_explanations(
            dataset, explanations_jsonl, out_path=out, allow_l0_mismatch=True
        )
        assert out.exists()
        assert result["features"][0]["explanation"] == "label-0"

    def test_no_sidecar_still_merges(self, explanations_jsonl, tmp_path):
        """Backward-compat: absent a sidecar, the merge proceeds unchanged (the
        guard must not fire on a dataset it cannot identify)."""
        dataset = tmp_path / DATASET_FILE.name
        shutil.copyfile(DATASET_FILE, dataset)
        out = tmp_path / "merged.json"
        result = merge_explanations(dataset, explanations_jsonl, out_path=out)
        assert out.exists()
        assert result["features"][0]["explanation"] == "label-0"
