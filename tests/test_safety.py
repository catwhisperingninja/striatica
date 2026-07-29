# striatica/tests/test_safety.py
"""P0 safety tests: model tier classification, semantic redaction, and pipeline gating."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.config import PUBLIC_TIER_MODELS, is_public_tier, SAEConfig
from pipeline.prepare import prepare_json


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def small_dataset(tmp_path: Path):
    """Create minimal features + explanations JSONL files for testing."""
    n = 20
    coords = np.random.default_rng(42).standard_normal((n, 3)).astype(np.float32)
    labels = np.array([0] * 8 + [1] * 8 + [-1] * 4)

    features_path = tmp_path / "features.jsonl"
    explanations_path = tmp_path / "explanations.jsonl"

    with open(features_path, "w") as f:
        for i in range(n):
            json.dump({
                "index": i,
                "maxActApprox": float(n - i),
                "frac_nonzero": 0.01,
                "topkCosSimIndices": list(range(min(i + 1, n), min(i + 6, n))),
                "pos_str": [f"token_{i}"],
                "neg_str": [],
            }, f)
            f.write("\n")

    with open(explanations_path, "w") as f:
        for i in range(n):
            json.dump({
                "index": i,
                "description": f"Feature {i} detects pattern related to concept_{i}",
            }, f)
            f.write("\n")

    return {
        "coords": coords,
        "labels": labels,
        "features_path": features_path,
        "explanations_path": explanations_path,
        "n": n,
    }


# ── Model Tier Classification ───────────────────────────────────────────

class TestModelTierClassification:
    """PUBLIC_TIER_MODELS and is_public_tier() gate semantic label inclusion."""

    def test_gpt2_small_is_public(self):
        assert is_public_tier("gpt2-small")

    def test_gpt2_alias_is_public(self):
        assert is_public_tier("gpt2")

    def test_pythia_70m_is_public(self):
        assert is_public_tier("pythia-70m")

    def test_pythia_70m_deduped_is_public(self):
        assert is_public_tier("pythia-70m-deduped")

    def test_gemma_2b_is_restricted(self):
        assert not is_public_tier("gemma-2b")

    def test_llama_3_8b_is_restricted(self):
        assert not is_public_tier("llama-3-8b")

    def test_mistral_7b_is_restricted(self):
        assert not is_public_tier("mistral-7b")

    def test_claude_is_restricted(self):
        """Any hypothetical Claude SAE should absolutely be restricted."""
        assert not is_public_tier("claude-3-haiku")

    def test_case_insensitive(self):
        """Model IDs should be matched case-insensitively."""
        assert is_public_tier("GPT2-Small")
        assert is_public_tier("GPT2-SMALL")
        assert not is_public_tier("Gemma-2B")

    def test_whitespace_trimmed(self):
        """Leading/trailing whitespace should be stripped."""
        assert is_public_tier("  gpt2-small  ")
        assert not is_public_tier("  gemma-2b  ")

    def test_empty_string_is_restricted(self):
        assert not is_public_tier("")

    def test_public_tier_is_frozen(self):
        """PUBLIC_TIER_MODELS must be immutable to prevent runtime tampering."""
        assert isinstance(PUBLIC_TIER_MODELS, frozenset)
        with pytest.raises(AttributeError):
            PUBLIC_TIER_MODELS.add("hacked-model")  # type: ignore[attr-defined]


# ── Semantic Redaction in prepare_json ──────────────────────────────────

class TestSemanticRedaction:
    """prepare_json with redact_semantics=True strips all explanations."""

    def test_redacted_output_has_no_explanations(self, small_dataset, tmp_path):
        """When redacted, every feature.explanation must be empty string."""
        d = small_dataset
        output = tmp_path / "redacted.json"
        result = prepare_json(
            d["coords"], d["labels"], d["features_path"], d["explanations_path"],
            output, model="gemma-2b", layer="test", redact_semantics=True,
        )
        for feature in result["features"]:
            assert feature["explanation"] == "", (
                f"Feature {feature['index']} has explanation despite redaction: "
                f"{feature['explanation']!r}"
            )

    def test_redacted_output_sets_flag(self, small_dataset, tmp_path):
        """semanticsRedacted must be True in output JSON."""
        d = small_dataset
        output = tmp_path / "redacted.json"
        result = prepare_json(
            d["coords"], d["labels"], d["features_path"], d["explanations_path"],
            output, model="gemma-2b", layer="test", redact_semantics=True,
        )
        assert result["semanticsRedacted"] is True

    def test_non_redacted_output_has_explanations(self, small_dataset, tmp_path):
        """When not redacted, explanations must be present."""
        d = small_dataset
        output = tmp_path / "public.json"
        result = prepare_json(
            d["coords"], d["labels"], d["features_path"], d["explanations_path"],
            output, model="gpt2-small", layer="test", redact_semantics=False,
        )
        non_empty = [f for f in result["features"] if f["explanation"]]
        assert len(non_empty) == d["n"], "All features should have explanations"

    def test_non_redacted_output_flag_false(self, small_dataset, tmp_path):
        """semanticsRedacted must be False when not redacting."""
        d = small_dataset
        output = tmp_path / "public.json"
        result = prepare_json(
            d["coords"], d["labels"], d["features_path"], d["explanations_path"],
            output, model="gpt2-small", layer="test", redact_semantics=False,
        )
        assert result["semanticsRedacted"] is False

    def test_redacted_preserves_geometry(self, small_dataset, tmp_path):
        """Redaction must NOT affect positions, clusters, or feature metadata."""
        d = small_dataset
        out_public = tmp_path / "public.json"
        out_redacted = tmp_path / "redacted.json"

        r_pub = prepare_json(
            d["coords"], d["labels"], d["features_path"], d["explanations_path"],
            out_public, model="test", layer="test", redact_semantics=False,
        )
        r_red = prepare_json(
            d["coords"], d["labels"], d["features_path"], d["explanations_path"],
            out_redacted, model="test", layer="test", redact_semantics=True,
        )

        # Positions identical
        assert r_pub["positions"] == r_red["positions"]
        # Cluster labels identical
        assert r_pub["clusterLabels"] == r_red["clusterLabels"]
        # Feature count identical
        assert r_pub["numFeatures"] == r_red["numFeatures"]
        # Non-semantic feature metadata identical
        for fp, fr in zip(r_pub["features"], r_red["features"]):
            assert fp["index"] == fr["index"]
            assert fp["maxAct"] == fr["maxAct"]
            assert fp["fracNonzero"] == fr["fracNonzero"]
            assert fp["topSimilar"] == fr["topSimilar"]
            assert fp["posTokens"] == fr["posTokens"]

    def test_redacted_json_on_disk_matches(self, small_dataset, tmp_path):
        """The JSON file on disk must also reflect redaction."""
        d = small_dataset
        output = tmp_path / "disk_check.json"
        prepare_json(
            d["coords"], d["labels"], d["features_path"], d["explanations_path"],
            output, model="gemma-2b", layer="test", redact_semantics=True,
        )
        with open(output) as f:
            disk_data = json.load(f)
        assert disk_data["semanticsRedacted"] is True
        for feature in disk_data["features"]:
            assert feature["explanation"] == ""

    def test_default_is_no_redaction(self, small_dataset, tmp_path):
        """prepare_json defaults to redact_semantics=False (backward compat)."""
        d = small_dataset
        output = tmp_path / "default.json"
        result = prepare_json(
            d["coords"], d["labels"], d["features_path"], d["explanations_path"],
            output, model="gpt2-small", layer="test",
            # redact_semantics NOT passed — should default to False
        )
        assert result["semanticsRedacted"] is False
        non_empty = [f for f in result["features"] if f["explanation"]]
        assert len(non_empty) == d["n"]


# ── Pipeline Gate Integration ───────────────────────────────────────────

class TestPipelineGate:
    """The CLI correctly gates semantic output based on model tier."""

    def test_cmd_model_redacts_restricted_model(self):
        """cmd_model should set redact=True for non-public-tier models."""
        from pipeline.config import is_public_tier
        # Simulate the logic from cmd_model
        model_id = "gemma-2b"
        include_semantics_flag = False  # user did NOT pass --include-semantics
        public = is_public_tier(model_id)
        include_semantics = include_semantics_flag if include_semantics_flag else public
        redact = not include_semantics
        assert redact is True

    def test_cmd_model_allows_public_model(self):
        """cmd_model should set redact=False for public-tier models."""
        from pipeline.config import is_public_tier
        model_id = "gpt2-small"
        include_semantics_flag = False
        public = is_public_tier(model_id)
        include_semantics = include_semantics_flag if include_semantics_flag else public
        redact = not include_semantics
        assert redact is False

    def test_include_semantics_overrides_restriction(self):
        """--include-semantics should override redaction for restricted models."""
        from pipeline.config import is_public_tier
        model_id = "gemma-2b"
        include_semantics_flag = True  # user passed --include-semantics
        public = is_public_tier(model_id)
        include_semantics = include_semantics_flag if include_semantics_flag else public
        redact = not include_semantics
        assert redact is False

    def test_demo_path_never_redacts(self):
        """striat demo uses GPT2_SMALL_L6 which is always public tier."""
        from pipeline.config import GPT2_SMALL_L6, is_public_tier
        assert is_public_tier(GPT2_SMALL_L6.model_id)


# ── Gemma 2 2B Public Tier (semantics_merge unit, 2026-07-28) ───────────


class TestGemma22bPublicTier:
    """gemma-2-2b is PUBLIC tier: Gemma Scope interpretability data is
    deliberately public and Neuronpedia hosts the full explanations. The
    exact-match rule means every OTHER Gemma id stays restricted."""

    def test_gemma_2_2b_is_public(self):
        assert is_public_tier("gemma-2-2b")

    def test_gemma_2_2b_case_insensitive(self):
        assert is_public_tier("Gemma-2-2B")
        assert is_public_tier("  gemma-2-2b  ")

    def test_gemma_2b_sae_era_id_still_restricted(self):
        """'gemma-2b' (Gemma 1 2B) is a DIFFERENT model — still restricted."""
        assert not is_public_tier("gemma-2b")

    def test_other_gemma_variants_still_restricted(self):
        assert not is_public_tier("gemma-2-9b")
        assert not is_public_tier("gemma-2-2b-it")
        assert not is_public_tier("gemma-7b")


# ── Fixture Secrecy Sweep (semantics_merge unit, 2026-07-28) ────────────
#
# Committed test fixtures under tests/fixtures/ derive from REAL Neuronpedia
# payloads with all free text neutralized at capture time: supernode labels
# became "label-N", prompts became "prompt-redacted", prompt tokens became
# "t0".."tN", and clerp was blanked. This sweep fails if any fixture (current
# or future) carries free text beyond those patterns — i.e. if semantic label
# text ever leaks into a committed fixture.

import re

FIXTURES_DIR = Path(__file__).parent / "fixtures"

_NEUTRAL_PATTERNS = (
    re.compile(r"^label-\d+([-_][a-z0-9]+)*$"),   # neutralized labels/descriptions
    re.compile(r"^prompt-redacted$"),             # neutralized prompts
    re.compile(r"^t\d+$"),                        # neutralized prompt tokens
)

# Exact catalog/enum strings that legitimately contain whitespace. Anything
# else with whitespace is treated as forbidden free text.
_ALLOWED_PHRASES = frozenset({
    "Per Layer Transcoders",         # sourceset catalog description
    "Google DeepMind",               # sourceset creatorName
    "Hanna and Piotrowski",          # graph metadata.info.creator_name
    "cross layer transcoder",        # feature_type enum
    "mlp reconstruction error",      # feature_type enum
})

# Keys whose string values carry semantic text in raw Neuronpedia payloads.
_SEMANTIC_KEYS = frozenset({
    "clerp", "ppClerp", "explanation", "prompt", "description", "title", "label",
})


def _is_neutral(value: str) -> bool:
    return value == "" or any(p.match(value) for p in _NEUTRAL_PATTERNS)


def fixture_secrecy_violations(doc, path: str = "") -> list[str]:
    """Return a list of 'path: reason' strings for free-text leaks in doc."""
    violations: list[str] = []

    def walk(obj, path: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{path}.{k}" if path else str(k)
                if k in _SEMANTIC_KEYS and isinstance(v, str):
                    if not (_is_neutral(v) or v in _ALLOWED_PHRASES):
                        violations.append(f"{p}: semantic key holds free text: {v!r}")
                if k == "supernodes" and isinstance(v, list):
                    for i, sn in enumerate(v):
                        if isinstance(sn, list) and sn and isinstance(sn[0], str):
                            if not _is_neutral(sn[0]):
                                violations.append(
                                    f"{p}[{i}][0]: supernode label not neutralized: {sn[0]!r}"
                                )
                walk(v, p)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")
        elif isinstance(obj, str):
            if any(ch.isspace() for ch in obj):
                if not (_is_neutral(obj) or obj in _ALLOWED_PHRASES):
                    violations.append(f"{path}: multi-word free text: {obj!r}")

    walk(doc, path)
    return violations


def _iter_fixture_docs():
    """Yield (label, parsed_doc) for every JSON/JSONL fixture file."""
    for fp in sorted(FIXTURES_DIR.glob("*.json")):
        with open(fp) as f:
            yield fp.name, json.load(f)
    for fp in sorted(FIXTURES_DIR.glob("*.jsonl")):
        with open(fp) as f:
            for lineno, line in enumerate(f, 1):
                if line.strip():
                    yield f"{fp.name}:{lineno}", json.loads(line)


class TestFixtureSecrecy:
    """No committed fixture may contain semantic free text."""

    def test_known_fixtures_present(self):
        """Guard: the sweep is not vacuous — the real fixtures exist."""
        names = {p.name for p in FIXTURES_DIR.glob("*.json")}
        assert {
            "neuronpedia_graph_gemma.json",
            "neuronpedia_graph_record_gemma.json",
            "neuronpedia_sourceset_gemma.json",
        } <= names

    def test_all_fixture_files_are_neutralized(self):
        all_violations = []
        for label, doc in _iter_fixture_docs():
            for v in fixture_secrecy_violations(doc):
                all_violations.append(f"{label} :: {v}")
        assert not all_violations, (
            "Semantic free text leaked into committed fixtures:\n  "
            + "\n  ".join(all_violations)
        )

    def test_sweep_detects_leaked_clerp(self):
        """The sweep itself must catch a raw (non-neutralized) clerp."""
        leaked = {"nodes": [{"node_id": "12_1_1", "clerp": "mentions of Texas cities"}]}
        assert fixture_secrecy_violations(leaked)

    def test_sweep_detects_leaked_prompt_and_supernode_label(self):
        leaked = {
            "metadata": {"prompt": "Fact: the capital of the state"},
            "qParams": {"supernodes": [["capital", "12_1_1"]]},
        }
        found = fixture_secrecy_violations(leaked)
        assert any("prompt" in v for v in found)
        assert any("supernode" in v for v in found)

    def test_sweep_detects_free_text_in_nested_list(self):
        leaked = {"prompt_tokens": ["t0", "some raw prompt text"]}
        assert fixture_secrecy_violations(leaked)

    def test_sweep_accepts_neutralized_document(self):
        clean = {
            "metadata": {"prompt": "prompt-redacted", "prompt_tokens": ["t0", "t1"]},
            "qParams": {"supernodes": [["label-3", "12_1_1"]]},
            "nodes": [{"clerp": "", "feature_type": "cross layer transcoder"}],
        }
        assert fixture_secrecy_violations(clean) == []
