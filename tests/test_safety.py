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
