"""Configuration for the data pipeline."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "frontend" / "public" / "data"

S3_BASE = "https://neuronpedia-datasets.s3.amazonaws.com"

@dataclass(frozen=True)
class SAEConfig:
    model_id: str
    layer: str          # e.g. "6-res-jb"
    sae_release: str    # SAELens release name
    sae_hook: str       # SAELens hook point
    num_batches: int    # S3 batch file count
    features_per_batch: int

GPT2_SMALL_L6 = SAEConfig(
    model_id="gpt2-small",
    layer="6-res-jb",
    sae_release="gpt2-small-res-jb",
    sae_hook="blocks.6.hook_resid_pre",
    num_batches=24,
    features_per_batch=1024,
)


# ── Safety: Model Tier Classification ──────────────────────────────────
#
# Models in PUBLIC_TIER have semantic labels (feature explanations) included
# in pipeline output. These are small, well-studied models whose safety
# circuits are either non-existent or already public knowledge.
#
# All other models are RESTRICTED by default: semantic labels are stripped
# from the pipeline output to prevent safety-relevant feature interpretations
# from being generated and potentially published or ingested into training
# data. Override with --include-semantics if you understand the implications.
#
# See: containment-architecture-20260309.md, Invention 5 (Redaction Engine)

PUBLIC_TIER_MODELS = frozenset({
    "gpt2-small",    # 117M params, 2019 — paper model, fully public
    "gpt2",          # alias
    "pythia-70m",    # 70M params — too small for meaningful safety circuits
    "pythia-70m-deduped",
})


def is_public_tier(model_id: str) -> bool:
    """Check if a model is in the public tier (safe for full semantic output)."""
    return model_id.lower().strip() in PUBLIC_TIER_MODELS
