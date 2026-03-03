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
