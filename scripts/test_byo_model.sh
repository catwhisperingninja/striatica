#!/bin/bash
# Test the BYO model pipeline with a small SAELens-supported model.
#
# Usage:
#   bash scripts/test_byo_model.sh          # Quick test: Pythia-70M (~150MB, ~1 min)
#   bash scripts/test_byo_model.sh --gemma   # Full test: Gemma 2B (~2GB, ~15-30 min, GPU recommended)

set -e

# Ensure the package is installed (idempotent, fast if already done)
poetry install --extras ml --quiet

if [ "$1" = "--gemma" ]; then
  echo "Running Gemma 2B test (15 batches × 1024 = 15,360 features)..."
  poetry run striat model \
    --model gemma-2b \
    --layer 12-res-jb \
    --sae-release gemma-2b-res-jb \
    --sae-hook blocks.12.hook_resid_post \
    --num-batches 15 \
    --features-per-batch 1024 \
    --device auto
else
  echo "Running Pythia-70M quick test (4 batches × 1024 = 4,096 features)..."
  poetry run striat model \
    --model pythia-70m-deduped \
    --layer 4-res-sm \
    --sae-release pythia-70m-deduped-res-sm \
    --sae-hook blocks.3.hook_resid_post \
    --num-batches 4 \
    --features-per-batch 1024 \
    --device auto
fi
