#!/bin/bash
# Quick test: run the BYO model pipeline with Pythia-70M (smallest SAELens-supported model)
# ~150MB download vs ~2GB for GPT-2 Small — much faster for testing
#
# Run from the repo root:
#   bash scripts/test_byo_model.sh

set -e

# Ensure the package is installed (idempotent, fast if already done)
poetry install --extras ml --quiet

poetry run striat model \
  --model pythia-70m-deduped \
  --layer 4-res-sm \
  --sae-release pythia-70m-deduped-res-sm \
  --sae-hook blocks.3.hook_resid_post \
  --num-batches 4 \
  --features-per-batch 1024