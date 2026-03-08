#!/bin/bash
# Run the BYO model pipeline with a standard SAELens Gemma-2B preset.
#
# Run from the repo root:
#   bash scripts/test_byo_model.sh

set -e

# Ensure the package is installed (idempotent, fast if already done)
poetry install --extras ml --quiet

poetry run striat model \
  --model gemma-2b \
  --layer 12-res-jb \
  --sae-release gemma-2b-res-jb \
  --sae-hook blocks.12.hook_resid_post \
  --num-batches 15 \
  --features-per-batch 1024