#!/bin/bash
# Quick test: run the BYO model pipeline with Pythia-70M (smallest SAELens-supported model)
# ~150MB download vs ~2GB for GPT-2 Small — much faster for testing
#
# Run from the striatica/ directory:
#   cd pipeline && bash /path/to/test_byo_model.sh

set -e

poetry run striat model \
  --model pythia-70m-deduped \
  --layer 4-res-sm \
  --sae-release pythia-70m-deduped-res-sm \
  --sae-hook blocks.4.hook_resid_pre \
  --num-batches 4 \
  --features-per-batch 1024