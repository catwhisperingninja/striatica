#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# striatica Lambda AI quick start
#
# Run this on a fresh Lambda AI GPU instance to process models.
# Assumes: Ubuntu, NVIDIA GPU, CUDA drivers pre-installed (Lambda default).
#
# Usage:
#   # SSH into your Lambda instance, then:
#   curl -sSL https://raw.githubusercontent.com/catwhisperingninja/striatica/main/scripts/lambda_quickstart.sh | bash -s -- --np-id gemma-2-2b/12-gemmascope-res-16k
#
#   # Or clone first and run locally:
#   ./scripts/lambda_quickstart.sh --np-id gemma-2-2b/12-gemmascope-res-16k
#   ./scripts/lambda_quickstart.sh --np-ids "gpt2-small/6-res-jb,gemma-2-2b/12-gemmascope-res-16k"
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

NP_ID=""
NP_IDS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --np-id)   NP_ID="$2"; shift 2 ;;
        --np-ids)  NP_IDS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --np-id <id> | --np-ids <id1,id2,...>"
            echo ""
            echo "Lambda AI quick start for striatica preprocessing."
            echo "Installs everything, runs the pipeline on GPU, outputs to ./output/"
            exit 0
            ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [[ -z "$NP_ID" && -z "$NP_IDS" ]]; then
    echo "Provide --np-id or --np-ids"
    exit 1
fi

echo ""
echo "  ░▒▓  s t r i a t i c a  ≡≡≡≡≡  ▓▒░"
echo "  Lambda AI Quick Start"
echo ""

# ── Check GPU ──
if command -v nvidia-smi &>/dev/null; then
    echo "  GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/    /'
    echo ""
else
    echo "  ⚠  No GPU detected. Running on CPU (will be slower)."
    echo ""
fi

# ── Clone or update repo ──
if [ -d "striatica" ]; then
    echo "  Updating striatica repo..."
    cd striatica
    git pull --rebase
else
    echo "  Cloning striatica..."
    git clone https://github.com/catwhisperingninja/striatica.git
    cd striatica
fi

# ── Install Poetry + dependencies ──
echo "  Installing dependencies..."
if ! command -v poetry &>/dev/null; then
    pip install --quiet poetry
fi
poetry install --extras ml --quiet

# ── Create output dir ──
mkdir -p output

# ── Detect device ──
DEVICE="cpu"
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda"
    echo "  Using CUDA"
else
    echo "  Using CPU (no CUDA available)"
fi

# ── Run pipeline ──
echo ""
if [[ -n "$NP_ID" && -z "$NP_IDS" ]]; then
    echo "  Processing: $NP_ID"
    echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    poetry run striat model --np-id "$NP_ID" --device "$DEVICE" --json-export
elif [[ -n "$NP_IDS" ]]; then
    echo "  Batch processing: $NP_IDS"
    echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    poetry run striat batch --np-ids "$NP_IDS" --device "$DEVICE" --continue-on-error
fi

echo ""
echo "  ✅  Done!"
echo "  Output files:"
ls -lh frontend/public/data/*.json 2>/dev/null | sed 's/^/    /' || echo "    (check frontend/public/data/)"
echo ""
echo "  To download results to your local machine:"
echo "    scp -r lambda-instance:~/striatica/frontend/public/data/*.json ./output/"
echo ""
