#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# striatica Lambda AI quick start (Docker)
#
# Run this on a fresh Lambda AI GPU instance to process models.
# Assumes: Docker + NVIDIA Container Toolkit pre-installed (Lambda default).
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
            echo "Builds Docker container, runs the pipeline on GPU, outputs to ./output/"
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
echo "  Lambda AI Quick Start (Docker)"
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

# ── Build Docker image ──
echo "  Building Docker image (this takes a few minutes the first time)..."
docker build -f Dockerfile.gpu -t striatica-gpu .

# ── Create output dir ──
mkdir -p output

# ── Run pipeline ──
echo ""
if [[ -n "$NP_ID" && -z "$NP_IDS" ]]; then
    echo "  Processing: $NP_ID"
    echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    docker run --gpus all -t --name striatica-run --rm \
        -v "$(pwd)/output:/app/output" \
        striatica-gpu \
        model --np-id "$NP_ID" --device cuda --json-export
elif [[ -n "$NP_IDS" ]]; then
    echo "  Batch processing: $NP_IDS"
    echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    docker run --gpus all -t --name striatica-run --rm \
        -v "$(pwd)/output:/app/output" \
        striatica-gpu \
        batch --np-ids "$NP_IDS" --device cuda --continue-on-error
fi

echo ""
echo "  ✅  Done!"
echo "  Output files:"
ls -lh output/*.json 2>/dev/null | sed 's/^/    /' || echo "    (check ./output/)"
echo ""
echo "  To download results to your local machine:"
echo "    scp -r lambda-instance:~/striatica/output/*.json ./output/"
echo ""
