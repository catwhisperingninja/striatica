#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# striatica cloud preprocessing launcher
#
# Runs the striatica pipeline on cloud instances. Handles Docker build,
# model selection, and output collection.
#
# Usage:
#   # Process a single model
#   ./scripts/cloud_preprocess.sh --np-id gemma-2-2b/12-gemmascope-res-16k
#
#   # Process multiple models
#   ./scripts/cloud_preprocess.sh --np-ids "gpt2-small/6-res-jb,gpt2-small/8-res-jb"
#
#   # GPU mode (for Lambda AI, Azure GPU, etc.)
#   ./scripts/cloud_preprocess.sh --gpu --np-id gemma-2-9b/20-gemmascope-res-16k
#
#   # Just show what you'd need (dry run)
#   ./scripts/cloud_preprocess.sh --plan --np-id gemma-2-2b/12-gemmascope-res-16k
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Defaults ──
GPU=false
PLAN_ONLY=false
NP_ID=""
NP_IDS=""
DEVICE="cpu"
OUTPUT_DIR="./output"
FORCE=false
CONTINUE_ON_ERROR=true

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)          GPU=true; DEVICE="cuda"; shift ;;
        --plan)         PLAN_ONLY=true; shift ;;
        --np-id)        NP_ID="$2"; shift 2 ;;
        --np-ids)       NP_IDS="$2"; shift 2 ;;
        --output)       OUTPUT_DIR="$2"; shift 2 ;;
        --force)        FORCE=true; shift ;;
        --no-continue)  CONTINUE_ON_ERROR=false; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --np-id ID          Single Neuronpedia ID (e.g. gpt2-small/6-res-jb)"
            echo "  --np-ids IDS        Comma-separated Neuronpedia IDs"
            echo "  --gpu               Use GPU Dockerfile and cuda device"
            echo "  --plan              Dry run: show instance recommendations only"
            echo "  --output DIR        Output directory (default: ./output)"
            echo "  --force             Reprocess even if output exists"
            echo "  --no-continue       Stop on first error"
            echo "  -h, --help          Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ── Validate ──
if [[ -z "$NP_ID" && -z "$NP_IDS" ]]; then
    echo "Error: provide --np-id or --np-ids"
    echo "Run with --help for usage"
    exit 1
fi

# Build the combined NP_IDS string
if [[ -n "$NP_ID" && -z "$NP_IDS" ]]; then
    NP_IDS="$NP_ID"
fi

# ── Instance Recommendations ──
print_recommendations() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  striatica — Cloud Instance Recommendations"
    echo "═══════════════════════════════════════════════════════"
    echo ""
    echo "  Models to process: $NP_IDS"
    echo "  Mode: $([ "$GPU" = true ] && echo "GPU" || echo "CPU")"
    echo ""

    if [ "$GPU" = true ]; then
        echo "  ┌─────────────────────────────────────────────────┐"
        echo "  │  GPU Instances                                  │"
        echo "  ├─────────────────────────────────────────────────┤"
        echo "  │                                                 │"
        echo "  │  Lambda AI (recommended for this project):      │"
        echo "  │    1x A100 80GB  — \$1.29/hr                    │"
        echo "  │    Good for: all models up to 27B               │"
        echo "  │    1x H100 80GB  — \$2.49/hr                    │"
        echo "  │    Good for: 70B models, fastest processing     │"
        echo "  │                                                 │"
        echo "  │  Azure:                                         │"
        echo "  │    Standard_NC24ads_A100_v4 (1x A100 80GB)      │"
        echo "  │    ~\$3.67/hr (pay-as-you-go)                   │"
        echo "  │    Standard_NC6s_v3 (1x V100 16GB)              │"
        echo "  │    ~\$3.06/hr — sufficient for models ≤2B       │"
        echo "  │                                                 │"
        echo "  │  AWS:                                           │"
        echo "  │    p4d.24xlarge (8x A100) — overkill but fast   │"
        echo "  │    g5.xlarge (1x A10G 24GB) — budget GPU option │"
        echo "  │                                                 │"
        echo "  └─────────────────────────────────────────────────┘"
    else
        echo "  ┌─────────────────────────────────────────────────┐"
        echo "  │  CPU Instances                                  │"
        echo "  ├─────────────────────────────────────────────────┤"
        echo "  │                                                 │"
        echo "  │  Small models (<32K features):                  │"
        echo "  │    4 vCPU, 16 GB RAM — ~\$0.20/hr              │"
        echo "  │    Azure: Standard_D4s_v5                       │"
        echo "  │    AWS:   m6i.xlarge                            │"
        echo "  │    Est. runtime: 30-60 min per model            │"
        echo "  │                                                 │"
        echo "  │  Medium models (32-65K features):               │"
        echo "  │    8 vCPU, 32 GB RAM — ~\$0.40/hr              │"
        echo "  │    Azure: Standard_D8s_v5                       │"
        echo "  │    AWS:   m6i.2xlarge                           │"
        echo "  │    Est. runtime: 1-3 hrs per model              │"
        echo "  │                                                 │"
        echo "  │  Large models (65K+ features):                  │"
        echo "  │    16 vCPU, 64 GB RAM — ~\$0.80/hr             │"
        echo "  │    Azure: Standard_D16s_v5                      │"
        echo "  │    AWS:   m6i.4xlarge                           │"
        echo "  │    Est. runtime: 2-6 hrs per model              │"
        echo "  │    Consider GPU instead for these               │"
        echo "  │                                                 │"
        echo "  └─────────────────────────────────────────────────┘"
    fi

    echo ""
    echo "  Estimated storage: ~25 MB per model (dataset + circuits + metadata)"
    echo "  Network: ~2 GB download per model from Neuronpedia S3"
    echo ""
    echo "  Quick start on your instance:"
    echo "    git clone https://github.com/catwhisperingninja/striatica.git"
    echo "    cd striatica"
    echo "    pip install poetry && poetry install --extras ml"
    echo "    poetry run striat model --np-id <your-model-id> --device $DEVICE"
    echo ""
    echo "  Or with Docker:"
    if [ "$GPU" = true ]; then
        echo "    docker build -f Dockerfile.gpu -t striatica-gpu ."
        echo "    docker run --gpus all -v \$(pwd)/output:/app/output striatica-gpu \\"
        echo "      model --np-id <your-model-id> --device cuda"
    else
        echo "    docker build -t striatica ."
        echo "    docker run -v \$(pwd)/output:/app/output striatica \\"
        echo "      model --np-id <your-model-id>"
    fi
    echo ""
}

# ── Plan mode: just show recommendations ──
if [ "$PLAN_ONLY" = true ]; then
    print_recommendations
    exit 0
fi

# ── Build Docker image ──
echo ""
echo "  Building Docker image..."
if [ "$GPU" = true ]; then
    DOCKERFILE="Dockerfile.gpu"
    IMAGE="striatica-pipeline-gpu"
else
    DOCKERFILE="Dockerfile"
    IMAGE="striatica-pipeline"
fi

docker build -f "$DOCKERFILE" -t "$IMAGE" .

# ── Create output directory ──
mkdir -p "$OUTPUT_DIR"

# ── Run pipeline ──
echo ""
echo "  Starting pipeline..."
echo "  Models: $NP_IDS"
echo "  Device: $DEVICE"
echo "  Output: $OUTPUT_DIR"
echo ""

DOCKER_ARGS="-v $(pwd)/$OUTPUT_DIR:/app/output"
if [ "$GPU" = true ]; then
    DOCKER_ARGS="--gpus all $DOCKER_ARGS"
fi

# Determine if single or batch
if [[ "$NP_IDS" == *","* ]]; then
    # Multiple models — use batch command
    CMD="batch --np-ids '$NP_IDS' --device $DEVICE"
    if [ "$FORCE" = true ]; then CMD="$CMD --force"; fi
    if [ "$CONTINUE_ON_ERROR" = true ]; then CMD="$CMD --continue-on-error"; fi
else
    # Single model
    CMD="model --np-id $NP_IDS --device $DEVICE --json-export"
fi

eval "docker run $DOCKER_ARGS $IMAGE $CMD"

echo ""
echo "  ✅  Done! Output in: $OUTPUT_DIR"
echo ""
