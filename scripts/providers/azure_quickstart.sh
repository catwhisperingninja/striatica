#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# striatica Azure quick start
#
# Provisions an Azure VM, runs the pipeline, and downloads results.
# Requires: az CLI authenticated (az login)
#
# Usage:
#   # CPU instance for small/medium models
#   ./scripts/azure_quickstart.sh --np-id gpt2-small/8-res-jb
#
#   # GPU instance for large models
#   ./scripts/azure_quickstart.sh --gpu --np-id gemma-2-9b/20-gemmascope-res-16k
#
#   # Just provision the VM (manual processing)
#   ./scripts/azure_quickstart.sh --provision-only --gpu
#
#   # Tear down when done
#   ./scripts/azure_quickstart.sh --teardown
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Config ──
RG="striatica-preprocessing"
LOCATION="eastus"
VM_NAME="striatica-worker"
GPU=false
PROVISION_ONLY=false
TEARDOWN=false
NP_ID=""
NP_IDS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)              GPU=true; shift ;;
        --provision-only)   PROVISION_ONLY=true; shift ;;
        --teardown)         TEARDOWN=true; shift ;;
        --np-id)            NP_ID="$2"; shift 2 ;;
        --np-ids)           NP_IDS="$2"; shift 2 ;;
        --location)         LOCATION="$2"; shift 2 ;;
        --rg)               RG="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --np-id ID        Single Neuronpedia ID"
            echo "  --np-ids IDS      Comma-separated Neuronpedia IDs"
            echo "  --gpu             Use GPU VM (NC-series with A100/V100)"
            echo "  --provision-only  Just create the VM, don't run pipeline"
            echo "  --teardown        Delete the resource group and all resources"
            echo "  --location LOC    Azure region (default: eastus)"
            echo "  --rg NAME         Resource group name (default: striatica-preprocessing)"
            echo ""
            echo "Instance types used:"
            echo "  CPU: Standard_D8s_v5  (8 vCPU, 32 GB RAM, ~\$0.38/hr)"
            echo "  GPU: Standard_NC6s_v3 (1x V100 16GB, ~\$3.06/hr)"
            echo ""
            echo "IMPORTANT: Run --teardown when done to avoid charges!"
            exit 0
            ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# ── Teardown ──
if [ "$TEARDOWN" = true ]; then
    echo "  Deleting resource group $RG and all resources..."
    az group delete --name "$RG" --yes --no-wait
    echo "  ✅  Teardown initiated (runs in background)"
    exit 0
fi

# ── Validate ──
if ! command -v az &>/dev/null; then
    echo "Error: Azure CLI (az) not found. Install: https://aka.ms/installazurecli"
    exit 1
fi

if [ "$PROVISION_ONLY" = false ] && [[ -z "$NP_ID" && -z "$NP_IDS" ]]; then
    echo "Error: provide --np-id, --np-ids, or --provision-only"
    exit 1
fi

# ── Select VM size ──
if [ "$GPU" = true ]; then
    VM_SIZE="Standard_NC6s_v3"
    IMAGE="Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest"
    echo "  GPU mode: $VM_SIZE (1x V100 16GB, ~\$3.06/hr)"
else
    VM_SIZE="Standard_D8s_v5"
    IMAGE="Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest"
    echo "  CPU mode: $VM_SIZE (8 vCPU, 32 GB RAM, ~\$0.38/hr)"
fi

# ── Provision ──
echo "  Creating resource group: $RG in $LOCATION"
az group create --name "$RG" --location "$LOCATION" --output none

echo "  Creating VM: $VM_NAME ($VM_SIZE)..."
VM_IP=$(az vm create \
    --resource-group "$RG" \
    --name "$VM_NAME" \
    --size "$VM_SIZE" \
    --image "$IMAGE" \
    --admin-username striatica \
    --generate-ssh-keys \
    --public-ip-sku Standard \
    --output tsv \
    --query publicIpAddress)

echo "  ✅  VM ready at: $VM_IP"

if [ "$PROVISION_ONLY" = true ]; then
    echo ""
    echo "  SSH in:  ssh striatica@$VM_IP"
    echo "  Then run:"
    echo "    curl -sSL https://raw.githubusercontent.com/catwhisperingninja/striatica/main/scripts/lambda_quickstart.sh | bash -s -- --np-id <your-id>"
    echo ""
    echo "  IMPORTANT: Run '$0 --teardown' when done to avoid charges!"
    exit 0
fi

# ── Setup and run on VM ──
echo "  Setting up striatica on VM..."

# Build the processing command
if [[ -n "$NP_ID" && -z "$NP_IDS" ]]; then
    STRIAT_CMD="model --np-id $NP_ID"
elif [[ -n "$NP_IDS" ]]; then
    STRIAT_CMD="batch --np-ids '$NP_IDS' --continue-on-error"
fi

ssh -o StrictHostKeyChecking=no "striatica@$VM_IP" bash -s <<REMOTE_SCRIPT
set -euo pipefail

# Install system deps
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip git

# GPU drivers (if GPU instance)
if command -v nvidia-smi &>/dev/null; then
    echo "GPU drivers already present"
elif [ "$GPU" = true ]; then
    echo "Installing CUDA drivers..."
    sudo apt-get install -y -qq nvidia-driver-535
    sudo modprobe nvidia || true
fi

# Clone and setup
git clone https://github.com/catwhisperingninja/striatica.git
cd striatica
pip install --quiet poetry
poetry install --extras ml --quiet

# Detect device
DEVICE="cpu"
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda"
fi

# Run pipeline
mkdir -p output
poetry run striat $STRIAT_CMD --device \$DEVICE --json-export
REMOTE_SCRIPT

# ── Download results ──
echo "  Downloading results..."
mkdir -p output
scp -o StrictHostKeyChecking=no "striatica@$VM_IP:~/striatica/frontend/public/data/*.json" ./output/

echo ""
echo "  ✅  Done! Results in: ./output/"
ls -lh ./output/*.json 2>/dev/null | sed 's/^/    /'
echo ""
echo "  ⚠  IMPORTANT: Run '$0 --teardown' to delete the VM and stop charges!"
echo ""
