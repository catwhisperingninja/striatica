#!/usr/bin/env bash
# vast_setup.sh — Vast.ai A100 instance bootstrap for striatica pipeline
# Run directly on the instance: bash vast_setup.sh
# Does NOT use Docker. Installs Python 3.12 via conda, then Poetry + deps.
set -euo pipefail

echo "=== striatica Vast.ai setup ==="
echo "Instance: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no nvidia-smi')"
echo ""

# ---------------------------------------------------------------
# 1. Python 3.12 via conda (Vast base image ships 3.10)
# ---------------------------------------------------------------
echo "[1/5] Installing Python 3.12 via conda..."
conda install -y python=3.12 -c conda-forge 2>&1 | tail -5
python --version  # should say 3.12.x


# ---------------------------------------------------------------
# 2. Poetry
# ---------------------------------------------------------------
echo "[2/5] Installing Poetry..."
pip install --quiet poetry
poetry --version

# ---------------------------------------------------------------
# 3. Clone repo and install deps
# ---------------------------------------------------------------
echo "[3/5] Cloning striatica and installing dependencies..."
cd /root
if [ -d "striatica" ]; then
    echo "  striatica/ already exists, pulling latest..."
    cd striatica && git pull && cd ..
else
    git clone https://github.com/catwhisperingninja/striatica.git
fi

cd /root/striatica

# Install all deps including ML extras
# CRITICAL: Do NOT run `poetry lock`. Use the committed lockfile as-is.
# Re-locking pulls newer dependency versions which silently changes UMAP
# output even with the same random_state.
poetry config virtualenvs.create false
poetry install --extras ml --no-interaction

# ---------------------------------------------------------------
# 4. Verify GPU access from Python
# ---------------------------------------------------------------
echo "[4/5] Verifying GPU access..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# ---------------------------------------------------------------
# 5. Download layer 12 transcoder weights (verify key names)
# ---------------------------------------------------------------
echo "[5/5] Downloading layer 12 transcoder weights to verify key names..."
python -c "
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

# Try a small download first to verify the key structure
# We don't know the exact L0 variant yet, so let's list what's available
from huggingface_hub import list_repo_tree

print('  Available L0 variants for layer 12, width_16k:')
entries = list(list_repo_tree('google/gemma-scope-2b-pt-transcoders', path_in_repo='layer_12/width_16k'))
for e in entries:
    if hasattr(e, 'path'):
        variant = e.path.split('/')[-1] if '/' in e.path else e.path
        print(f'    {variant}')

# Download the first available variant to check keys
first_variant = [e for e in list(list_repo_tree('google/gemma-scope-2b-pt-transcoders', path_in_repo='layer_12/width_16k')) if hasattr(e, 'path') and 'average_l0' in e.path]
if first_variant:
    variant_path = first_variant[0].path
    print(f'  Downloading {variant_path}/params.safetensors to check keys...')
    local = hf_hub_download('google/gemma-scope-2b-pt-transcoders', filename=f'{variant_path}/params.safetensors')
    weights = load_file(local)
    print(f'  Keys in safetensors file: {list(weights.keys())}')
    for k, v in weights.items():
        print(f'    {k}: shape={v.shape}, dtype={v.dtype}')
else:
    print('  WARNING: Could not find any L0 variants. Check repo structure.')
"

echo ""
echo "=== Setup complete ==="
echo "cd /root/striatica && python -m pytest tests/ -v"
