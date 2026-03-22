#!/usr/bin/env bash
# vast_setupv2.sh — Vast.ai A100 instance bootstrap for striatica pipeline
# Run directly on the instance: bash vast_setupv2.sh
# Does NOT use Docker or Poetry. Pip-installs deps directly for CUDA compat.
# Checks out triage-debug branch (has transcoder loading code + new tests).
set -euo pipefail

BRANCH="triage-debug"
VERSION_LOG="/root/striatica-installed-versions.txt"

# HuggingFace token — Gemma models are GATED and require authentication.
# Set HF_TOKEN before running this script, or it will be read from .env
if [ -z "${HF_TOKEN:-}" ]; then
    # Try loading from .env in the repo (after clone) or from root
    if [ -f /root/striatica/.env ]; then
        HF_TOKEN=$(grep -E '^HF_TOKEN=' /root/striatica/.env | cut -d= -f2- | tr -d '"' || true)
    elif [ -f /root/.env ]; then
        HF_TOKEN=$(grep -E '^HF_TOKEN=' /root/.env | cut -d= -f2- | tr -d '"' || true)
    fi
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN not set. Gemma models are gated — downloads WILL fail."
    echo "  Set it with: export HF_TOKEN=hf_your_token_here"
    echo "  Or add HF_TOKEN=hf_... to /root/.env"
    echo ""
fi
export HF_TOKEN="${HF_TOKEN:-}"

echo "=== striatica Vast.ai setup (pip, no Poetry) ==="
echo "Target branch: ${BRANCH}"
echo "Instance: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no nvidia-smi')"
echo ""

# ---------------------------------------------------------------
# 1. Python 3.12 via deadsnakes (Vast base image ships 3.10)
# ---------------------------------------------------------------
echo "[1/6] Installing Python 3.12..."
apt-get update -qq && apt-get install -y -qq software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get install -y -qq python3.12 python3.12-venv python3.12-dev curl git
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12
# Override conda's python — put our symlink FIRST in PATH
ln -sf /usr/bin/python3.12 /usr/local/bin/python
# Also override /usr/bin/python if it exists (conda often puts itself here)
ln -sf /usr/bin/python3.12 /usr/bin/python
# Make sure our python3.12 pip is the one that runs
export PATH="/usr/local/bin:$PATH"
hash -r  # clear bash's command cache
python --version
echo "  Python path: $(which python)"

# ---------------------------------------------------------------
# 2. Install deps (pip, no Poetry — Poetry breaks on A100/CUDA)
# ---------------------------------------------------------------
echo "[2/6] Installing PyTorch (CUDA 12.4)..."
python -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124

echo "[2/6] Installing project deps..."
# NOTE: For the Gemma 2 transcoder pipeline, we are generating NEW data
# (no existing positions to preserve), so unpinned versions are acceptable
# for this first run. We record what gets installed and pin from there.
python -m pip install --quiet \
    numpy scipy scikit-learn umap-learn hdbscan \
    requests huggingface-hub safetensors python-dotenv \
    sae-lens transformer-lens transformers \
    pytest ruff

# ---------------------------------------------------------------
# 3. Clone repo and checkout triage-debug
# ---------------------------------------------------------------
echo "[3/6] Cloning striatica and checking out ${BRANCH}..."
cd /root
if [ -d "striatica" ]; then
    echo "  striatica/ already exists, pulling latest..."
    cd striatica
    git fetch origin
    git checkout "${BRANCH}"
    git pull origin "${BRANCH}"
else
    git clone https://github.com/catwhisperingninja/striatica.git
    cd striatica
    git checkout "${BRANCH}"
fi

# ---------------------------------------------------------------
# 4. Record installed versions (pin from these for reproducibility)
# ---------------------------------------------------------------
echo "[4/6] Recording installed package versions..."
python -m pip freeze > "${VERSION_LOG}"
echo "  Versions saved to ${VERSION_LOG}"

# Print the ones that matter for UMAP reproducibility
echo ""
echo "  === UMAP REPRODUCIBILITY CHAIN ==="
for pkg in numpy scipy scikit-learn umap-learn hdbscan numba pynndescent; do
    ver=$(python -m pip show "${pkg}" 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "NOT INSTALLED")
    echo "  ${pkg}==${ver}"
done
echo ""
echo "  === ML STACK ==="
for pkg in torch sae-lens transformer-lens transformers safetensors huggingface-hub; do
    ver=$(python -m pip show "${pkg}" 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "NOT INSTALLED")
    echo "  ${pkg}==${ver}"
done
echo ""

# ---------------------------------------------------------------
# 5. Verify GPU access
# ---------------------------------------------------------------
echo "[5/6] Verifying GPU access..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  WARNING: No CUDA GPU detected!')
"

# ---------------------------------------------------------------
# 6. Probe transcoder weights on HuggingFace (discover L0 + keys)
# ---------------------------------------------------------------
echo "[6/6] Probing Gemmascope transcoder weights (layer 12)..."
# Repo is public (CC-BY-4.0), no auth required.
# Files are params.npz (NOT safetensors).
python -c "
import os, re
from huggingface_hub import hf_hub_download, list_repo_tree
import numpy as np

repo = 'google/gemma-scope-2b-pt-transcoders'
base_path = 'layer_12/width_16k'
token = os.environ.get('HF_TOKEN')  # optional, repo is public

# List available L0 variants for layer 12
print('  Available L0 variants for layer 12, width_16k:')
entries = list(list_repo_tree(repo, path_in_repo=base_path, token=token))

variant_dirs = []
for e in entries:
    if not hasattr(e, 'path'):
        continue
    dirname = e.path.split('/')[-1]
    if re.match(r'^average_l0_\d+$', dirname):
        variant_dirs.append(e.path)

for vd in sorted(variant_dirs):
    print(f'    {vd.split(\"/\")[-1]}')

if not variant_dirs:
    print('  WARNING: Could not discover variants via API.')
    print('  Trying known L0 values: 6, 60, 359, 604, 955')
    for l0 in [6, 60, 359, 604, 955]:
        variant_dirs.append(f'{base_path}/average_l0_{l0}')

# Download first available variant to inspect key names.
# Gemmascope transcoders use params.npz (NOT safetensors).
downloaded = False
for variant_path in sorted(variant_dirs):
    for filename in ['params.npz', 'params.safetensors']:
        full_path = f'{variant_path}/{filename}'
        print(f'  Trying {full_path}...')
        try:
            local = hf_hub_download(repo, filename=full_path, token=token)
            variant_name = variant_path.split('/')[-1]
            print(f'  SUCCESS — downloaded {filename} from {variant_name}')

            if filename.endswith('.safetensors'):
                from safetensors.numpy import load_file
                weights = load_file(local)
            else:
                weights = dict(np.load(local))

            print(f'  Keys: {list(weights.keys())}')
            for k, v in weights.items():
                print(f'    {k}: shape={v.shape}, dtype={v.dtype}')
            downloaded = True
            break
        except Exception as exc:
            print(f'    SKIP: {type(exc).__name__}')
    if downloaded:
        break

if downloaded:
    print(f'')
    print(f'  === ACTION REQUIRED ===')
    print(f'  If the decoder key is NOT \"W_dec\", update TRANSCODER_DECODER_KEY')
    print(f'  in pipeline/vectors.py to match the actual key name above.')
else:
    print(f'')
    print(f'  WARNING: Could not download any variant weights.')
    print(f'  Check repo: https://huggingface.co/{repo}/tree/main/{base_path}')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  cd /root/striatica"
echo "  python -m pytest tests/ -v              # run tests"
echo "  cat ${VERSION_LOG}                       # review installed versions"
echo ""
echo "Once L0 variant is confirmed, run the geometric pipeline:"
echo "  python scripts/process_gpt2_small.py --help   # (adapt for Gemma 2 transcoder)"
