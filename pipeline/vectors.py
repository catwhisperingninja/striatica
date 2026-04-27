# striatica/pipeline/vectors.py
"""Extract decoder weight vectors from SAELens pretrained SAEs and Gemmascope transcoders."""

import numpy as np

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None  # Available in Docker/GPU env; tests can monkeypatch


def load_decoder_vectors(
    release: str,
    sae_id: str,
    device: str = "cpu",
) -> np.ndarray:
    """Load SAE decoder weight matrix from HuggingFace via SAELens.

    Args:
        release: SAELens release name (e.g. "gpt2-small-res-jb")
        sae_id: Hook point ID (e.g. "blocks.6.hook_resid_pre")
        device: torch device

    Returns:
        numpy array of shape (num_features, hidden_dim)
    """
    from sae_lens import SAE

    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    # W_dec shape: (num_features, hidden_dim)
    vectors = sae.W_dec.detach().cpu().numpy()
    return vectors


# --- Key name for decoder weights in Gemmascope transcoder npz files.
# Verified 2026-03-22: files are params.npz (NOT safetensors).
# Key must be checked on first download — run the setup script's step 6
# or manually: dict(np.load("params.npz")).keys()
TRANSCODER_DECODER_KEY = "W_dec"


def load_transcoder_vectors(
    layer: int,
    l0_variant: int,
    repo_id: str = "google/gemma-scope-2b-pt-transcoders",
    width: str = "width_16k",
) -> np.ndarray:
    """Load transcoder decoder weight matrix from Gemmascope on HuggingFace.

    Downloads the params.npz file for a specific layer/L0-variant combination
    and returns the decoder weight matrix as a numpy array.

    Args:
        layer: Transformer layer index (e.g. 12).
        l0_variant: Average L0 sparsity variant (e.g. 6, 60, 359, 604, 955).
        repo_id: HuggingFace repository ID.
        width: Width variant directory name.

    Returns:
        numpy array of shape (num_features, d_model), e.g. (16384, 2304) for
        Gemma 2 2B width_16k.

    Raises:
        ValueError: If weights contain non-finite values or L0 variant not found.
        KeyError: If npz file has unexpected key names.
    """
    # Gemmascope transcoders use params.npz (NumPy), NOT safetensors
    filename = f"layer_{layer}/{width}/average_l0_{l0_variant}/params.npz"

    try:
        local_path = hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as e:
        from huggingface_hub.utils import EntryNotFoundError
        if isinstance(e, EntryNotFoundError):
            # Fall back to safetensors in case format changes
            try:
                filename_st = f"layer_{layer}/{width}/average_l0_{l0_variant}/params.safetensors"
                local_path = hf_hub_download(repo_id=repo_id, filename=filename_st)
                from safetensors.numpy import load_file
                weights = load_file(local_path)
                if TRANSCODER_DECODER_KEY not in weights:
                    raise KeyError(
                        f"Key '{TRANSCODER_DECODER_KEY}' not found. "
                        f"Available: {list(weights.keys())}"
                    )
                vectors = weights[TRANSCODER_DECODER_KEY]
                if not np.all(np.isfinite(vectors)):
                    raise ValueError("Weights contain non-finite values.")
                return vectors
            except EntryNotFoundError:
                raise ValueError(
                    f"L0 variant {l0_variant} not found for layer {layer} in {repo_id}. "
                    f"Tried: params.npz and params.safetensors"
                ) from e
        raise

    weights = dict(np.load(local_path))

    if TRANSCODER_DECODER_KEY not in weights:
        available_keys = list(weights.keys())
        raise KeyError(
            f"Expected decoder weight key '{TRANSCODER_DECODER_KEY}' not found in "
            f"npz file. Available keys: {available_keys}. "
            f"Update TRANSCODER_DECODER_KEY in pipeline/vectors.py to match."
        )

    vectors = weights[TRANSCODER_DECODER_KEY]

    if not np.all(np.isfinite(vectors)):
        raise ValueError(
            f"Transcoder decoder weights for layer {layer} L0={l0_variant} contain "
            f"non-finite values (NaN or inf). File may be corrupted."
        )

    return vectors
