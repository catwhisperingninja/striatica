# striatica/pipeline/vectors.py
"""Extract decoder weight vectors from SAELens pretrained SAEs."""

import numpy as np


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
