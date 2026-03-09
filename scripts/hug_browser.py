from sae_lens import SAE

sae = SAE.from_pretrained(
    release = "gemma-scope-2b-pt-res-canonical",
    sae_id = "layer_12/width_16k/canonical",
    device = "cuda"
)
