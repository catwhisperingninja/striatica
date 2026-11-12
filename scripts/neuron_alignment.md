# Neuron-alignment control (A-NAP-001)

This weights-only control asks whether the atlas's geometric patterns partly
reflect features that point in the same directions as individual host-model
neurons. It does not measure circuit faithfulness.

## How it complements the existing tools

- **UMAP is a map:** a 3D drawing of feature directions, which can distort distances.
- **PR and local-dimension estimates describe shape:** how many independent
  directions a population or neighborhood occupies, measured before projection.
- **Circuit traces describe connections:** attribution edges between feature
  instances for a particular input. They are not inferred from geometric proximity.
- **Neuron alignment checks an alternative explanation:** are some geometric
  patterns associated with features resembling the model's existing neurons?

Compare each feature direction with host-neuron directions. Sweep the alignment
cutoff rather than select a favorable binary split, and recompute PR on both
populations. Matched random dictionaries show accidental alignment. A weaker
match is not proof of a new useful feature; a stronger match is not proof that a
feature is useless. Neither alone validates a circuit.

## Reproduce with the existing Poetry environment

Do not update dependencies or the lockfile. Obtain these exact inputs through
Hugging Face, using an account with the Gemma license accepted:

| Input | Repository revision | File |
| --- | --- | --- |
| Dictionary | `google/gemma-scope-2b-pt-transcoders` at `50eec2f25c60545a9a74c1c3a26a0afdd0b4b872` | `layer_12/width_16k/average_l0_6/params.npz` |
| Host | `google/gemma-2-2b` at `c5ebcd40d208330abc697524c919956e692655cf` | `model-00002-of-00003.safetensors` and `config.json` |

The host is **pretrained, not instruct**. Only one host shard (about 5 GB) is
needed. `huggingface_hub.hf_hub_download` accepts a revision SHA; use the existing
Hugging Face cache and pass credentials through the environment or library, never
in shell arguments or output. No token or weight file belongs in Git.

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
poetry run python scripts/neuron_alignment.py \
  --dictionary /path/to/params.npz \
  --host-shard /path/to/model-00002-of-00003.safetensors \
  --config /path/to/config.json \
  --output data/audits/neuron-alignment/run-001 \
  --null-runs 3
```

The output directory must be new. The script verifies all three input SHA-256s,
reads `d_ff` from the actual config, extracts static L12 tensors directly from the
shard, and never instantiates or executes a model. Results stay in the gitignored
`data/` tree, separate from production `frontend/public/data/` datasets.

## Definitions

- **S1:** maximum signed decoder cosine to a physical neuron's output direction.
- **S2/S3:** separate encoder maxima to the gate and up directions. Positive row
  rescaling leaves cosine unchanged; the raw/unit-row grid matters for encoder PR.
- **S4:** agreement between independently maximizing neuron IDs. This is a
  diagnostic, not the paper's shared-candidate criterion.
- **S5:** concentration of responses across neuron directions, reported for both
  absolute-cosine mass and squared-cosine energy. This nonorthogonal frame does
  not give unique reconstruction coefficients.
- **Joint score:** the maximum, over shared neuron/sign candidates, of the smaller
  encoder/decoder cosine. It implements the literal pairs `(up+gate,+down)` and
  `(up−gate,−down)` from [arXiv:2610.24521v1, §§2.1 and 3.1](https://arxiv.org/html/2610.24521v1).

The [dictionary README at the pinned revision](https://huggingface.co/google/gemma-scope-2b-pt-transcoders/blob/50eec2f25c60545a9a74c1c3a26a0afdd0b4b872/README.md)
clarifies its normalization frame: pre-MLP RMSNorm gains are folded into MLP
input matrices, and its target is the post-MLP RMSNorm update. Primary comparisons
multiply host gate/up rows by the pre gain and host down rows by the post gain.
Raw-down S1 is retained as an explicitly mismatched-frame diagnostic. PR uses
released dictionary rows, with centering performed separately in each partition.

This is a new application to this per-layer dictionary, not reproduction of the
paper's Gemma Scope 2/Gemma 3 results or its Gemma-2 cross-layer transcoder results.
Its activation-weighted headline cannot be reproduced from static counts.

## Controls and limits

The actual config gives 2,304 model dimensions and 9,216 physical neurons.
**Before interpreting fractions:** distinct physical-neuron assignments have a
56.25% capacity relative to 16,384 latents; the 18,432 sign candidates give a
capped capacity of 100%. These do not limit many-to-one aligned fractions.

N1 is signed-cosine RMS (0.0208333), not a positive signed mean. N2 is the
positive-maximum approximation (0.0890181). **N3 is the empirical control:**
independent isotropic Gaussian decoder/encoder rows, each rescaled to its real
counterpart's norm. It preserves shape and norm profile, not the real geometry or
encoder/decoder dependence. Three draws give an empirical tail-resolution floor
of 0.25 under run exchangeability, not a significance claim.

**Null coverage is metric-specific.** Only `A_dec_frame_matched` (S1) and
`S5_squared_cos_pr` have matched-null PR curves. `paper_joint_score`,
`S5_abs_cos_pr` and `A_dec_raw_diagnostic` are **real-only, uncontrolled PR
curves**: descriptive measurements with no matched-null PR baseline. A saved
per-latent null score is not a null PR curve. The manifest's `pr_curve_controls`
records these lists, which also drive run-dependent curve selection.

Every PR point includes `center`, `row_normalize`, `weights` and population size.
Cutoffs cover −1–1 in 0.01 steps for cosine and 1–`d_ff` logarithmically for S5.
Empty populations have undefined PR (`null`). Changing population size changes
rank bounds. A same-cutoff null comparison is not automatically size matched.
Cosine spread is not PR uncertainty; no formal minimum detectable PR difference
is established. Historical hub-removal IDs were not recovered, so no three-way
hub-removal invariance is claimed.

The initial run's PR curve was non-flat: at S1 cutoff 0.2, raw uncentered decoder
PR was 617.409 below (5,995 rows) and 160.344 above (10,389 rows); at 0.7 it was
273.600 below (14,251 rows) and 243.365 above (2,133 rows). The full sweep and
preprocessing choices, not these illustrations, are the result. The baseline
remains 276.673488 uncentered and 317.567670 centered.

No activations, forward passes, Jacobians, interp-engine, UMAP generation,
production-data regeneration or causal claims are part of this control.
