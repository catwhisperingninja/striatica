# ERRATA

## 2026-03-25 — Circuit Methodology Revision

**Affects:** Circuit-specific claims in the paper (v10 draft) and circuit view
screenshots. Does NOT affect geometric findings.

### What is affected

The co-activation circuit extraction (`pipeline/circuits.py`) uses Jaccard
similarity over token positions to infer feature relationships. This is a
correlational heuristic, not a causal attribution method. The following claims
in the current paper draft are under revision:

- Multi-circuit membership counts (e.g., "in 6 circuits") — inflated by
  broadly-activating features that co-occur with everything
- Circuit role assignments (source/processing/sink by activation breadth) — no
  published basis; Olah et al. use layer position
- Edge directionality — no basis in the circuits literature
- Finding #6 ("circuit node roles correlate with local dimension; circuits share
  epistemic infrastructure") — not supported by current implementation

Ten specific features (#316, #979, #2039, #7496, #9088, #10423, #16196, #23111,
#23123, #23373) saturated all five co-activation circuits due to high activation
frequency, not computational relationship.

### What is NOT affected

All geometric findings are produced by a separate pipeline
(`process_gpt2_small.py`: PCA, UMAP, HDBSCAN, VGT on raw SAE decoder vectors)
that never touches the circuit code:

- VGT dimensional structure and growth curves
- Local intrinsic dimension estimates (participation ratio)
- Cluster topology and membership
- Convergence singularity observation (near-zero dimension cluster)
- Uncategorized interior population (51.4% of dictionary)
- Independence of local dimension and activation frequency (Pearson r = 0.03)

The main dataset JSON (`gpt2-small-6-res-jb.json`) contains positions, clusters,
local dimensions, and activation stats — all clean.

### What is being done

Integration with Neuronpedia Circuit Tracer for causal attribution is in
progress. The parsing infrastructure (`parse_neuronpedia_circuit()`) is
implemented and tested but not yet wired into the production pipeline. A revised
paper version will retain only claims backed by validated causal evidence.

The Gemma 2 2B transcoder pipeline is under active development. The
dimensionality reduction parameters require adaptation for higher-dimensional
input (2304-D vs 768-D). A three-level validation suite (structural integrity,
embedding quality, cross-model comparison) is implemented and operational.

### Tracking

- v3 plan: `img/correction/v3_transcoder-validate-semantics.md`
- Validation plan: `img/docs/transcoder-validation/VALIDATION_PLAN.md`
- Circuit contamination analysis: documented in `CLAUDE.md` under
  "Co-Activation Circuit Methodology — KNOWN LIMITATION"
