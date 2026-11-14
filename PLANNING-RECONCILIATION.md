# Planning reconciliation

Status: November 14, 2026 UTC. This is the public-safe product checkpoint, not a completed migration, scientific signoff, or publication claim.

## Completion goal

Striatica is not finished while its geometry pipeline depends on UMAP. The product goal is meaningful multilayer circuit and subgraph visualization, not a decoder point cloud alone.

Keep original-space dictionary geometry, display coordinates, imported attribution edges, and experimentally validated mechanisms distinct. Removing UMAP does not by itself establish geometric fidelity or circuit faithfulness.

## Delivered and unfinished

[PR #19](https://github.com/catwhisperingninja/striatica/pull/19) is merged. It delivered the lossless full-graph contract, graph privacy fixes, and weights-only neuron-alignment tooling.

The following remain unfinished:

- UMAP retirement: `pipeline/reduce.py` still fits UMAP, and the CLI clusters its projected coordinates.
- Integrated multilayer rendering of the full graph contract.
- Selection and validation of the replacement geometry/layout method.

The existing graph-contract checks and neuron-alignment control do not establish completion of those tasks.

## Replacement boundary

The recovered architecture separates three responsibilities:

1. Construct and measure geometry in the declared original decoder space.
2. Produce its 3D presentation without UMAP.
3. Measure what that presentation preserves.

A selected layout must use one declared shared decoder population, not independently fitted coordinate systems per layer. Preserve dictionary-feature identity separately from graph-node/token instances. Imported edge endpoints, direction, sign, and weights remain evidence independent of the layout.

Special nodes must retain their actual connections without being assigned invented decoder vectors. A display-dependent cluster must not silently become an original-space stratum.

The precise construction algorithm, selected layout, fidelity calculation, and justified acceptance cutoff have not yet been recovered or approved. This document does not select them by implication.

## One bounded acceptance case

Use the two existing real-graph inputs and their cached 810-feature decoder union. Run the selected no-UMAP route on CPU into disposable output outside production data.

Require:

- Finite 3D coordinates for every feature anchor, with a consistent position for each shared dictionary-feature key.
- Preserved graph instances, endpoints, directed signed weights, and explicit handling of non-feature nodes.
- Recorded input identity, preprocessing, layout parameters, and the agreed fidelity measurement.

Engineering checks and scientific fidelity must be reported separately. Passing identity/topology checks cannot substitute for an accepted fidelity criterion. The integrated viewer will also require inspection of rendered states and interactions.

Extend existing checks rather than building another acceptance framework or duplicating the graph-contract suite. The case is proposed, not executed or passed.

## Scope and next step

Recover the referenced technical design and later amendments, then implement one bounded no-UMAP slice. Missing source material is a retrieval gap, not proof that no design exists.

Keep product and paper work parallel; only claims that need missing evidence wait for it. Small, checkable outcomes are sufficient: new DAG, orchestration, or CI machinery is not a prerequisite.

Preserve the committed dependency locks, Poetry-only Python workflow, CPU-first verification, audit isolation, real application data, and semantic secrecy. Any stratified/TDA implementation entering `pipeline/` remains subject to the repository's evidence and external-validation gate. No production-data regeneration or publication is authorized by this document.
