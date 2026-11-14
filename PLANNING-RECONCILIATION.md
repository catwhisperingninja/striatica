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

Extend existing checks rather than building another acceptance framework or duplicating the graph-contract suite. The numerical case below has now passed its candidate screen; rendering and scientific fidelity remain unverified.

## New proposal: one cosine-neighbour spectral pilot

Draft recommendation, not a recovered historical decision or a completed migration. Stop archive searching. Try one candidate on the existing 810-feature union before changing the production reducer.

### Build one geometry graph and one layout

Use the cached L2-normalized decoder rows in their original 2,304-dimensional space. Connect each feature to its 30 nearest cosine neighbours, excluding itself. Symmetrize by retaining an undirected, unit-weight edge whenever either endpoint selected the other. These are geometry-neighbour edges, not attribution or causal edges.

Embed this graph once into three dimensions with the already-pinned scikit-learn `SpectralEmbedding`, using the precomputed adjacency, ARPACK solver and seed 42. Use one shared layout across all participating layers. No intermediate PCA, UMAP, new dependency, clustering or Euclidicity score enters this candidate. Thirty neighbours is an explicit initial design choice, not a cited scientific constant.

A read-only prerequisite check found that the cached 810-feature graph has one connected component and no self-edges. If another input is disconnected, report it; do not silently invent bridging edges or claim meaningful distances between components.

### One comparison and a stopping rule

Compare this candidate with deterministic full-SVD 3D PCA on the same normalized rows. PCA is only a yardstick, not the replacement or a fallback. For each feature, measure how many of its 30 original-space cosine neighbours remain among its 30 Euclidean neighbours in 3D. Report the mean retained fraction, per-layer means/counts, and CPU time for both layouts.

Run the one existing real-data acceptance case above. Require finite coordinates, correct feature-key joins and unchanged imported graph topology/weights; ensure the candidate route never invokes UMAP. Reuse existing graph-contract checks rather than duplicating them. Fit and compare layouts before attaching attribution edges, so those edges cannot determine the coordinates.

**Advance only if the engineering checks pass and spectral layout has strictly higher mean neighbour retention than PCA.** Otherwise stop after this one comparison and report the failure; no parameter sweep or automatic search for another method. If it advances, the next deliverable is a real rendered two-graph slice, with representative states inspected before calling it usable.

This is a cheap candidate-screening rule, not a scientific fidelity threshold. Evaluating the same neighbourhood used to build the layout is not independent functional validation. Report the absolute retention and layer differences even if the candidate wins; low retention cannot support spatial claims. Do not claim preserved topology, causal convergence, stratification or circuit faithfulness from this result.

### Deliverable and limits

One small pilot script, two disposable coordinate outputs and one short numeric report. No test framework, broad audit, CodeRabbit review, activation capture, production-data regeneration or default change. No viewer refactor before the numerical comparison earns it. The cached graph-union sample is not representative of a full atlas.

A successful pilot is only the first slice. Before production cutover, check CPU cost and fidelity on the complete flagship dictionary, resolve clustering outside display coordinates, integrate the graph viewer and obtain the required data/visual approvals. Existing stratified/TDA gates are not waived. UMAP-dependent production paths remain unfinished until replaced; removing locked dependencies is a separate approved change.

## Bounded pilot result — November 14, 2026 UTC

The single candidate passes the declared advancement screen. On the cached 810-feature union across 26 layers, spectral retains 4,843 of 24,300 directed neighbour relations (19.93%), versus PCA's 3,227 (13.28%). This is a 6.65-percentage-point gain, not scientific acceptance. Spectral retains fewer neighbours in layers 8, 10 and 11; roughly 80% of neighbours are still lost overall.

Single-thread CPU time: graph construction 0.144 s, spectral fit 0.153 s (0.296 s combined), PCA fit 0.729 s. These are one-run sample timings, not full-atlas performance estimates. Finite coordinates, unique/shared feature joins, all graph instances and directed signed weights passed. The two graphs retain 855/894 nodes and 34,801/42,338 edges respectively. Their 158/154 special nodes remain explicit without invented decoder coordinates. No UMAP was imported. Attribution edges were attached only after fitting both layouts.

Execution: `poetry run python .amp/in/orb-backup/no-umap-pilot/run.py`. An independent pairwise-distance check reproduced every feature's retained-neighbour count. Existing graph-contract checks: `poetry run pytest tests/test_traced_circuits.py -k full -q` returned **32 passed, 28 deselected**. No tests were added or modified.

Input SHA-256: `a6e4c7e1a003dadf35aefd9ecc1bd8de80c5bd24861fe551d3aaddf2f334cae2`. Decoder repository: `google/gemma-scope-2b-pt-transcoders`, revision `50eec2f25c60545a9a74c1c3a26a0afdd0b4b872`. Current source-set and graph-record identities were checked against the cached manifest; this validates declared identities, not historical capture weights. Only small identity/API metadata was refreshed; no model weights or graph payloads were downloaded.

Reusable script, coordinates, label-free graphs and complete numeric report with all layer counts/means are under `.amp/in/orb-backup/no-umap-pilot/`. These are local ignored pilot outputs, not production data or a committed implementation. The result earns the next small deliverable: render this same two-graph slice, preserving graph instances and handling special nodes explicitly. No viewer work or production replacement has happened yet.

Keep product and paper work parallel; only claims that need missing evidence wait for it. Small, checkable outcomes are sufficient: new DAG, orchestration, or CI machinery is not a prerequisite.

Preserve the committed dependency locks, Poetry-only Python workflow, CPU-first verification, audit isolation, real application data, and semantic secrecy. Any stratified/TDA implementation entering `pipeline/` remains subject to the repository's evidence and external-validation gate. No production-data regeneration or publication is authorized by this document.
