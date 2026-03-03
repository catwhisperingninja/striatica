#!/usr/bin/env python3
# striatica/scripts/process_gpt2_small.py
"""End-to-end: download + reduce + cluster + JSON for GPT2-small layer 6."""

import time
from pathlib import Path

from pipeline.config import GPT2_SMALL_L6, DATA_DIR, OUTPUT_DIR
from pipeline.download import download_features, download_explanations
from pipeline.vectors import load_decoder_vectors
from pipeline.reduce import reduce_to_3d
from pipeline.cluster import cluster_points
from pipeline.prepare import prepare_json

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cfg = GPT2_SMALL_L6
t_total = time.time()

# Step 1: Download metadata from Neuronpedia S3
features_path = DATA_DIR / f"{cfg.model_id}_{cfg.layer}_features.jsonl"
if not features_path.exists():
    t0 = time.time()
    print("━━━ Step 1a/6: Downloading feature metadata ━━━")
    download_features(cfg.model_id, cfg.layer, output_path=features_path)
    print(f"  Done in {time.time() - t0:.0f}s")
else:
    print(f"━━━ Step 1a/6: Using cached {features_path.name} ━━━")

explanations_path = DATA_DIR / f"{cfg.model_id}_{cfg.layer}_explanations.jsonl"
if not explanations_path.exists():
    t0 = time.time()
    print("━━━ Step 1b/6: Downloading explanations ━━━")
    download_explanations(cfg.model_id, cfg.layer, output_path=explanations_path)
    print(f"  Done in {time.time() - t0:.0f}s")
else:
    print(f"━━━ Step 1b/6: Using cached {explanations_path.name} ━━━")

# Step 2: Load decoder weight vectors from SAELens
t0 = time.time()
print("━━━ Step 2/6: Loading SAELens decoder vectors ━━━")
vectors = load_decoder_vectors(cfg.sae_release, cfg.sae_hook)
print(f"  Loaded {vectors.shape[0]} vectors of dim {vectors.shape[1]} in {time.time() - t0:.0f}s")

# Step 3: Dimensionality reduction
t0 = time.time()
print("━━━ Step 3/6: PCA + UMAP to 3D ━━━")
coords = reduce_to_3d(vectors, pca_dim=50)
print(f"  Done in {time.time() - t0:.0f}s")

# Step 4: Clustering
t0 = time.time()
print("━━━ Step 4/6: HDBSCAN clustering ━━━")
labels = cluster_points(coords)
print(f"  Done in {time.time() - t0:.0f}s")

# Step 5: Local dimension estimation
from pipeline.local_dim import estimate_local_dim, estimate_local_dim_vgt

t0 = time.time()
print("━━━ Step 5a/6: Local intrinsic dimension (Participation Ratio) ━━━")
local_dims = estimate_local_dim(vectors, method="pr")
print(f"  Done in {time.time() - t0:.0f}s")

t0 = time.time()
print("━━━ Step 5b/6: VGT growth curves for detail view ━━━")
_, growth_curves = estimate_local_dim_vgt(vectors, return_curves=True)
print(f"  Done in {time.time() - t0:.0f}s")

# Step 6: Assemble JSON
t0 = time.time()
print("━━━ Step 6/6: Preparing JSON for frontend ━━━")
output = OUTPUT_DIR / f"{cfg.model_id}-{cfg.layer}.json"
prepare_json(coords, labels, features_path, explanations_path, output,
             local_dimensions=local_dims, dim_method="pr", growth_curves=growth_curves)
print(f"  Done in {time.time() - t0:.0f}s")

elapsed = time.time() - t_total
minutes = int(elapsed // 60)
seconds = int(elapsed % 60)
print(f"\n{'━' * 40}")
print(f"Done! Total time: {minutes}m {seconds}s")
print(f"Output: {output}")
print("Run the frontend with: cd frontend && pnpm dev")
