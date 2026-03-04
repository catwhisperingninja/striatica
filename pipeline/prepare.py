# striatica/pipeline/prepare.py
"""Assemble final JSON for frontend consumption."""

from __future__ import annotations

import json
import time as _time
from pathlib import Path

import numpy as np


def prepare_json(
    coords: np.ndarray,
    labels: np.ndarray,
    features_jsonl: Path,
    explanations_jsonl: Path,
    output_path: Path,
    local_dimensions: np.ndarray | None = None,
    dim_method: str = "pr",
    growth_curves: list[dict] | None = None,
    model: str = "gpt2-small",
    layer: str = "6-res-jb",
) -> dict:
    """Assemble final JSON combining 3D coords, clusters, and metadata."""
    n = len(coords)
    _t_start = _time.monotonic()

    # Load feature metadata
    print(f"    Loading feature metadata from {features_jsonl.name}...")
    feature_meta = {}
    with open(features_jsonl) as f:
        for line in f:
            d = json.loads(line)
            idx = int(d["index"])
            feature_meta[idx] = {
                "maxAct": d.get("maxActApprox", 0),
                "fracNonzero": d.get("frac_nonzero", 0),
                "topSimilar": d.get("topkCosSimIndices", [])[:5],
                "posTokens": d.get("pos_str", [])[:5],
                "negTokens": d.get("neg_str", [])[:3],
            }
    print(f"    Loaded {len(feature_meta):,} feature records")

    # Load explanations
    print(f"    Loading explanations from {explanations_jsonl.name}...")
    explanations = {}
    with open(explanations_jsonl) as f:
        for line in f:
            d = json.loads(line)
            idx = int(d["index"])
            if idx not in explanations:
                explanations[idx] = d.get("description", "")
    print(f"    Loaded {len(explanations):,} explanations")

    # Flatten positions to interleaved array
    print(f"    Flattening {n:,} positions...")
    positions = coords.flatten().tolist()

    # Build cluster info
    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    print(f"    Building cluster centroids ({n_clusters} clusters)...")
    clusters = []
    for label in unique_labels:
        mask = labels == label
        centroid = coords[mask].mean(axis=0).tolist()
        clusters.append({
            "id": int(label),
            "count": int(mask.sum()),
            "centroid": centroid,
        })

    # Build per-feature metadata
    print(f"    Assembling {n:,} feature records...")
    features = []
    _last_log = _time.monotonic()
    _log_every = 15.0
    for i in range(n):
        _now = _time.monotonic()
        if i > 0 and _now - _last_log >= _log_every:
            pct = i / n * 100
            print(f"    [{pct:5.1f}%] {i:,}/{n:,} features assembled")
            _last_log = _now
        meta = feature_meta.get(i, {})
        features.append({
            "index": i,
            "explanation": explanations.get(i, ""),
            "maxAct": meta.get("maxAct", 0),
            "fracNonzero": meta.get("fracNonzero", 0),
            "topSimilar": meta.get("topSimilar", []),
            "posTokens": meta.get("posTokens", []),
            "negTokens": meta.get("negTokens", []),
        })

    result = {
        "model": model,
        "layer": layer,
        "numFeatures": n,
        "positions": positions,
        "clusterLabels": labels.tolist(),
        "clusters": clusters,
        "features": features,
    }

    # Add local dimension data if computed
    if local_dimensions is not None:
        result["localDimensions"] = local_dimensions.tolist()
        result["dimMethod"] = dim_method

    # Add VGT growth curves if computed (for DetailPanel chart)
    if growth_curves is not None:
        result["growthCurves"] = growth_curves

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"    Writing JSON to disk...")
    with open(output_path, "w") as f:
        json.dump(result, f)

    size_mb = output_path.stat().st_size / 1024 / 1024
    elapsed = _time.monotonic() - _t_start
    m, s = divmod(int(elapsed), 60)
    print(f"  Wrote {output_path} ({size_mb:.1f} MB) in {m}m{s:02d}s")
    return result


def prepare_json_minimal(
    coords: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    """Write positions-only JSON for fast testing without S3 data."""
    result = {
        "model": "test",
        "layer": "0",
        "numFeatures": len(coords),
        "positions": coords.flatten().tolist(),
        "clusterLabels": labels.tolist(),
        "clusters": [],
        "features": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f)
