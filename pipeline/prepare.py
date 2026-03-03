# striatica/pipeline/prepare.py
"""Assemble final JSON for frontend consumption."""

from __future__ import annotations

import json
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
    # Load feature metadata
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

    # Load explanations
    explanations = {}
    with open(explanations_jsonl) as f:
        for line in f:
            d = json.loads(line)
            idx = int(d["index"])
            if idx not in explanations:
                explanations[idx] = d.get("description", "")

    # Flatten positions to interleaved array
    positions = coords.flatten().tolist()

    # Build cluster info
    unique_labels = sorted(set(labels))
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
    n = len(coords)
    features = []
    for i in range(n):
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
    with open(output_path, "w") as f:
        json.dump(result, f)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Wrote {output_path} ({size_mb:.1f} MB)")
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
