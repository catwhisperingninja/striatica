# striatica/pipeline/circuits.py
"""Extract circuit data from GPT-2 Small using co-activation and similarity methods."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def extract_coactivation_circuit(
    prompt: str,
    model_name: str = "gpt2",
    sae_release: str = "gpt2-small-res-jb",
    sae_hook: str = "blocks.6.hook_resid_pre",
    top_k_features: int = 30,
    min_coactivation: float = 0.1,
    device: str = "cpu",
) -> dict:
    """Run GPT-2 + SAE on a prompt and extract co-activation circuits.

    Steps:
    1. Forward pass through GPT-2 via TransformerLens
    2. Encode residual stream activations through SAE
    3. Find top-k features by peak activation
    4. Build co-activation graph via Jaccard similarity over token positions
    5. Assign roles by activation breadth

    Returns:
        Dict matching CircuitData schema.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if top_k_features < 1:
        raise ValueError(f"top_k_features must be >= 1, got {top_k_features}")
    if not 0 <= min_coactivation <= 1:
        raise ValueError(f"min_coactivation must be in [0, 1], got {min_coactivation}")

    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    # Load models
    model = HookedTransformer.from_pretrained(model_name, device=device)
    sae = SAE.from_pretrained(release=sae_release, sae_id=sae_hook, device=device)

    # Forward pass
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens)

    # Get residual activations at the hook point
    residual = cache[sae_hook]  # shape: (1, seq_len, d_model)
    seq_len = residual.shape[1]

    # Encode each token position through SAE
    # Collect (feature_index, activation_value, token_position)
    feature_activations: dict[int, list[tuple[float, int]]] = {}
    for pos in range(seq_len):
        encoded = sae.encode(residual[:, pos, :])  # shape: (1, n_features)
        acts = encoded[0].detach().cpu().numpy()
        nonzero = np.nonzero(acts)[0]
        for feat_idx in nonzero:
            feat_idx_int = int(feat_idx)
            if feat_idx_int not in feature_activations:
                feature_activations[feat_idx_int] = []
            feature_activations[feat_idx_int].append((float(acts[feat_idx]), pos))

    # Select top_k features by peak activation
    if not feature_activations:
        return {
            "name": f"coact-{prompt[:30].replace(' ', '-').lower()}",
            "description": f"Co-activation circuit for: {prompt} (no features activated)",
            "nodes": [],
            "edges": [],
        }

    peak_acts = {
        idx: max(a for a, _ in positions)
        for idx, positions in feature_activations.items()
    }
    top_features = sorted(peak_acts, key=lambda x: peak_acts[x], reverse=True)[:top_k_features]
    top_set = set(top_features)

    # Build token position sets for Jaccard calculation
    position_sets: dict[int, set[int]] = {}
    for idx in top_features:
        position_sets[idx] = {pos for _, pos in feature_activations[idx]}

    # Activation breadth = number of token positions where feature fires
    breadths = {idx: len(position_sets[idx]) for idx in top_features}

    # Assign roles by breadth: top 1/3 broadest = source, bottom 1/3 = sink
    sorted_by_breadth = sorted(top_features, key=lambda x: breadths[x], reverse=True)
    n = len(sorted_by_breadth)
    third = max(1, n // 3)
    source_set = set(sorted_by_breadth[:third])
    sink_set = set(sorted_by_breadth[-third:])

    # Build nodes
    nodes = []
    for idx in top_features:
        if idx in source_set:
            role = "source"
        elif idx in sink_set:
            role = "sink"
        else:
            role = "intermediate"
        # Normalize activation to 0-1 range
        max_peak = max(peak_acts[f] for f in top_features)
        activation = peak_acts[idx] / max_peak if max_peak > 0 else 0
        nodes.append({
            "featureIndex": idx,
            "activation": round(activation, 4),
            "role": role,
        })

    # Build edges via Jaccard similarity
    edges = []
    for i, f1 in enumerate(top_features):
        for f2 in top_features[i + 1:]:
            s1, s2 = position_sets[f1], position_sets[f2]
            intersection = len(s1 & s2)
            union = len(s1 | s2)
            if union == 0:
                continue
            jaccard = intersection / union
            if jaccard >= min_coactivation:
                # Direction: broader feature → narrower feature
                if breadths[f1] >= breadths[f2]:
                    src, tgt = f1, f2
                else:
                    src, tgt = f2, f1
                edges.append({
                    "source": src,
                    "target": tgt,
                    "weight": round(jaccard, 4),
                })

    return {
        "name": f"coact-{prompt[:30].replace(' ', '-').lower()}",
        "description": f"Co-activation circuit for: {prompt}",
        "nodes": nodes,
        "edges": edges,
    }


def extract_similarity_circuit(
    features_jsonl: str | Path,
    seed_feature: int,
    depth: int = 2,
    top_k_neighbors: int = 5,
) -> dict:
    """Build a circuit from cosine similarity BFS starting at a seed feature.

    Uses topkCosSimIndices/Values from Neuronpedia feature metadata.
    BFS depth 0 = seed (source), depth 1 = intermediate, depth 2 = sink.

    Returns:
        Dict matching CircuitData schema.
    """
    features_jsonl = Path(features_jsonl)

    # Load similarity data from JSONL
    sim_indices: dict[int, list[int]] = {}
    sim_values: dict[int, list[float]] = {}
    max_acts: dict[int, float] = {}

    with open(features_jsonl) as f:
        for line in f:
            d = json.loads(line)
            idx = int(d["index"])
            # Skip self-reference (index 0 in topk is usually self with sim=1.0)
            raw_indices = d.get("topkCosSimIndices", [])
            raw_values = d.get("topkCosSimValues", [])
            # Filter out self-similarity
            filtered_i = []
            filtered_v = []
            for ki, kv in zip(raw_indices, raw_values):
                if int(ki) != idx:
                    filtered_i.append(int(ki))
                    filtered_v.append(float(kv))
            sim_indices[idx] = filtered_i
            sim_values[idx] = filtered_v
            max_acts[idx] = float(d.get("maxActApprox", 0))

    if seed_feature not in sim_indices:
        raise ValueError(f"Seed feature {seed_feature} not found in JSONL")

    # BFS from seed
    visited: dict[int, int] = {seed_feature: 0}  # feature_index -> depth
    queue = [seed_feature]
    edges = []

    for current_depth in range(depth):
        next_queue = []
        for feat_idx in queue:
            neighbors = sim_indices.get(feat_idx, [])[:top_k_neighbors]
            neighbor_sims = sim_values.get(feat_idx, [])[:top_k_neighbors]
            for neighbor, sim_val in zip(neighbors, neighbor_sims):
                edges.append({
                    "source": feat_idx,
                    "target": neighbor,
                    "weight": round(sim_val, 4),
                })
                if neighbor not in visited:
                    visited[neighbor] = current_depth + 1
                    next_queue.append(neighbor)
        queue = next_queue

    # Build nodes with roles based on BFS depth
    # Compute similarity to seed for activation values
    seed_sims: dict[int, float] = {seed_feature: 1.0}
    for feat_idx, d in visited.items():
        if feat_idx == seed_feature:
            continue
        # Use the max similarity value along any edge connecting to this node
        best_sim = 0.0
        for edge in edges:
            if edge["target"] == feat_idx:
                best_sim = max(best_sim, edge["weight"])
            elif edge["source"] == feat_idx:
                best_sim = max(best_sim, edge["weight"])
        seed_sims[feat_idx] = best_sim

    nodes = []
    for feat_idx, feat_depth in visited.items():
        if feat_depth == 0:
            role = "source"
        elif feat_depth < depth:
            role = "intermediate"
        else:
            role = "sink"
        nodes.append({
            "featureIndex": feat_idx,
            "activation": round(seed_sims.get(feat_idx, 0), 4),
            "role": role,
        })

    return {
        "name": f"sim-{seed_feature}",
        "description": f"Similarity circuit from feature #{seed_feature} (depth {depth})",
        "nodes": nodes,
        "edges": edges,
    }
