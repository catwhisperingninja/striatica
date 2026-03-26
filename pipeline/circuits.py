# striatica/pipeline/circuits.py
"""Extract circuit data using co-activation, similarity, and Neuronpedia causal methods."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _load_global_activation_frequencies(
    features_jsonl: str | Path | None,
) -> dict[int, float]:
    """Load frac_nonzero (global activation frequency) from features JSONL.

    Returns:
        Dict mapping feature index to frac_nonzero (0.0–1.0).
        Empty dict if features_jsonl is None or doesn't exist.
    """
    if features_jsonl is None:
        return {}
    path = Path(features_jsonl)
    if not path.exists():
        return {}
    freqs: dict[int, float] = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            idx = int(d["index"])
            freqs[idx] = float(d.get("frac_nonzero", 0))
    return freqs


def extract_coactivation_circuit(
    prompt: str,
    model_name: str = "gpt2",
    sae_release: str = "gpt2-small-res-jb",
    sae_hook: str = "blocks.6.hook_resid_pre",
    top_k_features: int = 30,
    min_coactivation: float = 0.1,
    max_breadth_ratio: float = 0.5,
    max_global_freq: float = 0.01,
    features_jsonl: str | Path | None = None,
    device: str = "cpu",
) -> dict:
    """Run GPT-2 + SAE on a prompt and extract co-activation circuits.

    Steps:
    1. Forward pass through GPT-2 via TransformerLens
    2. Encode residual stream activations through SAE
    3. Pre-filter broadly-activating features (global freq + per-prompt breadth)
    4. Find top-k features by peak activation
    5. Build co-activation graph via Jaccard similarity over token positions
    6. Assign roles by activation breadth

    Args:
        max_breadth_ratio: Max fraction of prompt tokens a feature can fire on
            before being excluded (per-prompt filter). Default 0.5 (50%).
        max_global_freq: Max frac_nonzero from corpus-level stats. Features
            activating on more than this fraction of all tokens are excluded.
            Default 0.01 (1%). Requires features_jsonl to be provided.
        features_jsonl: Path to features JSONL with frac_nonzero stats.
            If None, only per-prompt breadth filtering is used.

    Returns:
        Dict matching CircuitData schema.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if top_k_features < 1:
        raise ValueError(f"top_k_features must be >= 1, got {top_k_features}")
    if not 0 <= min_coactivation <= 1:
        raise ValueError(f"min_coactivation must be in [0, 1], got {min_coactivation}")
    if not 0 < max_breadth_ratio <= 1:
        raise ValueError(f"max_breadth_ratio must be in (0, 1], got {max_breadth_ratio}")
    if not 0 < max_global_freq <= 1:
        raise ValueError(f"max_global_freq must be in (0, 1], got {max_global_freq}")

    # Load global activation frequencies for pre-filtering
    global_freqs = _load_global_activation_frequencies(features_jsonl)

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

    # ── Filter broadly-activating features ──
    # Two-layer defense:
    # 1. Global frequency: exclude features with high frac_nonzero (corpus-level)
    # 2. Per-prompt breadth: exclude features firing on too many token positions
    #
    # This prevents broadly-activating "stop word" features from contaminating
    # every circuit with spurious cross-circuit convergences.
    n_before = len(feature_activations)

    # Layer 1: Global activation frequency filter
    if global_freqs:
        globally_filtered = {
            idx: positions
            for idx, positions in feature_activations.items()
            if global_freqs.get(idx, 0) <= max_global_freq
        }
        n_global_removed = n_before - len(globally_filtered)
        if n_global_removed > 0:
            print(f"    Filtered {n_global_removed} features with frac_nonzero > {max_global_freq}")
    else:
        globally_filtered = feature_activations

    # Layer 2: Per-prompt breadth filter
    breadth_threshold = max(1, int(seq_len * max_breadth_ratio))
    narrowed = {
        idx: positions
        for idx, positions in globally_filtered.items()
        if len(positions) <= breadth_threshold
    }
    n_breadth_removed = len(globally_filtered) - len(narrowed)
    if n_breadth_removed > 0:
        print(f"    Filtered {n_breadth_removed} features firing on > {max_breadth_ratio:.0%} of tokens")

    # Fall back to unfiltered if filtering would leave us with nothing
    if not narrowed:
        print("    WARNING: All features filtered out, falling back to unfiltered")
        narrowed = feature_activations

    peak_acts = {
        idx: max(a for a, _ in positions)
        for idx, positions in narrowed.items()
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


# ---------------------------------------------------------------------------
# Neuronpedia Circuit Tracer integration (Gemma 2 transcoders)
# ---------------------------------------------------------------------------

# Max local feature index for width_16k transcoders
_MAX_LOCAL_INDEX = 16384  # 0 through 16383


def extract_local_feature_index(global_index: int, layer: int) -> int:
    """Convert a Neuronpedia global transcoder feature index to a local index.

    Neuronpedia encodes global indices as: layer * 100000 + local_index.
    Layer 0 is special: indices are already local (0-16383).

    Args:
        global_index: The global feature index from node["feature"].
        layer: The expected layer number.

    Returns:
        The local feature index within the layer (0 to 16383 for width_16k).

    Raises:
        ValueError: If global index doesn't match the claimed layer, or if
            the derived local index is out of range.
    """
    if layer == 0:
        local_index = global_index
    else:
        expected_layer = global_index // 100000
        if expected_layer != layer:
            raise ValueError(
                f"Global index {global_index} does not match layer {layer} "
                f"(derived layer={expected_layer})"
            )
        local_index = global_index % 100000

    if local_index < 0 or local_index >= _MAX_LOCAL_INDEX:
        raise ValueError(
            f"Local index {local_index} out of range [0, {_MAX_LOCAL_INDEX}) "
            f"for global index {global_index}, layer {layer}"
        )

    return local_index


def parse_neuronpedia_circuit(
    data: dict,
    name: str,
    description: str,
    layer_filter: int,
) -> dict:
    """Parse a Neuronpedia Circuit Tracer attribution graph into Striatica schema.

    Translates the Neuronpedia JSON format (global feature indices, string layers,
    list-of-lists supernodes) into our internal circuit representation with local
    feature indices.

    Args:
        data: Raw JSON dict from Neuronpedia Circuit Tracer API.
        name: Name for the resulting circuit.
        description: Human-readable description.
        layer_filter: Only include nodes from this layer.

    Returns:
        Dict matching Striatica CircuitData schema with keys:
        name, description, type, source, nodes, edges, metadata.
    """
    # Build supernode role mapping: node_id -> role label
    supernode_roles: dict[str, str] = {}
    supernodes = data.get("qParams", {}).get("supernodes", [])
    for sn in supernodes:
        if not sn or len(sn) < 2:
            continue
        label = sn[0]  # first element is the label
        for node_id in sn[1:]:
            supernode_roles[node_id] = label

    # Filter nodes by layer and convert to local indices
    node_id_to_local: dict[str, int] = {}  # node_id -> local feature index
    nodes = []

    for node in data.get("nodes", []):
        node_layer = int(node["layer"])
        if node_layer != layer_filter:
            continue

        global_index = node["feature"]
        local_index = extract_local_feature_index(global_index, layer=node_layer)
        node_id = node["node_id"]
        node_id_to_local[node_id] = local_index

        role = supernode_roles.get(node_id, "unassigned")

        node_entry = {
            "featureIndex": local_index,
            "activation": node.get("activation", 0.0),
            "role": role,
        }
        if "influence" in node:
            node_entry["influence"] = node["influence"]

        nodes.append(node_entry)

    # Filter edges: both endpoints must be in-layer
    edges = []
    for link in data.get("links", []):
        source_id = link["source"]
        target_id = link["target"]
        if source_id in node_id_to_local and target_id in node_id_to_local:
            edges.append({
                "source": node_id_to_local[source_id],
                "target": node_id_to_local[target_id],
                "weight": link.get("weight", 0.0),
            })

    # Preserve source metadata
    metadata = {}
    if "metadata" in data:
        metadata = dict(data["metadata"])

    return {
        "name": name,
        "description": description,
        "type": "traced",
        "source": "neuronpedia",
        "nodes": nodes,
        "edges": edges,
        "metadata": metadata,
    }
