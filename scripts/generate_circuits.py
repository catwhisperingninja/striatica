#!/usr/bin/env python3
# striatica/scripts/generate_circuits.py
"""Generate circuit JSON files for the frontend circuit view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.config import GPT2_SMALL_L6, DATA_DIR, OUTPUT_DIR
from pipeline.circuits import extract_coactivation_circuit, extract_similarity_circuit

CIRCUITS_DIR = OUTPUT_DIR / "circuits"

DEFAULT_PROMPTS = [
    ("capital-of-france", "The capital of France is"),
    ("once-upon-a-time", "Once upon a time, there was a"),
    ("fibonacci-code", "def fibonacci(n):\n    if n <= 1:"),
    ("cat-sat-on", "The cat sat on the"),
    ("moon-landing", "In 1969, humans first landed on the"),
]


def write_circuit(circuit: dict, name: str) -> Path:
    """Write a circuit dict to JSON file."""
    CIRCUITS_DIR.mkdir(parents=True, exist_ok=True)
    path = CIRCUITS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(circuit, f, indent=2)
    print(f"  Wrote {path} ({len(circuit['nodes'])} nodes, {len(circuit['edges'])} edges)")
    return path


def write_manifest(entries: list[dict]) -> Path:
    """Write manifest.json listing all available circuits."""
    CIRCUITS_DIR.mkdir(parents=True, exist_ok=True)
    path = CIRCUITS_DIR / "manifest.json"
    with open(path, "w") as f:
        json.dump({"circuits": entries}, f, indent=2)
    print(f"  Wrote manifest with {len(entries)} circuits")
    return path


def pick_seed_features(features_jsonl: Path, count: int = 5) -> list[int]:
    """Pick seed features with highest maxAct from different clusters.

    Loads the frontend dataset to get cluster labels, then picks the
    highest-activation feature from each of the top clusters.
    """
    # Load features and their maxAct values
    features: list[tuple[int, float]] = []
    with open(features_jsonl) as f:
        for line in f:
            d = json.loads(line)
            features.append((int(d["index"]), float(d.get("maxActApprox", 0))))

    # Try to load cluster labels from the frontend dataset
    dataset_path = OUTPUT_DIR / f"{GPT2_SMALL_L6.model_id}-{GPT2_SMALL_L6.layer}.json"
    cluster_labels: dict[int, int] = {}
    if dataset_path.exists():
        with open(dataset_path) as f:
            data = json.load(f)
        for i, label in enumerate(data.get("clusterLabels", [])):
            cluster_labels[i] = label

    if cluster_labels:
        # Group by cluster, pick highest maxAct from each non-negative cluster
        by_cluster: dict[int, list[tuple[int, float]]] = {}
        for idx, act in features:
            cl = cluster_labels.get(idx, -1)
            if cl < 0:
                continue
            by_cluster.setdefault(cl, []).append((idx, act))

        # Sort clusters by their best feature's activation
        cluster_bests = []
        for cl, feats in by_cluster.items():
            best = max(feats, key=lambda x: x[1])
            cluster_bests.append((cl, best[0], best[1]))
        cluster_bests.sort(key=lambda x: x[2], reverse=True)

        seeds = [idx for _, idx, _ in cluster_bests[:count]]
    else:
        # Fallback: just pick top maxAct features
        features.sort(key=lambda x: x[1], reverse=True)
        seeds = [idx for idx, _ in features[:count]]

    return seeds


def _decontaminate_coact_circuits(circuits: dict[str, dict]) -> dict[str, dict]:
    """Remove broadly-activating features that appear in ALL co-activation circuits.

    Co-activation circuits use different prompts, so a feature appearing in
    every single one is a noisy broadly-activating feature, not a prompt-specific
    circuit member. We remove these and re-normalize activations.
    """
    if len(circuits) < 3:
        return circuits

    from collections import Counter

    # Count how many circuits each feature appears in
    feature_counts: Counter = Counter()
    for circuit in circuits.values():
        for node in circuit["nodes"]:
            feature_counts[node["featureIndex"]] += 1

    # Identify contaminating features (present in ALL circuits)
    total = len(circuits)
    contaminated = {feat for feat, count in feature_counts.items() if count >= total}

    if not contaminated:
        return circuits

    print(f"\n  Cross-circuit decontamination: removing {len(contaminated)} "
          f"broadly-activating features: {sorted(contaminated)}")

    # Remove contaminated features from each circuit
    cleaned = {}
    for name, circuit in circuits.items():
        clean_nodes = [n for n in circuit["nodes"] if n["featureIndex"] not in contaminated]
        clean_node_set = {n["featureIndex"] for n in clean_nodes}
        clean_edges = [
            e for e in circuit["edges"]
            if e["source"] in clean_node_set and e["target"] in clean_node_set
        ]
        # Re-normalize activations to 0-1
        if clean_nodes:
            max_act = max(n["activation"] for n in clean_nodes)
            if max_act > 0:
                for n in clean_nodes:
                    n["activation"] = round(n["activation"] / max_act, 4)

        cleaned[name] = {
            **circuit,
            "nodes": clean_nodes,
            "edges": clean_edges,
        }
        removed = len(circuit["nodes"]) - len(clean_nodes)
        if removed > 0:
            print(f"    {name}: removed {removed} nodes, "
                  f"{len(clean_nodes)} remain")

    return cleaned


def generate_batch_defaults() -> None:
    """Generate 5 co-activation + 5 similarity circuits."""
    cfg = GPT2_SMALL_L6
    features_jsonl = DATA_DIR / f"{cfg.model_id}_{cfg.layer}_features.jsonl"
    manifest_entries: list[dict] = []

    # --- Co-activation circuits ---
    print("=== Generating co-activation circuits ===")
    coact_circuits: dict[str, dict] = {}
    for name, prompt in DEFAULT_PROMPTS:
        circuit_name = f"coact-{name}"
        print(f"\n  Prompt: {prompt!r}")
        try:
            circuit = extract_coactivation_circuit(
                prompt=prompt,
                model_name="gpt2",
                sae_release=cfg.sae_release,
                sae_hook=cfg.sae_hook,
                top_k_features=30,
                min_coactivation=0.1,
                features_jsonl=features_jsonl,
                device="cpu",
            )
            circuit["name"] = circuit_name
            coact_circuits[circuit_name] = circuit
        except Exception as e:
            print(f"  ERROR generating {circuit_name}: {e}")

    # Cross-circuit decontamination: remove broadly-activating features
    coact_circuits = _decontaminate_coact_circuits(coact_circuits)

    for circuit_name, circuit in coact_circuits.items():
        write_circuit(circuit, circuit_name)
        manifest_entries.append({
            "id": circuit_name,
            "name": circuit_name,
            "type": "coactivation",
            "description": circuit["description"],
            "nodeCount": len(circuit["nodes"]),
            "edgeCount": len(circuit["edges"]),
            "path": f"/data/circuits/{circuit_name}.json",
        })

    # --- Similarity circuits ---
    print("\n=== Generating similarity circuits ===")
    if not features_jsonl.exists():
        print(f"  WARNING: {features_jsonl} not found, skipping similarity circuits")
    else:
        seeds = pick_seed_features(features_jsonl, count=5)
        print(f"  Seed features: {seeds}")
        for seed in seeds:
            circuit_name = f"sim-{seed}"
            print(f"\n  Seed: feature #{seed}")
            try:
                circuit = extract_similarity_circuit(
                    features_jsonl=features_jsonl,
                    seed_feature=seed,
                    depth=2,
                    top_k_neighbors=5,
                )
                write_circuit(circuit, circuit_name)
                manifest_entries.append({
                    "id": circuit_name,
                    "name": circuit_name,
                    "type": "similarity",
                    "description": circuit["description"],
                    "nodeCount": len(circuit["nodes"]),
                    "edgeCount": len(circuit["edges"]),
                    "path": f"/data/circuits/{circuit_name}.json",
                })
            except Exception as e:
                print(f"  ERROR generating {circuit_name}: {e}")

    write_manifest(manifest_entries)
    print(f"\nDone! Generated {len(manifest_entries)} circuits in {CIRCUITS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate circuit data for frontend")
    parser.add_argument("--type", choices=["coactivation", "similarity"], help="Circuit type")
    parser.add_argument("--prompt", help="Prompt text (for coactivation)")
    parser.add_argument("--seed-feature", type=int, help="Seed feature index (for similarity)")
    parser.add_argument("--name", help="Circuit identifier")
    parser.add_argument("--top-k", type=int, default=30, help="Top-k features (default: 30)")
    parser.add_argument("--min-weight", type=float, default=0.1, help="Min edge weight (default: 0.1)")
    parser.add_argument("--depth", type=int, default=2, help="BFS depth for similarity (default: 2)")
    parser.add_argument("--batch-defaults", action="store_true", help="Generate all default circuits")
    args = parser.parse_args()

    if args.batch_defaults:
        generate_batch_defaults()
        return

    if not args.type:
        parser.error("--type is required (or use --batch-defaults)")

    cfg = GPT2_SMALL_L6
    features_jsonl = DATA_DIR / f"{cfg.model_id}_{cfg.layer}_features.jsonl"

    if args.type == "coactivation":
        if not args.prompt:
            parser.error("--prompt is required for coactivation circuits")
        name = args.name or f"coact-custom"
        circuit = extract_coactivation_circuit(
            prompt=args.prompt,
            model_name="gpt2",
            sae_release=cfg.sae_release,
            sae_hook=cfg.sae_hook,
            top_k_features=args.top_k,
            min_coactivation=args.min_weight,
            features_jsonl=features_jsonl if features_jsonl.exists() else None,
            device="cpu",
        )
        circuit["name"] = name
        write_circuit(circuit, name)

    elif args.type == "similarity":
        if args.seed_feature is None:
            parser.error("--seed-feature is required for similarity circuits")
        name = args.name or f"sim-{args.seed_feature}"
        circuit = extract_similarity_circuit(
            features_jsonl=features_jsonl,
            seed_feature=args.seed_feature,
            depth=args.depth,
            top_k_neighbors=args.top_k,
        )
        write_circuit(circuit, name)

    print("Done!")


if __name__ == "__main__":
    main()
