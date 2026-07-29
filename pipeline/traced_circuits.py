"""Build FROZEN-schema traced circuits from Neuronpedia attribution graphs.

Takes a raw Neuronpedia graph (the full S3 JSON: {metadata, qParams, nodes,
links}), the small graph record from GET /api/graph/{model}/{slug}, the public
source-set JSON from GET /api/source-set/{model}/{name}, and the local dataset
metadata JSON, and produces a traced circuit dict per the frozen schema:

    {name, description, type: "traced", source: "neuronpedia",
     nodes: [{featureIndex, activation, role, influence, instances}],
     edges: [],  # always empty this sprint (forward DAG: no same-layer edges)
     metadata: {model, slug, sourceSet, hfRepo, width, l0, l0Verified, prompt,
                pruning, generation, layerFilter, crossLayerMembers, fetchedAt,
                pipelineVersion, labelProvenance}}

Semantic-label hygiene: raw graph nodes carry `clerp` (semantic text) which
must NEVER be emitted. ``scrub_check`` enforces this — the only string field
allowed on an output node is ``role`` (a Neuronpedia public-graph supernode
label, or "unassigned", or "group-{i}" when redact_roles=True).

The build is pure and deterministic: no wall-clock timestamps, no randomness,
and inputs are never mutated. Two builds of identical inputs are
byte-identical JSON.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from pathlib import Path

from pipeline.banner import detail, error, info, success
from pipeline.circuits import extract_local_feature_index, parse_neuronpedia_circuit
from pipeline.graph_fetch import validate_graph_identity

PIPELINE_VERSION = "0.4.0"
LABEL_PROVENANCE = "neuronpedia-public-graph"

# layer_{L}/width_{W}/average_l0_{N} — the L0 identity source
_SAE_ID_RE = re.compile(r"^layer_(\d+)/(width_[^/]+)/average_l0_(\d+)$")

# Frozen schema key sets (also enforced by scrub_check)
_TOP_LEVEL_KEYS = {"name", "description", "type", "source", "nodes", "edges", "metadata"}
_NODE_KEYS = {"featureIndex", "activation", "role", "influence", "instances"}
_METADATA_KEYS = {
    "model",
    "slug",
    "sourceSet",
    "hfRepo",
    "width",
    "l0",
    "l0Verified",
    "prompt",
    "pruning",
    "generation",
    "layerFilter",
    "crossLayerMembers",
    "fetchedAt",
    "pipelineVersion",
    "labelProvenance",
}

# Raw Neuronpedia fields carrying semantic text — must never appear anywhere
# in an emitted circuit.
_FORBIDDEN_KEYS = {"clerp", "pos_str", "neg_str"}

_MAX_LOCAL_INDEX = 16383  # width_16k: local feature indices are 0..16383


def _kept_layer_feature_ids(raw_graph: dict, layer: int) -> set[str]:
    """node_ids of the graph nodes kept as layer-``layer`` feature nodes.

    Mirrors the keep-filter in ``parse_neuronpedia_circuit``: the node's layer
    must parse as int and equal ``layer``, and its global feature index must
    yield a valid local index (excludes "mlp reconstruction error" nodes with
    feature=-1 and non-numeric layers like "E").
    """
    kept: set[str] = set()
    for node in raw_graph.get("nodes", []):
        try:
            node_layer = int(node["layer"])
        except (TypeError, ValueError):
            continue
        if node_layer != layer:
            continue
        try:
            extract_local_feature_index(node["feature"], layer=node_layer)
        except (TypeError, ValueError):
            continue
        kept.add(node["node_id"])
    return kept


def _resolve_role(roles: list[str]) -> str:
    """Aggregate per-instance roles into a single role for a feature.

    Most-frequent non-'unassigned' role wins regardless of how many instances
    are unassigned; ties break to the lexicographically smallest label. All
    unassigned -> 'unassigned'.
    """
    counts = Counter(r for r in roles if r != "unassigned")
    if not counts:
        return "unassigned"
    best = max(counts.values())
    return min(label for label, n in counts.items() if n == best)


def build_traced_circuit(
    raw_graph: dict,
    record: dict,
    source_set: dict,
    dataset_metadata: dict,
    *,
    name: str | None = None,
    description: str | None = None,
    layer: int = 12,
    allow_l0_mismatch: bool = False,
    redact_roles: bool = False,
) -> dict:
    """Build a frozen-schema traced circuit from a Neuronpedia graph.

    Args:
        raw_graph: Full Neuronpedia graph JSON {metadata, qParams, nodes, links}.
        record: Graph record from GET /api/graph/{model}/{slug}.
        source_set: Source-set JSON from GET /api/source-set/{model}/{name}.
        dataset_metadata: Local dataset *-metadata.json contents.
        name: Circuit name; defaults to the graph slug.
        description: Circuit description; defaults to a derived string.
        layer: Transformer layer to keep feature nodes from.
        allow_l0_mismatch: Forwarded to the identity validator.
        redact_roles: Replace supernode labels with "group-{i}" placeholders.

    Returns:
        Traced circuit dict per the frozen schema (see module docstring).

    Raises:
        ValueError: propagated from identity validation, or from scrub_check
            if the built circuit somehow carries semantic text.
    """
    # Identity validation — hard gate before any parsing. Return value is
    # deliberately unused: l0/l0Verified are recomputed below so the circuit
    # metadata is derived from the inputs alone.
    validate_graph_identity(
        record,
        raw_graph,
        source_set,
        dataset_metadata,
        layer=layer,
        allow_l0_mismatch=allow_l0_mismatch,
    )

    graph_md = raw_graph.get("metadata", {}) or {}
    slug = record.get("slug") or graph_md.get("slug") or "traced-circuit"
    model = record.get("modelId") or graph_md.get("scan") or ""

    circuit_name = name if name is not None else slug
    if description is not None:
        circuit_description = description
    else:
        circuit_description = (
            f"Layer-{layer} traced circuit from Neuronpedia attribution "
            f"graph '{slug}' ({model})"
        )

    # ── Parse + layer filter (shared, tested parser) ────────────────
    parsed = parse_neuronpedia_circuit(
        raw_graph,
        name=circuit_name,
        description=circuit_description,
        layer_filter=layer,
    )
    dropped_by_layer = parsed["metadata"]["droppedByLayer"]

    # ── Aggregate duplicate featureIndex instances ──────────────────
    # activation = max raw; influence follows the max-activation instance
    # (first occurrence wins exact ties — stable); instances = count;
    # role = most-frequent non-'unassigned', ties lexicographic.
    groups: dict[int, dict] = {}
    for inst in parsed["nodes"]:
        feature_index = inst["featureIndex"]
        raw_act = inst.get("activation")
        raw_act = float(raw_act) if raw_act is not None else 0.0
        group = groups.get(feature_index)
        if group is None:
            groups[feature_index] = {
                "activation": raw_act,
                "influence": inst.get("influence"),
                "instances": 1,
                "roles": [inst.get("role", "unassigned")],
            }
        else:
            group["instances"] += 1
            group["roles"].append(inst.get("role", "unassigned"))
            if raw_act > group["activation"]:
                group["activation"] = raw_act
                group["influence"] = inst.get("influence")

    instances_total = len(parsed["nodes"])
    nodes_kept = len(groups)
    feature_instances_collapsed = instances_total - nodes_kept

    # ── Per-circuit max-normalization of activation (after aggregation) ──
    max_act = max((g["activation"] for g in groups.values()), default=0.0)

    # ── Role redaction map (supernode order → group index) ──────────
    supernodes = (raw_graph.get("qParams", {}) or {}).get("supernodes", []) or []
    label_to_group = {
        sn[0]: f"group-{i}" for i, sn in enumerate(supernodes) if sn
    }

    nodes = []
    for feature_index in sorted(groups):
        group = groups[feature_index]
        role = _resolve_role(group["roles"])
        if redact_roles and role != "unassigned":
            role = label_to_group.get(role, "group-unknown")
        influence = group["influence"]
        nodes.append(
            {
                "featureIndex": feature_index,
                "activation": group["activation"] / max_act if max_act > 0 else 0.0,
                "role": role,
                "influence": float(influence) if influence is not None else None,
                "instances": group["instances"],
            }
        )

    # ── Edge accounting (edges are always emitted empty this sprint) ──
    # parse keeps only links with BOTH endpoints on kept in-layer feature
    # nodes; endpoints that collapse to the same featureIndex are self-loops.
    # In-layer edges between DIFFERENT features are structurally impossible
    # (forward DAG); if one ever appeared it would be folded into the
    # cross-layer dropped count below so the accounting invariant holds.
    edges_total = len(raw_graph.get("links", []))
    self_loops_dropped = sum(
        1 for e in parsed["edges"] if e["source"] == e["target"]
    )
    edges_kept = 0
    cross_layer_edges_dropped = edges_total - self_loops_dropped - edges_kept

    layer_filter_md = {
        "layer": layer,
        "nodesTotal": len(raw_graph.get("nodes", [])),
        "nodesKept": nodes_kept,
        "featureInstancesCollapsed": feature_instances_collapsed,
        "droppedByLayer": dict(dropped_by_layer),
        "edgesTotal": edges_total,
        "edgesKept": edges_kept,
        "crossLayerEdgesDropped": cross_layer_edges_dropped,
        "selfLoopsDropped": self_loops_dropped,
    }

    # ── Cross-layer supernode members (informational only) ──────────
    # Supernode members that are NOT kept in-layer feature nodes, keyed by
    # the node_id's layer prefix. Never emitted as featureIndex entries.
    kept_node_ids = _kept_layer_feature_ids(raw_graph, layer)
    cross_by_layer: dict[str, int] = {}
    cross_count = 0
    for sn in supernodes:
        for member in sn[1:]:
            if member in kept_node_ids:
                continue
            prefix = str(member).split("_", 1)[0]
            cross_by_layer[prefix] = cross_by_layer.get(prefix, 0) + 1
            cross_count += 1
    cross_layer_members = {"count": cross_count, "byLayer": cross_by_layer}

    # ── L0 identity fields (computed from inputs, not the validator) ──
    # Graph-side dictionary: the layer's saelensSaeId in the source set.
    graph_l0: int | None = None
    width: str | None = None
    hf_repo: str | None = None
    for source in source_set.get("sources", []):
        m = _SAE_ID_RE.match(source.get("saelensSaeId", ""))
        if m and int(m.group(1)) == layer:
            width = m.group(2)
            graph_l0 = int(m.group(3))
            hf_repo = source.get("hfRepoId")
            break

    transcoder_md = dataset_metadata.get("transcoder", {}) or {}
    dataset_l0 = transcoder_md.get("l0_variant")
    dataset_l0 = int(dataset_l0) if dataset_l0 is not None else None
    if graph_l0 is not None:
        l0 = graph_l0
        l0_verified = dataset_l0 is not None and graph_l0 == dataset_l0
    else:
        # Unverifiable from the source set — fall back to the dataset side.
        l0 = dataset_l0 if dataset_l0 is not None else -1
        l0_verified = False

    metadata = {
        "model": model,
        "slug": slug,
        "sourceSet": record.get("sourceSetName") or source_set.get("name") or "",
        "hfRepo": hf_repo or transcoder_md.get("repo_id") or "",
        "width": width or transcoder_md.get("width") or "",
        "l0": l0,
        "l0Verified": l0_verified,
        "prompt": graph_md.get("prompt", ""),
        "pruning": copy.deepcopy(graph_md.get("pruning_settings", {})),
        "generation": copy.deepcopy(graph_md.get("generation_settings", {})),
        "layerFilter": layer_filter_md,
        "crossLayerMembers": cross_layer_members,
        # From inputs only — never wall-clock time (build determinism).
        "fetchedAt": record.get("fetchedAt") or graph_md.get("fetchedAt") or None,
        "pipelineVersion": PIPELINE_VERSION,
        "labelProvenance": LABEL_PROVENANCE,
    }

    circuit = {
        "name": circuit_name,
        "description": circuit_description,
        "type": "traced",
        "source": "neuronpedia",
        "nodes": nodes,
        "edges": [],
        "metadata": metadata,
    }

    # Final gate: no semantic text may leave the builder. Called through the
    # module global so tests (and callers) can intercept it.
    scrub_check(circuit)
    return circuit


def scrub_check(circuit: dict) -> None:
    """Raise ValueError if a traced circuit carries any semantic-text leak.

    Enforces the frozen schema shape as the leak barrier: forbidden raw
    Neuronpedia fields (clerp/pos_str/neg_str) anywhere in the structure,
    unexpected keys at any schema level, and any node string field other
    than ``role`` are all rejected.
    """
    if not isinstance(circuit, dict):
        raise ValueError("scrub_check: circuit must be a dict")

    _reject_forbidden_keys(circuit, path="circuit")

    if set(circuit.keys()) != _TOP_LEVEL_KEYS:
        raise ValueError(
            f"scrub_check: top-level keys {sorted(circuit.keys())} do not "
            f"match the frozen schema {sorted(_TOP_LEVEL_KEYS)}"
        )
    for key in ("name", "description"):
        if not isinstance(circuit[key], str):
            raise ValueError(f"scrub_check: '{key}' must be a string")
    if circuit["type"] != "traced":
        raise ValueError("scrub_check: type must be 'traced'")
    if circuit["source"] != "neuronpedia":
        raise ValueError("scrub_check: source must be 'neuronpedia'")
    if circuit["edges"] != []:
        raise ValueError("scrub_check: edges must be [] (same-layer edges are "
                         "structurally impossible this sprint)")

    if not isinstance(circuit["nodes"], list):
        raise ValueError("scrub_check: nodes must be a list")
    for i, node in enumerate(circuit["nodes"]):
        if not isinstance(node, dict):
            raise ValueError(f"scrub_check: nodes[{i}] is not a dict")
        if set(node.keys()) != _NODE_KEYS:
            unexpected = sorted(set(node.keys()) - _NODE_KEYS)
            raise ValueError(
                f"scrub_check: nodes[{i}] has unexpected/missing fields "
                f"(unexpected: {unexpected}) — node keys must be exactly "
                f"{sorted(_NODE_KEYS)}"
            )
        fi = node["featureIndex"]
        if not isinstance(fi, int) or isinstance(fi, bool) or not (
            0 <= fi <= _MAX_LOCAL_INDEX
        ):
            raise ValueError(
                f"scrub_check: nodes[{i}].featureIndex {fi!r} is not a local "
                f"index in [0, {_MAX_LOCAL_INDEX}]"
            )
        act = node["activation"]
        if isinstance(act, bool) or not isinstance(act, (int, float)) or not (
            0.0 <= float(act) <= 1.0
        ):
            raise ValueError(
                f"scrub_check: nodes[{i}].activation {act!r} is not a float "
                f"in [0, 1]"
            )
        if not isinstance(node["role"], str):
            raise ValueError(f"scrub_check: nodes[{i}].role must be a string")
        infl = node["influence"]
        if infl is not None and (
            isinstance(infl, bool) or not isinstance(infl, (int, float))
        ):
            raise ValueError(
                f"scrub_check: nodes[{i}].influence {infl!r} must be float or "
                f"null"
            )
        inst = node["instances"]
        if not isinstance(inst, int) or isinstance(inst, bool) or inst < 1:
            raise ValueError(
                f"scrub_check: nodes[{i}].instances {inst!r} must be a "
                f"positive int"
            )
        # The ONLY string field allowed on a node is 'role' (guaranteed by
        # the exact key set + per-field type checks above).

    md = circuit["metadata"]
    if not isinstance(md, dict):
        raise ValueError("scrub_check: metadata must be a dict")
    if set(md.keys()) != _METADATA_KEYS:
        unexpected = sorted(set(md.keys()) - _METADATA_KEYS)
        missing = sorted(_METADATA_KEYS - set(md.keys()))
        raise ValueError(
            f"scrub_check: metadata keys mismatch (unexpected: {unexpected}, "
            f"missing: {missing})"
        )
    return None


def _reject_forbidden_keys(obj, path: str) -> None:
    """Recursively reject raw Neuronpedia semantic-text fields anywhere."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in _FORBIDDEN_KEYS:
                raise ValueError(
                    f"scrub_check: forbidden semantic field '{key}' found at "
                    f"{path} — semantic labels must never be emitted"
                )
            _reject_forbidden_keys(value, path=f"{path}.{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            _reject_forbidden_keys(value, path=f"{path}[{i}]")


# ── Output Writing ──────────────────────────────────────────────────

# Suffixes/names generate_datasets_manifest must never list.
_DATASETS_MANIFEST_NAME = "datasets.json"
_DATASETS_SKIP_SUFFIXES = ("-metadata.json", "-validation.json")


def _manifest_entry(circuit: dict, dataset_stem: str) -> dict:
    """Frozen-schema manifest entry for one traced circuit."""
    return {
        "id": circuit["name"],
        "name": circuit["name"],
        "type": "traced",
        "description": circuit["description"],
        "nodeCount": len(circuit["nodes"]),
        "edgeCount": len(circuit["edges"]),
        "path": f"/data/circuits/{dataset_stem}/{circuit['name']}.json",
        "model": circuit["metadata"]["model"],
        "slug": circuit["metadata"]["slug"],
    }


def write_traced_outputs(circuits: list[dict], out_root: Path, dataset_stem: str) -> Path:
    """Write traced circuit JSONs + manifest.json under the dataset-stem dir.

    Writes exactly one ``{circuit["name"]}.json`` per circuit plus a
    ``manifest.json`` into ``{out_root}/circuits/{dataset_stem}`` (created if
    missing), and touches nothing else under ``out_root``.

    Every circuit passes through ``scrub_check`` (via the module global, so
    it can be intercepted) BEFORE its file is written — semantic text never
    reaches disk. Circuit names that would resolve outside the dataset-stem
    directory (path traversal, absolute paths) are rejected before anything
    is written.

    Args:
        circuits: Frozen-schema traced circuit dicts (build_traced_circuit
            output), written in input order.
        out_root: Public data root, e.g. frontend/public/data.
        dataset_stem: Dataset stem naming the output directory, e.g.
            "gemma-2-2b-layer12-l0604".

    Returns:
        Path of the dataset-stem directory that was written.

    Raises:
        ValueError: on a hostile circuit name or a scrub_check failure; the
            offending circuit's file is never written.
    """
    out_root = Path(out_root)
    stem_dir = out_root / "circuits" / dataset_stem
    resolved_stem = stem_dir.resolve()

    # Name safety gate — validate ALL names before creating or writing
    # anything, so a hostile name can never place a file outside stem_dir.
    for circuit in circuits:
        name = circuit["name"]
        target = stem_dir / f"{name}.json"
        if target.resolve().parent != resolved_stem:
            error(f"Refusing circuit name outside dataset dir: {name!r}")
            raise ValueError(
                f"write_traced_outputs: circuit name {name!r} resolves "
                f"outside {stem_dir} — refusing to write"
            )

    stem_dir.mkdir(parents=True, exist_ok=True)
    info("Writing", f"{len(circuits)} traced circuit(s) -> {stem_dir}", emoji="🔗")

    for circuit in circuits:
        # Leak barrier: scrub via the MODULE GLOBAL, before the file exists.
        try:
            scrub_check(circuit)
        except ValueError:
            error(
                f"scrub_check failed for circuit '{circuit.get('name')}' — "
                f"aborting write, nothing emitted for this circuit"
            )
            raise
        target = stem_dir / f"{circuit['name']}.json"
        target.write_text(json.dumps(circuit, indent=2) + "\n")
        detail(f"wrote {target.name} ({len(circuit['nodes'])} nodes)")

    manifest = {
        "circuits": [_manifest_entry(c, dataset_stem) for c in circuits]
    }
    manifest_path = stem_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    detail(f"wrote {manifest_path.name} ({len(circuits)} circuit(s) listed)")
    success(f"Traced circuit outputs written to {stem_dir}")
    return stem_dir


def generate_datasets_manifest(public_data_dir: Path) -> Path:
    """Write {public_data_dir}/datasets.json listing top-level dataset JSONs.

    Scans only the TOP-LEVEL ``*.json`` files of ``public_data_dir`` —
    skipping directories (circuits/ included), ``*-metadata.json``,
    ``*-validation.json``, and ``datasets.json`` itself — and reads each
    file's top-level ``model``, ``layer`` and ``numFeatures`` keys.

    Writes ``datasets.json`` as a JSON array of
    ``{file, model, layer, numFeatures}`` entries sorted by ``file``
    ascending (``file`` is the basename), overwriting any stale
    ``datasets.json``. No other file is modified.

    Args:
        public_data_dir: The public data root, e.g. frontend/public/data.

    Returns:
        Path of the written datasets.json.
    """
    public_data_dir = Path(public_data_dir)
    entries = []
    for path in sorted(public_data_dir.iterdir()):
        if not path.is_file() or path.suffix != ".json":
            continue
        if path.name == _DATASETS_MANIFEST_NAME:
            continue
        if path.name.endswith(_DATASETS_SKIP_SUFFIXES):
            continue
        data = json.loads(path.read_text())
        entries.append(
            {
                "file": path.name,
                "model": data["model"],
                "layer": data["layer"],
                "numFeatures": data["numFeatures"],
            }
        )
        detail(f"indexed {path.name}")
    entries.sort(key=lambda entry: entry["file"])

    manifest_path = public_data_dir / _DATASETS_MANIFEST_NAME
    manifest_path.write_text(json.dumps(entries, indent=2) + "\n")
    success(f"datasets.json written ({len(entries)} dataset(s)) -> {manifest_path}")
    return manifest_path
