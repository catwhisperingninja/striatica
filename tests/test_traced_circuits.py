# striatica/tests/test_traced_circuits.py
"""Tests for pipeline/traced_circuits.py — traced circuit builder (unit: traced_builder).

All tests run against the REAL Neuronpedia fixtures in tests/fixtures/
(gemma-fact-dallas-austin graph, fetched live 2026-07-28; labels neutralized to
"label-N", clerp blanked, prompt redacted). Perturbation tests mutate deep
copies of the real fixture in memory (the sanctioned pattern — no synthetic
mock JSON files).

API CONTRACT (implementers: build pipeline/traced_circuits.py to this):

    build_traced_circuit(
        raw_graph: dict,          # full Neuronpedia graph {metadata, qParams, nodes, links}
        record: dict,             # GET /api/graph/{model}/{slug} record
        source_set: dict,         # GET /api/source-set/{model}/{name} JSON
        dataset_metadata: dict,   # local *-metadata.json contents
        *,
        name: str | None = None,          # default: record/graph slug
        description: str | None = None,   # default: non-empty derived string
        layer: int = 12,
        allow_l0_mismatch: bool = False,
        redact_roles: bool = False,
    ) -> dict                     # FROZEN traced circuit schema

    scrub_check(circuit: dict) -> None    # raises ValueError on any leak

Contract details these tests pin:

- traced_circuits must bind the identity validator into its own namespace via
  ``from pipeline.graph_fetch import validate_graph_identity`` so that
  monkeypatching ``pipeline.traced_circuits.validate_graph_identity`` intercepts
  it. build_traced_circuit calls it exactly once per build, passing record,
  source_set and dataset_metadata (any position/keyword) plus
  ``allow_l0_mismatch`` as a KEYWORD; any exception it raises propagates. Its
  return value must NOT be relied upon (the stub returns None).
- metadata.l0 / metadata.l0Verified are computed by build itself from the
  layer's saelensSaeId in source_set (average_l0_6 -> 6) vs
  dataset_metadata["transcoder"]["l0_variant"] (604) -> l0Verified False.
- Aggregation of duplicate featureIndex instances: activation = max raw,
  instances = instance count, influence = influence of the max-activation
  instance, role = most-frequent non-'unassigned' role with ties broken by
  lexicographically smallest label.
- Activation is per-circuit max-normalized AFTER aggregation (max node == 1.0).
- layerFilter.droppedByLayer counts every raw node not kept, keyed by
  str(node["layer"]) — including layer-12 nodes with no valid feature index
  (mlp reconstruction error nodes, feature=-1) under key "12". Invariant:
  nodesTotal == nodesKept + featureInstancesCollapsed + sum(droppedByLayer).
- edges is ALWAYS [] and edgesKept == 0; edgesTotal == edgesKept +
  crossLayerEdgesDropped + selfLoopsDropped. A link whose endpoints collapse to
  the same kept featureIndex counts as a self-loop.
- crossLayerMembers counts qParams.supernodes members that are not kept
  layer-12 feature nodes, keyed by the node_id's layer prefix (string before
  the first "_"). Informational only — never emitted as featureIndex entries.
- redact_roles=True maps each attached supernode label to "group-{i}" where i
  follows qParams.supernodes order; 'unassigned' is left unchanged; no original
  label string may survive anywhere in the circuit.
- build_traced_circuit calls scrub_check (module-level, by name) on its result.
- fetchedAt comes from inputs (None for these fixtures), never from wall-clock
  time: two builds of the same inputs must be byte-identical JSON.
"""

import copy
import inspect
import json
import sys
import types
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _load_fixture(filename: str) -> dict:
    with open(FIXTURES / filename) as f:
        return json.load(f)


_RAW_GRAPH = _load_fixture("neuronpedia_graph_gemma.json")
_RECORD = _load_fixture("neuronpedia_graph_record_gemma.json")
_SOURCE_SET = _load_fixture("neuronpedia_sourceset_gemma.json")

_DATASET_METADATA_PATH = (
    Path(__file__).parent.parent
    / "frontend"
    / "public"
    / "data"
    / "gemma-2-2b-layer12-l0604-metadata.json"
)
_DATASET_METADATA = json.loads(_DATASET_METADATA_PATH.read_text())

# ---------------------------------------------------------------------------
# pipeline.graph_fetch shim: the fetcher unit may not exist yet. Install a
# no-op stub module ONLY if the real one is absent, so pipeline.traced_circuits
# can be imported. Per-test behavior is always controlled by monkeypatching
# pipeline.traced_circuits.validate_graph_identity (see identity_calls).
# ---------------------------------------------------------------------------
try:
    import pipeline.graph_fetch  # noqa: F401
except ImportError:
    _stub_module = types.ModuleType("pipeline.graph_fetch")

    def _stub_validate_graph_identity(*args, **kwargs):
        return None

    _stub_module.validate_graph_identity = _stub_validate_graph_identity
    sys.modules["pipeline.graph_fetch"] = _stub_module

import pipeline.traced_circuits as traced_circuits
from pipeline.traced_circuits import build_traced_circuit, scrub_check


# ---------------------------------------------------------------------------
# Expected values computed from the REAL fixture contents (verified 2026-07-28)
# ---------------------------------------------------------------------------

# The 6 layer-12 cross-layer-transcoder feature nodes (layer-LOCAL indices).
L12_LOCALS = {2082, 2799, 8580, 10631, 12601, 12910}

# Raw activations of the 6 kept nodes, straight from the fixture.
RAW_ACT = {
    2082: 7.404547691345215,
    2799: 6.065004825592041,
    12910: 9.857479095458984,
    8580: 7.126477241516113,
    10631: 16.468292236328125,
    12601: 5.670328140258789,
}
MAX_RAW_ACT = 16.468292236328125  # feature 10631

RAW_INFL = {
    2082: 0.6934998631477356,
    2799: 0.6910012364387512,
    12910: 0.6598602533340454,
    8580: 0.6354414820671082,
    10631: 0.4975889325141907,
    12601: 0.6785522699356079,
}

# Fixture node census: 53 nodes = 12 layer-"12" (6 CLT + 6 mlp-error feature=-1)
# + 11 layer-"E" embedding + 5 layer-"27" logit + 25 layer-"0".
NODES_TOTAL = 53
EDGES_TOTAL = 250
DROPPED_BY_LAYER = {"E": 11, "27": 5, "0": 25, "12": 6}

# qParams.supernodes: 19 members total, none of them layer-12, keyed by the
# node_id layer prefix.
CROSS_LAYER_COUNT = 19
CROSS_LAYER_BY_LAYER = {
    "15": 1,
    "6": 2,
    "4": 2,
    "3": 1,
    "1": 1,
    "0": 1,
    "20": 1,
    "19": 2,
    "16": 2,
    "14": 1,
    "7": 2,
    "18": 1,
    "21": 1,
    "17": 1,
}

NODE_KEYS = {"featureIndex", "activation", "role", "influence", "instances"}
METADATA_KEYS = {
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
TOP_LEVEL_KEYS = {"name", "description", "type", "source", "nodes", "edges", "metadata"}

EXPECTED_LAYER_FILTER = {
    "layer": 12,
    "nodesTotal": NODES_TOTAL,
    "nodesKept": 6,
    "featureInstancesCollapsed": 0,
    "droppedByLayer": DROPPED_BY_LAYER,
    "edgesTotal": EDGES_TOTAL,
    "edgesKept": 0,
    "crossLayerEdgesDropped": 250,
    "selfLoopsDropped": 0,
}


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def raw_graph():
    return copy.deepcopy(_RAW_GRAPH)


@pytest.fixture
def record():
    return copy.deepcopy(_RECORD)


@pytest.fixture
def source_set():
    return copy.deepcopy(_SOURCE_SET)


@pytest.fixture
def dataset_metadata():
    return copy.deepcopy(_DATASET_METADATA)


@pytest.fixture
def identity_calls(monkeypatch):
    """Stub out the graph_fetch identity validator; record every call."""
    calls = []

    def _stub(*args, **kwargs):
        calls.append((args, kwargs))
        return None

    monkeypatch.setattr(traced_circuits, "validate_graph_identity", _stub)
    return calls


@pytest.fixture
def build(raw_graph, record, source_set, dataset_metadata, identity_calls):
    """Build against the (possibly test-mutated) real fixture graph.

    The l0 pairing is a CONFIRMED mismatch (graph dict average_l0_6 vs local
    dataset l0_604), so every happy-path build passes allow_l0_mismatch=True.
    """

    def _build(**overrides):
        kwargs = {"layer": 12, "allow_l0_mismatch": True}
        kwargs.update(overrides)
        return build_traced_circuit(
            raw_graph, record, source_set, dataset_metadata, **kwargs
        )

    return _build


def _node_map(circuit: dict) -> dict:
    return {n["featureIndex"]: n for n in circuit["nodes"]}


def _raw_node(graph: dict, node_id: str) -> dict:
    return next(n for n in graph["nodes"] if n["node_id"] == node_id)


def _add_instance(graph: dict, base_node_id: str, new_ctx: int,
                  activation: float, influence: float) -> str:
    """Add a second instance of a REAL layer-12 feature node at another ctx.

    Copies the real node dict and changes only node_id/ctx_idx plus the
    activation/influence (values reused from other real fixture nodes).
    """
    base = _raw_node(graph, base_node_id)
    dup = copy.deepcopy(base)
    prefix = base_node_id.rsplit("_", 1)[0]
    dup["node_id"] = f"{prefix}_{new_ctx}"
    dup["ctx_idx"] = new_ctx
    dup["activation"] = activation
    dup["influence"] = influence
    graph["nodes"].append(dup)
    return dup["node_id"]


# ---------------------------------------------------------------------------
# Fixture preconditions (guard against fixture drift — other tests build on
# these exact facts)
# ---------------------------------------------------------------------------


def test_fixture_preconditions():
    assert len(_RAW_GRAPH["nodes"]) == NODES_TOTAL
    assert len(_RAW_GRAPH["links"]) == EDGES_TOTAL
    labels = [sn[0] for sn in _RAW_GRAPH["qParams"]["supernodes"]]
    assert labels == ["label-0", "label-1", "label-2", "label-3", "label-4"]
    # No supernode member is a layer-12 node_id in the base fixture.
    members = [m for sn in _RAW_GRAPH["qParams"]["supernodes"] for m in sn[1:]]
    assert len(members) == CROSS_LAYER_COUNT
    assert not any(m.startswith("12_") for m in members)
    # No link connects two layer-12 nodes (forward DAG).
    l12_ids = {n["node_id"] for n in _RAW_GRAPH["nodes"] if str(n["layer"]) == "12"}
    assert not any(
        l["source"] in l12_ids and l["target"] in l12_ids for l in _RAW_GRAPH["links"]
    )


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_public_api_signature():
    sig = inspect.signature(build_traced_circuit)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == [
        "raw_graph",
        "record",
        "source_set",
        "dataset_metadata",
        "name",
        "description",
        "layer",
        "allow_l0_mismatch",
        "redact_roles",
    ]
    for p in params[:4]:
        assert p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert p.default is inspect.Parameter.empty
    kwonly = {p.name: p for p in params[4:]}
    for p in kwonly.values():
        assert p.kind is inspect.Parameter.KEYWORD_ONLY
    assert kwonly["name"].default is None
    assert kwonly["description"].default is None
    assert kwonly["layer"].default == 12
    assert kwonly["allow_l0_mismatch"].default is False
    assert kwonly["redact_roles"].default is False


# ---------------------------------------------------------------------------
# Frozen schema: top level and nodes
# ---------------------------------------------------------------------------


def test_frozen_schema_top_level(build):
    circuit = build()
    assert set(circuit.keys()) == TOP_LEVEL_KEYS
    assert circuit["type"] == "traced"
    assert circuit["source"] == "neuronpedia"
    assert circuit["edges"] == []
    # Default name is the record slug; default description is non-empty.
    assert circuit["name"] == "gemma-fact-dallas-austin"
    assert isinstance(circuit["description"], str)
    assert circuit["description"]


def test_name_description_override(build):
    circuit = build(name="dallas-austin-l12", description="Layer-12 slice")
    assert circuit["name"] == "dallas-austin-l12"
    assert circuit["description"] == "Layer-12 slice"


def test_nodes_are_local_layer12_features(build):
    circuit = build()
    assert len(circuit["nodes"]) == 6
    nm = _node_map(circuit)
    assert set(nm.keys()) == L12_LOCALS
    for node in circuit["nodes"]:
        assert set(node.keys()) == NODE_KEYS
        assert isinstance(node["featureIndex"], int)
        assert 0 <= node["featureIndex"] <= 16383
        assert node["instances"] == 1
        assert node["influence"] is None or isinstance(node["influence"], float)
    # Influence carried verbatim from the raw fixture nodes.
    for local, infl in RAW_INFL.items():
        assert nm[local]["influence"] == pytest.approx(infl)


def test_activation_normalized_per_circuit_max(build):
    circuit = build()
    nm = _node_map(circuit)
    for local, raw in RAW_ACT.items():
        assert 0.0 <= nm[local]["activation"] <= 1.0
        assert nm[local]["activation"] == pytest.approx(raw / MAX_RAW_ACT)
    assert nm[10631]["activation"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------


def test_roles_unassigned_when_no_supernode_attaches_to_layer12(build):
    # In the base fixture no supernode references a layer-12 node_id.
    circuit = build()
    assert all(n["role"] == "unassigned" for n in circuit["nodes"])


def test_roles_from_supernode_labels_attached_to_layer12(raw_graph, build):
    # Attach two REAL layer-12 node_ids to the (neutralized) label-0 supernode.
    raw_graph["qParams"]["supernodes"][0].extend(["12_2082_9", "12_2799_9"])
    circuit = build()
    nm = _node_map(circuit)
    assert nm[2082]["role"] == "label-0"
    assert nm[2799]["role"] == "label-0"
    for local in L12_LOCALS - {2082, 2799}:
        assert nm[local]["role"] == "unassigned"
    # Layer-12 supernode members are NOT cross-layer members.
    clm = circuit["metadata"]["crossLayerMembers"]
    assert clm["count"] == CROSS_LAYER_COUNT
    assert clm["byLayer"] == CROSS_LAYER_BY_LAYER


def test_role_aggregation_nonunassigned_beats_unassigned(raw_graph, build):
    # Three instances of feature 2799: one labeled, two unassigned.
    # Most-frequent NON-'unassigned' role wins regardless of unassigned count.
    _add_instance(raw_graph, "12_2799_9", 10,
                  activation=5.670328140258789, influence=0.6785522699356079)
    _add_instance(raw_graph, "12_2799_9", 8,
                  activation=7.126477241516113, influence=0.6354414820671082)
    raw_graph["qParams"]["supernodes"][3].append("12_2799_9")  # label-3
    circuit = build()
    nm = _node_map(circuit)
    assert nm[2799]["instances"] == 3
    assert nm[2799]["role"] == "label-3"


def test_role_aggregation_tie_break_is_lexicographic(raw_graph, build):
    # Two instances of feature 2082 with different labels, 1 vote each:
    # deterministic tie-break = lexicographically smallest label.
    dup_id = _add_instance(raw_graph, "12_2082_9", 10,
                           activation=6.065004825592041,
                           influence=0.6910012364387512)
    raw_graph["qParams"]["supernodes"][1].append("12_2082_9")  # label-1
    raw_graph["qParams"]["supernodes"][0].append(dup_id)       # label-0
    circuit = build()
    nm = _node_map(circuit)
    assert nm[2082]["instances"] == 2
    assert nm[2082]["role"] == "label-0"


# ---------------------------------------------------------------------------
# Aggregation of duplicate featureIndex instances
# ---------------------------------------------------------------------------


def test_aggregation_max_activation_instances_influence(raw_graph, build):
    # Second instance of feature 2082 at ctx 10 with a HIGHER raw activation
    # (values reused from real fixture nodes 12910).
    _add_instance(raw_graph, "12_2082_9", 10,
                  activation=9.857479095458984, influence=0.6598602533340454)
    circuit = build()
    nm = _node_map(circuit)
    assert set(nm.keys()) == L12_LOCALS  # still 6 unique features
    assert nm[2082]["instances"] == 2
    # Max raw activation wins, then per-circuit normalization.
    assert nm[2082]["activation"] == pytest.approx(9.857479095458984 / MAX_RAW_ACT)
    # Influence follows the max-activation instance.
    assert nm[2082]["influence"] == pytest.approx(0.6598602533340454)
    lf = circuit["metadata"]["layerFilter"]
    assert lf["nodesTotal"] == NODES_TOTAL + 1
    assert lf["nodesKept"] == 6
    assert lf["featureInstancesCollapsed"] == 1
    assert lf["droppedByLayer"] == DROPPED_BY_LAYER
    assert lf["nodesTotal"] == (
        lf["nodesKept"]
        + lf["featureInstancesCollapsed"]
        + sum(lf["droppedByLayer"].values())
    )


def test_self_loop_edges_dropped_and_counted(raw_graph, build):
    # A link between two instances of the SAME feature collapses to a
    # self-loop after aggregation: dropped and counted, never emitted.
    dup_id = _add_instance(raw_graph, "12_2082_9", 10,
                           activation=9.857479095458984,
                           influence=0.6598602533340454)
    real_weight = raw_graph["links"][0]["weight"]
    raw_graph["links"].append(
        {"source": "12_2082_9", "target": dup_id, "weight": real_weight}
    )
    circuit = build()
    assert circuit["edges"] == []
    lf = circuit["metadata"]["layerFilter"]
    assert lf["edgesTotal"] == EDGES_TOTAL + 1
    assert lf["edgesKept"] == 0
    assert lf["selfLoopsDropped"] == 1
    assert lf["crossLayerEdgesDropped"] == 250
    assert lf["edgesTotal"] == (
        lf["edgesKept"] + lf["crossLayerEdgesDropped"] + lf["selfLoopsDropped"]
    )


# ---------------------------------------------------------------------------
# Metadata block
# ---------------------------------------------------------------------------


def test_layer_filter_metadata_exact(build):
    circuit = build()
    assert circuit["metadata"]["layerFilter"] == EXPECTED_LAYER_FILTER


def test_cross_layer_members_informational(build):
    circuit = build()
    clm = circuit["metadata"]["crossLayerMembers"]
    assert set(clm.keys()) == {"count", "byLayer"}
    assert clm["count"] == CROSS_LAYER_COUNT
    assert clm["byLayer"] == CROSS_LAYER_BY_LAYER
    assert sum(clm["byLayer"].values()) == clm["count"]
    # Informational ONLY: node featureIndex entries stay pure layer-12.
    assert {n["featureIndex"] for n in circuit["nodes"]} == L12_LOCALS


def test_metadata_identity_fields(build):
    md = build()["metadata"]
    assert set(md.keys()) == METADATA_KEYS
    assert md["model"] == "gemma-2-2b"
    assert md["slug"] == "gemma-fact-dallas-austin"
    assert md["sourceSet"] == "gemmascope-transcoder-16k"
    assert md["hfRepo"] == "google/gemma-scope-2b-pt-transcoders"
    assert md["width"] == "width_16k"
    # Graph dictionary is layer_12/width_16k/average_l0_6 -> l0 = 6 ...
    assert md["l0"] == 6
    assert isinstance(md["l0"], int) and not isinstance(md["l0"], bool)
    # ... which does NOT match the local dataset's l0_604 dictionary.
    assert md["l0Verified"] is False
    assert md["prompt"] == "prompt-redacted"
    # pruning/generation blocks verbatim from graph metadata.
    assert md["pruning"] == {"node_threshold": 0.7, "edge_threshold": 0.9}
    assert md["generation"] == {
        "max_n_logits": 10,
        "desired_logit_prob": 0.95,
        "max_feature_nodes": 5000,
    }
    assert md["fetchedAt"] is None or isinstance(md["fetchedAt"], str)
    assert md["pipelineVersion"] == "0.4.0"
    assert md["labelProvenance"] == "neuronpedia-public-graph"


# ---------------------------------------------------------------------------
# Identity validation wiring
# ---------------------------------------------------------------------------


def test_identity_validator_called_with_inputs(
    build, identity_calls, record, source_set, dataset_metadata
):
    build()
    assert len(identity_calls) == 1
    args, kwargs = identity_calls[0]
    passed = list(args) + list(kwargs.values())
    assert any(v is record for v in passed)
    assert any(v is source_set for v in passed)
    assert any(v is dataset_metadata for v in passed)
    assert 12 in passed
    assert kwargs["allow_l0_mismatch"] is True


def test_identity_validator_failure_propagates(
    monkeypatch, raw_graph, record, source_set, dataset_metadata
):
    def _hard_fail(*args, **kwargs):
        raise ValueError(
            "graph dictionary average_l0_6 does not match dataset l0_604"
        )

    monkeypatch.setattr(traced_circuits, "validate_graph_identity", _hard_fail)
    with pytest.raises(ValueError, match="average_l0_6"):
        build_traced_circuit(
            raw_graph, record, source_set, dataset_metadata,
            layer=12, allow_l0_mismatch=False,
        )


# ---------------------------------------------------------------------------
# Role redaction
# ---------------------------------------------------------------------------


def test_redact_roles_maps_labels_to_group_indices(raw_graph, build):
    # label-0 (supernode index 0) -> group-0, label-1 (index 1) -> group-1.
    raw_graph["qParams"]["supernodes"][0].append("12_2799_9")
    raw_graph["qParams"]["supernodes"][1].append("12_2082_9")
    circuit = build(redact_roles=True)
    nm = _node_map(circuit)
    assert nm[2799]["role"] == "group-0"
    assert nm[2082]["role"] == "group-1"
    for local in L12_LOCALS - {2082, 2799}:
        assert nm[local]["role"] == "unassigned"
    # No supernode label text may survive anywhere in the circuit.
    assert "label-" not in json.dumps(circuit)


# ---------------------------------------------------------------------------
# Scrub check
# ---------------------------------------------------------------------------


def test_build_calls_scrub_check(
    monkeypatch, raw_graph, record, source_set, dataset_metadata, identity_calls
):
    def _sentinel(circuit):
        raise RuntimeError("sentinel-scrub-called")

    monkeypatch.setattr(traced_circuits, "scrub_check", _sentinel)
    with pytest.raises(RuntimeError, match="sentinel-scrub-called"):
        build_traced_circuit(
            raw_graph, record, source_set, dataset_metadata,
            layer=12, allow_l0_mismatch=True,
        )


def test_scrub_check_passes_on_clean_build(build):
    assert scrub_check(build()) is None


def test_scrub_check_rejects_clerp(build):
    circuit = build()
    circuit["nodes"][0]["clerp"] = "increases probability of Texas city names"
    with pytest.raises(ValueError):
        scrub_check(circuit)


@pytest.mark.parametrize("field", ["pos_str", "neg_str"])
def test_scrub_check_rejects_pos_neg_str(build, field):
    circuit = build()
    circuit["nodes"][1][field] = ["Dallas", "Austin"]
    with pytest.raises(ValueError):
        scrub_check(circuit)


def test_scrub_check_rejects_unexpected_node_string_field(build):
    circuit = build()
    circuit["nodes"][2]["note"] = "fires on city names"
    with pytest.raises(ValueError):
        scrub_check(circuit)


# ---------------------------------------------------------------------------
# Determinism / purity
# ---------------------------------------------------------------------------


def test_build_is_deterministic_and_does_not_mutate_inputs(
    record, source_set, dataset_metadata, identity_calls
):
    graph_a = copy.deepcopy(_RAW_GRAPH)
    graph_b = copy.deepcopy(_RAW_GRAPH)
    snapshot = copy.deepcopy(graph_a)
    circuit_a = build_traced_circuit(
        graph_a, record, source_set, dataset_metadata,
        layer=12, allow_l0_mismatch=True,
    )
    circuit_b = build_traced_circuit(
        graph_b, record, source_set, dataset_metadata,
        layer=12, allow_l0_mismatch=True,
    )
    # JSON-serializable and byte-identical across builds (fetchedAt must come
    # from inputs, never wall-clock time).
    assert json.dumps(circuit_a, sort_keys=True) == json.dumps(
        circuit_b, sort_keys=True
    )
    # Inputs are not mutated.
    assert graph_a == snapshot
