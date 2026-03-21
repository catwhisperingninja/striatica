# striatica/tests/test_causal_circuits.py
"""Tests for parsing Neuronpedia causal attribution graphs.

These tests use the ACTUAL Neuronpedia data format as observed from the
Circuit Tracer API (gemma-fact-dallas-austin circuit, fetched 2026-03-21).
Key format details verified against real data:
  - node.feature is a global int (layer * 100000 + local_index)
  - node.layer is a string, not int
  - qParams.supernodes is a list of lists: [["label", "id1", "id2", ...], ...]
  - links use node_id strings as source/target
"""

import pytest
from pipeline.circuits import parse_neuronpedia_circuit, extract_local_feature_index


def test_extract_local_feature_index():
    """Global transcoder feature index → local index within a layer.

    Neuronpedia encodes global indices as layer * 100000 + local_index.
    Layer 0 is special: indices are already local (0-16383).
    """
    # Layer 12: global 1202082 → local 2082
    assert extract_local_feature_index(1202082, layer=12) == 2082
    # Layer 0: global index IS the local index
    assert extract_local_feature_index(437, layer=0) == 437
    # Layer 25: global 2516326 → local 16326
    assert extract_local_feature_index(2516326, layer=25) == 16326
    # Edge case: first feature in a layer
    assert extract_local_feature_index(1200000, layer=12) == 0
    # Edge case: last feature in a layer (width_16k)
    assert extract_local_feature_index(1216383, layer=12) == 16383

    # Invalid: local index out of range for 16k width
    with pytest.raises(ValueError, match="out of range"):
        extract_local_feature_index(1220000, layer=12)

    # Invalid: global index doesn't match claimed layer
    with pytest.raises(ValueError, match="does not match layer"):
        extract_local_feature_index(1302082, layer=12)


def test_parse_neuronpedia_circuit_format():
    """Mock Neuronpedia JSON should translate to Striatica schema.

    Uses the REAL Neuronpedia format: node.feature is global int,
    node.layer is string, supernodes are list-of-lists.
    """
    mock_np_data = {
        "nodes": [
            {
                "node_id": "12_2082_1",
                "feature": 1202082,
                "layer": "12",
                "ctx_idx": 4,
                "feature_type": "cross layer transcoder",
                "activation": 0.8,
                "influence": 0.5,
                "clerp": "detects cities",
            },
            {
                "node_id": "12_4000_1",
                "feature": 1204000,
                "layer": "12",
                "ctx_idx": 5,
                "feature_type": "cross layer transcoder",
                "activation": 0.6,
                "influence": 0.3,
                "clerp": "detects states",
            },
        ],
        "links": [
            {
                "source": "12_2082_1",
                "target": "12_4000_1",
                "weight": 0.45,
            }
        ],
        "qParams": {
            "supernodes": [
                ["location", "12_2082_1", "12_4000_1"],
            ]
        },
    }

    circuit = parse_neuronpedia_circuit(
        mock_np_data,
        name="test-circuit",
        description="Test Circuit",
        layer_filter=12,
    )

    assert circuit["name"] == "test-circuit"
    assert circuit["description"] == "Test Circuit"
    assert circuit["type"] == "traced"
    assert len(circuit["nodes"]) == 2
    assert len(circuit["edges"]) == 1

    # Check nodes — featureIndex should be the LOCAL index (2082, not 1202082)
    node_indices = {n["featureIndex"]: n for n in circuit["nodes"]}
    assert 2082 in node_indices
    assert 4000 in node_indices

    assert node_indices[2082]["activation"] == 0.8
    assert node_indices[2082]["influence"] == 0.5
    assert node_indices[2082]["role"] == "location"

    # Check edges — source/target should be LOCAL indices
    edge = circuit["edges"][0]
    assert edge["source"] == 2082
    assert edge["target"] == 4000
    assert edge["weight"] == 0.45


def test_layer_filtering():
    """Nodes outside the specified layer should be excluded.

    Edges where either endpoint is outside the layer are also excluded.
    """
    mock_np_data = {
        "nodes": [
            {
                "node_id": "12_2082_1",
                "feature": 1202082,
                "layer": "12",
                "activation": 0.8,
                "influence": 0.5,
            },
            {
                "node_id": "13_4000_1",
                "feature": 1304000,
                "layer": "13",
                "activation": 0.6,
                "influence": 0.3,
            },
        ],
        "links": [
            {"source": "12_2082_1", "target": "13_4000_1", "weight": 0.45}
        ],
    }

    circuit = parse_neuronpedia_circuit(
        mock_np_data, name="test", description="test", layer_filter=12
    )

    # Should only include layer 12 nodes
    assert len(circuit["nodes"]) == 1
    assert circuit["nodes"][0]["featureIndex"] == 2082

    # Edge crosses layers — both endpoints must be in-layer to be included
    assert len(circuit["edges"]) == 0


def test_supernode_assignment_real_format():
    """Supernodes in real Neuronpedia format: list of lists, label first.

    Real example from dallas-austin circuit:
    ["capital", "15_4494_4", "6_4662_4", "4_7671_4"]
    """
    mock_np_data = {
        "nodes": [
            {"node_id": "12_100_1", "feature": 1200100, "layer": "12", "activation": 0.9, "influence": 0.7},
            {"node_id": "12_200_1", "feature": 1200200, "layer": "12", "activation": 0.7, "influence": 0.5},
            {"node_id": "12_300_1", "feature": 1200300, "layer": "12", "activation": 0.5, "influence": 0.3},
        ],
        "links": [],
        "qParams": {
            "supernodes": [
                ["geography", "12_100_1", "12_200_1"],
                ["syntax", "12_300_1"],
            ]
        },
    }

    circuit = parse_neuronpedia_circuit(
        mock_np_data, name="test", description="test", layer_filter=12
    )

    node_map = {n["featureIndex"]: n for n in circuit["nodes"]}
    assert node_map[100]["role"] == "geography"
    assert node_map[200]["role"] == "geography"
    assert node_map[300]["role"] == "syntax"


def test_nodes_without_supernode_get_default_role():
    """Nodes not in any supernode should get role 'unassigned'."""
    mock_np_data = {
        "nodes": [
            {"node_id": "12_500_1", "feature": 1200500, "layer": "12", "activation": 0.5, "influence": 0.2},
        ],
        "links": [],
        "qParams": {"supernodes": []},
    }

    circuit = parse_neuronpedia_circuit(
        mock_np_data, name="test", description="test", layer_filter=12
    )

    assert circuit["nodes"][0]["role"] == "unassigned"


def test_circuit_preserves_metadata():
    """Parsed circuit should include source metadata for traceability."""
    mock_np_data = {
        "nodes": [
            {"node_id": "12_1_1", "feature": 1200001, "layer": "12", "activation": 0.5, "influence": 0.3},
        ],
        "links": [],
        "metadata": {
            "slug": "gemma-fact-dallas-austin",
            "prompt": "<bos>Fact: The capital of the state containing Dallas is",
            "node_threshold": 0.6,
        },
    }

    circuit = parse_neuronpedia_circuit(
        mock_np_data, name="dallas-austin", description="test", layer_filter=12
    )

    assert circuit["source"] == "neuronpedia"
    assert "prompt" in circuit["metadata"]
    assert circuit["metadata"]["slug"] == "gemma-fact-dallas-austin"
