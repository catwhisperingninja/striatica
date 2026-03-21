# striatica/tests/test_causal_circuits.py
"""Tests for parsing Neuronpedia causal attribution graphs."""

import pytest
from pipeline.circuits import parse_neuronpedia_circuit, parse_node_id

def test_parse_node_id():
    """Node IDs like '12_2082_1' should be parsed to integer 2082."""
    assert parse_node_id("12_2082_1") == 2082
    assert parse_node_id("0_0_1") == 0
    assert parse_node_id("25_16383_1") == 16383
    
    with pytest.raises(ValueError):
        parse_node_id("invalid_id")

def test_parse_neuronpedia_circuit_format():
    """Mock Neuronpedia JSON should translate to Striatica schema."""
    mock_np_data = {
        "nodes": [
            {
                "node_id": "12_2082_1",
                "feature": 2082,
                "layer": 12,
                "activation": 0.8,
                "influence": 0.5,
                "clerp": "detects cities"
            },
            {
                "node_id": "12_4000_1",
                "feature": 4000,
                "layer": 12,
                "activation": 0.6,
                "influence": 0.3,
                "clerp": "detects states"
            }
        ],
        "links": [
            {
                "source": "12_2082_1",
                "target": "12_4000_1",
                "weight": 0.45
            }
        ],
        "qParams": {
            "supernodes": [
                {
                    "label": "location",
                    "features": ["12_2082_1", "12_4000_1"]
                }
            ]
        }
    }
    
    circuit = parse_neuronpedia_circuit(
        mock_np_data, 
        name="test-circuit", 
        description="Test Circuit",
        layer_filter=12
    )
    
    assert circuit["name"] == "test-circuit"
    assert circuit["description"] == "Test Circuit"
    assert len(circuit["nodes"]) == 2
    assert len(circuit["edges"]) == 1
    
    # Check nodes
    node_indices = {n["featureIndex"]: n for n in circuit["nodes"]}
    assert 2082 in node_indices
    assert 4000 in node_indices
    
    assert node_indices[2082]["activation"] == 0.8
    assert node_indices[2082]["role"] == "location"
    
    # Check edges
    edge = circuit["edges"][0]
    assert edge["source"] == 2082
    assert edge["target"] == 4000
    assert edge["weight"] == 0.45

def test_layer_filtering():
    """Nodes outside the specified layer should be filtered or marked as external."""
    mock_np_data = {
        "nodes": [
            {"node_id": "12_2082_1", "feature": 2082, "layer": 12, "activation": 0.8},
            {"node_id": "13_4000_1", "feature": 4000, "layer": 13, "activation": 0.6}
        ],
        "links": [
            {"source": "12_2082_1", "target": "13_4000_1", "weight": 0.45}
        ]
    }
    
    circuit = parse_neuronpedia_circuit(mock_np_data, name="test", description="test", layer_filter=12)
    
    # Should only include layer 12 nodes
    assert len(circuit["nodes"]) == 1
    assert circuit["nodes"][0]["featureIndex"] == 2082
    
    # Edges pointing to external nodes might be filtered or kept depending on implementation
    # Let's assume we filter edges where both endpoints aren't in the layer for now
    assert len(circuit["edges"]) == 0
