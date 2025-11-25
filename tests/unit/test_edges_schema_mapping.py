import json

from src.visualization.graph import edges_to_networkx, networkx_to_dict


def test_edges_schema_mapping():
    edges = [
        {"source": None, "source_id": "A", "target": "B", "hybrid_score": 0.5},
        {"source_index": 5, "target_id": 12, "hybrid_score": 0.3},
        {"source": "  svc1 ", "target": " svc2 ", "hybrid_score": 0.9},
        {"source": "", "source_id": None, "target": "", "note": "fallback"},
    ]

    G = edges_to_networkx(edges)
    d = networkx_to_dict(G)

    # we expect at least three valid edges (the last entry should produce
    # a fallback-generated edge rather than being dropped)
    assert len(d["edges"]) >= 3

    # all serialized edges must have non-empty trimmed source/target
    for edge in d["edges"]:
        assert edge["source"] is not None and str(edge["source"]).strip() != ""
        assert edge["target"] is not None and str(edge["target"]).strip() != ""

    # check trimming for the explicit svc1/svc2 case
    assert any(e["source"] == "svc1" and e["target"] == "svc2" for e in d["edges"])
