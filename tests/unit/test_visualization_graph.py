import json
import numpy as np
import networkx as nx

from src.visualization.graph import edges_to_networkx, networkx_to_dict


def test_edges_to_networkx_and_serialize():
    edges = [
        {"source": "a", "target": "b", "weight": 0.5, "metadata": {"count": 3}},
        {"source": "b", "target": "c", "weight": 0.75},
        {"source": "a", "target": "c", "weight": np.array([1.0])},
    ]

    G = edges_to_networkx(edges)
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 3

    d = networkx_to_dict(G)
    assert set(d.keys()) == {"nodes", "edges"}
    assert len(d["nodes"]) == 3
    assert len(d["edges"]) == 3

    # ensure numpy array was coerced to list/string in attributes
    found = [e for e in d["edges"] if e["source"] == "a" and e["target"] == "c"]
    assert found
    # weight should be present and serializable
    assert "weight" in found[0]
    json.dumps(d)  # should not raise
