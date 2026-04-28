import sqlite3
import json
from pathlib import Path

from scripts.visualize_graph import load_edges_from_db
from src.storage import EdgeStore


def test_load_edges_from_db(tmp_path):
    db_path = tmp_path / "edges.db"
    store = EdgeStore(str(db_path))
    # write a couple of edges
    def gen():
        for i in range(3):
            yield {
                "source_index": i,
                "source_id": f"s{i}",
                "target_id": f"t{i}",
                "source_timestamp": None,
                "target_timestamp": None,
                "time_delta_ms": None,
                "retrieval_distance": None,
                "retrieval_similarity": None,
                "semantic_cosine": None,
                "hybrid_score": 0.5,
                "alpha": 0.5,
                "target_metadata": {"component": f"svc{i}"},
            }

    written = store.write_edges(gen(), batch_size=2)
    assert written == 3

    # Use level="template" to avoid filtering by service (which is missing in this test data)
    edges = load_edges_from_db(str(db_path), limit=10, level="template")
    assert len(edges) == 3
    # ensure metadata parsed
    assert edges[0]["target_metadata"] is None or isinstance(edges[0]["target_metadata"], dict)
