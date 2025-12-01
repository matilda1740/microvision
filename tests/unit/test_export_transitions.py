import sqlite3
import tempfile

from src.storage import EdgeStore
from src.transitions import compute_and_persist_transitions, export_transitions_to_csv


def test_export_transitions_roundtrip(tmp_path):
    db_path = str(tmp_path / "edges_export.db")
    csv_path = str(tmp_path / "transitions.csv")

    store = EdgeStore(db_path)
    store.init_db()
    edges = [
        {"source_index": 0, "source_id": "X", "target_id": "Y", "time_delta_ms": 10.0},
        {"source_index": 1, "source_id": "X", "target_id": "Z", "time_delta_ms": 20.0},
        {"source_index": 2, "source_id": "X", "target_id": "Y", "time_delta_ms": 30.0},
    ]
    store.write_edges(edges)

    n = compute_and_persist_transitions(db_path, min_count=1)
    assert n >= 1

    exported = export_transitions_to_csv(db_path, csv_path)
    assert exported >= 1

    # basic file existence and header check
    with open(csv_path, "r", encoding="utf-8") as fh:
        hdr = fh.readline().strip()
        assert hdr.startswith("source_id,target_id,count")
