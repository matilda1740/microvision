import sqlite3
import tempfile

from src.storage import EdgeStore
from src.transitions import compute_and_persist_transitions


def _make_sample_edges():
    # produce a mix of edges where A->B occurs twice, A->C once, D->E once
    return [
        {"source_index": 0, "source_id": "A", "target_id": "B", "time_delta_ms": 100.0},
        {"source_index": 1, "source_id": "A", "target_id": "B", "time_delta_ms": 120.0},
        {"source_index": 2, "source_id": "A", "target_id": "C", "time_delta_ms": 200.0},
        {"source_index": 3, "source_id": "D", "target_id": "E", "time_delta_ms": 50.0},
    ]


def test_compute_and_persist_transitions(tmp_path):
    db_path = str(tmp_path / "edges_test.db")

    store = EdgeStore(db_path)
    store.init_db()

    edges = _make_sample_edges()
    n_written = store.write_edges(edges)
    assert n_written == 4

    # compute transitions
    written = compute_and_persist_transitions(db_path, min_count=1)
    assert written == 3  # (A->B), (A->C), (D->E)

    # inspect transitions table
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT source_id, target_id, count, prob, avg_time_delta_ms FROM transitions")
    rows = cur.fetchall()
    # build dict for assertions
    d = {(r[0], r[1]): (r[2], r[3], r[4]) for r in rows}

    assert ("A", "B") in d and d[("A", "B")][0] == 2
    # probability for A->B should be 2 / (2+1) = 0.666...
    assert abs(d[("A", "B")][1] - (2.0 / 3.0)) < 1e-6

    assert ("A", "C") in d and d[("A", "C")][0] == 1
    assert ("D", "E") in d and d[("D", "E")][0] == 1

    conn.close()
