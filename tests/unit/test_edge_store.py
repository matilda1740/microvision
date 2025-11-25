import sqlite3
import os
from src.storage.edge_store import EdgeStore
from src.retrieval.retriever import compute_candidate_edges_stream
import numpy as np
import pandas as pd


def test_edge_store_write_and_read(tmp_path):
    db_path = str(tmp_path / "edges.db")
    store = EdgeStore(db_path)
    store.init_db()

    # create minimal df and embeddings
    df = pd.DataFrame({"doc_id": ["a", "b"], "text": ["ta", "tb"]})
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])

    gen = compute_candidate_edges_stream(df, embeddings=embeddings, collection=None, top_k=2, threshold=0.0, batch_size=2, alpha=0.5)
    written = store.write_edges(gen, batch_size=1)

    assert written > 0

    # check DB rows
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT retrieval_similarity, semantic_cosine, hybrid_score, alpha, target_metadata FROM edges")
    rows = cur.fetchall()
    assert len(rows) == written
    for r in rows:
        retrieval_similarity, semantic_cosine, hybrid_score, alpha, target_metadata = r
        # hybrid should be numeric
        assert hybrid_score is not None
        assert alpha == 0.5

    conn.close()
    store.close()
