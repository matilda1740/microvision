import pytest
import numpy as np
import pandas as pd

from src.retrieval.retriever import compute_candidate_edges_stream


class MockCollection:
    """A minimal mock of a chroma collection's query API.

    It returns the same controlled response for each query in the batch.
    """

    def __init__(self, ids_per_query, metadatas_per_query, distances_per_query):
        self.ids_per_query = ids_per_query
        self.metadatas_per_query = metadatas_per_query
        self.distances_per_query = distances_per_query

    def query(self, query_embeddings=None, n_results=5, where=None, include=None, **kwargs):
        # Build Chroma-like response where each key maps to a list-of-lists
        qn = len(query_embeddings)
        return {
            "ids": [self.ids_per_query for _ in range(qn)],
            "metadatas": [self.metadatas_per_query for _ in range(qn)],
            "distances": [self.distances_per_query for _ in range(qn)],
        }


def make_small_df():
    df = pd.DataFrame({"doc_id": ["a", "b", "c"], "text": ["ta", "tb", "tc"]})
    return df


def test_compute_candidate_edges_stream_with_mock_collection():
    df = make_small_df()
    # simple embeddings (3x2)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    # Make mock responses: each query returns one neighbour (id 'b') with small distance
    mock_ids = ["b"]
    mock_metas = [{"service": "namenode"}]
    mock_dists = [0.1]  # distance -> similarity = 0.9

    collection = MockCollection(mock_ids, mock_metas, mock_dists)

    gen = compute_candidate_edges_stream(df, embeddings=embeddings, collection=collection, top_k=1, threshold=0.4, batch_size=2, alpha=0.7)
    edges = list(gen)

    assert len(edges) == len(df)  # each source returned one neighbour
    for e in edges:
        assert "source_index" in e and "target_id" in e and "hybrid_score" in e
        assert e["hybrid_score"] >= 0.4
        assert e["target_metadata"] == mock_metas[0]
        # hybrid should equal alpha * retrieval_similarity + (1-alpha) * semantic_cosine
        assert "retrieval_similarity" in e and "semantic_cosine" in e
        expected = 0.7 * e["retrieval_similarity"] + 0.3 * e["semantic_cosine"]
        assert abs(e["hybrid_score"] - expected) < 1e-6


def test_compute_candidate_edges_stream_local_fallback():
    df = make_small_df()
    # embeddings designed so cosine between 0 and 2
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])

    # Use a low threshold so we get some candidates
    gen = compute_candidate_edges_stream(df, embeddings=embeddings, collection=None, top_k=2, threshold=0.1, batch_size=2, alpha=0.4)
    edges = list(gen)

    assert len(edges) > 0
    # each yielded edge must have numeric score and source/target indices
    for e in edges:
        assert isinstance(e["source_index"], int)
        assert isinstance(e["target_id"], int)
        assert isinstance(e["hybrid_score"], float)
