"""Lightweight pipeline orchestration for running the backend and producing
artifacts that a UI can consume (edges DB and transitions CSV).

This module intentionally avoids heavy external dependencies and is suitable
for small-scale runs and unit tests. For larger workloads, run the
notebook/cluster encoding steps and provide embeddings to the runner.
"""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from src.storage import EdgeStore
from src.retrieval import compute_candidate_edges_stream
from src.transitions import compute_and_persist_transitions, export_transitions_to_csv


def run_pipeline_df(
    df,
    embeddings: Optional[np.ndarray],
    db_path: str,
    top_k: int = 5,
    threshold: float = 0.1,
    alpha: float = 0.5,
    batch_size: int = 128,
    export_csv: Optional[str] = None,
) -> Tuple[int, int]:
    """Run the minimal backend pipeline:

    - compute candidate edges (streaming)
    - write edges into SQLite via EdgeStore
    - compute order-1 transitions and persist
    - optionally export transitions to CSV

    Returns (edges_written, transitions_written).
    """
    store = EdgeStore(db_path)
    store.init_db()

    gen = compute_candidate_edges_stream(
        df,
        embeddings=embeddings,
        collection=None,
        top_k=top_k,
        threshold=threshold,
        batch_size=batch_size,
        alpha=alpha,
    )

    edges_written = store.write_edges(gen, batch_size=100)

    transitions_written = compute_and_persist_transitions(db_path, min_count=1)

    if export_csv:
        export_transitions_to_csv(db_path, export_csv)

    store.close()
    return edges_written, transitions_written


__all__ = ["run_pipeline_df"]
