"""Phase 2 runner: embeddings -> Chroma ingest -> retrieval -> persist -> transitions

This script picks up a merged templates CSV (produced by Phase 1) and
continues the pipeline: build metadatas, compute embeddings, ingest into
Chroma, compute candidate edges, persist edges to the edges DB and compute
transitions. It reuses helpers from the project and adds safety guards for
optional heavy dependencies.

Run from project root with the project's venv active. Example:

    ./.venv/bin/python scripts/run_phase2.py --merged data/sample_raw_merged.csv

Note: this step requires `sentence-transformers` and `chromadb` to be
installed in the venv.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.encode_chroma.encode_utils import df_to_metadatas
from src.retrieval.retriever import compute_candidate_edges_stream
from src.storage.edge_store import EdgeStore
from src.persistence.transitions import compute_and_persist_transitions
from config.settings import settings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--merged", required=False, default=getattr(settings, "DEFAULT_MERGED", "data/sample_raw_merged.csv"), help="Path to merged templates CSV")
    p.add_argument("--chroma-dir", default=getattr(settings, "DEFAULT_CHROMA_DIR", "data/chroma_db/chroma_smoke"))
    p.add_argument("--db", default=getattr(settings, "DEFAULT_DB", "data/edges/edges.db"))
    p.add_argument("--top-k", type=int, default=getattr(settings, "DEFAULT_TOP_K", 10))
    p.add_argument("--threshold", type=float, default=getattr(settings, "DEFAULT_THRESHOLD", 0.2))
    p.add_argument("--alpha", type=float, default=getattr(settings, "DEFAULT_ALPHA", 0.5))
    p.add_argument("--device", default=getattr(settings, "DEFAULT_DEVICE", None))
    p.add_argument("--model", default=None)
    p.add_argument("--clear-db", action="store_true", help="Clear the edges table in the target DB before writing (destructive)")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    return p.parse_args()


from src.encode_chroma.chroma import ingest_to_chroma_atomic


def main() -> int:
    args = parse_args()
    # configure logging early so modules can emit debug traces
    try:
        from config.logging_config import configure_logging

        configure_logging(debug=bool(getattr(args, "debug", False)))
    except Exception:
        # best effort: if logging helper not available, configure basic logging
        # Use the module-level `logging` import to avoid creating a local
        # variable named `logging` which would shadow the module reference
        # and cause UnboundLocalError when referenced earlier in this
        # function.
        logging.basicConfig(level=logging.DEBUG if getattr(args, "debug", False) else logging.INFO)
    merged_path = Path(args.merged)
    if not merged_path.exists():
        print(f"Merged CSV not found: {merged_path}")
        return 2

    logging.info("Loading merged CSV: %s", merged_path)
    df = pd.read_csv(merged_path)

    # ensure semantic_text exists
    if "semantic_text" not in df.columns:
        if "template_ids" in df.columns:
            df["semantic_text"] = df["template_ids"].apply(lambda v: ", ".join(v) if isinstance(v, list) else str(v))
        else:
            df["semantic_text"] = ""

    # Drop the list-valued `template_ids` column so downstream metadata and
    # embeddings don't carry list objects. We still preserve `template_id` as
    # the scalar canonical id emitted by the merge step.
    if "template_ids" in df.columns:
        df = df.drop(columns=["template_ids"])

    # build metadatas
    metas = df_to_metadatas(df)
    # sanitize metadatas
    metas_sanitized = []
    for i, m in enumerate(metas):
        if not isinstance(m, dict):
            m2 = {"idx": i}
        else:
            m2 = {k: v for k, v in m.items() if v is not None and v != ""}
        if not m2:
            m2 = {"semantic_text": str(df['semantic_text'].iloc[i]) if i < len(df) else f"row_{i}"}
        metas_sanitized.append(m2)
    metas = metas_sanitized

    # embeddings
    model_name = args.model if args.model else getattr(settings, "EMBEDDING_MODEL", "all-mpnet-base-v2")
    device = args.device if args.device is not None else getattr(settings, "DEFAULT_DEVICE", None)
    if device is None:
        device = "cuda" if (os.environ.get("CUDA_VISIBLE_DEVICES") or os.path.exists("/usr/local/cuda")) else "cpu"

    print(f"Encoding semantic_text using model={model_name} device={device} ...")
    from src.encode_chroma.encoder import encode_templates

    # compute embeddings via central helper (no ingest here)
    # Force recompute to avoid re-using a stale embeddings cache that may have
    # different length than the current input dataframe (common in test runs).
    embeddings_out_dir = getattr(settings, "DEFAULT_EMBEDDINGS_DIR", "data/edges")
    embeddings, index_df, _ = encode_templates(df, model=None, model_name=model_name, device=device, output_dir=embeddings_out_dir, chroma_dir=None, force_recompute=True)
    documents = df["semantic_text"].fillna("").astype(str).tolist()

    # ingest into chroma
    chroma_dir = Path(args.chroma_dir)
    ids = [str(i) for i in range(len(documents))]
    print(f"Ingesting {len(ids)} vectors into Chroma at {chroma_dir} ...")
    client, collection = ingest_to_chroma_atomic(chroma_dir, ids, metas, documents, embeddings)

    # retrieval: compute candidate edges
    top_k = int(args.top_k)
    alpha = float(args.alpha)
    threshold = float(args.threshold)
    # Ensure retriever knows about the canonical id column (template_id)
    id_column_candidates = ("template_id", "reqid", "template", "pid")
    gen = compute_candidate_edges_stream(
        df,
        embeddings=embeddings,
        collection=collection,
        top_k=top_k,
        threshold=threshold,
        batch_size=128,
        alpha=alpha,
        id_column_candidates=id_column_candidates,
    )

    # persist edges
    db_path = args.db
    store = EdgeStore(db_path)
    store.init_db()
    if getattr(args, "clear_db", False):
        print(f"--clear-db specified: clearing edges table in {db_path} before writing")
        store.clear_edges(reset_sequence=True)
    print(f"Writing edges to {db_path} ...")
    edges_written = store.write_edges(gen, batch_size=500)
    print(f"Edges written: {edges_written}")

    # compute transitions
    transitions_written = compute_and_persist_transitions(db_path, min_count=1)
    print(f"Transitions written: {transitions_written}")

    store.close()
    print("Phase 2 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
