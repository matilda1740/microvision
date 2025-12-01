"""Scalable semantic retrieval and candidate-edge generation utilities.

Previously part of src.enrichment; moved to src.retrieval to separate concerns.
"""
from __future__ import annotations

from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence, Tuple
import math
import numpy as np
import logging
from config.settings import settings
from src.utils.time_utils import get_canonical_timestamp, to_iso_string

try:
    # chromadb is optional for unit tests that don't touch vector DBs
    import chromadb  # type: ignore
except Exception:
    chromadb = None  # type: ignore

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kw: x  # type: ignore


def _batch_indices(n: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield start, end


def batch_query_chroma(
    collection: Any,
    query_embeddings: Sequence[Sequence[float]],
    top_k: int = 5,
    metadata_filter: Optional[Dict[str, Any]] = None,
    include_distances: bool = True,
    batch_size: int = 256,
) -> List[Dict[str, Any]]:
    if collection is None:
        raise ValueError("collection is required for batch_query_chroma")

    q_embs = np.asarray(query_embeddings)
    results: List[Dict[str, Any]] = []

    for start, end in _batch_indices(len(q_embs), batch_size):
        batch = q_embs[start:end].tolist()
        try:
            # newer chroma versions validate the include list; 'ids' is not
            # allowed there. Request metadatas and distances explicitly and
            # fall back if the collection/query implementation differs.
            out = collection.query(
                query_embeddings=batch,
                n_results=top_k,
                where=metadata_filter,
                include=["metadatas", "distances"],
            )
        except TypeError:
            out = collection.query(
                query_embeddings=batch,
                n_results=top_k,
                where=metadata_filter,
            )

        # make safe defaults sized to the batch so missing keys won't IndexError
        defaults = {
            "ids": [None] * len(batch),
            "metadatas": [None] * len(batch),
            "distances": [None] * len(batch),
        }

        results.extend(
            [{k: out.get(k, defaults[k])[i] for k in ("ids", "metadatas", "distances")} for i in range(len(batch))]
        )

    return results


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0 or math.isnan(denom):
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_candidate_edges_stream(
    df,  # pandas.DataFrame like
    embeddings: Optional[np.ndarray] = None,
    collection: Any = None,
    top_k: int = 5,
    threshold: float = 0.4,
    batch_size: int = 128,
    alpha: float = 0.5,
    id_column_candidates: Sequence[str] = ("doc_id", "id", "orig_idx", "index"),
) -> Generator[Dict[str, Any], None, None]:
    try:
        import pandas as pd
    except Exception:
        pd = None  # type: ignore

    n = len(df)

    id_col = None
    for col in id_column_candidates:
        if col in df.columns:
            id_col = col
            break

    # Prefer an explicit canonical timestamp column when present
    timestamp_candidates = ("timestamp_canonical", "timestamp", "time", "ts", "created_at")
    ts_col = None
    for c in timestamp_candidates:
        if c in df.columns:
            ts_col = c
            break

    id_to_index = None
    if id_col is not None:
        id_to_index = {str(v): int(i) for i, v in enumerate(df[id_col].values)}
    logger = logging.getLogger(__name__)

    if collection is not None:
        if embeddings is None:
            raise ValueError("embeddings array required when querying a collection")

        q_embs = np.asarray(embeddings)
        results = batch_query_chroma(collection, q_embs, top_k=top_k, batch_size=batch_size)

        for src_idx, retrieved in enumerate(results):
            src_id = None if id_col is None else df.iloc[src_idx][id_col]
            ids = retrieved.get("ids", [])
            metas = retrieved.get("metadatas", [])
            dists = retrieved.get("distances", []) if retrieved.get("distances") is not None else [None] * len(ids)

            for tgt_idx_local, tgt_id in enumerate(ids):
                meta = metas[tgt_idx_local] if tgt_idx_local < len(metas) else None
                dist = dists[tgt_idx_local] if tgt_idx_local < len(dists) else None
                sim = None if dist is None else max(0.0, 1.0 - float(dist))

                semantic_cosine = None
                tgt_index = None
                canonical_target_id = None

                # Prefer matching a canonical id value (e.g. template_id) returned
                # directly by the collection. This ensures target_id stored in the
                # edges table is the meaningful template identifier.
                if id_to_index is not None and tgt_id is not None and str(tgt_id) in id_to_index:
                    tgt_index = id_to_index[str(tgt_id)]
                    try:
                        if id_col is not None:
                            canonical_target_id = str(df.iloc[tgt_index][id_col])
                    except Exception:
                        canonical_target_id = None

                # If we didn't resolve by canonical id, try interpreting the
                # returned id as a numeric positional index (some ingestion
                # pipelines use stringified integers as ids).
                if tgt_index is None and tgt_id is not None:
                    try:
                        maybe_idx = int(tgt_id)
                        if 0 <= maybe_idx < len(df):
                            tgt_index = maybe_idx
                            try:
                                if id_col is not None:
                                    canonical_target_id = str(df.iloc[tgt_index][id_col])
                            except Exception:
                                canonical_target_id = None
                    except Exception:
                        # not a numeric id — leave tgt_index as None
                        pass

                # Compute semantic cosine when we have a resolved target index
                if tgt_index is not None:
                    try:
                        semantic_cosine = _cosine_sim(q_embs[src_idx], np.asarray(embeddings)[tgt_index])
                    except Exception:
                        semantic_cosine = None

                source_timestamp = None
                target_timestamp = None
                time_delta_ms = None
                if ts_col is not None:
                    try:
                        source_timestamp = df.iloc[src_idx][ts_col]
                    except Exception:
                        source_timestamp = None
                if isinstance(meta, dict):
                    for k in ("timestamp", "time", "ts", "created_at"):
                        if k in meta and meta[k] is not None:
                            target_timestamp = meta[k]
                            break
                # compute canonical timestamps for storage and delta computation
                source_timestamp_canonical = None
                target_timestamp_canonical = None
                try:
                    if source_timestamp is not None:
                        stc = get_canonical_timestamp(source_timestamp)
                        source_timestamp_canonical = None if stc is None else stc.strftime("%Y-%m-%dT%H:%M:%SZ")
                    if target_timestamp is not None:
                        ttc = get_canonical_timestamp(target_timestamp)
                        target_timestamp_canonical = None if ttc is None else ttc.strftime("%Y-%m-%dT%H:%M:%SZ")
                    if source_timestamp_canonical is not None and target_timestamp_canonical is not None:
                        import pandas as _pd

                        st = _pd.to_datetime(source_timestamp_canonical)
                        tt = _pd.to_datetime(target_timestamp_canonical)
                        time_delta_ms = float((tt - st).total_seconds() * 1000.0)
                    else:
                        time_delta_ms = None
                except Exception:
                    time_delta_ms = None

                retrieval_distance = dist
                retrieval_similarity = None if retrieval_distance is None else max(0.0, 1.0 - float(retrieval_distance))

                if retrieval_similarity is not None and semantic_cosine is not None:
                    hybrid = float(alpha * retrieval_similarity + (1.0 - alpha) * semantic_cosine)
                elif retrieval_similarity is not None:
                    hybrid = float(retrieval_similarity)
                elif semantic_cosine is not None:
                    hybrid = float(semantic_cosine)
                else:
                    hybrid = None

                if hybrid is None:
                    continue

                if hybrid >= threshold:
                    # decide stored target_id: prefer canonical_target_id, then
                    # DataFrame lookup by tgt_index, then fall back to raw returned id
                    if canonical_target_id is not None:
                        stored_target_id = canonical_target_id
                    elif tgt_index is not None and id_col is not None:
                        try:
                            stored_target_id = str(df.iloc[tgt_index][id_col])
                        except Exception:
                            stored_target_id = tgt_id
                    else:
                        stored_target_id = tgt_id

                    # Log unresolved cases for later debugging: when we could not
                    # map the returned id to a canonical template id.
                    if id_col is not None and (stored_target_id is None or (isinstance(stored_target_id, str) and str(stored_target_id) not in id_to_index)):
                        logger.debug("unresolved target id mapping: src_idx=%s returned_id=%r tgt_index=%r canonical=%r", src_idx, tgt_id, tgt_index, canonical_target_id)

                    yield {
                        "source_index": int(src_idx),
                        "source_id": src_id,
                        "target_id": stored_target_id,
                        "target_metadata": meta,
                        "source_timestamp": source_timestamp,
                        "target_timestamp": target_timestamp,
                        "source_timestamp_canonical": source_timestamp_canonical,
                        "target_timestamp_canonical": target_timestamp_canonical,
                        "time_delta_ms": time_delta_ms,
                        "retrieval_distance": retrieval_distance,
                        "retrieval_similarity": retrieval_similarity,
                        "semantic_cosine": semantic_cosine,
                        "hybrid_score": hybrid,
                        "alpha": float(alpha),
                        "target_semantic_text": (meta.get("semantic_text") if isinstance(meta, dict) else None),
                        "source_semantic_text": (df.iloc[src_idx]["semantic_text"] if "semantic_text" in df.columns else None),
                    }

        return

    # Fallback: local all-pairs comparison if no collection provided
    if embeddings is None:
         raise ValueError("embeddings array required for local comparison")
    
    emb = np.asarray(embeddings)
    # normalize for cosine similarity
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    emb_norm = emb / norms
    
    for start, end in _batch_indices(n, batch_size):
        batch = emb_norm[start:end]
        sims = np.dot(batch, emb_norm.T)
        
        for i_local in range(len(batch)):
            src_idx = start + i_local
            row = sims[i_local]
            candidate_idxs = np.where(row >= threshold)[0]
            
            for tgt_idx in candidate_idxs:
                if tgt_idx == src_idx:
                    continue
                
                retrieval_similarity = float(row[tgt_idx])
                retrieval_distance = max(0.0, 1.0 - retrieval_similarity)
                # For local embeddings, retrieval_similarity IS the semantic cosine
                semantic_cosine = retrieval_similarity
                hybrid = float(semantic_cosine)

                source_timestamp = None
                target_timestamp = None
                time_delta_ms = None
                
                if ts_col is not None:
                    try:
                        source_timestamp = df.iloc[src_idx][ts_col]
                        target_timestamp = df.iloc[tgt_idx][ts_col]
                    except Exception:
                        pass
                
                # Compute time delta if possible
                try:
                    stc = get_canonical_timestamp(source_timestamp)
                    ttc = get_canonical_timestamp(target_timestamp)
                    if stc is not None and ttc is not None:
                        time_delta_ms = float((ttc - stc).total_seconds() * 1000.0)
                except Exception:
                    time_delta_ms = None

                tgt_id_val = df.iloc[tgt_idx][id_col] if (id_col is not None) else int(tgt_idx)
                
                yield {
                    "source_index": int(src_idx),
                    "source_id": None if id_col is None else df.iloc[src_idx][id_col],
                    "target_id": tgt_id_val,
                    "target_metadata": None,
                    "source_timestamp": source_timestamp,
                    "target_timestamp": target_timestamp,
                    "source_timestamp_canonical": to_iso_string(get_canonical_timestamp(source_timestamp)),
                    "target_timestamp_canonical": to_iso_string(get_canonical_timestamp(target_timestamp)),
                    "time_delta_ms": time_delta_ms,
                    "retrieval_distance": retrieval_distance,
                    "retrieval_similarity": retrieval_similarity,
                    "semantic_cosine": semantic_cosine,
                    "hybrid_score": hybrid,
                    "alpha": float(alpha),
                    "target_semantic_text": (df.iloc[tgt_idx]["semantic_text"] if "semantic_text" in df.columns else None),
                    "source_semantic_text": (df.iloc[src_idx]["semantic_text"] if "semantic_text" in df.columns else None),
                }


__all__ = ["batch_query_chroma", "compute_candidate_edges_stream"]
