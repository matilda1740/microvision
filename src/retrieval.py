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

        # Ensure embeddings is a numpy array
        embeddings = np.asarray(embeddings)

        # Initialize Sparse Retriever (TF-IDF)
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Fit on all target documents
            corpus = df["semantic_text"].fillna("").astype(str).tolist()
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(corpus)
            has_sparse = True
        except ImportError:
            logger.warning("scikit-learn not found. Sparse retrieval disabled.")
            has_sparse = False
            tfidf_matrix = None
            vectorizer = None

        # Strategy 3: Metadata-Aware Hard Filtering
        # Group by service to apply {"service": {"$ne": current_service}}
        service_col = "service"
        groups = []

        if service_col in df.columns:
            # Handle valid services
            unique_services = df[service_col].dropna().unique()
            for service in unique_services:
                # Ensure we're working with the boolean mask as a numpy array
                mask = (df[service_col] == service).values
                indices = np.where(mask)[0]
                # Apply filter only if service is a non-empty string
                if isinstance(service, str) and service:
                    groups.append((indices, {"service": {"$ne": service}}))
                else:
                    groups.append((indices, None))
            
            # Handle missing services (NaN/None)
            mask_na = df[service_col].isna().values
            if mask_na.any():
                indices = np.where(mask_na)[0]
                groups.append((indices, None))
        else:
            groups.append((np.arange(n), None))

        for indices, metadata_filter in groups:
            if len(indices) == 0:
                continue

            group_embeddings = embeddings[indices]
            
            # 1. Dense Retrieval (Chroma)
            dense_results = batch_query_chroma(
                collection, 
                group_embeddings, 
                top_k=top_k, 
                metadata_filter=metadata_filter, 
                batch_size=batch_size
            )

            # 2. Sparse Retrieval (TF-IDF) - Batch
            sparse_sims_batch = None
            if has_sparse:
                # Transform source batch
                source_corpus = df.iloc[indices]["semantic_text"].fillna("").astype(str).tolist()
                source_tfidf = vectorizer.transform(source_corpus)
                # Compute similarity against ALL targets (small dataset assumption)
                sparse_sims_batch = cosine_similarity(source_tfidf, tfidf_matrix)

            for local_idx, retrieved in enumerate(dense_results):
                src_idx = indices[local_idx]
                src_id = None if id_col is None else df.iloc[src_idx][id_col]
                
                # Collect candidates from Dense
                dense_ids = retrieved.get("ids", [])
                dense_dists = retrieved.get("distances", []) if retrieved.get("distances") is not None else [None] * len(dense_ids)
                dense_metas = retrieved.get("metadatas", [])
                
                candidates = {} # tgt_index -> {dense_score, sparse_score, meta, id}

                # Process Dense Candidates
                for i, tgt_id in enumerate(dense_ids):
                    if tgt_id is None: continue
                    
                    # Resolve tgt_index
                    tgt_index = None
                    if id_to_index is not None and str(tgt_id) in id_to_index:
                        tgt_index = id_to_index[str(tgt_id)]
                    elif tgt_id is not None:
                        try:
                            maybe_idx = int(tgt_id)
                            if 0 <= maybe_idx < len(df):
                                tgt_index = maybe_idx
                        except: pass
                    
                    if tgt_index is None: continue
                    
                    dist = dense_dists[i]
                    dense_score = max(0.0, 1.0 - float(dist)) if dist is not None else 0.0
                    
                    candidates[tgt_index] = {
                        "dense_score": dense_score,
                        "sparse_score": 0.0,
                        "meta": dense_metas[i] if i < len(dense_metas) else None,
                        "tgt_id": tgt_id
                    }

                # Process Sparse Candidates (Top K)
                if has_sparse and sparse_sims_batch is not None:
                    row = sparse_sims_batch[local_idx]
                    # Filter out self (src_idx) and same-service (if metadata_filter active)
                    # We can't easily apply metadata_filter to sparse without iterating
                    # So we just check the service column manually
                    
                    # Get Top 2*K candidates to ensure we have enough after filtering
                    top_sparse_indices = np.argsort(row)[::-1][:top_k*2]
                    
                    current_service = df.iloc[src_idx].get("service")
                    
                    for tgt_idx in top_sparse_indices:
                        if tgt_idx == src_idx: continue
                        
                        # Apply Hard Filter (Same Service)
                        tgt_service = df.iloc[tgt_idx].get("service")
                        if current_service and tgt_service and current_service == tgt_service:
                            continue
                            
                        sparse_score = float(row[tgt_idx])
                        if sparse_score < 0.1: continue # Ignore weak matches
                        
                        if tgt_idx not in candidates:
                            # Fetch metadata for new sparse candidate
                            # We need to reconstruct what Chroma would return
                            # We have the DF row, so we can build it
                            tgt_row = df.iloc[tgt_idx]
                            meta = {
                                "service": tgt_row.get("service"),
                                "component": tgt_row.get("component"),
                                "semantic_text": tgt_row.get("semantic_text"),
                                "timestamp": tgt_row.get(ts_col) if ts_col else None
                            }
                            tgt_id = tgt_row[id_col] if id_col else str(tgt_idx)
                            
                            candidates[tgt_idx] = {
                                "dense_score": 0.0, # Will be updated if we compute cosine
                                "sparse_score": sparse_score,
                                "meta": meta,
                                "tgt_id": tgt_id
                            }
                        else:
                            candidates[tgt_idx]["sparse_score"] = sparse_score

                # Final Scoring and Yielding
                # We need to compute dense score for sparse-only candidates if we want true hybrid
                # But for speed, we might skip it or compute it on the fly
                
                final_candidates = []
                for tgt_idx, data in candidates.items():
                    # If dense_score is 0 (sparse only), we should compute it for fairness
                    if data["dense_score"] == 0.0:
                        try:
                            data["dense_score"] = _cosine_sim(embeddings[src_idx], embeddings[tgt_idx])
                        except: pass
                    
                    # Hybrid Score Formula
                    # We use a weighted sum. 
                    # Note: dense_score is Cosine (0-1), sparse_score is Cosine (0-1)
                    hybrid_score = alpha * data["dense_score"] + (1.0 - alpha) * data["sparse_score"]
                    
                    if hybrid_score >= threshold:
                        final_candidates.append((hybrid_score, tgt_idx, data))
                
                # Sort by hybrid score descending
                final_candidates.sort(key=lambda x: x[0], reverse=True)
                
                # Yield Top K
                for score, tgt_index, data in final_candidates[:top_k]:
                    
                    # ... (Rest of the yielding logic is similar, just using data dict)
                    meta = data["meta"]
                    tgt_id = data["tgt_id"]
                    
                    canonical_target_id = None
                    if id_col is not None:
                         try:
                             canonical_target_id = str(df.iloc[tgt_index][id_col])
                         except: pass
                    
                    stored_target_id = canonical_target_id if canonical_target_id else tgt_id

                    # Timestamp logic (same as before)
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
                        "retrieval_distance": 1.0 - data["dense_score"], # Approx
                        "retrieval_similarity": data["dense_score"],
                        "semantic_cosine": data["dense_score"],
                        "hybrid_score": score,
                        "alpha": float(alpha),
                        "target_semantic_text": (meta.get("semantic_text") if isinstance(meta, dict) else None),
                        "source_semantic_text": (df.iloc[src_idx]["semantic_text"] if "semantic_text" in df.columns else None),
                        "source_service": df.iloc[src_idx].get("service"),
                        "source_component": df.iloc[src_idx].get("component"),
                        "target_service": df.iloc[tgt_index].get("service"),
                        "target_component": df.iloc[tgt_index].get("component"),
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
                    "source_service": df.iloc[src_idx].get("service"),
                    "source_component": df.iloc[src_idx].get("component"),
                    "target_service": df.iloc[tgt_idx].get("service"),
                    "target_component": df.iloc[tgt_idx].get("component"),
                }


__all__ = ["batch_query_chroma", "compute_candidate_edges_stream"]
