"""Encoding utilities: compute embeddings, cache, and optionally ingest to Chroma.

This module is intentionally defensive: heavy deps (sentence-transformers,
chromadb) are imported lazily to keep unit tests fast.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import hashlib
import json
import pandas as pd

from src.encode_chroma.encode_utils import df_to_metadatas


def _ensure_dir(path: Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def encode_templates(
    input_df: pd.DataFrame,
    model=None,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    output_dir: str = "data/datasets/embeddings",
    collection_name: str = "smoke_collection",
    chroma_dir: Optional[str] = None,
    force_recompute: bool = False,
    batch_size: Optional[int] = None,
) -> Tuple[np.ndarray, pd.DataFrame, Optional[object]]:
    """Encode templates and optionally persist to Chroma.

    Returns (embeddings, index_df, collection_or_None).

    - input_df must contain a `semantic_text` column.
    - If `model` is None, `model_name` must be provided and SentenceTransformer
      will be loaded lazily.
    - If `chroma_dir` is provided, the function will attempt to ingest the
      embeddings using the chroma helper and return the collection object.
    """
    if "semantic_text" not in input_df.columns:
        raise KeyError("input_df must contain a 'semantic_text' column")

    out_dir = Path(output_dir)
    _ensure_dir(out_dir)
    emb_path = out_dir / "embeddings.npy"
    index_path = out_dir / "embeddings_index.csv"

    # Try cached reload (validate metadata)
    meta_path = out_dir / "embeddings.meta.json"
    if not force_recompute and emb_path.exists() and index_path.exists() and meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            # compute current text hash and count
            cur_count = len(texts)
            h = hashlib.sha256()
            for t in texts:
                h.update(t.encode("utf-8", errors="surrogatepass"))
                h.update(b"\n")
            cur_hash = h.hexdigest()
            if int(meta.get("count", -1)) == cur_count and meta.get("hash") == cur_hash:
                embeddings = np.load(emb_path)
                df_index = pd.read_csv(index_path)
                collection = None
                return embeddings, df_index, collection
        except Exception:
            # fall back to recomputing
            pass

    # prepare model
    if model is None:
        if model_name is None:
            raise ValueError("Either model instance or model_name must be provided")
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, device=device if device is not None else "cpu")
        except Exception as e:
            raise RuntimeError("sentence-transformers is required to encode templates") from e

    # auto choose batch_size
    if batch_size is None:
        batch_size = 128 if (device and device.lower() in ("cuda", "mps")) else 64

    texts = input_df["semantic_text"].fillna("").astype(str).tolist()

    # encode in batches to avoid memory spikes
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embs.append(emb)
    if embs:
        embeddings = np.vstack(embs)
    else:
        embeddings = np.zeros((0, 0))

    # save cache atomically and write metadata
    try:
        from src.utils.atomic_write import atomic_save_npy, atomic_write_csv, atomic_write_json

        atomic_save_npy(emb_path, embeddings)
        atomic_write_csv(index_path, input_df)
        # write metadata: count and hash of texts
        h = hashlib.sha256()
        for t in texts:
            h.update(t.encode("utf-8", errors="surrogatepass"))
            h.update(b"\n")
        meta = {"count": len(texts), "hash": h.hexdigest()}
        atomic_write_json(meta_path, meta)
    except Exception:
        # best-effort fallback to inplace writes
        np.save(emb_path, embeddings)
        input_df.to_csv(index_path, index=False)
        try:
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump({"count": len(texts), "hash": h.hexdigest()}, fh)
        except Exception:
            pass

    # Build metadatas using centralized helper (coercion applied there)
    metas = df_to_metadatas(input_df)

    collection = None
    if chroma_dir is not None:
        # ingest into chroma using the helper to keep ingestion atomic and consistent
        try:
            from src.encode_chroma.chroma import ingest_to_chroma_atomic

            ids = [str(i) for i in range(len(texts))]
            client, collection = ingest_to_chroma_atomic(Path(chroma_dir), ids, metas, texts, embeddings)
        except Exception:
            # non-fatal here; return embeddings and index and let caller decide
            collection = None

    return embeddings, input_df, collection
