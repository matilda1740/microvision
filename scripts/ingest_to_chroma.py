"""Small utility to ingest embeddings DataFrame into a ChromaDB collection
while ensuring metadata timestamps are serialized to ISO-8601.

Usage (from project root):
    python scripts/ingest_to_chroma.py --csv data/embeddings.csv --persist data/chroma --collection my_col
"""
from __future__ import annotations

import argparse
import pandas as pd
from tqdm import tqdm

try:
    import chromadb
except Exception:  # pragma: no cover - optional runtime dependency
    chromadb = None


def repopulate_chroma_from_csv(csv_path: str, persist_dir: str, collection_name: str, batch_size: int = 128, update_service_map: bool = False):
    if chromadb is None:
        raise RuntimeError("chromadb is required to run this script")

    df = pd.read_csv(csv_path)

    # parse embedding column if it's serialized
    if "embedding" in df.columns:
        try:
            # keep as Python lists for Chroma
            df["embedding"] = df["embedding"].apply(lambda x: x if isinstance(x, list) else eval(x))
        except Exception:
            pass

    ids = df["embedding_id"].tolist() if "embedding_id" in df.columns else [str(i) for i in range(len(df))]
    docs = df.get("semantic_text", [None] * len(df))
    embs = df["embedding"].tolist() if "embedding" in df.columns else None

    from src.encode_chroma.encode_utils import df_to_metadatas

    meta_cols = [c for c in df.columns if c not in ("embedding", "semantic_text", "structured_metadata", "Timestamp")]
    # Allow optional updates to the service map during ingestion (opt-in default)
    metadatas = df_to_metadatas(df, meta_cols=meta_cols, update_service_map=update_service_map)

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name=collection_name)

    for i in tqdm(range(0, len(ids), batch_size), desc="Ingesting to Chroma"):
        kwargs = {"ids": ids[i:i+batch_size], "metadatas": metadatas[i:i+batch_size]}
        if docs is not None:
            kwargs["documents"] = docs[i:i+batch_size]
        if embs is not None:
            kwargs["embeddings"] = embs[i:i+batch_size]
        collection.add(**kwargs)

    print(f"✅ Inserted {len(ids)} vectors into collection '{collection_name}' at '{persist_dir}'")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--persist", required=True)
    p.add_argument("--collection", required=True)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--update-service-map", action="store_true", help="Opt-in: update the service name map from this ingest")
    args = p.parse_args()

    repopulate_chroma_from_csv(args.csv, args.persist, args.collection, args.batch_size, update_service_map=args.update_service_map)


if __name__ == "__main__":
    main()
