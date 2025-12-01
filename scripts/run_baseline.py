"""Baseline pipeline runner using TF-IDF (Syntactic) instead of BERT (Semantic).

This script implements the "Control Group" for the thesis evaluation.
It uses:
1. TF-IDF Vectorization on log templates.
2. Cosine Similarity to find edges.
3. Standard EdgeStore for persistence.

Usage:
    python scripts/run_baseline.py --input data/merged_templates.csv --db data/edges/baseline_edges.db
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Generator, Dict, Any

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.storage import EdgeStore


def load_data(input_path: Path) -> pd.DataFrame:
    """Load merged templates CSV."""
    print(f"Loading data from {input_path}...")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    # Ensure we have the template column
    if "template" not in df.columns:
        # Fallback to 'content' or 'message' if template is missing, but warn
        if "content" in df.columns:
            print("Warning: 'template' column missing, using 'content'")
            df["template"] = df["content"]
        else:
            raise ValueError(f"Input CSV must have 'template' column. Found: {df.columns}")
            
    # Fill NaNs
    df["template"] = df["template"].fillna("")
    return df


def compute_tfidf_edges(df: pd.DataFrame, top_k: int = 10, threshold: float = 0.2) -> Generator[Dict[str, Any], None, None]:
    """
    Compute edges using TF-IDF and Cosine Similarity.
    Yields edge dictionaries compatible with EdgeStore.
    """
    print("Vectorizing templates using TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df["template"])
    
    n_samples = tfidf_matrix.shape[0]
    print(f"Computed TF-IDF matrix: {n_samples} samples, {tfidf_matrix.shape[1]} features")
    
    print(f"Computing cosine similarity (Top-K={top_k}, Threshold={threshold})...")
    
    # We process in chunks to avoid OOM on large datasets if necessary, 
    # but for typical thesis benchmarks (thousands of logs), full matrix is fine.
    # Using sklearn's cosine_similarity which returns a dense matrix.
    # For very large N, we might need a sparse approach, but let's stick to simplicity first.
    
    # To avoid OOM with N^2, we can iterate row by row
    for i in range(n_samples):
        # Compute similarity for one row against all others
        # tfidf_matrix is sparse, so this is efficient
        sim_scores = cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix).flatten()
        
        # Get indices of top_k scores
        # We want to exclude self-match (i) usually, or keep it? 
        # The pipeline usually keeps it but we filter self-loops later.
        # Let's get top_k + 1 to account for self
        if top_k >= n_samples:
            top_indices = np.argsort(sim_scores)[::-1]
        else:
            # argpartition is faster for top-k
            top_indices = np.argpartition(sim_scores, -(top_k+1))[-(top_k+1):]
            # sort them by score descending
            top_indices = top_indices[np.argsort(sim_scores[top_indices])[::-1]]
            
        source_row = df.iloc[i]
        
        for j in top_indices:
            if i == j:
                continue
                
            score = float(sim_scores[j])
            if score < threshold:
                continue
                
            target_row = df.iloc[j]
            
            # Construct edge dict matching EdgeStore schema
            yield {
                "source_id": str(source_row.get("template_id", i)),
                "target_id": str(target_row.get("template_id", j)),
                "weight": score,
                "source_timestamp": str(source_row.get("timestamp", "")),
                "target_timestamp": str(target_row.get("timestamp", "")),
                "source_component": str(source_row.get("component", "")),
                "target_component": str(target_row.get("component", "")),
                "source_service": str(source_row.get("service", "")),
                "target_service": str(target_row.get("service", "")),
                "hybrid_score": score, # For baseline, hybrid is just tf-idf score
                "semantic_cosine": score,
                "retrieval_similarity": score
            }
            
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{n_samples} rows...", end="\r")
            
    print(f"\nFinished computing edges.")


def main():
    parser = argparse.ArgumentParser(description="Run TF-IDF Baseline Pipeline")
    parser.add_argument("--input", type=Path, default=Path("data/merged_templates.csv"), help="Path to input CSV")
    parser.add_argument("--db", type=Path, default=Path("data/edges/baseline_edges.db"), help="Path to output SQLite DB")
    parser.add_argument("--top-k", type=int, default=10, help="Top K neighbors")
    parser.add_argument("--threshold", type=float, default=0.1, help="Similarity threshold")
    parser.add_argument("--clear-db", action="store_true", help="Clear existing DB before writing")
    
    args = parser.parse_args()
    
    # 1. Load Data
    try:
        df = load_data(args.input)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Init DB
    store = EdgeStore(str(args.db))
    store.init_db()
    if args.clear_db:
        print(f"Clearing DB at {args.db}...")
        store.clear_edges(reset_sequence=True)

    # 3. Compute & Store Edges
    t0 = time.time()
    edge_gen = compute_tfidf_edges(df, top_k=args.top_k, threshold=args.threshold)
    
    count = store.write_edges(edge_gen, batch_size=1000)
    duration = time.time() - t0
    
    print(f"\nBaseline Run Complete.")
    print(f" - Input: {args.input}")
    print(f" - Output: {args.db}")
    print(f" - Edges Written: {count}")
    print(f" - Time Taken: {duration:.2f}s")
    
    store.close()


if __name__ == "__main__":
    main()
