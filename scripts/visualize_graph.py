"""Small utility to load edges DB and produce an interactive HTML graph.

This module exposes helpers so it can be used as a script or imported from
the Streamlit app. Heavy deps (pyvis) are imported lazily.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any

from src.visualization.graph import edges_to_networkx, write_pyvis_html, networkx_to_dict


def load_edges_from_db(db_path: str, limit: int = 10000) -> List[Dict[str, Any]]:
    """Load recent edges from the edges DB and return a list of edge dicts.

    The function is defensive: it parses JSON metadata stored in `target_metadata`.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT source_id, target_id, time_delta_ms, hybrid_score, alpha, target_metadata FROM edges ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    edges = []
    for src, tgt, td, hybrid, alpha, meta in rows:
        try:
            meta_obj = json.loads(meta) if meta else None
        except Exception:
            meta_obj = None
        edges.append({
            "source": src,
            "target": tgt,
            "time_delta_ms": td,
            "hybrid_score": hybrid,
            "alpha": alpha,
            "target_metadata": meta_obj,
        })
    conn.close()
    return edges


def build_and_write_html(db_path: str, out_path: str, limit: int = 10000):
    edges = load_edges_from_db(db_path, limit=limit)
    G = edges_to_networkx(edges)
    # ensure pyvis available inside writer; write_pyvis_html will raise if missing
    write_pyvis_html(G, out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True, help="Path to edges DB")
    p.add_argument("--out", required=True, help="Output HTML path")
    p.add_argument("--limit", type=int, default=10000)
    args = p.parse_args()

    build_and_write_html(args.db, args.out, limit=args.limit)


if __name__ == "__main__":
    main()
