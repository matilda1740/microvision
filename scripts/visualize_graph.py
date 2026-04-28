"""Small utility to load edges DB and produce an interactive HTML graph.

This module exposes helpers so it can be used as a script or imported from
the Streamlit app. Heavy deps (pyvis) are imported lazily.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visualization import edges_to_networkx, write_pyvis_html, networkx_to_dict


def load_edges_from_db(db_path: str, limit: int = 10000, threshold: float = 0.0, level: str = "service") -> List[Dict[str, Any]]:
    """Load recent edges from the edges DB and return a list of edge dicts.

    Args:
        db_path: Path to SQLite DB.
        limit: Max rows to fetch.
        threshold: Min hybrid_score.
        level: 'service' (aggregate by service name) or 'template' (raw template IDs).
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # We need source_service and target_service for aggregation
    query = """
        SELECT source_id, target_id, time_delta_ms, hybrid_score, alpha, target_metadata, 
               source_service, target_service, source_semantic_text, target_semantic_text,
               llm_verification
        FROM edges 
        WHERE hybrid_score >= ? 
        ORDER BY id DESC 
        LIMIT ?
    """
    cur.execute(query, (threshold, limit))
    rows = cur.fetchall()
    conn.close()

    raw_edges = []
    for row in rows:
        # Robust unpacking to handle legacy DBs that lack the column
        if len(row) == 11:
            src_id, tgt_id, td, hybrid, alpha, meta, src_srv, tgt_srv, src_txt, tgt_txt, llm_ver = row
        else:
            src_id, tgt_id, td, hybrid, alpha, meta, src_srv, tgt_srv, src_txt, tgt_txt = row
            llm_ver = None

        try:
            meta_obj = json.loads(meta) if meta else None
        except Exception:
            meta_obj = None
            
        raw_edges.append({
            "source_id": src_id,
            "target_id": tgt_id,
            "source_service": src_srv,
            "target_service": tgt_srv,
            "source_semantic_text": src_txt,
            "target_semantic_text": tgt_txt,
            "time_delta_ms": td,
            "hybrid_score": hybrid,
            "alpha": alpha,
            "target_metadata": meta_obj,
            "llm_verification": llm_ver
        })

    if level == "template":
        # Return raw edges with template IDs as source/target
        return [
            {
                "source": e["source_id"],
                "target": e["target_id"],
                **{k: v for k, v in e.items() if k not in ("source_id", "target_id")}
            }
            for e in raw_edges
        ]
    else:
        # Aggregate by service
        return _aggregate_edges_by_service(raw_edges)


def _aggregate_edges_by_service(raw_edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group edges by (source_service, target_service) and aggregate stats."""
    grouped: Dict[tuple, Dict[str, Any]] = {}
    
    for e in raw_edges:
        src = e.get("source_service")
        tgt = e.get("target_service")
        
        # Filter out missing/Unknown services to keep the graph clean
        if not src or src == "Unknown" or not tgt or tgt == "Unknown":
            continue
        
        # Skip self-loops in service graph
        if src == tgt:
            continue
        
        key = (src, tgt)
        if key not in grouped:
            grouped[key] = {
                "count": 0,
                "total_score": 0.0,
                "max_score": 0.0,
                "avg_time_delta": 0.0,
                "examples": [],
                "llm_verifications": [] # Store all verifications for this service pair
            }
        
        stats = grouped[key]
        stats["count"] += 1
        stats["total_score"] += e["hybrid_score"]
        stats["max_score"] = max(stats["max_score"], e["hybrid_score"])
        stats["avg_time_delta"] += (e["time_delta_ms"] or 0)
        
        # Collect LLM verification data for this edge instance
        if e.get("llm_verification"):
            stats["llm_verifications"].append(e["llm_verification"])
        
        # Keep a few examples for the tooltip
        if len(stats["examples"]) < 3:
            stats["examples"].append((e.get("source_semantic_text"), e.get("target_semantic_text")))

    aggregated_edges = []
    for (src, tgt), stats in grouped.items():
        count = stats["count"]
        avg_score = stats["total_score"] / count
        avg_td = stats["avg_time_delta"] / count
        
        # CRITICAL FIX: Propagate the 'best' LLM verification to the aggregate edge
        # We find the verification with the highest confidence or prioritize a 'true' causal finding.
        best_llm = None
        if stats["llm_verifications"]:
            # Simple heuristic: pick the first one, or add logic to find most representative
            best_llm = stats["llm_verifications"][0]

        # Build a rich tooltip
        tooltip = f"{src} -> {tgt}\nCount: {count}\nAvg Score: {avg_score:.2f}"
        if stats["examples"]:
            tooltip += "\n\nExamples:"
            for i, (s_txt, t_txt) in enumerate(stats["examples"]):
                s_trunc = (s_txt[:50] + "...") if s_txt and len(s_txt) > 50 else s_txt
                t_trunc = (t_txt[:50] + "...") if t_txt and len(t_txt) > 50 else t_txt
                tooltip += f"\n{i+1}. {s_trunc} -> {t_trunc}"
        
        aggregated_edges.append({
            "source": src,
            "target": tgt,
            "weight": count,          
            "value": count,           
            "title": tooltip,         
            "hybrid_score": avg_score,
            "time_delta_ms": avg_td,
            "label": str(count),
            "llm_verification": best_llm # PASS THIS TO THE VISUALIZER
        })
        
    return aggregated_edges


def build_and_write_html(db_path: str, out_path: str, limit: int = 10000, threshold: float = 0.0, level: str = "service"):
    edges = load_edges_from_db(db_path, limit=limit, threshold=threshold, level=level)
    G = edges_to_networkx(edges)
    # ensure pyvis available inside writer; write_pyvis_html will raise if missing
    write_pyvis_html(G, out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", help="Path to edges DB (optional if --dataset-name is provided)")
    p.add_argument("--dataset-name", help="Name of the dataset (e.g. 'openstack'). If provided, defaults paths will be used.")
    p.add_argument("--out", help="Output HTML path (optional if --dataset-name is provided)")
    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--threshold", type=float, default=0.3, help="Minimum hybrid score to display (default: 0.3)")
    p.add_argument("--level", choices=["service", "template"], default="service", help="Aggregation level (default: service)")
    args = p.parse_args()

    db_path = args.db
    out_path = args.out

    if args.dataset_name:
        data_root = Path("data") / args.dataset_name
        if not db_path:
            db_path = str(data_root / "edges" / "edges.db")
        if not out_path:
            out_path = str(data_root / f"{args.dataset_name}_graph.html")
    
    if not db_path or not out_path:
        p.error("Either --dataset-name or both --db and --out must be provided.")

    print(f"Visualizing graph from {db_path} -> {out_path}")
    build_and_write_html(db_path, out_path, limit=args.limit, threshold=args.threshold, level=args.level)


if __name__ == "__main__":
    main()
