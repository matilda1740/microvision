"""Evaluation script for MicroVision pipeline.

This script performs three main functions:
1. Generates a "Ground Truth" graph from the parsed logs by correlating 'reqid' across services (Runtime GT).
2. Loads a static "Knowledge Base" of architectural dependencies (Static GT) to augment the runtime truth.
3. Compares the inferred semantic graph (from edges.db) against this Hybrid Ground Truth to calculate
   Precision, Recall, and F1-Score.

Feature: Automated Knowledge Base Expansion
    The script can identify "High Confidence False Positives"—edges that the model is very sure about
    (high hybrid score) but are missing from the current Ground Truth. These are saved to
    `suggested_additions.json` for human review, allowing for a "Human-in-the-Loop" improvement cycle.

Limitations:
    - The Static Ground Truth mechanism currently relies on a dataset-specific JSON file 
      (`data/knowledge_base/openstack_architecture.json`).
    - This logic is tightly coupled to the OpenStack domain. Adapting this pipeline to a new 
      system (e.g., Kubernetes logs) requires manually creating a corresponding architecture file.

Usage:
    python scripts/evaluate_pipeline.py --parsed-logs data/parsed_sample.csv --edges-db data/edges/edges.db --suggest-updates
"""
import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Set, Tuple

import pandas as pd

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_parsed_logs(csv_path: Path) -> pd.DataFrame:""
    print(f"Loading parsed logs from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Fix for OpenStack logs where 'service' might be misparsed as 'INFO'/'WARN'
    # Extract service from the filename in 'raw' column: "nova-api.log.1..." -> "nova-api"
    if 'raw' in df.columns:
        # Regex to capture the part before .log
        # The raw column might be quoted, so handle optional quote
        extracted_service = df['raw'].astype(str).str.extract(r'^"?([a-zA-Z0-9-]+)\.log')
        
        # If we successfully extracted something, use it
        if not extracted_service.empty and extracted_service[0].notna().any():
            print("Refining 'service' column by extracting from filename in 'raw'...")
            df['service'] = extracted_service[0]
    
    # Ensure required columns exist
    required_cols = {'reqid', 'service', 'timestamp'}
    if not required_cols.issubset(df.columns):
        # If service is still missing, try to use component
        if 'component' in df.columns and 'service' not in df.columns:
             df['service'] = df['component']
        
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Input CSV missing required columns: {required_cols - set(df.columns)}")
    
    # Filter out logs without request IDs
    df = df.dropna(subset=['reqid'])
    df = df[df['reqid'] != '']
    
    # Convert timestamp to datetime for sorting
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Loaded {len(df)} logs with valid reqids.")
    return df


def generate_ground_truth(df: pd.DataFrame) -> Set[Tuple[str, str]]:
    """
    Generate ground truth edges based on reqid propagation.
    
    Logic:
    - Group by 'reqid'.
    - Sort by timestamp.
    - If Service A appears before Service B in the same reqid trace, 
      and they are different services, assume A -> B.
    
    Returns:
        Set of (source, target) tuples.
    """
    print("Generating ground truth from trace IDs...")
    ground_truth_edges = set()
    
    # Group by request ID
    grouped = df.groupby('reqid')
    
    for reqid, group in grouped:
        if len(group) < 2:
            continue
            
        # Sort by time to determine order
        sorted_group = group.sort_values('timestamp')
        services = sorted_group['service'].tolist()
        
        # Create edges between consecutive services in the trace
        # Note: This assumes a linear chain. For more complex concurrency, 
        # this is a simplified approximation (A->B, B->C).
        for i in range(len(services) - 1):
            src = services[i]
            dst = services[i+1]
            
            if src != dst:
                ground_truth_edges.add((src, dst))
                
    print(f"Identified {len(ground_truth_edges)} unique ground truth edges from traces.")
    return ground_truth_edges


import json

def get_static_ground_truth() -> Set[Tuple[str, str]]:
    """
    Returns a set of architecturally known dependencies for OpenStack.
    These serve as a fallback to 'rescue' valid edges that might be missing
    from the specific trace sample (Hybrid Ground Truth).
    """
    # Load from external JSON file
    kb_path = Path("data/knowledge_base/openstack_architecture.json")
    if kb_path.exists():
        try:
            with open(kb_path, "r") as f:
                data = json.load(f)
                # Convert list of lists to set of tuples
                return {tuple(edge) for edge in data}
        except Exception as e:
            print(f"Warning: Failed to load static knowledge base: {e}")
            return set()
    else:
        print(f"Warning: Static knowledge base not found at {kb_path}")
        return set()


def load_inferred_edges(db_path: Path, threshold: float = 0.0) -> Dict[Tuple[str, str], float]:
    """Load inferred edges from the SQLite database with their scores."""
    print(f"Loading inferred edges from {db_path}...")
    if not db_path.exists():
        raise FileNotFoundError(f"Edges DB not found at {db_path}")
        
    conn = sqlite3.connect(db_path)
    # Updated query to match actual schema: source_component/target_component or source_service/target_service
    # We'll try to use service if available, else component
    try:
        query = "SELECT source_service, target_service, hybrid_score FROM edges WHERE hybrid_score >= ?"
        cursor = conn.execute(query, (threshold,))
    except sqlite3.OperationalError:
        # Fallback if service columns don't exist (older schema?)
        print("Warning: 'source_service' column not found. Falling back to 'source_component'.")
        query = "SELECT source_component, target_component, hybrid_score FROM edges WHERE hybrid_score >= ?"
        cursor = conn.execute(query, (threshold,))
    
    inferred_edges = {}
    for row in cursor:
        src, dst, weight = row
        # Filter out None values and self-loops
        if src and dst and src != dst:
            # Keep the highest score if multiple edges exist between same pair
            if (src, dst) in inferred_edges:
                inferred_edges[(src, dst)] = max(inferred_edges[(src, dst)], weight)
            else:
                inferred_edges[(src, dst)] = weight
            
    conn.close()
    print(f"Loaded {len(inferred_edges)} inferred edges (threshold >= {threshold}).")
    return inferred_edges


def calculate_metrics(ground_truth: Set[Tuple[str, str]], inferred: Set[Tuple[str, str]]) -> Dict[str, float]:
    """Calculate Precision, Recall, and F1 using Hybrid Ground Truth."""
    
    # 1. Load Static Architecture Truth
    static_gt = get_static_ground_truth()
    
    # 2. Combine Trace GT and Static GT for a comprehensive 'Valid Edges' set
    # We treat an edge as valid if it appears in EITHER the runtime traces OR the static architecture.
    valid_ground_truth = ground_truth.union(static_gt)
    
    # Intersection: Edges found in both
    true_positives = valid_ground_truth.intersection(inferred)
    tp_count = len(true_positives)
    
    # Precision: TP / (TP + FP) -> How many inferred edges were correct?
    if len(inferred) > 0:
        precision = tp_count / len(inferred)
    else:
        precision = 0.0
        
    # Recall: TP / (TP + FN) -> How many ground truth edges did we find?
    # Note: We calculate recall against the COMBINED ground truth.
    if len(valid_ground_truth) > 0:
        recall = tp_count / len(valid_ground_truth)
    else:
        recall = 0.0
        
    # F1 Score
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
        
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp_count,
        "fp": len(inferred) - tp_count,
        "fn": len(valid_ground_truth) - tp_count,
        "trace_gt_count": len(ground_truth),
        "static_gt_count": len(static_gt),
        "combined_gt_count": len(valid_ground_truth)
    }


def suggest_updates(inferred_dict: Dict[Tuple[str, str], float], 
                   trace_gt: Set[Tuple[str, str]], 
                   static_gt: Set[Tuple[str, str]],
                   output_path: Path = Path("suggested_additions.json"),
                   min_confidence: float = 0.85):
    """
    Identify high-confidence inferred edges that are missing from both Ground Truth sets.
    Save them to a JSON file for human review.
    
    NOTE: This feature is designed to assist in building the static knowledge base.
    It assumes that high-confidence predictions (score >= 0.85) that are not in the
    current ground truth are likely missing architectural links rather than model errors.
    This assumption holds well for the OpenStack dataset but should be verified manually.
    """
    print(f"\nScanning for potential Knowledge Base additions (Confidence >= {min_confidence})...")
    
    known_edges = trace_gt.union(static_gt)
    suggestions = []
    
    for edge, score in inferred_dict.items():
        if edge not in known_edges and score >= min_confidence:
            suggestions.append({
                "source": edge[0],
                "target": edge[1],
                "score": float(score)
            })
            
    # Sort by score descending
    suggestions.sort(key=lambda x: x["score"], reverse=True)
    
    if suggestions:
        print(f"Found {len(suggestions)} high-confidence candidates.")
        try:
            with open(output_path, "w") as f:
                json.dump(suggestions, f, indent=2)
            print(f"Suggestions saved to {output_path}")
            print("Review these edges and add valid ones to data/knowledge_base/openstack_architecture.json")
        except Exception as e:
            print(f"Failed to save suggestions: {e}")
    else:
        print("No new candidates found meeting the confidence threshold.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MicroVision Pipeline")
    parser.add_argument("--parsed-logs", type=Path, default=Path("data/parsed_sample.csv"), help="Path to parsed logs CSV")
    parser.add_argument("--edges-db", type=Path, default=Path("data/edges/edges.db"), help="Path to inferred edges SQLite DB")
    parser.add_argument("--threshold", type=float, default=0.0, help="Minimum weight threshold for inferred edges")
    parser.add_argument("--suggest-updates", action="store_true", help="Generate suggestions for Knowledge Base additions")
    
    args = parser.parse_args()
    
    # 1. Generate Ground Truth
    try:
        df = load_parsed_logs(args.parsed_logs)
        ground_truth = generate_ground_truth(df)
    except Exception as e:
        print(f"Error generating ground truth: {e}")
        return

    if not ground_truth:
        print("No ground truth edges found. Cannot evaluate.")
        return

    # 2. Load Inferred Edges
    try:
        inferred_dict = load_inferred_edges(args.edges_db, args.threshold)
        inferred_set = set(inferred_dict.keys())
    except Exception as e:
        print(f"Error loading inferred edges: {e}")
        return

    # 3. Calculate Metrics
    metrics = calculate_metrics(ground_truth, inferred_set)
    
    print("\n" + "="*30)
    print("EVALUATION RESULTS (Hybrid GT)")
    print("="*30)
    print(f"Trace GT Edges:     {metrics['trace_gt_count']}")
    print(f"Static GT Edges:    {metrics['static_gt_count']}")
    print(f"Combined GT Edges:  {metrics['combined_gt_count']}")
    print(f"Inferred Edges:     {len(inferred_set)}")
    print(f"Correct Matches:    {metrics['tp']}")
    print("-" * 30)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print("="*30)

    # Optional: Print missed edges for debugging
    if metrics['fn'] > 0:
        print("\nTop 5 Missed Edges (False Negatives):")
        # Re-calculate valid_ground_truth locally for display
        static_gt = get_static_ground_truth()
        valid_ground_truth = ground_truth.union(static_gt)
        missed = list(valid_ground_truth - inferred_set)[:5]
        for edge in missed:
            print(f"  {edge[0]} -> {edge[1]}")

    # Optional: Print inferred edges for debugging
    if len(inferred_set) > 0:
        print("\nTop 5 Inferred Edges (for debugging):")
        for edge in list(inferred_set)[:5]:
            print(f"  {edge[0]} -> {edge[1]}")
            
    # 4. Suggest Updates
    if args.suggest_updates:
        suggest_updates(inferred_dict, ground_truth, get_static_ground_truth())


if __name__ == "__main__":
    main()
