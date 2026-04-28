"""
Sensitivity Analysis Script for MicroVision

This script performs a parameter sweep on the 'hybrid_score' threshold to generate
Precision-Recall curves. This analysis is critical for academic evaluation to demonstrate
the trade-off between sensitivity (finding all edges) and specificity (avoiding false positives).

Methodology:
1. Load the generated edges from `edges.db`.
2. Load the Ground Truth (Hybrid).
3. Iterate threshold T from 0.0 to 1.0 in steps of 0.05.
4. Calculate True Positives (TP), False Positives (FP), False Negatives (FN).
5. Output a CSV/Table suitable for plotting.

Usage:
    python scripts/sensitivity_analysis.py
"""
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pathlib import Path
import sys

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_pipeline import load_gold_standard_csv, generate_ground_truth, load_parsed_logs
from config.settings import Settings

def run_sensitivity_analysis():
    # Paths
    # Convert string path to Path object if necessary, or ensure Settings.DATA_DIR is a Path
    data_dir_path = Path(Settings.DATA_DIR)
    edges_db_path = data_dir_path / "openstack/edges/edges.db"
    gold_standard_path = data_dir_path / "openstack/gold_standard_edges.csv"
    
    print(f"--- MicroVision Sensitivity Analysis ---")
    
    # 1. Load Inferred Edges
    if not edges_db_path.exists():
        print(f"Error: Database not found at {edges_db_path}")
        return

    conn = sqlite3.connect(edges_db_path)
    df_edges = pd.read_sql("SELECT source_service, target_service, hybrid_score FROM edges", conn)
    conn.close()
    
    print(f"Loaded {len(df_edges)} inferred edges from database.")

    # 2. Load Ground Truth
    # We use the hybrid verification logic: Trace GT + Static Knowledge Base
    # For this script's simplicity, we will assume strict comparison against the loaded CSV
    # because recreating the full hybrid logic here would code-duplicate evaluate_pipeline.
    # However, for the best academic rigor, we should use the *exact* same GT set.
    gt_edges = load_gold_standard_csv(gold_standard_path)
    
    # Add Static GT (Simple Manual Injection to match evaluate_pipeline methodology)
    # In a full refactor, this list should be imported from a shared constant.
    static_gt = {
        ('nova-api', 'nova-scheduler'),
        ('nova-scheduler', 'nova-compute'),
        ('nova-compute', 'nova-conductor'),
        ('nova-conductor', 'nova-db'),
        ('nova-api', 'nova-conductor')
    }
    gt_edges.update(static_gt)
    
    print(f"Total Ground Truth Edges for Evaluation: {len(gt_edges)}")
    
    # 3. Parameter Sweep
    results = []
    thresholds = np.arange(0.0, 1.0, 0.05)
    
    print("\nRunning Parameter Sweep...")
    print(f"{'Threshold':<10} | {'Prec':<10} | {'Recall':<10} | {'F1':<10} | {'TP':<5} | {'FP':<5}")
    print("-" * 65)
    
    for t in thresholds:
        # Filter edges by current threshold
        current_edges = df_edges[df_edges['hybrid_score'] >= t]
        
        # Convert to set of tuples for O(1) lookup
        inferred_set = set(zip(current_edges['source_service'], current_edges['target_service']))
        
        # Calculate Metrics
        true_positives = inferred_set.intersection(gt_edges)
        tp_count = len(true_positives)
        fp_count = len(inferred_set) - tp_count
        fn_count = len(gt_edges) - tp_count
        
        prec = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        f1 = (2 * prec * recall) / (prec + recall) if (prec + recall) > 0 else 0.0
        
        # Store
        results.append({
            "threshold": t,
            "precision": prec,
            "recall": recall,
            "f1": f1,
            "tp": tp_count,
            "fp": fp_count
        })
        
        print(f"{t:.2f}       | {prec:.4f}     | {recall:.4f}     | {f1:.4f}     | {tp_count:<5} | {fp_count:<5}")

    # 4. Find Optimal Point
    df_results = pd.DataFrame(results)
    best_f1 = df_results.loc[df_results['f1'].idxmax()]
    
    print("-" * 65)
    print(f"\nOptimal Threshold: {best_f1['threshold']:.2f}")
    print(f"Max F1 Score:      {best_f1['f1']:.4f}")
    
    # Save results
    output_path = data_dir_path / "sensitivity_analysis.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\nFull breakdown saved to: {output_path}")

    # 5. Generate Plot
    print("\\nGenerating Sensitivity Plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(df_results["threshold"], df_results["precision"], label="Precision", marker=".")
    plt.plot(df_results["threshold"], df_results["recall"], label="Recall", marker=".")
    plt.plot(df_results["threshold"], df_results["f1"], label="F1 Score", linewidth=3, color="black")
    
    plt.title("Constraint Relaxation Analysis (Vector Search Sensitivity)")
    plt.xlabel("Hybrid Score Threshold")
    plt.ylabel("Score")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    
    # Save Plot
    docs_dir = ROOT / "docs" / "images"
    docs_dir.mkdir(parents=True, exist_ok=True)
    plot_path = docs_dir / "sensitivity_plot.png"
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")

if __name__ == "__main__":
    run_sensitivity_analysis()
