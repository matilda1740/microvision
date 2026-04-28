# MicroVision User Guide

This guide provides step-by-step instructions for running the MicroVision pipeline, interpreting the results, and visualizing the generated dependency graph.

## 1. Prerequisites

Ensure your environment is set up and the virtual environment is active:
```bash
source venv/bin/activate
```

## 2. Running the Pipeline

The core pipeline is managed by `scripts/run_full_pipeline.py`.

### A. Full Execution (Recommended)
This runs the end-to-end process: Parsing -> Enrichment -> Embedding -> Retrieval.

```bash
python scripts/run_full_pipeline.py \
    --source data/OpenStack_full.log \
    --sample 207636 \
    --validate \
    --clear-db \
    --format-name OpenStack
```

**Parameters:**
*   `--source`: Path to the raw log file.
*   `--sample`: Number of lines to process. Use `207636` for the full dataset.
*   `--validate`: Runs the Cross-Encoder validation step.
*   `--clear-db`: Resets the ChromaDB vector store before running.

## 3. Visualizing Results

The project includes an interactive Streamlit application for exploring the dependency graph.

### Launch the Visualizer
```bash
streamlit run apps/streamlit_visualize.py
```

### Interpreting the Graph
The graph displays services as nodes and dependencies as edges. The edges are color-coded based on validation against Ground Truth:

*   **<span style="color:green">Green Edges</span> (True Positive)**:
    *   The model inferred this edge, AND it was confirmed by the Ground Truth (Trace ID).
    *   *Example*: `nova-api` -> `nova-compute`.
*   **<span style="color:grey">Grey Edges</span> (Inferred)**:
    *   The model inferred this edge based on semantic similarity, but it was NOT found in the trace data.
    *   *Validation*: These may be valid dependencies that simply didn't generate a shared Request ID (e.g., subtle state changes).
*   **<span style="color:red">Red Edges</span> (False Positive)**:
    *   The model inferred this edge, but it contradicts the known Static Architecture (e.g., `nova-compute` talking to a frontend UI directly).

## 4. Evaluation

To generate precision/recall metrics:

```bash
python scripts/evaluate_pipeline.py --gold-standard data/openstack/gold_standard_edges.csv
```

**Output:**
*   **Precision**: % of inferred edges that are correct.
*   **Recall**: % of expected edges that were found.
*   **F1 Score**: Harmonic mean of Precision and Recall.

## 5. Artifact Locations

*   **Inferred Edges**: `data/openstack/edges/edges.db` (SQLite)
*   **Vector Store**: `data/openstack/chroma_db/`
*   **Ground Truth**: `data/openstack/gold_standard_edges.csv`
*   **Run History**: `data/run_manifests/`
