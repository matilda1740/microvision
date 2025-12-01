# MicroVision - Microservice Dependency Mapping

A semantic log analysis system for automated microservice dependency discovery.

## Quick Start

1. Activate environment: `source venv/bin/activate`
2. Run the full pipeline:
   ```bash
   # For random sampling (exploration):
   python scripts/run_full_pipeline.py --source data/OpenStack_full.log --sample 2000 --validate --clear-db --format-name OpenStack
   
   # For contiguous sampling (better for trace reconstruction/evaluation):
   python scripts/run_full_pipeline.py --source data/OpenStack_full.log --sample 50000 --contiguous --validate --clear-db --format-name OpenStack
   ```
3. Evaluate results:
   ```bash
   python scripts/evaluate_pipeline.py
   ```


## Project Structure

The codebase is organized into modular components to separate concerns:

*   **`apps/`**: Streamlit web applications for visualization and administration.
*   **`config/`**: Centralized configuration settings (`settings.py`) and logging setup.
*   **`data/`**: Directory for raw logs, intermediate CSVs, and ChromaDB storage (git-ignored).
*   **`docs/`**: Documentation and architectural diagrams.
*   **`scripts/`**: Executable scripts for running pipelines, evaluation, and maintenance.
    *   `run_full_pipeline.py`: Main entry point for the end-to-end pipeline.
    *   `evaluate_pipeline.py`: Script for calculating precision/recall against ground truth.
*   **`src/`**: Core source code.
    *   **`encode_chroma/`**: Logic for generating embeddings and interacting with ChromaDB.
    *   **`enrichment/`**: Modules for cleaning templates and merging structured metadata.
    *   **`parsing/`**: Log parsing logic using `Drain3` (`MetadataDrainParser`).
    *   **`pipeline/`**: Orchestration logic (`stages.py`) defining the data flow steps.
    *   **`retrieval.py`**: Semantic retrieval logic for finding candidate edges.
    *   **`storage.py`**: SQLite interface (`EdgeStore`) for persisting the dependency graph.
    *   **`transitions.py`**: Logic for computing and storing state transitions.
    *   **`utils/`**: Shared utilities, including centralized timestamp handling (`time_utils.py`).
    *   **`validation.py`**: Cross-encoder validation logic.
    *   **`visualization.py`**: Graph generation using NetworkX and PyVis.
*   **`tests/`**: Unit and integration tests.

## Documentation

*   [Architecture & Results](docs/ARCHITECTURE_AND_RESULTS.md): Detailed overview of the data flow, challenges, and current evaluation metrics.

