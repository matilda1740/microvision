# Semantic Optimization Strategy: Dynamic Context Injection

## Overview
This document details the "Dynamic Context Injection" strategy implemented to optimize semantic retrieval in the MicroVision pipeline. This strategy addresses the "Vocabulary Mismatch" problem where logs from dependent services share few common words, leading to missed connections (False Negatives).

## The Problem
Standard vector search relies on semantic similarity. However, a source log (e.g., `Connection refused`) and a target log (e.g., `Server shutting down`) may have high semantic distance despite being causally linked. Furthermore, generic error messages (e.g., `NullPointerException`) appear in all services, causing False Positives.

## The Solution: Dynamic Context Injection
Instead of embedding the raw log template, we inject a "Context Header" into the text before embedding. This header contains:
1.  **Service Identity**: Explicitly states which service generated the log.
2.  **Dynamic Keywords**: Automatically extracts the top 5 most distinctive words for that service to act as semantic "hooks".

### Implementation Logic
The logic is implemented in `src/pipeline/stages.py` within the `compute_embeddings_for_use_df` function.

1.  **Grouping**: Logs are grouped by `service`.
2.  **Keyword Extraction (TF-IDF-like)**:
    *   Aggregates all log text for a service.
    *   Tokenizes and removes stopwords (e.g., `info`, `error`, `pid`).
    *   Calculates the top 5 most frequent words.
3.  **Injection**:
    *   **Format**: `[<Service Name> | <Keyword 1> <Keyword 2> ...] <Original Log Message>`
    *   **Example**: `[ts-ui-dashboard | nginx emerg host upstream] nginx: emerg host not found in upstream...`

## Benefits
1.  **Dynamic**: Adapts automatically to changes in log vocabulary (no hardcoded dictionaries).
2.  **Context-Aware**: Pushes generic errors into service-specific vector clusters, reducing cross-service noise.
3.  **Zero-Config**: Requires no manual rule creation.

## Status
*   **Implemented**: Yes (`src/pipeline/stages.py`)
*   **Active**: Yes (Default behavior in pipeline)
*   **Evaluation**: Validated on `OpenStack` dataset. Successfully injects relevant keywords (e.g., `nova-api`, `cinder-api` for services; `rabbitmq` for messaging).

---

# Semantic Optimization Strategy: Hybrid Search (Dense + Sparse)

## Overview
This section details the "Hybrid Search" strategy, which combines Dense Vector Retrieval (ChromaDB) with Sparse Keyword Retrieval (TF-IDF/BM25). This approach ensures that the pipeline captures both semantic meaning (Dense) and exact keyword matches (Sparse).

## The Problem
Dense vector models (like `all-mpnet-base-v2`) are excellent at capturing conceptual similarity but can sometimes miss exact keyword matches, especially for technical terms or specific error codes (e.g., `Error 10054`). Conversely, keyword search misses synonyms but is precise for exact matches. Relying on only one method leads to either low recall or low precision.

## The Solution: Hybrid Retrieval
We implement a dual-path retrieval system in `src/retrieval.py`:

1.  **Dense Path**: Uses ChromaDB to find candidates based on vector cosine similarity.
2.  **Sparse Path**: Uses `scikit-learn`'s `TfidfVectorizer` to find candidates based on keyword overlap.
3.  **Fusion**: Candidates from both paths are merged, and a weighted hybrid score is computed.

### Implementation Logic
The logic is implemented in `src/retrieval.py` within `compute_candidate_edges_stream`.

1.  **Sparse Indexing**:
    *   A `TfidfVectorizer` is fitted on the target log templates.
    *   Stopwords are removed to focus on technical terms.
2.  **Dual Retrieval**:
    *   **Dense**: Queries ChromaDB for the top $K$ semantic matches.
    *   **Sparse**: Computes Cosine Similarity on TF-IDF vectors for the top $K$ keyword matches.
3.  **Score Fusion**:
    *   The final score is calculated as:
        $$ Score_{hybrid} = \alpha \cdot Score_{dense} + (1 - \alpha) \cdot Score_{sparse} $$
    *   Where $\alpha$ (alpha) is a configurable parameter (default 0.5).

## Benefits
1.  **Robustness**: Catches connections that are either semantically related OR share specific technical keywords.
2.  **Precision**: Sparse search acts as a filter for "hallucinated" semantic matches that share no actual terms.
3.  **Zero-Latency Overhead**: TF-IDF is extremely fast and runs in parallel with the vector search.

## Status
*   **Implemented**: Yes (`src/retrieval.py`)
*   **Active**: Yes (Default behavior in pipeline)
*   **Evaluation**: Validated on `OpenStack` dataset. The system successfully retrieves candidates using both methods and fuses the scores.
