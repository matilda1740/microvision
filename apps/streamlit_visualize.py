"""Streamlit app to run the MicroVision pipeline and visualize the edges DB.

This page contains two main sections on the same page:
- ⚙️ Pipeline Execution: run the full pipeline and display stage-level
  progress using st.spinner and success messages.
- Graph Visualizer: generate and embed a pyvis HTML from the edges DB.

The visualizer does not currently depend on the uploader/consumer; the
Run button will execute the existing `scripts/run_full_pipeline.py` and
watch its stdout to update the UI.
"""
from __future__ import annotations

import sys
import tempfile
import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import json
import streamlit as st
import streamlit.components.v1 as components

# Ensure top-level repo packages are importable when Streamlit runs the app.
# Prefer proper packaging (pip install -e .) but fall back to adding the
# repository root to sys.path if imports fail at runtime.
try:
    from scripts.visualize_graph import load_edges_from_db
    from src.visualization import edges_to_networkx, write_pyvis_html
    from src.storage import EdgeStore
    from src.retrieval import compute_candidate_edges_stream
    from src.utils.atomic_write import atomic_save_npy, atomic_write_json
    from scripts.evaluate_pipeline import load_parsed_logs, generate_ground_truth, load_inferred_edges, calculate_metrics
    from src.pipeline.stages import (
        sample_source,
        parse_sample_rows,
        load_parsed_df,
        preprocess_and_merge,
        build_metadatas_from_aggregated,
        compute_embeddings_for_use_df,
        ingest_embeddings_atomic,
        compute_and_persist_edges,
    )
except Exception:
    import pathlib as _pathlib

    repo_root = _pathlib.Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    # Retry imports
    from scripts.visualize_graph import load_edges_from_db
    from src.visualization import edges_to_networkx, write_pyvis_html
    from src.storage import EdgeStore
    from src.retrieval import compute_candidate_edges_stream
    from src.utils.atomic_write import atomic_save_npy, atomic_write_json
    from scripts.evaluate_pipeline import load_parsed_logs, generate_ground_truth, load_inferred_edges, calculate_metrics
    from src.pipeline.stages import (
        sample_source,
        parse_sample_rows,
        load_parsed_df,
        preprocess_and_merge,
        build_metadatas_from_aggregated,
        compute_embeddings_for_use_df,
        ingest_embeddings_atomic,
        compute_and_persist_edges,
    )


st.title("Microvision: Pipeline & Graph")

# Sidebar Configuration
st.sidebar.header("Pipeline Settings")

# Source Selection (Moved to Sidebar for dynamic defaults)
source_options = {
    "Sampling Mode": "data/sample_raw.csv",
    "Full Dataset": "data/OpenStack_full.log"
}

# Initialize session state for source tracking
if "last_source" not in st.session_state:
    st.session_state.last_source = "Sampling Mode"

selected_source_label = st.sidebar.selectbox("Log source", list(source_options.keys()), index=0)
source = source_options[selected_source_label]

# Update defaults if source changed
if st.session_state.last_source != selected_source_label:
    st.session_state.last_source = selected_source_label
    if selected_source_label == "Full Dataset":
        st.session_state.sample = 200000
        st.session_state.threshold = 0.40
        st.session_state.validate = True
    else:
        st.session_state.sample = 1000
        st.session_state.threshold = 0.20
        st.session_state.validate = False

# Ensure keys exist if they weren't set by the update block (e.g. first run)
if "sample" not in st.session_state: st.session_state.sample = 1000
if "top_k" not in st.session_state: st.session_state.top_k = 10
if "threshold" not in st.session_state: st.session_state.threshold = 0.2
if "alpha" not in st.session_state: st.session_state.alpha = 0.5
if "validate" not in st.session_state: st.session_state.validate = False
if "resume" not in st.session_state: st.session_state.resume = True

sample = st.sidebar.number_input("Sample size", min_value=1, step=1, key="sample")
top_k = st.sidebar.number_input("Top K", min_value=1, step=1, key="top_k")
threshold = st.sidebar.number_input("Threshold", min_value=0.0, max_value=1.0, format="%.2f", key="threshold")
alpha = st.sidebar.number_input("Alpha (hybrid)", min_value=0.0, max_value=1.0, format="%.2f", key="alpha")
validate = st.sidebar.checkbox("Enable Cross-Encoder Validation (RAG)", help="Re-rank edges using a Cross-Encoder model. Slower but more accurate.", key="validate")
resume = st.sidebar.checkbox("Smart Resume", help="Skip sampling, parsing, and preprocessing if data/parsed_sample.csv and data/merged_templates.csv exist.", key="resume")

st.subheader("⚙️ Pipeline Execution")

# Prefer centralized defaults from config.settings when available
db_default = getattr(__import__("config.settings", fromlist=["settings"]).settings, "DEFAULT_DB", "data/edges/edges_smoke.db")
chroma_default = getattr(__import__("config.settings", fromlist=["settings"]).settings, "DEFAULT_CHROMA_DIR", "data/chroma_db/chroma_smoke")
# Hidden configuration
db_path = db_default
chroma_dir = chroma_default
device = None


def _run_pipeline_direct(
    source: str,
    sample: int,
    db_path: str,
    chroma_dir: str,
    top_k: int,
    threshold: float,
    alpha: float,
    device: str | None = None,
    validate: bool = False,
    resume: bool = False,
) -> None:
    """Run pipeline stages directly using the refactored functions in
    `scripts.run_full_pipeline.py` and update Streamlit UI spinners.

    This removes the previous subprocess/stdout parsing approach and calls
    the stage functions directly, which is more robust and testable.
    """

    from src.pipeline.stages import (
        sample_source,
        parse_sample_rows,
        load_parsed_df,
        preprocess_and_merge,
        build_metadatas_from_aggregated,
        compute_embeddings_for_use_df,
        ingest_embeddings_atomic,
        compute_and_persist_edges,
    )
    from src.transitions import compute_and_persist_transitions
    from src.parsing.metadata_drain_parser import MetadataDrainParser

    parsing_ph = st.empty()
    preproc_ph = st.empty()
    embed_ph = st.empty()
    retrieval_ph = st.empty()
    raw_log_expander = st.expander("Show pipeline logs")
    log_lines: list[str] = []

    def _append_log(msg: str) -> None:
        nonlocal log_lines
        log_lines.append(msg)
        if len(log_lines) > 500:
            log_lines = log_lines[-500:]
        # display latest logs in the expander as plain text
        with raw_log_expander:
            for ln in log_lines[-200:]:
                st.text(ln)

    try:
        # 0) Clear DB to prevent mixing runs
        # Only clear if NOT resuming, OR if we want to refresh edges even when resuming.
        # Usually if we resume parsing, we still want to regenerate edges based on new params (threshold etc).
        # So clearing DB is correct.
        store = EdgeStore(db_path)
        store.clear_edges()
        store.close()
        _append_log(f"Cleared edges DB: {db_path}")

        # 1) Sampling + Parsing
        parsed_path = Path("data/parsed_sample.csv")
        templates_path = Path("data/parsed_templates.csv")
        
        if resume and parsed_path.exists():
            st.info(f"Resuming: Found {parsed_path}, skipping parsing.")
            _append_log(f"Resuming from {parsed_path}")
            parsed_df = load_parsed_df(parsed_path)
            parsing_ph.success(f"Loaded {len(parsed_df)} parsed log entries from disk.")
        else:
            with st.spinner(f"Sampling {sample} rows from {source}..."):
                _append_log(f"Sampling source={source} sample={sample}")
                sampled = sample_source(Path(source), int(sample))
                _append_log(f"Sampled {len(sampled)} rows")

            # Parsing with Progress Bar
            parse_msg = st.empty()
            parse_bar = st.progress(0)
            parse_msg.text("Parsing logs...")
            
            save_every = max(1, int(sample) // 4)
            parser = MetadataDrainParser(structured_csv=str(parsed_path), templates_csv=str(templates_path), save_every=save_every, mode="fresh")
            
            total_rows = len(sampled)
            for i, row in enumerate(sampled):
                raw_line = row.get("raw") or row.get("content") or " ".join([str(v) for v in row.values()])
                parser.process_line(raw_line, i + 1)
                
                if total_rows > 0 and (i % 100 == 0 or i == total_rows - 1):
                    pct = int((i + 1) / total_rows * 100)
                    parse_bar.progress(pct, text=f"Parsing: {pct}% ({i+1}/{total_rows})")
            
            parser.finalize()
            parse_bar.empty()
            parse_msg.empty()
            
            parsed_df = load_parsed_df(parsed_path)
            parsing_ph.success(f"Parsed {len(parsed_df)} log entries.")

        # 2) Preprocessing: cleaning/merge/metadatas
        cleaned_path = Path("data/cleaned_templates.csv")
        merged_out = Path("data/merged_templates.csv")
        
        if resume and merged_out.exists() and parsed_path.exists():
            st.info(f"Resuming: Found {merged_out}, skipping preprocessing.")
            aggregated = pd.read_csv(merged_out)
            _append_log(f"Loaded {len(aggregated)} aggregated rows from disk.")
            metas = build_metadatas_from_aggregated(aggregated)
            preproc_ph.success(f"Loaded {len(aggregated)} aggregated templates from disk.")
        else:
            with st.spinner("Preprocessing Parsed logs..."):
                cleaned, aggregated = preprocess_and_merge(parsed_df, cleaned_path, merged_out)
                _append_log(f"Cleaned={len(cleaned)} merged={len(aggregated)}")
                _append_log(f"Aggregated columns: {list(aggregated.columns)}")
                metas = build_metadatas_from_aggregated(aggregated)
                preproc_ph.success(f"Preprocessed templates: {len(aggregated)} rows")

        # 3) Embedding
        with embed_ph.container():
            model_name = getattr(__import__("config.settings", fromlist=["settings"]).settings, "EMBEDDING_MODEL", "all-mpnet-base-v2")
            if device is None or device == "":
                import os as _os

                device = "cuda" if (_os.environ.get("CUDA_VISIBLE_DEVICES") or _os.path.exists("/usr/local/cuda")) else "cpu"

            # Attempt to load embeddings from disk cache (validate metadata)
            emb_default = getattr(__import__("config.settings", fromlist=["settings"]).settings, "DEFAULT_EMBEDDINGS_DIR", "data/edges")
            out_dir = Path(emb_default)
            emb_path = out_dir / "embeddings.npy"
            meta_path = out_dir / "embeddings.meta.json"
            docs = aggregated["semantic_text"].fillna("").astype(str).tolist()

            def _compute_hash(texts: list[str]) -> str:
                import hashlib as _hashlib

                h = _hashlib.sha256()
                for t in texts:
                    h.update(t.encode("utf-8", errors="surrogatepass"))
                    h.update(b"\n")
                return h.hexdigest()

            cached_embeddings = None
            try:
                if emb_path.exists() and meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if int(meta.get("count", -1)) == len(docs) and meta.get("hash") == _compute_hash(docs):
                        cached_embeddings = np.load(emb_path)
            except Exception:
                cached_embeddings = None

            if cached_embeddings is not None and len(cached_embeddings) == len(docs):
                embeddings = cached_embeddings
                documents = docs
                _append_log(f"Loaded cached embeddings ({len(documents)} docs)")
            else:
                # Load model as a cached resource
                @st.cache_resource
                def get_model(name: str, device_str: str):
                    try:
                        from sentence_transformers import SentenceTransformer

                        return SentenceTransformer(name, device=device_str)
                    except Exception as e:
                        raise RuntimeError(f"sentence-transformers failed to import: {e}") from e

                model = get_model(model_name, device)

                # encode in batches and show progress
                batch_size = 64
                n = len(docs)
                progress = st.progress(0)
                encs = []
                for i in range(0, n, batch_size):
                    batch = docs[i : i + batch_size]
                    arr = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
                    encs.append(arr)
                    progress.progress(min(100, int((i + len(batch)) / max(1, n) * 100)))
                if encs:
                    embeddings = np.vstack(encs)
                else:
                    embeddings = np.zeros((0, 0))

                # save cache atomically
                try:
                    atomic_save_npy(emb_path, embeddings)
                    atomic_write_json(meta_path, {"count": len(docs), "hash": _compute_hash(docs)})
                except Exception:
                    pass

                documents = docs
                _append_log(f"Computed embeddings for {len(documents)} documents")

            # 4) Ingest into Chroma (optional)
            ids = [str(i) for i in range(len(documents))]
            client = None
            collection = None
            if chroma_dir:
                try:
                    res = ingest_embeddings_atomic(Path(chroma_dir), ids, metas, documents, embeddings)
                    # The wrapper may return (client, collection, reused) or (client, collection)
                    if isinstance(res, tuple) and len(res) == 3:
                        client, collection, reused = res
                    elif isinstance(res, tuple):
                        client, collection = res
                        reused = False
                    else:
                        client = res
                        collection = None
                        reused = False

                    if reused:
                        _append_log(f"Reused cached Chroma client for {chroma_dir}")

                    _append_log(f"Ingested {len(ids)} vectors into Chroma at {chroma_dir}")
                except Exception as e:
                    _append_log(f"Chroma ingest failed: {e}")

            embed_ph.success(f"Embedded {len(documents)} documents.")

        # 5) Retrieval & persist
        with retrieval_ph.container():
            # Stream retrieval generator and write edges in batches while updating a progress bar
            N = len(aggregated)
            progress = st.progress(0)
            store = EdgeStore(db_path)
            store.init_db()
            # Ensure retriever knows which id columns to consider (include `template_id` from parser)
            id_column_candidates = ("template_id", "reqid", "template", "pid")
            gen = compute_candidate_edges_stream(
                aggregated, 
                embeddings=embeddings, 
                collection=collection, 
                top_k=int(top_k), 
                threshold=float(threshold), 
                batch_size=128, 
                alpha=float(alpha),
                id_column_candidates=id_column_candidates
            )
            
            if validate:
                with st.spinner("Validating edges with Cross-Encoder..."):
                    try:
                        from src.validation import SemanticValidator
                        # Cache the validator to avoid reloading model on every run
                        @st.cache_resource
                        def get_validator():
                            return SemanticValidator()
                        
                        validator = get_validator()
                        # Validation requires consuming the generator to batch process
                        candidates = list(gen)
                        _append_log(f"Validating {len(candidates)} candidate edges...")
                        validated = validator.validate_edges(candidates)
                        gen = iter(validated)
                        _append_log("Validation complete.")
                    except Exception as e:
                        _append_log(f"Validation failed: {e}")
                        st.warning(f"Validation failed: {e}. Proceeding with unvalidated edges.")

            batch = []
            processed_sources = set()
            written = 0
            for e in gen:
                batch.append(e)
                processed_sources.add(int(e.get("source_index", -1)) if e.get("source_index") is not None else -1)
                if len(batch) >= 250:
                    written += store.write_edges(iter(batch), batch_size=500)
                    batch = []
                # update progress by unique source indices seen
                prog = min(1.0, len([s for s in processed_sources if s >= 0]) / max(1, N))
                progress.progress(int(prog * 100))
            if batch:
                written += store.write_edges(iter(batch), batch_size=500)
            _append_log(f"Edges written: {written}")
            transitions_written = compute_and_persist_transitions(db_path, min_count=1)
            _append_log(f"Transitions written: {transitions_written}")
            retrieval_ph.success(f"Edges written: {written}")

        # 6) Manifest
        try:
            run_id = datetime.datetime.utcnow().strftime("run_%Y%m%dT%H%M%SZ")
            manifest_dir = Path("data/run_manifests")
            manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = manifest_dir / f"{run_id}.json"
            manifest = {"run_id": run_id, "started_at": datetime.datetime.utcnow().isoformat() + "Z", "source": source, "sample": int(sample), "events": [], "metrics": {}}
            manifest["finished_at"] = datetime.datetime.utcnow().isoformat() + "Z"
            manifest["edges_db"] = str(db_path)
            from src.utils.atomic_write import atomic_write_json

            atomic_write_json(manifest_path, manifest)
            _append_log(f"Wrote manifest: {manifest_path}")
        except Exception as _:
            pass

    except Exception as e:
        # Surface the error in the UI and return failure so the caller
        # doesn't incorrectly show a success message.
        st.error(f"Pipeline failed: {e}")
        _append_log(f"ERROR: {e}")
        return False

    # Write manifest and finish
    try:
        run_id = datetime.datetime.utcnow().strftime("run_%Y%m%dT%H%M%SZ")
        manifest_dir = Path("data/run_manifests")
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{run_id}.json"
        manifest = {"run_id": run_id, "started_at": datetime.datetime.utcnow().isoformat() + "Z", "source": source, "sample": int(sample), "events": [], "metrics": {}}
        manifest["finished_at"] = datetime.datetime.utcnow().isoformat() + "Z"
        manifest["edges_db"] = str(db_path)
        from src.utils.atomic_write import atomic_write_json

        atomic_write_json(manifest_path, manifest)
        _append_log(f"Wrote manifest: {manifest_path}")
    except Exception:
        pass

    return True


# Visualization settings
# (Moved to Dependency Graph tab)



def render_graph(edges_list, height=600):
    """Render the PyVis graph component."""
    G = edges_to_networkx(edges_list)
    if G is not None:
        tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
        tmp.close()
        try:
            out = write_pyvis_html(G, tmp.name)
            html = Path(tmp.name).read_text(encoding="utf-8")
            # Diagnostic: show generated HTML header comment
            if "<!-- graph-renderer:" in html:
                hdr = html.split("<!-- graph-renderer:", 1)[1].split("-->", 1)[0]
                # st.text(f"HTML banner:{hdr.strip()}") # Optional debug
            components.html(html, height=height)
        except Exception as e:
            st.error(
                "Could not render visualization with pyvis: %s. "
                "Install optional deps in the venv: `pip install pyvis jinja2`" % e
            )

def render_inspector(edges_list):
    """Render the Edge Inspector dataframe and reasoning."""
    st.write(f"Loaded {len(edges_list)} edges from DB")
    
    if len(edges_list) > 0:
        # Convert to DataFrame for easier viewing
        df_display = pd.DataFrame(edges_list)
        
        # Select relevant columns if they exist
        cols = ["source", "target", "hybrid_score", "time_delta_ms", "source_semantic_text", "target_semantic_text"]
        
        available_cols = [c for c in cols if c in df_display.columns]
        if not available_cols:
            # Fallback for aggregated view
            available_cols = [c for c in ["source", "target", "weight", "hybrid_score", "title"] if c in df_display.columns]
        
        st.dataframe(df_display[available_cols], use_container_width=True)
        
        # If we have raw text (template level), show a few examples of reasoning
        if "source_semantic_text" in df_display.columns:
            st.markdown("### Sample Reasoning (Top 5 Edges)")
            for i, row in df_display.head(5).iterrows():
                with st.expander(f"{row.get('source', '?')} -> {row.get('target', '?')} (Score: {row.get('hybrid_score', 0):.2f})"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**Source Template:**")
                        st.info(row.get("source_semantic_text", "N/A"))
                    with col_b:
                        st.markdown("**Target Template:**")
                        st.success(row.get("target_semantic_text", "N/A"))
                    st.caption(f"Time Delta: {row.get('time_delta_ms', 'N/A')} ms")


# Initialize session state for graph persistence
if "show_graph" not in st.session_state:
    st.session_state.show_graph = False

if st.button("Run MicroVision", type="primary"):
    st.info("Starting full pipeline (direct mode). Logs will appear in 'Show pipeline logs'.")
    ok = _run_pipeline_direct(
        source=source, 
        sample=int(sample), 
        db_path=db_path, 
        chroma_dir=chroma_dir, 
        top_k=int(top_k), 
        threshold=float(threshold), 
        alpha=float(alpha), 
        device=device if device else None,
        validate=validate,
        resume=resume
    )
    
    if ok:
        st.success("Full pipeline run completed.")
        st.session_state.show_graph = True
    else:
        st.error("Pipeline failed — see logs above for details.")

st.markdown("---")
tab_graph, tab_inspector, tab_eval = st.tabs(["Dependency Graph", "Edge Inspector", "Evaluation"])

# Shared Data Holder
edges_list = []

with tab_graph:
    st.subheader("Visualization Settings")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        limit = st.number_input("Max edges", value=1000, min_value=10, step=10, key="viz_limit")
    with c2:
        viz_threshold = st.number_input("Min Score", value=0.3, min_value=0.0, max_value=1.0, step=0.05, key="viz_threshold")
    with c3:
        viz_level = st.selectbox("Level", ["service", "template"], index=0, key="viz_level")
    with c4:
        graph_height = st.number_input("Height (px)", value=600, min_value=400, max_value=1200, step=50, key="viz_height")
    
    if Path(db_path).exists():
        if st.button("Refresh Graph"):
            st.session_state.show_graph = True
    
    if st.session_state.show_graph and Path(db_path).exists():
        try:
            with st.spinner("Rendering graph..."):
                edges = load_edges_from_db(str(db_path), limit=int(limit), threshold=float(viz_threshold), level=viz_level)
                edges_list = list(edges)
                render_graph(edges_list, height=graph_height)
        except Exception as e:
            st.error(f"Failed loading edges: {e}")
    elif not Path(db_path).exists():
        st.info("Run the pipeline to generate the graph.")

with tab_inspector:
    if st.session_state.show_graph and edges_list:
        render_inspector(edges_list)
    elif st.session_state.show_graph and not edges_list and Path(db_path).exists():
         st.warning("No edges found matching the current criteria.")
    else:
        st.info("No edges loaded. Run pipeline or refresh graph.")

with tab_eval:
    st.subheader("Evaluation (Ground Truth vs Inferred)")
    if st.button("Run Evaluation"):
        st.info("Running evaluation...")
        try:
            # Use the same parsed logs path as the pipeline uses
            parsed_logs_path = Path("data/parsed_sample.csv")
            if not parsed_logs_path.exists():
                 st.error(f"Parsed logs not found at {parsed_logs_path}. Run the pipeline first.")
            else:
                 # 1. Generate Ground Truth
                 df = load_parsed_logs(parsed_logs_path)
                 ground_truth = generate_ground_truth(df)
                 
                 if not ground_truth:
                     st.warning("No ground truth edges found in the current sample (requires multiple services in same reqid trace).")
                 else:
                     st.write(f"**Ground Truth Edges:** {len(ground_truth)}")
                     
                     # 2. Load Inferred Edges
                     # Use the visualization threshold to filter inferred edges
                     inferred = load_inferred_edges(Path(db_path), threshold=float(viz_threshold))
                     st.write(f"**Inferred Edges** (threshold={viz_threshold}): {len(inferred)}")
                     
                     # 3. Calculate Metrics
                     metrics = calculate_metrics(ground_truth, inferred)
                     
                     c1, c2, c3 = st.columns(3)
                     c1.metric("Precision", f"{metrics['precision']:.4f}")
                     c2.metric("Recall", f"{metrics['recall']:.4f}")
                     c3.metric("F1 Score", f"{metrics['f1']:.4f}")
                     
                     with st.expander("Detailed Metrics"):
                         st.write(f"Correct Matches (TP): {metrics['tp']}")
                         st.write(f"False Positives (FP): {metrics['fp']}")
                         st.write(f"Missed Edges (FN): {metrics['fn']}")
                         
                         if metrics['fn'] > 0:
                             st.markdown("#### Top Missed Edges (False Negatives)")
                             missed = list(ground_truth - inferred)[:10]
                             for src, dst in missed:
                                 st.text(f"{src} -> {dst}")

        except Exception as e:
            st.error(f"Evaluation failed: {e}")
