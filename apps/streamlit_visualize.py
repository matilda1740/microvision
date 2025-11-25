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

import re
import sys
import subprocess
import tempfile
import time
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
    from src.visualization.graph import edges_to_networkx, write_pyvis_html
    from src.storage.edge_store import EdgeStore
    from src.retrieval.retriever import compute_candidate_edges_stream
    from src.utils.atomic_write import atomic_save_npy, atomic_write_json
except Exception:
    import pathlib as _pathlib

    repo_root = _pathlib.Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    # Retry imports
    from scripts.visualize_graph import load_edges_from_db
    from src.visualization.graph import edges_to_networkx, write_pyvis_html
    from src.storage.edge_store import EdgeStore
    from src.retrieval.retriever import compute_candidate_edges_stream
    from src.utils.atomic_write import atomic_save_npy, atomic_write_json


st.title("Microvision: Pipeline & Graph")

st.subheader("⚙️ Pipeline Execution")

# Pipeline inputs (same flags as scripts/run_full_pipeline.py)
source = st.text_input("Log source (file or directory)", value="data/sample_raw.csv")
sample = st.number_input("Sample size", value=1000, min_value=1, step=1)
db_path = st.text_input("Edges DB path", value="data/edges_full_pipeline.db")
chroma_dir = st.text_input("Chroma persist dir", value="data/datasets/embeddings/chroma_smoke")
top_k = st.number_input("Top K", value=10, min_value=1, step=1)
threshold = st.number_input("Threshold", value=0.2, min_value=0.0, max_value=1.0, format="%.2f")
alpha = st.number_input("Alpha (hybrid)", value=0.5, min_value=0.0, max_value=1.0, format="%.2f")
device = st.text_input("Device (leave blank for auto)", value="")


def _run_pipeline_direct(
    source: str,
    sample: int,
    db_path: str,
    chroma_dir: str,
    top_k: int,
    threshold: float,
    alpha: float,
    device: str | None = None,
) -> None:
    """Run pipeline stages directly using the refactored functions in
    `scripts.run_full_pipeline.py` and update Streamlit UI spinners.

    This removes the previous subprocess/stdout parsing approach and calls
    the stage functions directly, which is more robust and testable.
    """

    from scripts.run_full_pipeline import (
        sample_source,
        parse_sample_rows,
        load_parsed_df,
        preprocess_and_merge,
        build_metadatas_from_aggregated,
        compute_embeddings_for_use_df,
        ingest_embeddings_atomic,
        compute_and_persist_edges,
    )
    from src.persistence.transitions import compute_and_persist_transitions

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
        # 1) Sampling + Parsing
        with st.spinner("Parsing logs..."):
            _append_log(f"Sampling source={source} sample={sample}")
            sampled = sample_source(Path(source), int(sample))
            _append_log(f"Sampled {len(sampled)} rows")

            parsed_path = Path("data/parsed_sample.csv")
            templates_path = Path("data/parsed_templates.csv")
            save_every = max(1, int(sample) // 4)
            parse_sample_rows(sampled, parsed_path, templates_path, log_format=None, save_every=save_every)
            parsed_df = load_parsed_df(parsed_path)
            parsing_ph.success(f"Parsed {len(parsed_df)} log entries.")

        # 2) Preprocessing: cleaning/merge/metadatas
        with st.spinner("Preprocessing Parsed logs..."):
            cleaned_path = Path("data/cleaned_templates.csv")
            merged_out = Path("data/merged_templates.csv")
            cleaned, aggregated = preprocess_and_merge(parsed_df, cleaned_path, merged_out)
            _append_log(f"Cleaned={len(cleaned)} merged={len(aggregated)}")
            metas = build_metadatas_from_aggregated(aggregated)
            preproc_ph.success(f"Preprocessed templates: {len(aggregated)} rows")

        # 3) Embedding
        with embed_ph.container():
            model_name = getattr(__import__("config.settings", fromlist=["settings"]).settings, "EMBEDDING_MODEL", "all-mpnet-base-v2")
            if device is None or device == "":
                import os as _os

                device = "cuda" if (_os.environ.get("CUDA_VISIBLE_DEVICES") or _os.path.exists("/usr/local/cuda")) else "cpu"

            # Attempt to load embeddings from disk cache (validate metadata)
            out_dir = Path("data/datasets/embeddings")
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
                        raise RuntimeError("sentence-transformers required") from e

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
            gen = compute_candidate_edges_stream(aggregated, embeddings=embeddings, collection=collection, top_k=int(top_k), threshold=float(threshold), batch_size=128, alpha=float(alpha))
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


if st.button("Run MicroVision"):
    st.info("Starting full pipeline (direct mode). Logs will appear in 'Show pipeline logs'.")
    ok = _run_pipeline_direct(source=source, sample=int(sample), db_path=db_path, chroma_dir=chroma_dir, top_k=int(top_k), threshold=float(threshold), alpha=float(alpha), device=device if device else None)
    if ok:
        # On success, prompt user to view results in the Graph Visualizer and
        # provide a quick action that pre-fills the visualization controls.
        st.success("Full pipeline run completed (see logs above for details).")
        # Previously we offered a quick "Open Graph Visualizer" jump here.
        # That shortcut has been removed to simplify the UI and avoid
        # unexpected auto-generation in the user's session.
    else:
        st.error("Pipeline failed — see logs above for details.")


st.markdown("---")

st.subheader("Graph Visualizer")

# Visualization controls
viz_db_path = st.text_input("Edges DB for visualization", value=db_path, key="viz_db")
limit = st.number_input("Max edges to load", value=1000, min_value=10, step=10, key="viz_limit")

if st.button("Generate visualization"):
    p = Path(viz_db_path)
    if not p.exists():
        st.error(f"Edges DB not found: {viz_db_path}")
    else:
        st.info("Loading edges and building graph...")
        try:
            edges = load_edges_from_db(str(p), limit=int(limit))
            # Diagnostic: show counts and sample edges before building graph
            edges_list = list(edges)
            st.write(f"Loaded {len(edges_list)} edges from DB")
            if len(edges_list) > 0:
                # show first few edges to help debug (trim long values)
                sample_edges = edges_list[:10]
                st.json(sample_edges)

            G = edges_to_networkx(edges_list)
        except Exception as e:
            st.error(f"Failed loading edges or building graph: {e}")
            G = None

        if G is not None:
            tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
            tmp.close()
            try:
                out = write_pyvis_html(G, tmp.name)
                html = Path(tmp.name).read_text(encoding="utf-8")
                # Diagnostic: show generated HTML header comment and a short snippet
                if "<!-- graph-renderer:" in html:
                    hdr = html.split("<!-- graph-renderer:", 1)[1].split("-->", 1)[0]
                    st.text(f"HTML banner:{hdr.strip()}")
                st.text(f"Generated HTML size: {len(html)} bytes")
                components.html(html, height=800)
            except Exception as e:
                # Common pyvis failures stem from missing optional deps (jinja2)
                st.error(
                    "Could not render visualization with pyvis: %s. "
                    "Install optional deps in the venv: `pip install pyvis jinja2` "
                    "or view the raw edges in the Logs/DB." % e
                )

# Note: the previous auto-generate-on-jump logic has been removed since
# the quick-jump button is no longer provided. Visualization is generated
# only when the user explicitly clicks "Generate visualization".
