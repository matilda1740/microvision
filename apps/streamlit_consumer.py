"""Streamlit app scaffold for uploading logs, enqueueing jobs and polling status.

Run locally with:
    streamlit run apps/streamlit_consumer.py

This scaffold shows the minimal UI: file uploader, validation options, enqueue
button, job status polling, and a placeholder for the graph view.
"""
import os
import time
import tempfile

try:
    import streamlit as st
except Exception:
    st = None

import threading
from pathlib import Path
import pandas as pd
import tempfile
import time

def _ensure_project_on_path():
    """Make a best-effort attempt to put the repository root on sys.path.

    Streamlit sometimes executes scripts with different working directories or
    via a reloader which can make relative imports fail. Try several heuristics:
      - repo root = parent of the apps/ folder (i.e. two levels up from this file)
      - current working directory (Path.cwd())
      - climb parents until we find a folder that contains a 'src' directory

    This function is defensive and non-fatal; if it cannot find a suitable root
    the subsequent imports will raise the usual ModuleNotFoundError which
    surfaces to the user.
    """
    try:
        import sys
        from pathlib import Path

        candidates = []
        # 1) parent of the apps folder (likely repo root)
        try:
            candidates.append(Path(__file__).resolve().parents[1])
        except Exception:
            pass
        # 2) current working directory
        try:
            candidates.append(Path.cwd())
        except Exception:
            pass
        # 3) climb parents from this file and cwd looking for a 'src' directory
        for base in list(candidates):
            p = base
            for _ in range(6):
                if (p / "src").exists():
                    candidates.append(p)
                    break
                p = p.parent

        # dedupe and try to insert the most plausible candidate first
        seen = set()
        for c in candidates:
            try:
                cstr = str(c)
                if cstr in seen:
                    continue
                seen.add(cstr)
                if cstr not in sys.path:
                    sys.path.insert(0, cstr)
            except Exception:
                continue
    except Exception:
        # non-fatal
        pass


# ensure project is importable before attempting src imports
_ensure_project_on_path()

from src.enrichment.jobs import enqueue_job, get_job_status
from src.parsing.metadata_drain_parser import MetadataDrainParser
from config.settings import settings


MAX_BYTES_DEFAULT = 5_000_000
MAX_LINES_DEFAULT = 10000


def _background_runner(fn, key, *args, **kwargs):
    """Run `fn(*args, **kwargs)` in a background thread and store result in st.session_state[key].

    Stored value is a dict with keys: status (running|done|failed), result, error, started_at, finished_at
    """
    def _target():
        try:
            st.session_state[key] = {"status": "running", "started_at": time.time(), "result": None, "error": None}
            res = fn(*args, **kwargs)
            st.session_state[key]["status"] = "done"
            st.session_state[key]["result"] = res
            st.session_state[key]["finished_at"] = time.time()
        except Exception as e:
            st.session_state[key]["status"] = "failed"
            st.session_state[key]["error"] = str(e)
            st.session_state[key]["finished_at"] = time.time()

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    return t


@st.cache_data
def parse_log_file_to_df(path: str, format_name: str | None = None, sample: int | None = None, save_every: int = 250):
    """Parse a raw log file into a structured dataframe using MetadataDrainParser.

    Parsed outputs are written into a stable cache directory controlled by
    `settings.PARSE_CACHE_DIR` (defaults to `data/cache/parses`). The function
    is memoized by Streamlit's `st.cache_data` so repeated calls (including
    across reruns) will reuse the cached results. Returns (parsed_df,
    templates_df, parsed_path_str).
    """
    # determine cache directory (configurable via settings.PARSE_CACHE_DIR)
    cache_dir = getattr(settings, "PARSE_CACHE_DIR", "data/cache/parses")
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # deterministic cache naming to allow reuse: hash of path + args
    import hashlib

    key_src = f"{str(Path(path).resolve())}|format={format_name}|sample={sample}|save_every={save_every}"
    h = hashlib.sha1(key_src.encode("utf-8")).hexdigest()[:12]
    run_dir = cache_path / h
    run_dir.mkdir(parents=True, exist_ok=True)
    parsed_path = run_dir / "parsed_sample.csv"
    templates_path = run_dir / "parsed_templates.csv"

    # resolve optional format mapping from settings
    log_format = None
    if format_name:
        log_format = getattr(settings, "LOG_FORMAT_MAPPINGS", {}).get(format_name)

    parser = MetadataDrainParser(log_format=log_format, structured_csv=str(parsed_path), templates_csv=str(templates_path), save_every=save_every, mode="fresh")

    # stream file line-by-line to avoid loading huge files into memory
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for i, line in enumerate(fh):
            parser.process_line(line.rstrip("\n"), i + 1)
    parser.finalize()

    # read outputs
    parsed_df = pd.read_csv(parsed_path)
    templates_df = pd.read_csv(templates_path) if templates_path.exists() else pd.DataFrame()
    return parsed_df, templates_df, str(parsed_path)


@st.cache_resource
def get_embedding_model(model_name: str, device: str | None = None):
    """Load and cache a SentenceTransformer embedding model instance.

    This uses Streamlit's cache_resource so the heavy model is loaded once per session/worker.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        raise RuntimeError("sentence-transformers must be installed to load models")
    if device is None:
        device = "cuda" if (os.environ.get("CUDA_VISIBLE_DEVICES") or Path("/usr/local/cuda").exists()) else "cpu"
    return SentenceTransformer(model_name, device=device)


def main():
    if st is None:
        print("Streamlit is not installed. Install it to run the app: pip install streamlit")
        return

    st.title("MicroVision: Semantic Log Dependency Mapper")
    st.markdown("A framework for semantic microservice dependency discovery from distributed logs.")

    st.markdown("Upload a log file (max size configurable). Small files are processed inline; larger files are queued.")

    uploaded = st.file_uploader("Upload log file", type=["log", "txt", "csv"], help="Max size configurable below")

    jobs_db = st.text_input("Jobs DB path", value="data/jobs.db")
    edges_db = st.text_input("Edges DB path", value="data/edges.db")
    # mapping path override (optional)
    map_path_override = st.text_input("Service map path (optional override)", value="")
    # session toggle: show display names vs raw tokens
    if "use_display_names" not in st.session_state:
        st.session_state["use_display_names"] = True
    use_display_names = st.checkbox("Use human-friendly display names (if available)", value=st.session_state["use_display_names"])
    st.session_state["use_display_names"] = use_display_names

    max_bytes = st.number_input("Max upload bytes", value=MAX_BYTES_DEFAULT, step=1024 * 10)
    max_lines = st.number_input("Max upload lines", value=MAX_LINES_DEFAULT, step=100)

    if uploaded is not None:
        # save to temp file
        tmpdir = tempfile.mkdtemp()
        temp_path = os.path.join(tmpdir, uploaded.name)
        with open(temp_path, "wb") as fh:
            fh.write(uploaded.getbuffer())

        st.success(f"Saved upload to {temp_path}")

        if st.button("Enqueue for processing"):
            # enqueue with validation turned on
            try:
                job_id = enqueue_job(jobs_db, temp_path, edges_db, params={}, validate=True, max_bytes=int(max_bytes), max_lines=int(max_lines))
                st.session_state["last_job_id"] = job_id
                st.success(f"Enqueued job {job_id}")
            except Exception as e:
                st.error(f"Failed to enqueue: {e}")

    job_id = st.session_state.get("last_job_id")
    if job_id:
        st.subheader("Job status")
        status = get_job_status(jobs_db, job_id)
        if status is None:
            st.info("No job found")
        else:
            st.write(status)
            if status["status"] in ("queued", "running"):
                if st.button("Refresh status"):
                    st.experimental_rerun()
                st.info("Auto-refresh every 5s")
                st.experimental_set_query_params(job_id=job_id)
                time.sleep(1)
                st.experimental_rerun()
            elif status["status"] == "done":
                st.success("Job completed — view or refresh the graph below.")
                # Graph view controls
                import sqlite3
                import networkx as nx
                import numpy as np

                st.markdown("### Graph view & re-ranking")
                top_n = st.number_input("Top N edges to show", value=50, min_value=5, max_value=500, step=5)
                beta = st.slider("Sequence prior weight (beta)", min_value=0.0, max_value=1.0, value=0.5)
                min_hybrid = st.number_input("Min hybrid score", value=0.0, min_value=0.0, max_value=1.0, step=0.01)

                if st.button("Refresh graph"):
                    st.experimental_rerun()

                # load edges and transitions from DB
                try:
                    # Offload DB-heavy retrieval to a background thread so the UI stays responsive
                    def fetch_graph_data(edges_db_path, min_hybrid_val, limit):
                        import sqlite3 as _sqlite
                        conn2 = _sqlite.connect(edges_db_path)
                        cur2 = conn2.cursor()
                        cur2.execute(
                            "SELECT source_id, target_id, hybrid_score, target_metadata FROM edges WHERE hybrid_score IS NOT NULL AND hybrid_score >= ? ORDER BY hybrid_score DESC LIMIT ?",
                            (float(min_hybrid_val), int(limit)),
                        )
                        rows2 = cur2.fetchall()

                        cur2.execute("SELECT source_id, target_id, prob FROM transitions")
                        trans_rows2 = cur2.fetchall()
                        trans_map2 = {(r[0], r[1]): float(r[2]) for r in trans_rows2}

                        conn2.close()
                        return {"rows": rows2, "trans_map": trans_map2}

                    key = f"graph_fetch_{edges_db}_{min_hybrid}_{top_n}"
                    # if not already running/fetched, start background job
                    state = st.session_state.get(key)
                    if state is None or state.get("status") in ("failed", "done"):
                        _background_runner(fetch_graph_data, key, edges_db, min_hybrid, int(top_n * 5))
                        st.info("Fetching graph data in background. Refresh in a few seconds.")
                        st.stop()

                    fetched = st.session_state.get(key)
                    if not fetched or fetched.get("status") != "done":
                        st.info("Graph data is still loading... please refresh in a few seconds")
                        st.stop()

                    rows = fetched["result"]["rows"]
                    trans_map = fetched["result"]["trans_map"]

                    # aggregate top candidates and collect metadata labels
                    from src.enrichment.labels import extract_service_label_from_metadata, update_service_map_with_token
                    import os

                    # if provided, set env var so labels.resolve picks it up
                    if map_path_override:
                        os.environ["SERVICE_NAME_MAP_PATH"] = map_path_override

                    node_label_map = {}
                    edge_list = []
                    for src, tgt, hybrid, target_meta in rows:
                        # try to extract a service label for the target node
                        try:
                            label = extract_service_label_from_metadata(target_meta, prefer_two_level=True)
                            if label:
                                node_label_map.setdefault(tgt, label)
                        except Exception:
                            pass

                        seq_prob = trans_map.get((src, tgt), 0.0)
                        final_score = (1.0 - beta) * float(hybrid) + beta * float(seq_prob)
                        edge_list.append((src, tgt, float(hybrid), float(seq_prob), float(final_score)))

                    # sort by final_score and take top_n
                    edge_list.sort(key=lambda x: x[4], reverse=True)
                    edge_list = edge_list[:int(top_n)]

                    if not edge_list:
                        st.info("No edges to display (try lowering min hybrid or increasing Top N)")
                    else:
                        # Build graph
                        G = nx.DiGraph()
                        for src, tgt, h, s, f in edge_list:
                            G.add_node(src)
                            G.add_node(tgt)
                            G.add_edge(src, tgt, weight=f, hybrid=h, seq_prob=s)

                        # try interactive pyvis if available
                        try:
                            from pyvis.network import Network

                            net = Network(height="600px", width="100%", directed=True)
                            for n in G.nodes():
                                deg = max(1, G.in_degree(n) + G.out_degree(n))
                                raw_lbl = str(n)
                                lbl = node_label_map.get(n, raw_lbl)
                                # respect session toggle
                                display_lbl = lbl if st.session_state.get("use_display_names", True) else raw_lbl
                                net.add_node(n, label=str(display_lbl), size=10 + deg * 3)
                            for u, v, d in G.edges(data=True):
                                w = d.get("weight", 1.0)
                                title = f"hybrid={d.get('hybrid'):.3f}, seq={d.get('seq_prob'):.3f}, final={d.get('weight'):.3f}"
                                net.add_edge(u, v, value=float(w), title=title)

                            net.repulsion(node_distance=200, central_gravity=0.33)
                            html = net.generate_html()
                            st.components.v1.html(html, height=650, scrolling=True)
                        except Exception:
                            # fallback to matplotlib static image
                            import matplotlib.pyplot as plt

                            plt.figure(figsize=(10, 6))
                            pos = nx.spring_layout(G, seed=42)
                            weights = [d.get("weight", 1.0) for _, _, d in G.edges(data=True)]
                            labels = {}
                            for n in G.nodes():
                                raw_lbl = str(n)
                                lbl = node_label_map.get(n, raw_lbl)
                                labels[n] = lbl if st.session_state.get("use_display_names", True) else raw_lbl
                            nx.draw(G, pos, labels=labels, with_labels=True, node_size=500, node_color="skyblue", width=np.maximum(weights, 0.1))
                            st.pyplot(plt)

                except Exception as e:
                    st.error(f"Unable to load graph: {e}")
            elif status["status"] == "failed":
                st.error(f"Job failed: {status.get('result_msg')}")


if __name__ == "__main__":
    main()
