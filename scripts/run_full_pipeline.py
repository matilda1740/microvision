"""Full pipeline runner (safe, reviewable, not executed automatically).

This script implements an end-to-end pipeline (sampling -> parse -> cleaning -> merge ->
embed -> ingest into an isolated Chroma directory -> query -> persist -> transitions).

It contains safety features:
- per-run manifest written to data/run_manifests/<run_id>.json
- Chroma ingestion happens in a tmp directory and is atomically moved into place
- an edge-count guard that will auto-adjust top_k to avoid explosion

Run from project root with the project's venv active.
"""
from __future__ import annotations

import argparse
import sys
import csv
import json
import os
import random
import time
import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
# ensure project root on path (so `from src...` imports work when running the script)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parsing.metadata_drain_parser import MetadataDrainParser
from src.encode_chroma.encode_utils import df_to_metadatas
from src.enrichment.merger import merge_structured_metadata, MergeError
from src.retrieval import compute_candidate_edges_stream
from src.storage import EdgeStore
from config.settings import settings
from src.parsing.regex_utils import extract_service_from_path
from src.pipeline.stages import (
    sample_source,
    parse_sample_rows,
    load_parsed_df,
    preprocess_and_merge,
    build_metadatas_from_aggregated,
    compute_embeddings_for_use_df,
    ingest_embeddings_atomic,
    compute_and_persist_edges,
    adjust_top_k,
    validate_sample_df,
    reservoir_sample_csv,
)


## Refactored stage functions -------------------------------------------------
# Moved to src/pipeline/stages.py


from src.enrichment.cleaning import minimal_cleaning
from src.encode_chroma.chroma import ingest_to_chroma_atomic





def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--sample", type=int, default=getattr(settings, "DEFAULT_SAMPLE", 1000))
    p.add_argument("--contiguous", action="store_true", help="If set, take the first N rows/lines (preserve temporal order) instead of random sampling")
    p.add_argument("--db", default=getattr(settings, "DEFAULT_DB", "data/edges/edges_smoke.db"))
    p.add_argument("--chroma-dir", default=getattr(settings, "DEFAULT_CHROMA_DIR", "data/chroma_db/chroma_smoke"))
    p.add_argument("--top-k", type=int, default=getattr(settings, "DEFAULT_TOP_K", 10))
    p.add_argument("--threshold", type=float, default=getattr(settings, "DEFAULT_THRESHOLD", 0.2))
    p.add_argument("--alpha", type=float, default=getattr(settings, "DEFAULT_ALPHA", 0.5))
    p.add_argument("--device", default=getattr(settings, "DEFAULT_DEVICE", None))
    p.add_argument("--clear-db", action="store_true", help="Clear the edges table in the target DB before writing (destructive)")
    p.add_argument("--model", default=None, help="Override embedding model name (overrides config.settings.EMBEDDING_MODEL)")
    p.add_argument("--format-name", default=None, help="Optional log format name to pick from config.settings.LOG_FORMAT_MAPPINGS (e.g. OpenStack)")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    # optional guard overrides
    p.add_argument("--max-total-edges", type=int, default=None, help="Override MAX_TOTAL_EDGES from config")
    p.add_argument("--per-source-top-k-cap", type=int, default=None, help="Override PER_SOURCE_TOP_K_CAP from config")
    p.add_argument("--inspect", action="store_true", help="Print the `use_df` sample and pause before continuing")
    p.add_argument("--inspect-format", choices=["table", "plain", "json"], default="table", help="Format to display the `use_df` sample when --inspect is used")
    p.add_argument("--validate", action="store_true", help="Enable Cross-Encoder validation step")
    p.add_argument("--batch-size", type=int, default=128, help="Batch size for embedding generation")
    p.add_argument("--num-workers", type=int, default=1, help="Number of worker processes for parallel embedding (CPU only)")
    args = p.parse_args()

    # configure logging early so modules can emit debug traces
    try:
        from config.logging_config import configure_logging

        configure_logging(debug=bool(getattr(args, "debug", False)))
    except Exception:
        import logging

        logging.basicConfig(level=logging.DEBUG if getattr(args, "debug", False) else logging.INFO)

    src = Path(args.source)

    run_id = datetime.datetime.utcnow().strftime("run_%Y%m%dT%H%M%SZ")
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": datetime.datetime.utcnow().isoformat() + "Z",
        "source": str(src),
        "sample": int(args.sample),
        "top_k_requested": int(args.top_k),
        "threshold": float(args.threshold),
        "alpha": float(args.alpha),
        "events": [],
    }

    print(f"Sampling {args.sample} rows from {src} ... (contiguous={getattr(args, 'contiguous', False)})")
    t0 = time.time()
    sampled = reservoir_sample_csv(src, args.sample, contiguous=getattr(args, "contiguous", False))
    manifest["metrics"] = {"stages": {}}
    manifest["metrics"]["stages"]["sampling"] = {"duration_s": time.time() - t0, "rows": len(sampled)}
    raw_df = pd.DataFrame(sampled)

    # validate sampled input
    try:
        validate_sample_df(raw_df)
    except Exception as e:
        raise SystemExit(f"Input validation failed: {e}")

    # parsing using the notebook-derived MetadataDrainParser in fresh mode
    parsed_path = Path("data/parsed_sample.csv")
    templates_path = Path("data/parsed_templates.csv")
    parsed_path.parent.mkdir(parents=True, exist_ok=True)

    save_every = max(1, int(args.sample) // 4)
    t0 = time.time()
    # Determine log_format for MetadataDrainParser: CLI override by format name -> None (auto-detect)
    log_format = None
    if getattr(args, "format_name", None):
        try:
            log_format = getattr(settings, "LOG_FORMAT_MAPPINGS", {}).get(args.format_name)
        except Exception:
            log_format = None
        # warn if user supplied a format name that doesn't exist in settings
        if log_format is None:
            print(f"Warning: format_name '{args.format_name}' not found in config.settings.LOG_FORMAT_MAPPINGS; proceeding with auto-detect.", file=sys.stderr)

    parser = MetadataDrainParser(log_format=log_format, structured_csv=str(parsed_path), templates_csv=str(templates_path), save_every=save_every, mode="fresh")

    for i in range(len(raw_df)):
        row = raw_df.iloc[i].to_dict()
        raw_line = row.get("raw") or row.get("content") or " ".join([str(v) for v in row.values()])
        parser.process_line(raw_line, i + 1)
    # ensure buffer is flushed and templates exported
    parser.finalize()

    # read parsed results back
    parsed_df = pd.read_csv(parsed_path)
    
    # Fix for OpenStack logs where 'service' might be misparsed as 'INFO'/'WARN'
    # Extract service from the filename in 'raw' column: "nova-api.log.1..." -> "nova-api"
    if 'raw' in parsed_df.columns:
        # Regex to capture the part before .log
        # Matches 'nova-api' in 'nova-api.log' or '/var/log/nova/nova-api.log'
        # Also handles quoted strings like "nova-api.log..."
        extracted_service = parsed_df['raw'].astype(str).str.extract(r'(?:^|/|")([a-z0-9_-]+)\.log')
        
        if not extracted_service.empty and extracted_service[0].notna().any():
            print("Refining 'service' column by extracting from filename in 'raw'...")
            parsed_df['service'] = extracted_service[0]
            
    manifest["metrics"]["stages"]["parsing"] = {"duration_s": time.time() - t0, "rows": len(parsed_df)}
    print(f"Wrote parsed sample to {parsed_path}")
    manifest["events"].append({"ts": time.time(), "event": "parsed_sample_written", "path": str(parsed_path), "rows": len(parsed_df)})

    # 2) Cleaning/semantic grouping
    t0 = time.time()
    print("Running minimal cleaning/semantic grouping ...")
    cleaned = minimal_cleaning(parsed_df)
    cleaned_path = Path("data/cleaned_templates.csv")
    # write cleaned templates atomically
    try:
        from src.utils.atomic_write import atomic_write_csv

        atomic_write_csv(cleaned_path, cleaned)
    except Exception:
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(cleaned_path, index=False)
    print(f"Wrote cleaned templates to {cleaned_path}")
    manifest["events"].append({"ts": time.time(), "event": "cleaned_templates_written", "path": str(cleaned_path), "rows": len(cleaned)})
    manifest["metrics"]["stages"]["cleaning"] = {"duration_s": time.time() - t0, "rows": len(cleaned)}

    # 3) Merge with structured metadata
    merged_out = Path("data/merged_templates.csv")
    print("Running merge_structured_metadata")
    t0 = time.time()
    try:
        aggregated = merge_structured_metadata(cleaned, parsed_df, str(merged_out))
        print(f"Merged templates written to {merged_out} ({len(aggregated)} rows)")
        # Quick sanity: count non-empty timestamp_canonical values in the merged output
        try:
            if "timestamp_canonical" in aggregated.columns:
                non_empty = int(aggregated['timestamp_canonical'].fillna('').astype(str).str.strip().replace('', pd.NA).notna().sum())
            else:
                non_empty = 0
            print(f"Merged templates: timestamp_canonical non-empty count = {non_empty} / {len(aggregated)}")
            manifest["events"].append({
                "ts": time.time(),
                "event": "merged_timestamp_canonical_count",
                "count_non_empty": non_empty,
                "rows": int(len(aggregated)),
            })
        except Exception:
            # best-effort reporting; do not fail the pipeline on diagnostics
            pass
        manifest["events"].append({"ts": time.time(), "event": "merged_templates_written", "path": str(merged_out), "rows": len(aggregated)})
        manifest["metrics"]["stages"]["merge"] = {"duration_s": time.time() - t0, "rows": len(aggregated)}
    except MergeError as e:
        raise SystemExit(f"Merge failed: {e}")

    # 4) Metadata helpers
    use_df = aggregated.copy()

    # Ensure semantic_text exists (fall back to joining template_ids if needed),
    # then drop the list-valued `template_ids` to avoid passing lists into
    # metadata/embedding helpers which expect scalar fields.
    if "semantic_text" not in use_df.columns:
        use_df["semantic_text"] = use_df.get("template_ids").apply(lambda lst: ", ".join(lst) if isinstance(lst, list) else "")
    if "template_ids" in use_df.columns:
        use_df = use_df.drop(columns=["template_ids"])

    # Optionally print the DataFrame we'll pass to retrieval and pause so a
    # local developer can inspect it before heavy steps (embeddings/chroma).
    if getattr(args, "inspect", False):
        try:
            print("\nDEBUG: `use_df` shape:", getattr(use_df, "shape", None))
            sample = use_df.head(20)
            fmt = getattr(args, "inspect_format", "table")
            if fmt == "table":
                try:
                    print(sample.to_string())
                except Exception:
                    print(repr(sample))
            elif fmt == "plain":
                try:
                    # CSV-style plain output
                    print(sample.to_csv(index=False))
                except Exception:
                    print(repr(sample))
            elif fmt == "json":
                try:
                    import json as _json

                    recs = sample.to_dict(orient="records")
                    print(_json.dumps(recs, ensure_ascii=False, indent=2))
                except Exception:
                    try:
                        print(sample.to_string())
                    except Exception:
                        print(repr(sample))
        except Exception as _e:
            print("DEBUG: could not render use_df:", _e)

        # Pause execution here so you can inspect the output in your terminal
        # and then press Enter to continue the pipeline.
        try:
            input("DEBUG: Paused after building use_df. Press Enter to continue (or Ctrl-C to abort): \n")
        except KeyboardInterrupt:
            print("\nDEBUG: aborting as requested by user (KeyboardInterrupt)")
            raise SystemExit(130)
    # At this point `use_df` no longer contains `template_ids` (we dropped it above).
    # `semantic_text` must exist (either provided by merge or constructed earlier).
    if "semantic_text" not in use_df.columns:
        use_df["semantic_text"] = ""

    t0 = time.time()
    metas = df_to_metadatas(use_df)
    
    manifest["events"].append({"ts": time.time(), "event": "metadatas_built", "rows": len(metas)})
    manifest["metrics"]["stages"]["metadatas"] = {"duration_s": time.time() - t0, "rows": len(metas)}

    # Ensure metadatas are non-empty dicts for Chroma (Chroma rejects empty metadata dicts).
    # Fill minimal fallback metadata when needed (semantic_text) so ingestion doesn't fail.
    metas_sanitized = []
    def _is_nonempty(v):
        # Avoid relying on truthiness of numpy arrays which raises a DeprecationWarning.
        if v is None:
            return False
        # strings/bytes
        if isinstance(v, (str, bytes)):
            return v != ""
        # common iterable types
        if isinstance(v, (list, tuple, dict, set)):
            return len(v) > 0
        # numpy arrays: check .size when available
        try:
            import numpy as _np

            if isinstance(v, _np.ndarray):
                return v.size > 0
        except Exception:
            pass
        # otherwise treat scalars as non-empty
        return True

    for i, m in enumerate(metas):
        if not isinstance(m, dict):
            m2 = {"idx": i}
        else:
            # drop null/empty values using explicit non-empty test
            m2 = {k: v for k, v in m.items() if _is_nonempty(v)}
        if not m2:
            # fallback: include semantic_text or index to satisfy chroma metadata validation
            txt = use_df["semantic_text"].iloc[i] if ("semantic_text" in use_df.columns and i < len(use_df)) else None
            m2 = {"semantic_text": str(txt) if txt is not None else f"row_{i}"}
        metas_sanitized.append(m2)
    metas = metas_sanitized

    # 5) Embeddings
    # Use centralized encode helper to compute embeddings (we'll not ingest here so
    # we can sanitize metadatas consistently and then call the atomic ingest).
    model_name = args.model if args.model else getattr(settings, "EMBEDDING_MODEL", "all-mpnet-base-v2")
    device = args.device if args.device is not None else getattr(settings, "DEFAULT_DEVICE", None)
    if device is None:
        device = "cuda" if (os.environ.get("CUDA_VISIBLE_DEVICES") or os.path.exists("/usr/local/cuda")) else "cpu"

    t0 = time.time()
    print(f"Encoding semantic_text using model={model_name} device={device} ...")
    from src.encode_chroma.encoder import encode_templates

    # compute embeddings; do not ask the helper to ingest (chroma_dir=None)
    embeddings_out_dir = getattr(settings, "DEFAULT_EMBEDDINGS_DIR", "data/edges")
    embeddings, index_df, _ = encode_templates(use_df, model=None, model_name=model_name, device=device, output_dir=embeddings_out_dir, chroma_dir=None, batch_size=args.batch_size, num_workers=args.num_workers)
    documents = use_df["semantic_text"].fillna("").astype(str).tolist()
    manifest["metrics"]["stages"]["embeddings"] = {"duration_s": time.time() - t0, "docs": len(documents)}

    # 6) Ingest into Chroma (atomic)
    chroma_dir = Path(args.chroma_dir)
    ids = [str(i) for i in range(len(documents))]
    t0 = time.time()
    print(f"Ingesting {len(ids)} vectors into Chroma at {chroma_dir} ...")
    manifest["events"].append({"ts": time.time(), "event": "chroma_ingest_started", "tmp_dir": str(chroma_dir)})
    client, collection = ingest_to_chroma_atomic(chroma_dir, ids, metas, documents, embeddings)
    print("Chroma ingestion completed")
    manifest["events"].append({"ts": time.time(), "event": "chroma_ingest_completed", "persist_dir": str(chroma_dir), "count": len(ids)})
    manifest["metrics"]["stages"]["chroma_ingest"] = {"duration_s": time.time() - t0, "count": len(ids)}

    # 7) Retrieval: compute candidate edges using Chroma collection
    # Edge-count guard: estimate N * top_k and auto-adjust to avoid explosion
    N = len(use_df)
    requested_top_k = int(args.top_k)
    top_k = adjust_top_k(N, requested_top_k, max_total_edges=args.max_total_edges, per_source_top_k_cap=args.per_source_top_k_cap)
    est = N * requested_top_k
    if top_k != requested_top_k:
        print(f"Estimated edges (N*top_k) = {est} exceeds cap; Auto-adjusting top_k -> {top_k}")
        manifest["events"].append({"ts": time.time(), "event": "edge_count_guard_adjust", "requested_top_k": requested_top_k, "adjusted_top_k": top_k, "N": N, "est": est, "max_total_edges": args.max_total_edges})
    else:
        manifest["events"].append({"ts": time.time(), "event": "edge_count_guard_no_adjust", "requested_top_k": requested_top_k, "N": N, "est": est})

    print(f"Streaming candidate edges from Chroma with top_k={top_k} ...")
    # Ensure retriever can find the parser's id column (template_id)
    id_column_candidates = ("template_id", "reqid", "template", "pid")
    gen = compute_candidate_edges_stream(
        use_df,
        embeddings=embeddings,
        collection=collection,
        top_k=top_k,
        threshold=args.threshold,
        batch_size=128,
        alpha=args.alpha,
        id_column_candidates=id_column_candidates,
    )

    # 7.5) Validation (Optional RAG Step)
    if getattr(args, "validate", False):
        print("Validation enabled: Re-ranking edges with Cross-Encoder...")
        from src.validation import SemanticValidator
        
        # Initialize validator (this might download the model on first run)
        validator = SemanticValidator()
        
        # We need to consume the generator to validate in batches
        # Note: This breaks streaming for the validation step, but that's unavoidable for cross-encoding
        candidates = list(gen)
        print(f"Validating {len(candidates)} candidate edges...")
        
        # Validate
        validated_edges = validator.validate_edges(candidates)
        
        # Filter if needed (e.g. score > 0 for MS MARCO)
        # For now, we just pass them through with the new score, 
        # but you could filter: [e for e in validated_edges if e['validation_score'] > 0]
        
        # Re-create generator
        gen = (e for e in validated_edges)
        manifest["events"].append({"ts": time.time(), "event": "validation_completed", "count": len(candidates)})

    # 8) Persist edges
    db_path = args.db
    store = EdgeStore(db_path)
    store.init_db()
    if getattr(args, "clear_db", False):
        print(f"--clear-db specified: clearing edges table in {db_path} before writing")
        store.clear_edges(reset_sequence=True)
    print(f"Writing edges to {db_path} ...")
    t0 = time.time()
    edges_written = store.write_edges(gen, batch_size=500)
    print(f"Edges written: {edges_written}")
    manifest["events"].append({"ts": time.time(), "event": "edges_written", "count": edges_written, "db": str(db_path)})
    manifest["metrics"]["stages"]["edges_written"] = {"duration_s": time.time() - t0, "count": edges_written}

    # 9) Compute transitions
    from src.transitions import compute_and_persist_transitions
    t0 = time.time()
    transitions_written = compute_and_persist_transitions(db_path, min_count=1)
    print(f"Transitions written: {transitions_written}")
    manifest["events"].append({"ts": time.time(), "event": "transitions_written", "count": transitions_written})
    manifest["metrics"]["stages"]["transitions"] = {"duration_s": time.time() - t0, "count": transitions_written}

    store.close()

    # 10) Summarize artifacts
    print("Run complete. Artifacts:")
    print(f" - cleaned templates: {cleaned_path}")
    print(f" - merged templates: {merged_out}")
    print(f" - chroma dir: {chroma_dir}")
    print(f" - edges DB: {db_path}")

    # write manifest to disk
    try:
        from src.utils.atomic_write import atomic_write_json

        manifest_dir = Path("data/run_manifests")
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{run_id}.json"
        manifest["finished_at"] = datetime.datetime.utcnow().isoformat() + "Z"
        manifest["edges_db"] = str(db_path)
        atomic_write_json(manifest_path, manifest)
        print(f"Wrote run manifest: {manifest_path}")
    except Exception as e:
        print(f"Could not write manifest: {e}")


if __name__ == "__main__":
    main()
