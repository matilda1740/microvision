from __future__ import annotations

import csv
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from config.settings import settings
from src.encode_chroma.chroma import ingest_to_chroma_atomic
from src.encode_chroma import chroma as _chroma_mod
from src.encode_chroma.encode_utils import df_to_metadatas
from src.encode_chroma.encoder import encode_templates
from src.enrichment.cleaning import minimal_cleaning
from src.enrichment.merger import merge_structured_metadata
from src.parsing.metadata_drain_parser import MetadataDrainParser
from src.parsing.regex_utils import extract_service_from_path
from src.retrieval import compute_candidate_edges_stream
from src.storage import EdgeStore


def reservoir_sample_csv(path: Path, sample_size: int, seed: int = 42, contiguous: bool = False) -> List[Dict[str, Any]]:
    """Sample lines/rows from a file.

    By default this uses reservoir sampling to produce a random sample. If
    ``contiguous`` is True we read the first ``sample_size`` rows in order
    (preserving temporal continuity for trace reconstruction).
    """
    rng = random.Random(seed)
    sample: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        # Try to detect whether file has a CSV header. If not, treat each
        # line as a raw log message and sample accordingly.
        try:
            start = fh.read(8192)
            fh.seek(0)
            has_header = csv.Sniffer().has_header(start)
        except Exception:
            has_header = False

        if has_header:
            reader = csv.DictReader(fh)
            if contiguous:
                # take first N rows in order
                for i, row in enumerate(reader):
                    if i >= sample_size:
                        break
                    sample.append(row)
            else:
                for i, row in enumerate(reader):
                    if i < sample_size:
                        sample.append(row)
                    else:
                        j = rng.randint(0, i)
                        if j < sample_size:
                            sample[j] = row
        else:
            # treat each physical line as a single-column 'raw' value
            if contiguous:
                for i, line in enumerate(fh):
                    if i >= sample_size:
                        break
                    row = {"raw": line.rstrip("\n")}
                    sample.append(row)
            else:
                for i, line in enumerate(fh):
                    row = {"raw": line.rstrip("\n")}
                    if i < sample_size:
                        sample.append(row)
                    else:
                        j = rng.randint(0, i)
                        if j < sample_size:
                            sample[j] = row

    return sample


def sample_source(src: Path, sample_size: int, seed: int = 42) -> list[dict]:
    """Sample lines/rows from a source CSV or text file. Returns list of rows (dict).

    This is thin wrapper around reservoir_sample_csv kept for backwards
    compatibility with the original script behaviour.
    """
    return reservoir_sample_csv(src, sample_size, seed)


def validate_sample_df(df: pd.DataFrame) -> None:
    """Simple input validation for the sampled dataframe.

    Raises ValueError with actionable guidance when checks fail.
    """
    if df is None or len(df) == 0:
        raise ValueError("Sampled dataframe is empty. Check source path and sample size.")
    # require at least one of 'raw' or 'content' columns
    if not ("raw" in df.columns or "content" in df.columns):
        raise ValueError("Sampled dataframe must contain either 'raw' or 'content' columns for parsing.\nSuggestion: run the notebook parser to produce these columns.")


def parse_sample_rows(sampled: list[dict], parsed_path: Path, templates_path: Path, log_format: object | None = None, save_every: int = 250) -> None:
    """Run MetadataDrainParser over sampled rows and write parsed CSVs.

    Side-effects: writes parsed_path and templates_path.
    """
    parser = MetadataDrainParser(log_format=log_format, structured_csv=str(parsed_path), templates_csv=str(templates_path), save_every=save_every, mode="fresh")
    for i in range(len(sampled)):
        row = sampled[i]
        raw_line = row.get("raw") or row.get("content") or " ".join([str(v) for v in row.values()])
        parser.process_line(raw_line, i + 1)
    parser.finalize()


def load_parsed_df(parsed_path: Path) -> pd.DataFrame:
    return pd.read_csv(parsed_path)


def preprocess_and_merge(parsed_df: pd.DataFrame, save_cleaned: Path, merged_out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run cleaning and merge with structured metadata. Returns (cleaned, aggregated).
    Also writes cleaned and merged CSVs to disk.
    """
    # Fix for OpenStack logs where 'service' might be misparsed as 'INFO'/'WARN'
    # Extract service from the filename in 'raw' column: "nova-api.log.1..." -> "nova-api"
    print(f"DEBUG: parsed_df columns: {parsed_df.columns}")
    if 'raw' in parsed_df.columns:
        # Regex to capture the part before .log
        # Matches 'nova-api' in 'nova-api.log' or '/var/log/nova/nova-api.log'
        # Also handles quoted strings like "nova-api.log..."
        # Use shared helper for consistency
        extracted_service = parsed_df['raw'].apply(extract_service_from_path)
        
        if extracted_service.notna().any():
            print("Refining 'service' column by extracting from filename in 'raw'...")
            parsed_df['service'] = extracted_service
            
            # Save the patched parsed_df back to disk so edge_store can see it
            # We don't have parsed_path here, but we can infer it or pass it.
            # For now, let's just write to the default location "data/parsed_sample.csv"
            # since that's what the pipeline uses.
            try:
                parsed_df.to_csv("data/parsed_sample.csv", index=False)
                print("Saved refined parsed logs to data/parsed_sample.csv")
            except Exception as e:
                print(f"Warning: could not save refined parsed logs: {e}")
    cleaned = minimal_cleaning(parsed_df)
    save_cleaned.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(save_cleaned, index=False)
    aggregated = merge_structured_metadata(cleaned, parsed_df, str(merged_out))
    return cleaned, aggregated


def build_metadatas_from_aggregated(aggregated: pd.DataFrame) -> list[dict]:
    use_df = aggregated.copy()
    if "semantic_text" not in use_df.columns:
        use_df["semantic_text"] = use_df.get("template_ids").apply(lambda lst: ", ".join(lst) if isinstance(lst, list) else "")
    metas = df_to_metadatas(use_df)
    # sanitize to avoid empty metadata dicts
    metas_sanitized = []

    def _is_nonempty(v):
        if v is None:
            return False
        if isinstance(v, (str, bytes)):
            return v != ""
        if isinstance(v, (list, tuple, dict, set)):
            return len(v) > 0
        try:
            import numpy as _np

            if isinstance(v, _np.ndarray):
                return v.size > 0
        except Exception:
            pass
        return True

    for i, m in enumerate(metas):
        if not isinstance(m, dict):
            m2 = {"idx": i}
        else:
            m2 = {k: v for k, v in m.items() if _is_nonempty(v)}
        if not m2:
            txt = use_df["semantic_text"].iloc[i] if ("semantic_text" in use_df.columns and i < len(use_df)) else None
            m2 = {"semantic_text": str(txt) if txt is not None else f"row_{i}"}
        metas_sanitized.append(m2)
    return metas_sanitized


def compute_embeddings_for_use_df(use_df: pd.DataFrame, model_name: str | None, device: str | None, output_dir: str | None = None, batch_size: int = 128, num_workers: int = 1) -> tuple[list[list[float]], pd.DataFrame, list[str]]:
    """Compute embeddings using centralized helper and return (embeddings, index_df, documents).
    The helper mirrors the previous in-script call: it returns embeddings, index_df and a model object (ignored).
    """
    out_dir = output_dir or getattr(settings, "DEFAULT_EMBEDDINGS_DIR", "data/edges")
    embeddings, index_df, _ = encode_templates(use_df, model=None, model_name=model_name, device=device, output_dir=out_dir, chroma_dir=None, batch_size=batch_size, num_workers=num_workers)
    documents = use_df["semantic_text"].fillna("").astype(str).tolist()
    return embeddings, index_df, documents


def ingest_embeddings_atomic(chroma_dir: Path, ids: list[str], metas: list[dict], documents: list[str], embeddings: list[list[float]]):
    """Call the central atomic ingest helper and return (client, collection).
    """
    # Detect whether a cached client existed prior to this call so callers
    # (e.g. UI) can display a helpful message.
    pre_cached = str(chroma_dir) in getattr(_chroma_mod, '_CHROMA_CLIENT_CACHE', {})
    client, collection = ingest_to_chroma_atomic(chroma_dir, ids, metas, documents, embeddings)
    reused = bool(pre_cached and client is not None)
    return client, collection, reused


def compute_and_persist_edges(use_df: pd.DataFrame, embeddings: list[list[float]], collection, db_path: str, top_k: int, threshold: float, alpha: float, clear_db: bool = False) -> int:
    # Ensure retriever knows which id columns to consider (include `template_id` from parser)
    id_column_candidates = ("template_id", "reqid", "template", "pid")
    gen = compute_candidate_edges_stream(
        use_df,
        embeddings=embeddings,
        collection=collection,
        top_k=top_k,
        threshold=threshold,
        batch_size=128,
        alpha=alpha,
        id_column_candidates=id_column_candidates,
    )
    store = EdgeStore(db_path)
    store.init_db()
    try:
        if clear_db:
            store.clear_edges(reset_sequence=True)
        written = store.write_edges(gen, batch_size=500)
    finally:
        store.close()
    return written


def adjust_top_k(N: int, requested_top_k: int, max_total_edges: int | None = None, per_source_top_k_cap: int | None = None) -> int:
    """Return adjusted top_k using settings or passed thresholds.

    This encapsulates the guard logic and is unit-testable.
    """
    if max_total_edges is None:
        max_total_edges = getattr(settings, "MAX_TOTAL_EDGES", 200000)
    if per_source_top_k_cap is None:
        per_source_top_k_cap = getattr(settings, "PER_SOURCE_TOP_K_CAP", 100)

    est = N * int(requested_top_k)
    if est > max_total_edges:
        new_top_k = max(1, max_total_edges // max(1, N))
        new_top_k = min(new_top_k, per_source_top_k_cap)
        return new_top_k
    return int(requested_top_k)
