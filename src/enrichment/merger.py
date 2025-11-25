"""Refactored merge_structured_metadata function extracted from the notebook.

This version focuses on correctness and small, testable building blocks:
- explicit validation of required columns
- separate helpers for parsing/normalizing ids
- conservative remap behavior with logging via exceptions or returns
"""
from __future__ import annotations

import ast
import re
from collections import Counter
from typing import List, Optional
import numpy as np

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def parse_list_like(value):
    """Parse a value that may represent a list of ids.

    Accepts native lists, Python-list-like strings (``"[1,2]"``), and
    comma-separated strings. Returns a list (possibly empty) and never ``None``.
    """
    # handle explicit None / pandas NA / numpy nan
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        # in case pd.isna fails for exotic types, continue
        pass

    # native list/tuple/set/ndarray/Series -> return cleaned list
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        out = []
        for v in list(value):
            try:
                if pd.isna(v):
                    continue
            except Exception:
                pass
            out.append(v)
        return out
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
            except Exception:
                s = s.strip("[]")
        if "," in s:
            return [p.strip() for p in s.split(",") if p.strip()]
        return [s]
    return [value]


def normalize_tid(val: object) -> str:
    """Normalize a template id value to a stable string representation.

    Examples:
        - ``123.0`` -> ``"123"``
        - Surrounding quotes and brackets are stripped.

    Returns an empty string for missing values.
    """
    if pd.isna(val):
        return ""
    s = str(val).strip()
    s = s.strip("'\"[]")
    s = re.sub(r"\s+", " ", s)
    if re.fullmatch(r"\d+\.0+", s):
        try:
            s = str(int(float(s)))
        except Exception:
            pass
    return s


class MergeError(Exception):
    """Generic error raised for merge-related validation failures."""


def merge_structured_metadata(
    cleaned_df: pd.DataFrame,
    structured_df: pd.DataFrame,
    output_path: str,
    ids_col: str = "template_ids",
    id_col_single: str = "template_id",
    semantic_col: str = "semantic_text",
    meta_cols: Optional[List[str]] = None,
    try_simple_remap: bool = True,
    try_fuzzy_remap: bool = True,
) -> pd.DataFrame:
    """Merge cleaned templates and structured metadata into aggregated rows.

    This function is a testable, side-effecting implementation of the
    notebook's merge logic. It performs the following steps:
      - validation of required columns
      - normalization and explosion of ``template_ids``
      - left-join against structured metadata
      - conservative remapping heuristics and optional fuzzy remap
      - semantic majority-fill fallback
      - aggregation back to one row per ``semantic_text``

    Args:
        cleaned_df: DataFrame of cleaned templates (must contain ``semantic_text`` and either
            ``template_ids`` or ``template_id``).
        structured_df: DataFrame of structured metadata (must contain ``template_id``).
        output_path: CSV path where the aggregated results will be written.
        ids_col: Column in ``cleaned_df`` holding template id lists.
        id_col_single: Column name used for single template ids.
        semantic_col: Column name for semantic text grouping.
        meta_cols: List of metadata columns to preserve/aggregate; defaults to common fields.
        try_simple_remap: If True, apply numeric/prefix heuristics to remap missing ids.
        try_fuzzy_remap: If True, attempt fuzzy remapping using rapidfuzz (if installed).

    Returns:
        The aggregated DataFrame (also written to ``output_path``).
    """
    # shallow copies to avoid mutating inputs
    cleaned = cleaned_df.copy()
    structured = structured_df.copy()

    if id_col_single not in structured.columns:
        raise MergeError(f"structured_df must contain '{id_col_single}' column")

    if ids_col not in cleaned.columns and id_col_single not in cleaned.columns:
        raise MergeError(f"cleaned_df must contain '{ids_col}' or '{id_col_single}'")

    if meta_cols is None:
        meta_cols = [c for c in ["service", "component", "level", "timestamp"] if c in structured.columns]

    # ensure template_ids is a list
    if ids_col not in cleaned.columns and id_col_single in cleaned.columns:
        cleaned[ids_col] = cleaned[id_col_single].apply(lambda v: [v] if not pd.isna(v) else [])

    cleaned[ids_col] = cleaned[ids_col].apply(parse_list_like)

    exploded = cleaned.explode(ids_col).rename(columns={ids_col: id_col_single}).reset_index(drop=True)
    exploded = exploded.dropna(subset=[id_col_single])
    exploded["_tid_raw"] = exploded[id_col_single]
    exploded[id_col_single] = exploded[id_col_single].apply(normalize_tid)

    structured = structured.copy()
    structured["_tid_raw"] = structured[id_col_single]
    structured[id_col_single] = structured[id_col_single].apply(lambda v: normalize_tid('' if pd.isna(v) else v))

    merged = exploded.merge(structured, on=id_col_single, how="left", suffixes=("", "_struct"))

    primary_meta_col = next((c for c in meta_cols if c in merged.columns), None)

    # compute mapping structures used for remapping
    struct_by_id = structured.groupby(id_col_single).first()
    struct_ids = set(structured[id_col_single].dropna().astype(str).unique())

    # simple heuristic remap
    if try_simple_remap and primary_meta_col is not None:
        missing_mask = merged[primary_meta_col].isna()
        if missing_mask.any():
            missing_ids = sorted(set(merged.loc[missing_mask, id_col_single].astype(str).unique()))
            remap_map = {}
            prefixes = ["template_", "tpl", "t"]
            for mid in missing_ids:
                if not mid:
                    continue
                try:
                    m_int = int(float(mid))
                    cand = str(m_int)
                    if cand in struct_ids:
                        remap_map[mid] = cand
                        continue
                except Exception:
                    pass
                # endswith heuristic
                found = next((sid for sid in struct_ids if sid.endswith(mid) and len(sid) - len(mid) <= 6), None)
                if found:
                    remap_map[mid] = found
                    continue
                for p in prefixes:
                    cand = p + mid
                    if cand in struct_ids:
                        remap_map[mid] = cand
                        break

            if remap_map:
                # create remapped id column and only apply mappings for the
                # explicit metadata columns we care about (avoid writing other
                # structured columns like numeric metrics into textual fields)
                merged["_remapped_tid"] = merged[id_col_single].map(lambda x: remap_map.get(str(x), x))
                # restrict mapping to meta_cols that exist in structured and merged
                cols_to_map = [c for c in meta_cols if c in struct_by_id.columns and c in merged.columns]
                for col in cols_to_map:
                    try:
                        mapping = struct_by_id[col].to_dict()
                        # map using the remapped tid series; only fill missing cells
                        merged.loc[missing_mask, col] = merged.loc[missing_mask, "_remapped_tid"].map(mapping)
                    except Exception:
                        logger.exception("Failed to remap column %s during simple remap", col)
                # cleanup helper column
                if "_remapped_tid" in merged.columns:
                    merged = merged.drop(columns=["_remapped_tid"])

    # fuzzy remap placeholder: do not auto-apply if rapidfuzz not available
    if try_fuzzy_remap:
        try:
            from rapidfuzz import process, fuzz  # type: ignore
        except Exception:
            process = None
            fuzz = None
        if process is not None and primary_meta_col is not None:
            missing_mask = merged[primary_meta_col].isna()
            if missing_mask.any():
                missing_ids = [m for m in sorted(set(merged.loc[missing_mask, id_col_single].astype(str).unique())) if m]
                struct_ids_list = list(structured[id_col_single].dropna().astype(str).unique())
                fuzzy_map = {}
                for mid in missing_ids:
                    best = process.extractOne(mid, struct_ids_list, scorer=fuzz.ratio)
                    if best is None:
                        continue
                    cand, score, _ = best
                    if score >= 85:
                        fuzzy_map[mid] = cand
                if fuzzy_map:
                    merged["_remapped_tid_fuzzy"] = merged[id_col_single].map(lambda x: fuzzy_map.get(str(x), x))
                    cols_to_map = [c for c in meta_cols if c in struct_by_id.columns and c in merged.columns]
                    for col in cols_to_map:
                        try:
                            mapping = struct_by_id[col].to_dict()
                            merged.loc[missing_mask, col] = merged.loc[missing_mask, "_remapped_tid_fuzzy"].map(mapping)
                        except Exception:
                            logger.exception("Failed to remap column %s during fuzzy remap", col)
                    if "_remapped_tid_fuzzy" in merged.columns:
                        merged = merged.drop(columns=["_remapped_tid_fuzzy"])

    # semantic majority-fill fallback
    if primary_meta_col is not None:
        still_missing_mask = merged[primary_meta_col].isna()
        if still_missing_mask.any():
            filled = merged[~merged[primary_meta_col].isna()]
            if not filled.empty:
                modes = {}
                for meta in meta_cols:
                    mode_series = filled.groupby(semantic_col)[meta].agg(
                        lambda s: Counter(s.dropna()).most_common(1)[0][0] if len(s.dropna()) > 0 else None
                    )
                    modes[meta] = mode_series.to_dict()
                for meta in meta_cols:
                    merged.loc[still_missing_mask, meta] = merged.loc[still_missing_mask, semantic_col].map(modes[meta])

    # aggregate back to semantic_text
    def unique_list(series):
        flat = []
        for v in series.dropna().tolist():
            if isinstance(v, list):
                for it in v:
                    if it not in flat:
                        flat.append(it)
            else:
                if v not in flat:
                    flat.append(v)
        return flat

    def unique_timestamps(series):
        """Collect unique timestamp-like values only, filtering out IPs or other stray tokens.

        Accepts pandas.Timestamp objects, ISO-like date/time strings, or lists containing those.
        Non-timestamp strings (e.g. IPs like '10.11.28.181') are ignored.
        Returns a list of ISO-8601 strings where possible.
        """
        flat = []
        date_re = re.compile(r"\d{4}-\d{2}-\d{2}")
        for v in series.dropna().tolist():
            candidates = v if isinstance(v, list) else [v]
            for it in candidates:
                if it is None:
                    continue
                # pandas Timestamp
                try:
                    if hasattr(it, "isoformat") and hasattr(it, "tzinfo"):
                        s = str(it.isoformat())
                    else:
                        s = str(it)
                except Exception:
                    s = str(it)
                if not s:
                    continue
                # accept if looks like a date (YYYY-MM-DD) or full ISO
                if date_re.search(s):
                    # normalize to plain ISO-like string
                    # attempt to parse to pandas Timestamp for consistent formatting
                    try:
                        ts = pd.to_datetime(s, errors="coerce")
                        if pd.notna(ts):
                            s = ts.isoformat()
                        else:
                            s = s
                    except Exception:
                        s = s
                    if s not in flat:
                        flat.append(s)
        return flat

    # Aggregate per template id (one row per template_id). For each template id
    # we select a representative semantic_text (the most frequent one) and
    # aggregate metadata columns as unique lists. This produces a canonical
    # scalar id per output row which downstream systems expect.
    agg_dict = {"occurrences": "sum"}
    for c in meta_cols:
        if c in merged.columns:
            if c == "timestamp":
                agg_dict[c] = lambda s, col=c: unique_timestamps(s)
            else:
                agg_dict[c] = (lambda s, col=c: unique_list(s))

    # Determine the most-frequent semantic_text for each template id
    counts = merged.groupby([id_col_single, semantic_col]).size().reset_index(name="count")
    counts = counts.sort_values([id_col_single, "count"], ascending=[True, False])
    primary_semantic = counts.drop_duplicates(id_col_single).set_index(id_col_single)[semantic_col].to_dict()

    # Group by the template id to produce one row per id
    grouped = merged.groupby(id_col_single).agg(agg_dict).reset_index()

    # Attach representative semantic_text and keep template_ids as a single-element list
    grouped[semantic_col] = grouped[id_col_single].map(lambda v: primary_semantic.get(v, ""))
    grouped["template_ids"] = grouped[id_col_single].apply(lambda v: [v] if pd.notna(v) and str(v) != "" else [])

    # Prefer Drain-provided occurrences (if available in the structured dataframe).
    # structured may contain an 'occurrences' column coming from Drain's cluster sizes
    # (e.g. from parsed_templates.csv). When present, prefer that as the canonical
    # occurrences value; otherwise fall back to the aggregated count from cleaning.
    try:
        if "occurrences" in struct_by_id.columns:
            occ_map = struct_by_id["occurrences"].to_dict()

            def _use_drain_occ(x):
                val = occ_map.get(x)
                if val is None:
                    return np.nan
                try:
                    return int(val)
                except Exception:
                    return val

            grouped["occurrences_drain"] = grouped[id_col_single].map(_use_drain_occ)
            # Replace occurrences with Drain values when available, else keep aggregated sum
            if "occurrences" in grouped.columns:
                grouped["occurrences"] = grouped["occurrences_drain"].where(pd.notna(grouped["occurrences_drain"]), grouped["occurrences"])
            else:
                grouped["occurrences"] = grouped["occurrences_drain"].fillna(0).astype(int)
            grouped = grouped.drop(columns=["occurrences_drain"])
    except Exception:
        # non-fatal: keep the existing aggregated occurrences if anything goes wrong
        logger.exception("Could not prefer Drain occurrences; keeping aggregated occurrences")

    # Reorder and persist
    cols = [semantic_col, id_col_single, "template_ids", "occurrences"] + [c for c in meta_cols if c in grouped.columns]
    aggregated = grouped[cols]
    aggregated.to_csv(output_path, index=False)

    # Diagnostics: report remap/missing stats and write sample unresolved ids
    try:
        diag = {}
        total_exploded = len(exploded)
        unique_exploded_ids = exploded[id_col_single].astype(str).nunique()
        mapped_mask = merged[primary_meta_col].notna() if primary_meta_col is not None else merged[id_col_single].notna()
        mapped_count = mapped_mask.sum()
        unresolved_ids = sorted(set(merged.loc[~mapped_mask, id_col_single].astype(str).unique()))
        diag["total_exploded_rows"] = int(total_exploded)
        diag["unique_exploded_ids"] = int(unique_exploded_ids)
        diag["mapped_rows"] = int(mapped_count)
        diag["unresolved_id_count"] = int(len(unresolved_ids))

        # write a small sample of unresolved ids to a diagnostics CSV next to output_path
        try:
            import json as _json
            out_dir = __import__("os").path.dirname(output_path)
            diag_path = __import__("os").path.join(out_dir, "merge_diagnostics.json")
            _json.dump(diag, open(diag_path, "w"), indent=2)
            # also dump a CSV sample if unresolved exist
            if unresolved_ids:
                sample_ids = unresolved_ids[:200]
                sample_path = __import__("os").path.join(out_dir, "unresolved_template_ids.csv")
                with open(sample_path, "w", encoding="utf-8") as fh:
                    fh.write("template_id\n")
                    for sid in sample_ids:
                        fh.write(f"{sid}\n")
        except Exception:
            # non-fatal diagnostics failure should not break merge
            pass
    except Exception:
        pass

    return aggregated
