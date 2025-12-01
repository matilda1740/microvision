"""Cleaning and semantic grouping utilities for the enrichment pipeline.

This module centralizes the conservative cleaning/grouping logic previously
defined inline in `scripts/run_full_pipeline.py` so other runners can reuse it.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd


def clean_template(text: str, preserve_symbols: str = ":=0123456789") -> str:
    """Normalize a template string according to the provided rules.

    This implements the exact behaviour you supplied:
    - remove existing "<*>" placeholders
    - normalize UUIDs -> "<*>"
    - normalize long hex-like ids (20+ hex chars) -> "<>*"
    - remove unwanted symbols except those in `preserve_symbols`, plus / . -
    - collapse whitespace and lowercase
    """
    if text is None:
        return ""
    # 1) Normalize tokens inside angle brackets that are NOT the <*> placeholder
    #    e.g. turn '<REQ_ID>' -> 'req_id' but keep '<*>' as-is for the placeholder
    def _replace_angle(m: re.Match) -> str:
        inner = m.group(1)
        if inner is None:
            return ""
        inner_s = inner.strip()
        if inner_s == "*":
            return "<*>"
        # sanitize inner token: keep word chars, hyphen and underscore, then lowercase
        token = re.sub(r"[^\w-]", "", inner_s).strip().lower()
        # convert tokens like REQ_ID or REQ-123 -> req_id
        token = re.sub(r"[-]+", "_", token)
        return token

    text = re.sub(r"<\s*([^>]+?)\s*>", _replace_angle, text)

    # 2) Mask common timestamp patterns (ISO date and datetime variants)
    #    e.g. 2017-05-16 or 2017-05-16 13:53:08 or 2017-05-16_13:53:08
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}(?:[ _T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?\b", "<TS>", text)

    # 3) REMOVE ALL PLACEHOLDER PATTERNS (we replaced <*> above if present)
    text = re.sub(r"<\*>", "<TS>", text)

    # 4) NORMALIZE UUIDS (keep as placeholder)
    text = re.sub(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", "<ID>", text, flags=re.IGNORECASE)
    # 5) NORMALIZE OTHER ALPHANUMERIC IDS (long hex-like)
    text = re.sub(r"\b[0-9a-f]{20,}\b", "<ID>", text, flags=re.IGNORECASE)

    # 5b) Strip common log filename rotation / suffixes: keep 'foo.log' from 'foo.log.1.ts'
    text = re.sub(r"(\b[\w\-.]+\.log)(?:\.\d+)?(?:\.ts)?\b", r"\1", text)

    # 5c) Replace IPv4 addresses with a generic placeholder
    text = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<IP>", text)

    # 6) Normalize 'req' patterns: req-1234, req- - - -, req_foo -> req_id
    text = re.sub(r"\breq(?:[-_\s]*[0-9a-zA-Z-_]*)*\b", "req_id", text, flags=re.IGNORECASE)

    # 7) Mask long standalone integer tokens that look like PIDs (3-6 digits)
    text = re.sub(r"\b\d{3,6}\b", "<PID>", text)

    # 8) REMOVE ALL UNWANTED SYMBOLS EXCEPT THE ALLOWED SYMBOLS
    pattern = rf"[^\w\s{re.escape(preserve_symbols)}/.-]"
    text = re.sub(pattern, "", text)

    # 8b) Simplify HTTP/version tokens and API 'id' path fragments: http/1.1 -> http, /v2/id/ -> /v2/
    text = re.sub(r"http/\d+(?:\.\d+)*", "http", text, flags=re.IGNORECASE)
    text = re.sub(r"/v(\d+)/id/", r"/v\1/", text)

    # 8c) Collapse repeated common placeholders/tokens (e.g. 'ts ts ts' -> 'ts')
    #      This uses a conservative whitelist so we don't collapse meaningful repeats.
    text = re.sub(r"\b(?P<t>ts|pid|id|req_id|<TS>|<PID>|<ID>|<IP>)\b(?:\s+(?P=t))+", r"\1", text, flags=re.IGNORECASE)

    # 8d) Collapse repeated occurrences of '- - -' and similar hyphen runs to single '-'
    text = re.sub(r"(?:-\s*){2,}", "-", text)

    # 8e) Collapse multiple adjacent <IP> placeholders into a single <IP>
    text = re.sub(r"(<IP>)(?:[\s,]+\1)+", r"\1", text, flags=re.IGNORECASE)

    # 9) Collapse repeated hyphens/underscores and whitespace
    text = re.sub(r"[-_]{2,}", "_", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text.lower()



def minimal_cleaning(cleaned_rows_df: pd.DataFrame) -> pd.DataFrame:
    """Conservative cleaning/semantic grouping.

    Produces rows with 'semantic_text', 'template_ids' and 'occurrences'.
    Uses `clean_template` for normalization so cleaned artifacts match the
    provided canonical normalizer.
    """
    df = cleaned_rows_df.copy()
    if "template" not in df.columns and "content" in df.columns:
        df = df.rename(columns={"content": "template"})

    def norm(s: Any) -> str:
        if s is None or (isinstance(s, float) and pd.isna(s)):
            return ""
        return clean_template(str(s))

    df["_norm"] = df.get("template", "").apply(norm) if "template" in df.columns else df.apply(lambda r: norm(" ".join([str(v) for v in r.values()])), axis=1)

    groups = df.groupby("_norm")
    out_rows = []
    for name, g in groups:
        semantic_text = name if name else (g.get("template").dropna().iloc[0] if "template" in g.columns and not g.get("template").dropna().empty else "")
        template_ids = []
        if "template_id" in g.columns:
            for v in g["template_id"].dropna().unique().tolist():
                if v is None:
                    continue
                s = str(v).strip()
                if s not in template_ids:
                    template_ids.append(s)
        
        # Preserve service and component metadata
        # We take the most frequent value (mode) or the first non-null
        row_data = {"semantic_text": semantic_text, "template_ids": template_ids, "occurrences": len(g)}
        
        for col in ["service", "component"]:
            if col in g.columns:
                # Get most common value, ignoring NaNs
                modes = g[col].dropna().mode()
                if not modes.empty:
                    row_data[col] = modes[0]
                else:
                    row_data[col] = None
                    
        out_rows.append(row_data)

    out_df = pd.DataFrame(out_rows)
    return out_df
