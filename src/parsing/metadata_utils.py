"""Helpers for extracting lightweight metadata from free-text log content.

This module centralizes regex patterns and metadata-building logic so the
main MetadataDrainParser can remain focused on parsing and template mining.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pandas as pd

# Default field configuration used when none supplied
FIELD_CONFIG = {
    "core_fields": ["Component", "Level", "Method", "URL"],
    "enrich_fields": [
        "ReqID",
        "UserID",
        "TenantID",
        "IP",
        "Status",
        "ResponseLength",
        "ResponseTime",
        "Service",
    ],
    "metadata_fields": [
        "Component",
        "Level",
        "Pid",
        "ReqID",
        "UserID",
        "TenantID",
        "IP",
        "Status",
        "Method",
        "URL",
        "ResponseLength",
        "ResponseTime",
        "Service",
    ],
}

# Cached regexes for extracting common fields from free-text content
EXTRA_FIELD_PATTERNS = {
    # date and time fragments commonly found in log lines (ISO-like)
    # these are captured separately so the timestamp reconstructor can combine them
    "Date": re.compile(r"(\d{4}-\d{2}-\d{2})", re.IGNORECASE),
    "Time": re.compile(r"(\d{2}:\d{2}:\d{2}(?:\.\d+)?)", re.IGNORECASE),
    "ReqID": re.compile(r"\[req-([\w-]+)\b", re.IGNORECASE),
    "UserID": re.compile(r"(?:user[-_]?id)\s*[:=]\s*([\w-]+)", re.IGNORECASE),
    "TenantID": re.compile(r"(?:tenant[-_]?id)\s*[:=]\s*([\w-]+)", re.IGNORECASE),
    "IP": re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b", re.IGNORECASE),
    "Status": re.compile(r"(?:status)\s*[:=]\s*(\d{3})", re.IGNORECASE),
    "Method": re.compile(r"\b(GET|POST|PUT|DELETE|PATCH|OPTIONS)\b", re.IGNORECASE),
    "URL": re.compile(r"(https?://[^\s]+|/[\w./-]+)", re.IGNORECASE),
    "ResponseLength": re.compile(r"len[:=]\s*(\d+)", re.IGNORECASE),
    "ResponseTime": re.compile(r"time[:=]\s*([\d\.]+)", re.IGNORECASE),
}


def extract_and_build_metadata(
    content: str,
    component: Optional[str] = None,
    base_row: Optional[pd.Series] = None,
    metadata_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Extract metadata from a free-text content string and merge with a base row.

    Returns a dict with lower-cased keys only (e.g. 'service', 'reqid', 'ip').
    """
    meta: Dict[str, Any] = {}

    # 1) regex-based extraction (lowercase keys)
    for field, pattern in EXTRA_FIELD_PATTERNS.items():
        try:
            match = pattern.search(content)
        except Exception:
            match = None
        if match:
            meta[field.lower()] = match.group(1).strip()

    # 2) merge base_row metadata if provided (prefer extracted values)
    if base_row is not None and metadata_fields:
        for f in metadata_fields:
            if f in base_row and pd.notna(base_row[f]):
                key = f.lower()
                if key not in meta:
                    meta[key] = base_row[f]

    # 3) timestamp reconstruction
    # Accept both lower- and upper-cased base_row keys since callers may
    # canonicalize keys to lowercase before passing the series.
    def _get_base(key):
        if base_row is None:
            return ""
        # try exact, then lowercase, then capitalized
        for k in (key, key.lower(), key.capitalize()):
            try:
                if k in base_row and pd.notna(base_row[k]):
                    return base_row[k]
            except Exception:
                continue
        return ""

    date_str = meta.get("date") or _get_base("Date")
    time_str = meta.get("time") or _get_base("Time")
    date_str = str(date_str).strip()
    time_str = str(time_str).strip()
    if date_str and time_str:
        # Be flexible with time formats: try a forgiving parse first so we accept
        # timestamps with or without fractional seconds. Normalize to UTC ISO string.
        ts = pd.to_datetime(f"{date_str} {time_str}", errors="coerce", utc=True)
        if pd.notna(ts):
            try:
                meta["timestamp"] = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                meta["timestamp"] = str(ts.isoformat())
        else:
            meta["timestamp"] = None
    else:
        meta["timestamp"] = None

    # 4) service shortname from component
    if "service" not in meta and component:
        depth = 2
        parts = component.split('.') if isinstance(component, str) else []
        selected = parts[: min(depth, len(parts))]
        service = ".".join(selected) if selected else component
        meta["service"] = str(service).replace('_', '-')

    return meta
