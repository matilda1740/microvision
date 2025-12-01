"""Helpers for extracting lightweight metadata from free-text log content.

This module centralizes regex patterns and metadata-building logic so the
main MetadataDrainParser can remain focused on parsing and template mining.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pandas as pd

from src.parsing.regex_utils import normalize_service_from_component, extract_fields

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


def is_numeric_like(x: Any) -> bool:
    """Check if a value looks like a number (float or int)."""
    try:
        if x is None:
            return False
        # handle lists/arrays conservatively
        if isinstance(x, (list, tuple)):
            return False
        float(str(x))
        return True
    except Exception:
        return False


def extract_and_build_metadata(
    content: str,
    component: Optional[str] = None,
    base_row: Optional[pd.Series] = None,
    metadata_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Extract metadata from a free-text content string and merge with a base row.

    Returns a dict with lower-cased keys only (e.g. 'service', 'reqid', 'ip').
    """
    # 1) regex-based extraction (lowercase keys)
    meta = extract_fields(content)
    # filter out None values
    meta = {k: v for k, v in meta.items() if v is not None}

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
        try:
            ts = pd.to_datetime(f"{date_str} {time_str}", errors="coerce")
            if pd.notna(ts):
                meta["timestamp"] = str(ts.isoformat())
            else:
                meta["timestamp"] = None
        except Exception:
            meta["timestamp"] = None
    else:
        meta["timestamp"] = None

    # 4) service shortname from component
    if "service" not in meta and component:
        service = normalize_service_from_component(component)
        if service:
            meta["service"] = service

    return meta
