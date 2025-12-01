"""Shared timestamp manipulation utilities.

This module centralizes logic for parsing, normalizing, and selecting canonical
timestamps from various input formats (lists, pandas Series, strings).
"""
from __future__ import annotations

from typing import Any, List, Optional, Union, Sequence
import logging

try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore

from config.settings import settings

logger = logging.getLogger(__name__)


def parse_timestamp_sequence(val: Any) -> List[Any]:
    """Convert a value (list, string, Series, scalar) into a list of pandas Timestamps.

    Returns a list of valid, UTC-localized pandas.Timestamp objects.
    """
    if pd is None:
        return []
    
    if val is None:
        return []

    seq = []
    # Handle various container types
    if isinstance(val, (list, tuple)):
        seq = list(val)
    elif isinstance(val, (pd.Index, pd.Series)):
        seq = list(val)
    elif isinstance(val, str) and "," in val:
        seq = [s.strip() for s in val.split(",") if s.strip()]
    else:
        # Treat as scalar
        seq = [val]

    parsed = []
    for element in seq:
        try:
            # Coerce to UTC timestamp
            ts = pd.to_datetime(element, utc=True, errors="coerce")
            if not pd.isna(ts):
                parsed.append(ts)
        except Exception:
            continue
    
    return parsed


def get_canonical_timestamp(val: Any, policy: Optional[str] = None) -> Optional[Any]:
    """Select a single representative timestamp from a value or collection.

    Args:
        val: A scalar timestamp, string, or collection of them.
        policy: Selection policy ('median', 'latest', 'earliest', 'first').
                If None, defaults to settings.DEFAULT_TIMESTAMP_POLICY.

    Returns:
        A pandas.Timestamp (UTC) or None if no valid timestamps found.
    """
    if pd is None:
        return None

    parsed = parse_timestamp_sequence(val)
    if not parsed:
        return None

    if policy is None:
        policy = getattr(settings, "DEFAULT_TIMESTAMP_POLICY", "median")

    if policy == "latest":
        return max(parsed)
    elif policy == "earliest":
        return min(parsed)
    elif policy == "first":
        # 'first' implies original order, but parse_timestamp_sequence flattens 
        # and filters. If order matters strictly, we need to re-parse the 
        # original sequence one by one until a match is found.
        # For efficiency, we'll just take the first valid one we found.
        return parsed[0]
    
    # Default: median
    parsed_sorted = sorted(parsed)
    m = len(parsed_sorted)
    if m % 2 == 1:
        return parsed_sorted[m // 2]
    
    # Even number: midpoint between two middle values
    t0 = parsed_sorted[m // 2 - 1].value
    t1 = parsed_sorted[m // 2].value
    mid = (int(t0) + int(t1)) // 2
    return pd.to_datetime(mid, unit="ns", utc=True)


def to_iso_string(ts: Any) -> Optional[str]:
    """Safe helper to convert a pandas Timestamp to an ISO-8601 string (UTC)."""
    if ts is None:
        return None
    try:
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return str(ts)
