"""Regex utilities and extraction helpers used by the log parsing pipeline.

This module centralizes compiled regexes and provides small, well-tested helpers
for extracting common fields from log content.

Keep the patterns conservative and provide a small validate_examples() helper
for unit tests.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Pattern

# Precompiled patterns for extra metadata fields (case-insensitive)
# Keep them conservative and test them in unit tests (see tests/unit/test_regex.py)
EXTRA_FIELD_PATTERNS: Dict[str, Pattern] = {
    "reqid": re.compile(r"\[req-([\w-]+)\b", re.IGNORECASE),
    "userid": re.compile(r"(?:user[-_]?id)\s*[:=]\s*([\w-]+)", re.IGNORECASE),
    "tenantid": re.compile(r"(?:tenant[-_]?id)\s*[:=]\s*([\w-]+)", re.IGNORECASE),
    "ip": re.compile(r"\b(25[0-5]|2[0-4]\d|1?\d{1,2})(?:\.(25[0-5]|2[0-4]\d|1?\d{1,2})){3}\b", re.IGNORECASE),
    "status": re.compile(r"\bstatus\s*[:=]\s*(\d{3})\b", re.IGNORECASE),
    "method": re.compile(r"\b(GET|POST|PUT|DELETE|PATCH|OPTIONS)\b", re.IGNORECASE),
    "url": re.compile(r"(https?://[^\s]+|/[^\s,;]+)", re.IGNORECASE),
    "responselength": re.compile(r"len\s*[:=]\s*(\d+)", re.IGNORECASE),
    "responsetime": re.compile(r"time\s*[:=]\s*([0-9]+\.?[0-9]*)", re.IGNORECASE),
    # Date and Time fragments for timestamp reconstruction
    "date": re.compile(r"(\d{4}-\d{2}-\d{2})", re.IGNORECASE),
    "time": re.compile(r"(\d{2}:\d{2}:\d{2}(?:\.\d+)?)", re.IGNORECASE),
}


def extract_fields(content: str) -> Dict[str, Optional[str]]:
    """Extract known fields from a log content string.

    This function applies a set of conservative, precompiled regular expressions
    to the provided `content` string and returns a mapping from field name to
    the captured value (or ``None`` when a pattern does not match).

    Args:
        content: The log message content to inspect.

    Returns:
        A dictionary whose keys are the same as ``EXTRA_FIELD_PATTERNS`` and
        whose values are the first captured group for each pattern or ``None``.

    Example:
        >>> extract_fields("[req-123] user_id=alice GET /api len:123")
        {"reqid": "123", "userid": "alice", "method": "GET", "url": "/api", ...}
    """
    if not content:
        return {k: None for k in EXTRA_FIELD_PATTERNS.keys()}

    out: Dict[str, Optional[str]] = {}
    for name, pattern in EXTRA_FIELD_PATTERNS.items():
        m = pattern.search(content)
        out[name] = m.group(1).strip() if m else None
    return out


def normalize_service_from_component(component: Optional[str], depth: int = 2) -> Optional[str]:
    """Derive a short service name from a dotted component string.

    The function extracts the first ``depth`` parts from a dotted component
    identifier and returns a normalized short name. Underscores are converted
    to hyphens to align with common service naming.

    Args:
        component: A dotted component string, e.g. ``"nova.virt.driver"``.
        depth: Number of leading parts to keep (minimum 1).

    Returns:
        A normalized service shortname (e.g. ``"nova.virt"``) or ``None``
        when ``component`` is falsy.
    """
    if not component:
        return None
    parts = [p for p in str(component).split(".") if p]
    depth = max(1, min(depth, len(parts)))
    return ".".join(parts[:depth]).replace("_", "-")


def extract_service_from_path(path_str: str) -> Optional[str]:
    """Extract service name from a file path or filename.

    Matches patterns like:
      - "nova-api.log" -> "nova-api"
      - "/var/log/nova/nova-compute.log" -> "nova-compute"
      - "neutron-server.log" -> "neutron-server"
    
    Handles quoted strings as well.

    Args:
        path_str: The file path or filename string.

    Returns:
        The extracted service name or None if no match found.
    """
    if not path_str:
        return None
    
    # Regex to capture the part before .log
    # Matches 'nova-api' in 'nova-api.log' or '/var/log/nova/nova-api.log'
    # Also handles quoted strings like "nova-api.log..."
    # (?:^|/|") : Start of string, forward slash, or quote
    # ([a-z0-9_-]+) : Capture group for service name (alphanumeric, underscore, dash)
    # \.log : Literal .log extension
    match = re.search(r'(?:^|/|")([a-z0-9_-]+)\.log', str(path_str))
    if match:
        return match.group(1)
    return None


# Small helper for unit tests to validate pattern behavior on examples
def validate_examples(examples: Dict[str, str]) -> Dict[str, Optional[str]]:
    """Apply :func:`extract_fields` to example inputs for quick validation.

    This helper is primarily used by unit tests to run multiple example strings
    through the same extraction logic and return the combined extraction map.
    """
    # flatten examples and apply extract_fields on values joined by space
    combined = " ".join(examples.values())
    return extract_fields(combined)
