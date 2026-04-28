"""Log parser module extracted from the notebook.

Provides a LogParser class which performs a single-line parse, metadata extraction
and returns a normalized dictionary for downstream processing. The class takes a
`template_miner` instance (Drain3 TemplateMiner) as a dependency so it can be
unit-tested with a mock.

This module focuses on correctness and validation (Phase 1).
"""
from __future__ import annotations

import re
from .regex_utils import extract_fields, normalize_service_from_component


def log_format_to_regex(log_format: str) -> str:
    """Convert a LogPai-style format string into a compiled regex string.

    The returned regex uses named capture groups for each token found in the
    format string. The function preserves a loose match for the ``Content``
    token (greedy) and uses non-greedy matches for other tokens.

    Args:
        log_format: A format string with tokens like ``"<Date> <Time> <Content>"``.

    Returns:
        A string containing the regular expression to match lines of that format.
    """
    tokens = re.findall(r"<([^>]+)>", log_format)
    regex = re.escape(log_format)
    for t in tokens:
        esc = re.escape(f"<{t}>")
        if t.lower() == "content":
            repl = rf"(?P<{t}>.*)"
        else:
            repl = rf"(?P<{t}>.+?)"
        regex = regex.replace(esc, repl, 1)
    regex = regex.replace(r"\ ", r"\s+")
    return rf"^{regex}$"


def parse_line_with_format(line: str, log_format: Optional[str]) -> Dict[str, Optional[str]]:
    """Parse a single log line with the provided log format.

    Args:
        line: Raw log line text.
        log_format: The LogPai-style format string. When ``None`` the function
            will return a simple dict with the original content.

    Returns:
        A dict of parsed named groups. On parse failure returns a dict with
        ``{"ParseError": True, "Raw": line, "Content": line}``.
    """
    if not log_format:
        return {"Content": line}
    regex = log_format_to_regex(log_format)
    m = re.match(regex, line.strip())
    if not m:
        return {"ParseError": True, "Raw": line, "Content": line}
    return {k: v for k, v in m.groupdict().items()}

