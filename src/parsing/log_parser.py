"""Log parser module extracted from the notebook.

Provides a LogParser class which performs a single-line parse, metadata extraction
and returns a normalized dictionary for downstream processing. The class takes a
`template_miner` instance (Drain3 TemplateMiner) as a dependency so it can be
unit-tested with a mock.

This module focuses on correctness and validation (Phase 1).
"""
from __future__ import annotations

import re
from typing import Dict, Optional
import pandas as pd
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence
from drain3 import TemplateMiner

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


class LogParser:
    """Stateless line parser that enriches with extracted metadata.

    Usage:
        parser = LogParser(log_format=..., template_miner=template_miner)
        row = parser.process_line(raw_line, line_no)
    """

    def __init__(self, log_format: Optional[str] = None, template_miner: Optional[TemplateMiner] = None):
        self.log_format = log_format
        self.template_miner = template_miner

    def process_line(self, raw_line: str, line_no: int) -> Dict[str, Optional[object]]:
        """Parse one log line and return a normalized dict with metadata and template info.

        The returned dictionary contains a minimal, stable set of keys that can
        be persisted to CSV or further enriched. Keys include: ``line_no``,
        ``raw``, ``content``, ``template_id``, ``template``, ``service``,
        ``component``, ``timestamp`` and any extraction keys (e.g. ``reqid``).

        Args:
            raw_line: The raw log line string.
            line_no: Line number (useful for tracing back to source file).

        Returns:
            A dict with parsed/enriched fields.
        """
        parsed_meta = parse_line_with_format(raw_line, self.log_format)
        content = parsed_meta.get("Content") or raw_line
        # prefer lowercase metadata keys produced by the extractor; fallback to
        # canonical 'Component' when present for compatibility
        component = parsed_meta.get("component") or parsed_meta.get("Component")

        # extract extra fields from content
        extracted = extract_fields(content)

        # attempt to get service name from component when missing
        service = extracted.get("service") if isinstance(extracted.get("service"), str) else None
        if not service:
            service = normalize_service_from_component(component)

        # timestamp validation: try to read Date and Time if provided
        ts = None
        date_str = parsed_meta.get("Date")
        time_str = parsed_meta.get("Time")
        if date_str and time_str:
            try:
                ts = pd.to_datetime(f"{date_str} {time_str}", errors="coerce")
                if pd.isna(ts):
                    ts = None
            except Exception:
                ts = None

        # template mining (if available)
        template_mined = None
        cluster_id = None
        if self.template_miner is not None:
            try:
                res = self.template_miner.add_log_message(content)
                template_mined = res.get("template_mined")
                cluster_id = res.get("cluster_id")
            except Exception:
                # don't fail parsing due to template miner errors; include error info
                template_mined = None
                cluster_id = None

        row = {
            "line_no": line_no,
            "raw": raw_line.rstrip("\n"),
            "content": content,
            "template_id": cluster_id,
            "template": template_mined,
            # normalized metadata
            "service": service,
            "component": component,
            "timestamp": ts,
        }

        # add extracted fields (converted to lower-case keys)
        for k, v in extracted.items():
            row[k] = v
        return row
