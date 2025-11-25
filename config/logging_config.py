"""Central logging configuration helper used by runners and scripts.

Provide a tiny, testable helper to configure root logging consistently and
make it easy to enable DEBUG traces from the CLI with a --debug flag.
"""
from __future__ import annotations

import logging
from typing import Optional


def configure_logging(debug: bool = False, level: Optional[str] = None) -> None:
    """Configure root logging for the process.

    Args:
        debug: if True, set level to DEBUG. Otherwise uses `level` or INFO.
        level: optional explicit level name (e.g. 'INFO', 'DEBUG').
    """
    fmt = "%(asctime)s %(levelname)-5s [%(name)s] %(message)s"
    if debug:
        lvl = logging.DEBUG
    elif level is not None:
        lvl = getattr(logging, level.upper(), logging.INFO)
    else:
        lvl = logging.INFO

    root = logging.getLogger()
    # avoid double-configuring when called multiple times
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(handler)
    root.setLevel(lvl)
