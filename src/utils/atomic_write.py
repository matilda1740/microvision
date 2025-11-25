"""Atomic write helpers.

Provide small helpers to write files atomically by writing to a temporary
file in the same directory and then moving it into place with os.replace.
This ensures that readers never see half-written files.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:
    np = None


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    _ensure_parent(path)
    # write to a temporary file in same directory then replace
    fd, tmp = tempfile.mkstemp(dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    _ensure_parent(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(text)
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def atomic_write_json(path: Path, obj: Any, indent: int = 2) -> None:
    text = json.dumps(obj, indent=indent)
    atomic_write_text(path, text)


def atomic_save_npy(path: Path, arr) -> None:
    """Save numpy array atomically.

    Uses a temporary file in the same directory and os.replace to make the
    save atomic.
    """
    _ensure_parent(path)
    # If numpy isn't available, fall back to writing repr
    if np is None:
        atomic_write_text(path, repr(arr))
        return

    fd, tmp = tempfile.mkstemp(dir=str(path.parent))
    try:
        # use file descriptor to let numpy write binary data
        with os.fdopen(fd, "wb") as fh:
            np.save(fh, arr)
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def atomic_write_csv(path: Path, df) -> None:
    """Write a pandas DataFrame to CSV atomically.

    The df may be a pandas DataFrame with a to_csv() method.
    """
    _ensure_parent(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            df.to_csv(fh, index=False)
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass
