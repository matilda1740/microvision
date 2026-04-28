"""Utilities to extract a sane service name / label from metadata.

The OpenStack logs tend to use a "component" field with dotted names like
"nova.compute.manager" or "neutron.agent.linux.interface". This module
provides heuristics to extract a stable, human-friendly service label from
those fields with fallbacks for other logging formats.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

try:
    from pathlib import Path
    _MAP_PATH = Path(__file__).parent / "service_name_map.json"
    if _MAP_PATH.exists():
        with open(_MAP_PATH, "r", encoding="utf-8") as fh:
            _DISPLAY_MAP: Dict[str, str] = json.load(fh)
    else:
        _DISPLAY_MAP = {}
except Exception:
    _DISPLAY_MAP = {}


_SEPARATORS_RE = re.compile(r"[\s\/:|\\]+|\.|_")


def _normalize_token(tok: str) -> str:
    tok = tok.strip()
    tok = tok.lower()
    # remove common noise like trailing module suffixes
    tok = re.sub(r"\b(module|manager|handler|service|api)\b$", "", tok)
    tok = tok.strip("._- ")
    return tok


def extract_service_name_from_component(component: Optional[str], prefer_two_level: bool = False) -> Optional[str]:
    """Extract a canonical service name from component-like strings.

    Heuristics:
    - Split on dots and common separators.
    - Prefer the first token as the primary service (e.g. 'nova' from 'nova.compute.manager').
    - If prefer_two_level=True and there are at least two tokens, join first two
      tokens as 'service.subservice' (e.g. 'nova.compute').
    - Strip common noise words and return None for empty inputs.
    """
    if not component:
        return None

    if isinstance(component, bytes):
        try:
            component = component.decode("utf-8", errors="ignore")
        except Exception:
            component = str(component)

    comp = str(component)
    comp = comp.strip()
    if not comp:
        return None

    # If the component looks like JSON (some logs embed JSON), try to parse
    if comp.startswith("{") and comp.endswith("}"):
        try:
            parsed = json.loads(comp)
            # try common fields
            for k in ("component", "service", "name", "app", "service_name"):
                if k in parsed and parsed[k]:
                    return extract_service_name_from_component(parsed[k], prefer_two_level=prefer_two_level)
        except Exception:
            # fallback to plain text parsing
            pass

    # Split on dots and other separators
    parts = [p for p in _SEPARATORS_RE.split(comp) if p]
    parts = [_normalize_token(p) for p in parts if p and _normalize_token(p)]
    if not parts:
        return None

    if prefer_two_level and len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"

    return parts[0]


def extract_service_label_from_metadata(meta: Any, prefer_two_level: bool = False) -> Optional[str]:
    """Given a metadata object (dict or JSON string), attempt to determine a
    human-friendly service label.

    This tries several common metadata keys and falls back to parsing a
    'component' field when available.
    """
    if meta is None:
        return None

    # If a JSON string was stored, try to load it
    if isinstance(meta, str):
        try:
            parsed = json.loads(meta)
            meta_obj: Dict[str, Any] = parsed if isinstance(parsed, dict) else {}
        except Exception:
            # not JSON, treat as raw component string
            return extract_service_name_from_component(meta, prefer_two_level=prefer_two_level)
    elif isinstance(meta, dict):
        meta_obj = meta
    else:
        # unknown type, stringify and try to parse
        return extract_service_name_from_component(str(meta), prefer_two_level=prefer_two_level)

    # Check common keys
    for key in ("service", "service_name", "component", "component_name", "app", "name"):
        val = meta_obj.get(key)
        if val:
            token = extract_service_name_from_component(val, prefer_two_level=prefer_two_level)
            if token:
                # prefer display map when available
                disp = _DISPLAY_MAP.get(token)
                return disp if disp is not None else token

    # fallback: inspect all values and pick first plausible candidate
    for v in meta_obj.values():
        if isinstance(v, str) and "." in v:
            # a dotted token looks promising
            token = extract_service_name_from_component(v, prefer_two_level=prefer_two_level)
            if token and token in _DISPLAY_MAP:
                return _DISPLAY_MAP[token]
            return token

    return None


__all__ = [
    "extract_service_name_from_component",
    "extract_service_label_from_metadata",
]


def _pretty_display_from_token(token: str) -> str:
    """Create a human-friendly display name from a token like 'nova.compute'.

    Examples:
      nova -> 'Nova'
      nova.compute -> 'Nova Compute'
    """
    if not token:
        return token
    parts = [p for p in token.replace("/", ".").split(".") if p]
    parts = [p.replace("_", " ").replace("-", " ") for p in parts]
    parts = [p.title() for p in parts]
    return " ".join(parts)


def update_service_map_with_token(token: str, map_path: Optional[str] = None, display: Optional[str] = None) -> Optional[str]:
    """Ensure the mapping file contains a display name for token; if absent,
    generate a pretty display name and append it to the JSON map on disk.

    Returns the display name ensured for the token, or None on failure.
    """
    if not token:
        return None

    try:
        from pathlib import Path
        import os
        import tempfile

        # Determine map path: explicit arg -> env var -> config settings -> default next to module
        if map_path:
            path = Path(map_path)
        else:
            env_map = os.environ.get("SERVICE_NAME_MAP_PATH")
            if env_map:
                path = Path(env_map)
            else:
                # prefer centralized config.settings
                try:
                    from config import settings as _settings

                    path = Path(getattr(_settings, "SERVICE_NAME_MAP_PATH", Path(__file__).parent / "service_name_map.json"))
                except Exception:
                    path = Path(__file__).parent / "service_name_map.json"
        audit_path = path.with_suffix(".audit.log")
        lock_path = path.with_suffix(".lock")

        # Ensure parent exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Read existing map
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data: Dict[str, str] = json.load(fh)
        except Exception:
            data = {}

        if token in data:
            return data[token]

        # decide display name: explicit display takes precedence, otherwise pretty-generate
        disp = display if display is not None else _pretty_display_from_token(token)
        data[token] = disp

        # Attempt a safe, atomic write with advisory lock where available
        try:
            # Acquire lock if possible (Unix) or use portalocker for cross-platform
            lock_fd = None
            _use_fcntl = False
            _use_portalocker = False
            try:
                import fcntl

                _use_fcntl = True
            except Exception:
                _use_fcntl = False

            if not _use_fcntl:
                try:
                    import portalocker  # type: ignore

                    _use_portalocker = True
                except Exception:
                    _use_portalocker = False

            if _use_fcntl:
                import fcntl

                lock_fd = open(str(lock_path), "w+")
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
            elif _use_portalocker:
                import portalocker  # type: ignore

                lock_fd = open(str(lock_path), "w+")
                try:
                    portalocker.lock(lock_fd, portalocker.LOCK_EX)
                except Exception:
                    # fallback to no lock
                    pass
            else:
                lock_fd = None

            # write to temp file then rename
            dirpath = str(path.parent)
            fd, tmpname = tempfile.mkstemp(prefix=path.name, dir=dirpath)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as tmpf:
                    json.dump(data, tmpf, indent=2, ensure_ascii=False)
                    tmpf.flush()
                    os.fsync(tmpf.fileno())
                os.replace(tmpname, str(path))
            finally:
                # ensure tmp removed if exists
                try:
                    if os.path.exists(tmpname):
                        os.remove(tmpname)
                except Exception:
                    pass

            # append audit record
            try:
                with open(audit_path, "a", encoding="utf-8") as ah:
                    from datetime import datetime

                    ah.write(f"{datetime.utcnow().isoformat()}Z\t{token}\t{disp}\n")
            except Exception:
                # non-fatal
                pass

        finally:
            try:
                if lock_fd is not None:
                    try:
                        # try to unlock both fcntl and portalocker
                        try:
                            import fcntl

                            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
                        except Exception:
                            pass
                        try:
                            import portalocker  # type: ignore

                            try:
                                portalocker.unlock(lock_fd)
                            except Exception:
                                pass
                        except Exception:
                            pass
                    except Exception:
                        pass
                    try:
                        lock_fd.close()
                    except Exception:
                        pass
                    try:
                        if lock_path.exists():
                            lock_path.unlink()
                    except Exception:
                        pass
            except Exception:
                pass

        return disp
    except Exception:
        return None
