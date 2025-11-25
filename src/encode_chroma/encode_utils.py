"""Helpers to build safe metadata dicts for vector store ingestion.

This module provides a small utility to convert a pandas DataFrame into a
list of metadata dicts suitable for adding to a vector DB collection. It
ensures datetime-like columns are serialized to ISO-8601 (UTC) strings so
downstream consumers can parse them reliably.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence
import logging

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas required for full functionality
    pd = None  # type: ignore
try:
    import numpy as np
except Exception:  # pragma: no cover - numpy optional here
    np = None

logger = logging.getLogger(__name__)


def df_to_metadatas(
    df,
    meta_cols: Optional[Sequence[str]] = None,
    timestamp_candidates: Sequence[str] = ("timestamp", "time", "ts", "created_at"),
    update_service_map: bool = False,
) -> List[Dict[str, Any]]:
    """Convert DataFrame rows to list of metadata dicts with ISO-8601 timestamps.

    Args:
        df: pandas DataFrame or any object supporting column access by name and
            row iteration (df.iloc[...] used when pandas is available).
        meta_cols: optional list of column names to include as metadata. If
            omitted we include a small safe default set (intersection with df).
        timestamp_candidates: column names that should be treated/serialized
            as timestamps when present.

    Returns:
        List of dicts where datetime-like values have been converted to
        strings in ISO-8601 Zulu (UTC) form when possible. NaNs/NaTs map to
        None.
    """
    if pd is None:
        raise RuntimeError("pandas is required for df_to_metadatas")

    # choose metadata columns
    if meta_cols is None:
        default_cols = [
            "template",
            "component",
            "doc_id",
            "id",
            "line_no",
            "orig_idx",
            "timestamp",
        ]
        meta_cols = [c for c in default_cols if c in df.columns]
    else:
        meta_cols = [c for c in meta_cols if c in df.columns]

    metas: List[Dict[str, Any]] = []

    for i in range(len(df)):
        row_meta: Dict[str, Any] = {}
        for col in meta_cols:
            val = df.iloc[i][col]
            # normalize pandas NaT / NaN -> None
            try:
                na = pd.isna(val)
                # only treat as scalar NA when pd.isna returns a boolean-like
                if isinstance(na, bool) or (np is not None and isinstance(na, np.bool_)):
                    if na:
                        row_meta[col] = None
                        continue
                # otherwise, pd.isna returned array-like (e.g., for Index/ndarray)
            except Exception:
                # not a pandas scalar
                pass

            # If this column looks like a timestamp candidate, try to parse
            if col in timestamp_candidates:
                try:
                    # handle list-like timestamp values (pandas Index/Series, lists, ndarrays)
                    if pd is not None and isinstance(val, (pd.Index, pd.Series)):
                        seq = list(val)
                    elif np is not None and isinstance(val, np.ndarray):
                        seq = val.tolist()
                    elif isinstance(val, (list, tuple)):
                        seq = list(val)
                    else:
                        seq = None

                    if seq is not None:
                        parsed_strs = []
                        for element in seq:
                                try:
                                    ts2 = pd.to_datetime(element, errors="coerce")
                                    if pd.isna(ts2):
                                        continue
                                    try:
                                        ts2 = ts2.tz_convert("UTC") if ts2.tzinfo is not None else ts2.tz_localize("UTC")
                                    except Exception:
                                        pass
                                    parsed_strs.append(ts2.strftime("%Y-%m-%dT%H:%M:%SZ"))
                                except Exception:
                                    # log and continue — some list elements may be non-datetime
                                    logger.debug("df_to_metadatas: could not parse timestamp element=%r for col=%s (row=%d)", element, col, i)
                                    continue
                        if len(parsed_strs) == 0:
                            row_meta[col] = None
                        elif len(parsed_strs) == 1:
                            row_meta[col] = parsed_strs[0]
                        else:
                            row_meta[col] = ", ".join(parsed_strs)
                    else:
                        ts = pd.to_datetime(val, errors="coerce")
                        if pd.isna(ts):
                            row_meta[col] = None
                        else:
                            # convert to UTC then format as ISO Zulu
                            try:
                                ts = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
                            except Exception:
                                # If already tz-aware or cannot localize, ignore
                                pass
                            row_meta[col] = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
                except Exception:
                    # fallback: stringify
                    logger.debug("df_to_metadatas: timestamp fallback stringify for col=%s row=%d val=%r", col, i, val)
                    row_meta[col] = str(val)
            else:
                # not a timestamp; coerce common types to python primitives
                # Handle list-like and pandas index/series types which can
                # appear after group/agg operations. Chroma expects metadata
                # values to be simple scalars (str/int/float/bool) or None.
                try:
                    # pandas Index/Series
                    if pd is not None and isinstance(val, (pd.Index, pd.Series)):
                        seq = list(val)
                        # flatten singletons
                        if len(seq) == 1:
                            row_val = seq[0]
                        else:
                            row_val = ", ".join([str(x) for x in seq])
                        logger.debug("df_to_metadatas: coerced pandas Index/Series for col=%s row=%d type=%s", col, i, type(val))
                        row_meta[col] = None if pd.isna(row_val) else (row_val if isinstance(row_val, (int, float, bool, str)) else str(row_val))
                        continue
                    # numpy arrays
                    if np is not None and isinstance(val, np.ndarray):
                        seq = val.tolist()
                        if len(seq) == 1:
                            row_val = seq[0]
                        else:
                            row_val = ", ".join([str(x) for x in seq])
                        logger.debug("df_to_metadatas: coerced numpy.ndarray for col=%s row=%d shape=%s", col, i, getattr(val, 'shape', None))
                        row_meta[col] = row_val if isinstance(row_val, (int, float, bool, str)) else str(row_val)
                        continue
                    # python lists/tuples
                    if isinstance(val, (list, tuple)):
                        if len(val) == 0:
                            row_meta[col] = None
                        elif len(val) == 1:
                            single = val[0]
                            logger.debug("df_to_metadatas: coerced list/tuple singleton for col=%s row=%d", col, i)
                            row_meta[col] = single if isinstance(single, (int, float, bool, str)) else str(single)
                        else:
                            logger.debug("df_to_metadatas: coerced list/tuple for col=%s row=%d len=%d", col, i, len(val))
                            row_meta[col] = ", ".join([str(x) for x in val])
                        continue

                    if isinstance(val, (int, float, bool, str)):
                        row_meta[col] = val
                    else:
                        # fallback to string representation for unknown types
                        logger.debug("df_to_metadatas: fallback stringify for col=%s row=%d type=%s", col, i, type(val))
                        row_meta[col] = None if val is None else str(val)
                except Exception:
                    # defensive fallback
                    logger.exception("df_to_metadatas: unexpected error coercing column=%s row=%d", col, i)
                    row_meta[col] = None if val is None else str(val)

        # additional: attempt to extract a service token/display and optionally
        # update a shared mapping file so the UI can show friendly names.
        try:
            from src.enrichment.labels import extract_service_name_from_component, update_service_map_with_token, _pretty_display_from_token

            comp = None
            for candidate in ("component", "service", "service_name", "component_name", "app", "name"):
                if candidate in row_meta and row_meta[candidate]:
                    comp = row_meta[candidate]
                    break

            token = extract_service_name_from_component(comp, prefer_two_level=True) if comp else None
            if token:
                # store token and a best-effort display
                row_meta["service_token"] = token
                # if update_service_map is requested, persist mapping; else just generate display
                if update_service_map:
                    disp = update_service_map_with_token(token)
                else:
                    disp = _pretty_display_from_token(token)
                row_meta["service_display"] = disp
        except Exception:
            # non-fatal; metadata remains as-is
            pass

        metas.append(row_meta)

    return metas
