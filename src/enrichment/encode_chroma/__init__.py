"""Compatibility shim: re-export the encode_chroma package moved to `src.encode_chroma`.

Some users and scripts import `src.enrichment.encode_chroma.*`. During a
refactor the package was moved to `src/encode_chroma`. To avoid breaking many
call sites, this module re-exports the main symbols from the new location.
"""
from __future__ import annotations

try:  # local import from new location
    from src.encode_chroma.encode_utils import df_to_metadatas  # type: ignore
    from src.encode_chroma.chroma import ingest_to_chroma_atomic, load_chroma_collection  # type: ignore
    from src.encode_chroma.encoder import encode_templates  # type: ignore

    __all__ = [
        "df_to_metadatas",
        "ingest_to_chroma_atomic",
        "load_chroma_collection",
        "encode_templates",
    ]
except Exception:  # pragma: no cover - best-effort shim, allow import errors to surface elsewhere
    # falling back to empty exports keeps import-time failures clearer in callers
    __all__ = []
"""Encoding helpers and Chroma utilities.

This package contains helpers to encode semantic templates, cache embeddings,
and ingest/load collections from Chroma. Designed to be lazy about heavy
dependencies so unit tests that don't touch embeddings remain fast.
"""
from .encoder import encode_templates
from .chroma import ingest_to_chroma_atomic, load_chroma_collection

__all__ = [
    "encode_templates",
    "ingest_to_chroma_atomic",
    "load_chroma_collection",
]
