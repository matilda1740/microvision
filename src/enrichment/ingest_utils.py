"""Compatibility wrapper: re-export ingest utilities from src.ingest.

The implementation was moved to ``src.ingest.ingest_utils``. This thin
wrapper keeps older imports working while the real implementation lives
outside of the enrichment package.
"""
from __future__ import annotations

from src.encode_chroma.encode_utils import df_to_metadatas  # re-export

__all__ = ["df_to_metadatas"]
