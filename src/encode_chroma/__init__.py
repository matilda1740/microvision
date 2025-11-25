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
