"""Edge store utilities: simple SQLite writer for incremental edges.

Provides a small EdgeStore class that initializes an SQLite database (WAL)
and writes edge dicts in batches. Designed to work with the streaming
generator produced by `compute_candidate_edges_stream`.
"""
from __future__ import annotations

import sqlite3
import json
from typing import Iterable, Dict, Any, Optional
import logging
try:
    import numpy as np
except Exception:
    np = None

logger = logging.getLogger(__name__)


class EdgeStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def init_db(self) -> None:
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = self.conn.cursor()
        # enable WAL for better concurrency
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_index INTEGER,
                source_id TEXT,
                target_id TEXT,
                source_timestamp TEXT,
                target_timestamp TEXT,
                time_delta_ms REAL,
                retrieval_distance REAL,
                retrieval_similarity REAL,
                semantic_cosine REAL,
                hybrid_score REAL,
                alpha REAL,
                target_metadata TEXT,
                created_at DATETIME DEFAULT (CURRENT_TIMESTAMP)
            )
            """
        )
        self.conn.commit()

    def write_edges(self, edges: Iterable[Dict[str, Any]], batch_size: int = 100) -> int:
        """Write edge dicts into the database. Returns number of rows written."""
        if self.conn is None:
            self.init_db()
        cur = self.conn.cursor()
        inserted = 0
        batch = []
        for e in edges:
            target_meta = e.get("target_metadata")
            try:
                meta_txt = json.dumps(target_meta, ensure_ascii=False) if target_meta is not None else None
            except Exception:
                meta_txt = str(target_meta)
            # coerce common numpy/pandas scalar types to plain python scalars
            def _coerce_val(x):
                if x is None:
                    return None
                # plain python scalars
                if isinstance(x, (int, float, str, bool)):
                    return x
                try:
                    if np is not None:
                        if isinstance(x, np.integer):
                            return int(x)
                        if isinstance(x, np.floating):
                            return float(x)
                        if isinstance(x, np.bool_):
                            return bool(x)
                        if isinstance(x, np.ndarray):
                            logger.debug("edge_store: coercing numpy.ndarray field to JSON string; shape=%s", getattr(x, "shape", None))
                            return json.dumps(x.tolist(), ensure_ascii=False)
                except Exception:
                    pass
                # dict/list-like -> JSON string
                try:
                    logger.debug("edge_store: coercing field of type %s to JSON", type(x))
                    return json.dumps(x, ensure_ascii=False)
                except Exception:
                    logger.debug("edge_store: falling back to str() for type %s", type(x))
                    return str(x)

            row = (
                _coerce_val(e.get("source_index")),
                _coerce_val(e.get("source_id")),
                _coerce_val(e.get("target_id")),
                _coerce_val(e.get("source_timestamp")),
                _coerce_val(e.get("target_timestamp")),
                _coerce_val(e.get("time_delta_ms")),
                _coerce_val(e.get("retrieval_distance")),
                _coerce_val(e.get("retrieval_similarity")),
                _coerce_val(e.get("semantic_cosine")),
                _coerce_val(e.get("hybrid_score")),
                _coerce_val(e.get("alpha")),
                meta_txt,
            )
            batch.append(row)
            if len(batch) >= batch_size:
                cur.executemany(
                    "INSERT INTO edges (source_index, source_id, target_id, source_timestamp, target_timestamp, time_delta_ms, retrieval_distance, retrieval_similarity, semantic_cosine, hybrid_score, alpha, target_metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    batch,
                )
                self.conn.commit()
                inserted += len(batch)
                batch = []

        if batch:
            cur.executemany(
                "INSERT INTO edges (source_index, source_id, target_id, source_timestamp, target_timestamp, time_delta_ms, retrieval_distance, retrieval_similarity, semantic_cosine, hybrid_score, alpha, target_metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
            self.conn.commit()
            inserted += len(batch)

        return inserted

    def fetch_recent(self, limit: int = 1000):
        if self.conn is None:
            self.init_db()
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM edges ORDER BY id DESC LIMIT ?", (limit,))
        return cur.fetchall()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None


__all__ = ["EdgeStore"]
