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
                target_component TEXT,
                target_service TEXT,
                source_component TEXT,
                source_service TEXT,
                source_timestamp_canonical TEXT,
                target_timestamp_canonical TEXT,
                target_semantic_text TEXT,
                created_at DATETIME DEFAULT (CURRENT_TIMESTAMP)
            )
            """
        )
        self.conn.commit()
        # Ensure legacy DBs get the new columns: add if missing
        try:
            cur.execute("PRAGMA table_info(edges)")
            cols = [r[1] for r in cur.fetchall()]
            if "source_timestamp_canonical" not in cols:
                cur.execute("ALTER TABLE edges ADD COLUMN source_timestamp_canonical TEXT")
            if "target_timestamp_canonical" not in cols:
                cur.execute("ALTER TABLE edges ADD COLUMN target_timestamp_canonical TEXT")
            if "target_semantic_text" not in cols:
                cur.execute("ALTER TABLE edges ADD COLUMN target_semantic_text TEXT")
            if "target_component" not in cols:
                cur.execute("ALTER TABLE edges ADD COLUMN target_component TEXT")
            if "target_service" not in cols:
                cur.execute("ALTER TABLE edges ADD COLUMN target_service TEXT")
            if "source_component" not in cols:
                cur.execute("ALTER TABLE edges ADD COLUMN source_component TEXT")
            if "source_service" not in cols:
                cur.execute("ALTER TABLE edges ADD COLUMN source_service TEXT")
            self.conn.commit()
        except Exception:
            # non-fatal; if ALTER fails we proceed — write_edges will still try to insert
            pass
        # attempt to load merged_templates mapping (optional) to resolve source metadata
        try:
            import pandas as _pd
            mt_path = "data/merged_templates.csv"
            self._merged_comp_map = {}
            self._merged_serv_map = {}
            if _pd and _pd.io.common.file_exists(mt_path):
                _mt = _pd.read_csv(mt_path, dtype=str).fillna("")
                # store as JSON list string when possible for canonical matching
                for tid, comp, serv in zip(_mt.get("template_id", []), _mt.get("component", []), _mt.get("service", [])):
                    t = str(tid)
                    # normalize component: if looks like list repr, try to parse
                    try:
                        import ast as _ast

                        parsed = _ast.literal_eval(comp) if isinstance(comp, str) and comp.strip().startswith("[") else None
                        if isinstance(parsed, (list, tuple)):
                            self._merged_comp_map[t] = json.dumps([str(x) for x in parsed], ensure_ascii=False)
                        elif isinstance(comp, str) and "," in comp:
                            parts = [p.strip() for p in comp.split(",") if p.strip()]
                            self._merged_comp_map[t] = json.dumps(parts, ensure_ascii=False)
                        elif isinstance(comp, str) and comp:
                            self._merged_comp_map[t] = json.dumps([comp], ensure_ascii=False)
                        else:
                            self._merged_comp_map[t] = None
                    except Exception:
                        self._merged_comp_map[t] = json.dumps([str(comp)]) if comp is not None else None
                    self._merged_serv_map[t] = str(serv) if serv is not None and serv != "" else None
            # also attempt to read parsed_sample.csv to resolve source mappings (prefer this for source metadata)
            try:
                ps_path = "data/parsed_sample.csv"
                self._parsed_comp_map = {}
                self._parsed_serv_map = {}
                if _pd and _pd.io.common.file_exists(ps_path):
                    _ps = _pd.read_csv(ps_path, dtype=str).fillna("")
                    for tid, comp, serv in zip(_ps.get("template_id", []), _ps.get("component", []), _ps.get("service", [])):
                        t = str(tid)
                        # parsed sample component is often scalar or list-like string; store as JSON list string
                        try:
                            import ast as _ast

                            parsed = _ast.literal_eval(comp) if isinstance(comp, str) and comp.strip().startswith("[") else None
                            if isinstance(parsed, (list, tuple)):
                                self._parsed_comp_map[t] = json.dumps([str(x) for x in parsed], ensure_ascii=False)
                            elif isinstance(comp, str) and "," in comp:
                                parts = [p.strip() for p in comp.split(",") if p.strip()]
                                self._parsed_comp_map[t] = json.dumps(parts, ensure_ascii=False)
                            elif isinstance(comp, str) and comp:
                                self._parsed_comp_map[t] = json.dumps([comp], ensure_ascii=False)
                            else:
                                self._parsed_comp_map[t] = None
                        except Exception:
                            self._parsed_comp_map[t] = json.dumps([str(comp)]) if comp is not None else None
                        self._parsed_serv_map[t] = str(serv) if serv is not None and serv != "" else None
                else:
                    self._parsed_comp_map = {}
                    self._parsed_serv_map = {}
            except Exception:
                self._parsed_comp_map = {}
                self._parsed_serv_map = {}
        except Exception:
            self._merged_comp_map = {}
            self._merged_serv_map = {}

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

            # extract component/service from target metadata when available
            target_component = None
            target_service = None
            try:
                if isinstance(target_meta, dict):
                    comp = target_meta.get("component")
                    # normalize component to JSON list string when possible
                    try:
                        import ast as _ast

                        if isinstance(comp, str) and comp.strip().startswith("["):
                            parsed = _ast.literal_eval(comp)
                            if isinstance(parsed, (list, tuple)):
                                target_component = json.dumps([str(x) for x in parsed], ensure_ascii=False)
                            else:
                                target_component = json.dumps([str(parsed)], ensure_ascii=False)
                        elif isinstance(comp, (list, tuple)):
                            target_component = json.dumps([str(x) for x in comp], ensure_ascii=False)
                        elif isinstance(comp, str) and "," in comp:
                            parts = [p.strip() for p in comp.split(",") if p.strip()]
                            target_component = json.dumps(parts, ensure_ascii=False)
                        elif isinstance(comp, str) and comp:
                            target_component = json.dumps([comp], ensure_ascii=False)
                        else:
                            target_component = None
                    except Exception:
                        target_component = json.dumps([str(comp)]) if comp is not None else None

                    # service can be under several keys depending on upstream
                    target_service = target_meta.get("service") or target_meta.get("service_token") or target_meta.get("service_display")
            except Exception:
                target_component = None
                target_service = None

            # resolve source component/service: prefer parsed_sample mapping, fall back to merged_templates
            source_component = None
            source_service = None
            try:
                src_id = e.get("source_id")
                if src_id is not None:
                    sid = str(src_id)
                    # prefer parsed mapping
                    source_component = self._parsed_comp_map.get(sid) if hasattr(self, "_parsed_comp_map") else None
                    source_service = self._parsed_serv_map.get(sid) if hasattr(self, "_parsed_serv_map") else None
                    if not source_component:
                        source_component = self._merged_comp_map.get(sid) if hasattr(self, "_merged_comp_map") else None
                    if not source_service:
                        source_service = self._merged_serv_map.get(sid) if hasattr(self, "_merged_serv_map") else None
            except Exception:
                source_component = None
                source_service = None

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
                _coerce_val(target_component),
                _coerce_val(target_service),
                _coerce_val(source_component),
                _coerce_val(source_service),
                _coerce_val(e.get("source_timestamp_canonical")),
                _coerce_val(e.get("target_timestamp_canonical")),
                _coerce_val(e.get("target_semantic_text")),
            )
            batch.append(row)
            if len(batch) >= batch_size:
                cur.executemany(
                    "INSERT INTO edges (source_index, source_id, target_id, source_timestamp, target_timestamp, time_delta_ms, retrieval_distance, retrieval_similarity, semantic_cosine, hybrid_score, alpha, target_metadata, target_component, target_service, source_component, source_service, source_timestamp_canonical, target_timestamp_canonical, target_semantic_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    batch,
                )
                self.conn.commit()
                inserted += len(batch)
                batch = []

        if batch:
            cur.executemany(
                "INSERT INTO edges (source_index, source_id, target_id, source_timestamp, target_timestamp, time_delta_ms, retrieval_distance, retrieval_similarity, semantic_cosine, hybrid_score, alpha, target_metadata, target_component, target_service, source_component, source_service, source_timestamp_canonical, target_timestamp_canonical, target_semantic_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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

    def clear_edges(self, reset_sequence: bool = True) -> None:
        """Remove all rows from the edges table. Optionally reset sqlite_sequence for the table.

        This is a destructive operation. It is provided to support workflows
        where the user wants to start the edges DB fresh for a new run.
        """
        if self.conn is None:
            self.init_db()
        cur = self.conn.cursor()
        try:
            cur.execute("DELETE FROM edges")
            self.conn.commit()
            if reset_sequence:
                try:
                    cur.execute("DELETE FROM sqlite_sequence WHERE name='edges'")
                    self.conn.commit()
                except Exception:
                    # Not fatal; some sqlite builds may not expose sqlite_sequence
                    pass
            try:
                # attempt WAL checkpoint to truncate WAL files
                cur.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass
        except Exception:
            self.conn.rollback()
            raise

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None


__all__ = ["EdgeStore"]
