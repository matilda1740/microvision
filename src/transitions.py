"""Order-1 transition counter and persistence utilities.

Moved from src.enrichment.transitions to src.persistence to separate persistence
concerns from enrichment logic.
"""
from __future__ import annotations

import sqlite3
from typing import Dict, Tuple, List


def compute_and_persist_transitions(db_path: str, min_count: int = 1) -> int:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='edges'"
    )
    if cur.fetchone() is None:
        raise RuntimeError("No 'edges' table found in DB")

    cur.execute(
        """
        SELECT source_id, target_id, COUNT(*) as cnt, AVG(time_delta_ms) as avg_dt, MAX(created_at) as last_seen
        FROM edges
        WHERE source_id IS NOT NULL AND target_id IS NOT NULL
        GROUP BY source_id, target_id
        """
    )
    rows = cur.fetchall()

    counts: Dict[Tuple[str, str], Tuple[int, float, str]] = {}
    totals: Dict[str, int] = {}
    for src, tgt, cnt, avg_dt, last_seen in rows:
        cnt = int(cnt)
        counts[(src, tgt)] = (cnt, float(avg_dt) if avg_dt is not None else None, last_seen)
        totals[src] = totals.get(src, 0) + cnt

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS transitions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT,
            target_id TEXT,
            count INTEGER,
            prob REAL,
            avg_time_delta_ms REAL,
            last_seen TEXT,
            created_at DATETIME DEFAULT (CURRENT_TIMESTAMP)
        )
        """
    )

    cur.execute("DELETE FROM transitions")

    to_insert: List[Tuple[str, str, int, float, float, str]] = []
    for (src, tgt), (cnt, avg_dt, last_seen) in counts.items():
        if cnt < min_count:
            continue
        total = totals.get(src, 0)
        prob = float(cnt) / float(total) if total > 0 else 0.0
        to_insert.append((src, tgt, cnt, prob, avg_dt if avg_dt is not None else None, last_seen))

    if to_insert:
        cur.executemany(
            "INSERT INTO transitions (source_id, target_id, count, prob, avg_time_delta_ms, last_seen) VALUES (?, ?, ?, ?, ?, ?)",
            to_insert,
        )

    conn.commit()
    written = len(to_insert)
    conn.close()
    return written


def export_transitions_to_csv(db_path: str, csv_path: str) -> int:
    import csv

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='transitions'")
    if cur.fetchone() is None:
        conn.close()
        raise RuntimeError("No 'transitions' table found in DB")

    cur.execute(
        "SELECT source_id, target_id, count, prob, avg_time_delta_ms, last_seen, created_at FROM transitions ORDER BY source_id, prob DESC"
    )
    rows = cur.fetchall()

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source_id", "target_id", "count", "prob", "avg_time_delta_ms", "last_seen", "created_at"])
        for r in rows:
            writer.writerow(list(r))

    conn.close()
    return len(rows)


__all__ = ["compute_and_persist_transitions", "export_transitions_to_csv"]
