"""Simple SQLite-backed job queue and worker helpers.

This module provides a lightweight job table and helpers to enqueue jobs
and process them with a worker. It's intentionally dependency-light so it
can run inside a Streamlit process or a simple background worker.
"""
from __future__ import annotations

import sqlite3
import json
import time
from typing import Optional, Dict, Any, Sequence
from pathlib import Path
import subprocess
import sys

from src.pipeline.runner import run_pipeline_df


def init_jobs_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            status TEXT,
            input_path TEXT,
            edges_db TEXT,
            params TEXT,
            result_msg TEXT,
            created_at DATETIME DEFAULT (CURRENT_TIMESTAMP),
            updated_at DATETIME DEFAULT (CURRENT_TIMESTAMP)
        )
        """
    )
    conn.commit()
    conn.close()


def validate_upload(
    input_path: str,
    max_bytes: int = 5_000_000,
    max_lines: int = 10000,
    secret_patterns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Validate an uploaded file for size, line count and simple secret patterns.

    Raises RuntimeError on validation failure. Returns a small dict with file stats on success.
    """
    import os
    import re

    if secret_patterns is None:
        secret_patterns = [r"AKIA[0-9A-Z]{16}", r"-----BEGIN PRIVATE KEY-----", r"password\s*=", r"api_key\s*[:=]"]

    if not os.path.exists(input_path):
        raise RuntimeError("Uploaded file does not exist")

    stat = os.stat(input_path)
    if stat.st_size > max_bytes:
        raise RuntimeError(f"Uploaded file too large: {stat.st_size} bytes > {max_bytes}")

    # count lines up to max_lines+1 to avoid scanning huge files
    lines = 0
    found_secrets = []
    with open(input_path, "r", encoding="utf-8", errors="ignore") as fh:
        for i, line in enumerate(fh):
            if i <= max_lines:
                lines += 1
                for pat in secret_patterns:
                    if re.search(pat, line):
                        found_secrets.append(pat)
            else:
                # we scanned beyond allowed lines
                lines += 1
                break

    if lines > max_lines:
        raise RuntimeError(f"Uploaded file has too many lines: {lines} > {max_lines}")

    if found_secrets:
        raise RuntimeError(f"Uploaded file contains possible secrets: {found_secrets}")

    return {"size": stat.st_size, "lines": lines}


def enqueue_job(jobs_db: str, input_path: str, edges_db: str, params: Optional[Dict[str, Any]] = None, validate: bool = False, max_bytes: int = 5_000_000, max_lines: int = 10000) -> int:
    """Enqueue a job to process an uploaded file.

    If `validate` is True the uploaded file will be checked for allowed size,
    line count and simple secret patterns before enqueuing. Validation is
    conservative and intended as a first-line defense for UI uploads.
    """
    if validate:
        validate_upload(input_path, max_bytes=max_bytes, max_lines=max_lines)

    init_jobs_db(jobs_db)
    conn = sqlite3.connect(jobs_db)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO jobs (status, input_path, edges_db, params) VALUES (?, ?, ?, ?)",
        ("queued", str(input_path), str(edges_db), json.dumps(params or {})),
    )
    job_id = cur.lastrowid
    conn.commit()
    conn.close()
    return job_id


def get_job_status(jobs_db: str, job_id: int) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(jobs_db)
    cur = conn.cursor()
    cur.execute("SELECT id, status, input_path, edges_db, params, result_msg, created_at, updated_at FROM jobs WHERE id = ?", (job_id,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "id": row[0],
        "status": row[1],
        "input_path": row[2],
        "edges_db": row[3],
        "params": json.loads(row[4] or "{}"),
        "result_msg": row[5],
        "created_at": row[6],
        "updated_at": row[7],
    }


def _claim_next_job(jobs_db: str) -> Optional[Dict[str, Any]]:
    """Atomically claim the oldest queued job and mark it running. Returns the job row or None."""
    conn = sqlite3.connect(jobs_db)
    cur = conn.cursor()
    # Use a transaction to atomically select and update
    cur.execute("BEGIN EXCLUSIVE")
    cur.execute("SELECT id, input_path, edges_db, params FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1")
    row = cur.fetchone()
    if row is None:
        conn.commit()
        conn.close()
        return None
    job_id, input_path, edges_db, params_text = row
    cur.execute("UPDATE jobs SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", ("running", job_id))
    conn.commit()
    conn.close()
    return {"id": job_id, "input_path": input_path, "edges_db": edges_db, "params": json.loads(params_text or "{}")} 


def _update_job(jobs_db: str, job_id: int, status: str, result_msg: Optional[str] = None) -> None:
    conn = sqlite3.connect(jobs_db)
    cur = conn.cursor()
    cur.execute("UPDATE jobs SET status = ?, result_msg = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (status, result_msg, job_id))
    conn.commit()
    conn.close()


def worker_once(jobs_db: str) -> Optional[int]:
    """Claim one job and run it. Returns job id processed or None if none available."""
    job = _claim_next_job(jobs_db)
    if job is None:
        return None

    job_id = job["id"]
    try:
        # load input CSV and run pipeline; accept that input_path might be CSV with embedding column
        input_path = Path(job["input_path"])
        params = job.get("params") or {}

        # load embeddings file if present
        import pandas as pd
        import ast
        import numpy as np

        df = pd.read_csv(str(input_path))
        embeddings = None
        # If this is a raw-log CSV (has 'raw' column and no 'embedding'),
        # run the full pipeline script as a subprocess. This keeps the
        # worker lightweight while allowing the full end-to-end flow to run
        # in a deterministic subprocess using the same Python interpreter.
        if ("raw" in df.columns or "content" in df.columns) and ("embedding" not in df.columns):
            try:
                # Locate the pipeline script relative to repo root (3 parents up from this file)
                script_path = Path(__file__).resolve().parents[3] / "scripts" / "run_full_pipeline.py"
                run_args = [sys.executable, str(script_path), "--source", str(input_path), "--db", str(job["edges_db"])]
                # map common params through if provided
                if "sample" in params:
                    run_args += ["--sample", str(int(params.get("sample")))]
                if "top_k" in params:
                    run_args += ["--top-k", str(int(params.get("top_k")))]
                if "threshold" in params:
                    run_args += ["--threshold", str(float(params.get("threshold")))]
                if "alpha" in params:
                    run_args += ["--alpha", str(float(params.get("alpha")))]
                if "chroma_dir" in params:
                    run_args += ["--chroma-dir", str(params.get("chroma_dir"))]
                if "model" in params:
                    run_args += ["--model", str(params.get("model"))]

                proc = subprocess.run(run_args, capture_output=True, text=True)
                if proc.returncode != 0:
                    raise RuntimeError(f"run_full_pipeline failed (rc={proc.returncode}): {proc.stderr}\n{proc.stdout}")
                # try to pick a succinct success message from stdout
                out_lines = proc.stdout.strip().splitlines() if proc.stdout else []
                msg = out_lines[-1] if out_lines else "run_full_pipeline completed successfully"
                _update_job(jobs_db, job_id, "done", result_msg=f"run_full_pipeline: {msg}")
            except Exception as e:
                _update_job(jobs_db, job_id, "failed", result_msg=str(e))
            return job_id
        if "embedding" in df.columns:
            def _parse(x):
                if isinstance(x, list):
                    return x
                try:
                    return ast.literal_eval(x)
                except Exception:
                    return None
            df["embedding"] = df["embedding"].apply(_parse)
            embeddings = np.array([e for e in df["embedding"].tolist()])

        edges_db = job["edges_db"]
        # pipeline parameters
        top_k = int(params.get("top_k", 5))
        threshold = float(params.get("threshold", 0.1))
        alpha = float(params.get("alpha", 0.5))
        batch_size = int(params.get("batch_size", 128))

        edges_written, transitions_written = run_pipeline_df(
            df,
            embeddings=embeddings,
            db_path=edges_db,
            top_k=top_k,
            threshold=threshold,
            alpha=alpha,
            batch_size=batch_size,
        )

        _update_job(jobs_db, job_id, "done", result_msg=f"edges={edges_written},transitions={transitions_written}")
    except Exception as e:
        _update_job(jobs_db, job_id, "failed", result_msg=str(e))
    return job_id


def worker_loop(jobs_db: str, poll_interval: float = 1.0) -> None:
    init_jobs_db(jobs_db)
    while True:
        processed = worker_once(jobs_db)
        if processed is None:
            time.sleep(poll_interval)
        else:
            # continue immediately to pick up more work
            continue


__all__ = ["init_jobs_db", "enqueue_job", "get_job_status", "worker_once", "worker_loop"]
