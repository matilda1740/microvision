import sqlite3
from pathlib import Path

from src.enrichment.jobs import init_jobs_db, enqueue_job, get_job_status, worker_once


def test_enqueue_and_worker_once(tmp_path: Path):
    jobs_db = str(tmp_path / "jobs.db")
    input_csv = str(tmp_path / "sample.csv")
    edges_db = str(tmp_path / "edges.db")

    # create a small CSV with embedding column
    with open(input_csv, "w", encoding="utf-8") as fh:
        fh.write("doc_id,embedding\n")
        fh.write("a, [1.0, 0.0]\n")
        fh.write("b, [0.0, 1.0]\n")

    init_jobs_db(jobs_db)
    job_id = enqueue_job(jobs_db, input_csv, edges_db, params={"top_k": 1, "threshold": 0.0, "alpha": 0.5, "batch_size": 2})
    assert isinstance(job_id, int)

    status = get_job_status(jobs_db, job_id)
    assert status is not None and status["status"] == "queued"

    processed = worker_once(jobs_db)
    assert processed == job_id

    status = get_job_status(jobs_db, job_id)
    assert status is not None and status["status"] in ("done", "failed")
