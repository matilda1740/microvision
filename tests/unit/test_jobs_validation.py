from pathlib import Path

from src.enrichment.jobs import init_jobs_db, enqueue_job, validate_upload


def test_validate_upload_and_enqueue(tmp_path: Path):
    jobs_db = str(tmp_path / "jobs.db")
    small_file = tmp_path / "small.log"
    large_file = tmp_path / "large.log"

    # create small file
    with open(small_file, "w", encoding="utf-8") as fh:
        fh.write("line1\nline2\n")

    # create large file (line count > max_lines)
    with open(large_file, "w", encoding="utf-8") as fh:
        for i in range(20000):
            fh.write(f"l{i}\n")

    # validation on small file should pass
    stats = validate_upload(str(small_file), max_bytes=1024 * 10, max_lines=1000)
    assert stats["lines"] == 2

    init_jobs_db(jobs_db)
    job_id = enqueue_job(jobs_db, str(small_file), str(tmp_path / "edges.db"), validate=True, max_bytes=1024 * 10, max_lines=1000)
    assert isinstance(job_id, int)

    # validation on large file should raise
    try:
        validate_upload(str(large_file), max_bytes=1024 * 10, max_lines=1000)
        raised = False
    except RuntimeError:
        raised = True
    assert raised
