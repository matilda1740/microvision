import shutil
import subprocess
import sys
import sqlite3
from pathlib import Path
import pytest

# skip if heavy optional deps aren't available
sentence_transformers = pytest.importorskip("sentence_transformers")
chromadb = pytest.importorskip("chromadb")

ROOT = Path(__file__).resolve().parents[2]


def test_phase2_smoke_runs_and_writes_edges(tmp_path):
    """Run the Phase 2 runner in a temp workspace and assert edges DB created.

    This is a smoke/integration test: it will be skipped when optional heavy
    dependencies (sentence-transformers or chromadb) are not installed.
    """
    merged_src = ROOT / "data" / "sample_raw_merged.csv"
    if not merged_src.exists():
        pytest.skip("sample merged CSV not present; skip smoke Phase 2 test")

    merged = tmp_path / "merged.csv"
    shutil.copy(merged_src, merged)

    chroma_dir = tmp_path / "chroma"
    db_path = tmp_path / "edges.db"

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_phase2.py"),
        "--merged",
        str(merged),
        "--chroma-dir",
        str(chroma_dir),
        "--db",
        str(db_path),
        "--top-k",
        "5",
        "--threshold",
        "0.1",
        "--device",
        "cpu",
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    # dump stdout/stderr on failure to help debugging
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
    assert proc.returncode == 0, "Phase 2 runner failed"

    assert db_path.exists(), "edges DB not created"

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # assert table exists and has expected columns
    cur.execute("PRAGMA table_info(edges)")
    cols = [r[1] for r in cur.fetchall()]
    for expected in ("id", "source_index", "source_id", "target_id", "hybrid_score", "semantic_cosine"):
        assert expected in cols, f"Missing expected column {expected} in edges table"

    cur.execute("SELECT COUNT(*) FROM edges")
    count = cur.fetchone()[0]
    conn.close()

    assert count > 0, "No edges written by Phase 2 runner"
