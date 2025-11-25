import numpy as np
import sqlite3
from pathlib import Path

from src.pipeline.runner import run_pipeline_df


def test_run_pipeline_df(tmp_path: Path):
    # create minimal dataframe like object (pandas available in test env)
    import pandas as pd

    df = pd.DataFrame({
        "doc_id": ["a", "b", "c"],
        "timestamp": ["2025-11-12T00:00:00Z", "2025-11-12T00:00:01Z", "2025-11-12T00:00:02Z"],
    })

    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])

    db_path = str(tmp_path / "pipeline_edges.db")
    csv_out = str(tmp_path / "transitions_out.csv")

    edges_written, transitions_written = run_pipeline_df(
        df,
        embeddings=embeddings,
        db_path=db_path,
        top_k=2,
        threshold=0.0,
        alpha=0.5,
        batch_size=2,
        export_csv=csv_out,
    )

    assert edges_written > 0
    assert transitions_written > 0

    # verify DB tables exist
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM edges")
    ecount = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM transitions")
    tcount = cur.fetchone()[0]
    conn.close()

    assert ecount == edges_written
    assert tcount == transitions_written

    # verify CSV file present
    assert Path(csv_out).exists()
