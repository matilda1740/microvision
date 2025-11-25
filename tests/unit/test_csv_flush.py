import pandas as pd
from pathlib import Path
from src.parsing.metadata_drain_parser import MetadataDrainParser


def test_csv_flush_preserves_header_order(tmp_path: Path):
    """Simulate two flushes with differing row keys and assert the
    canonical header is preserved and subsequent rows align to it.

    This test avoids running the Drain3 miner by directly using the
    parser instance and manipulating its buffer; flush_to_csv is exercised
    to confirm the reindexing behavior implemented in the parser.
    """
    out = tmp_path / "out.csv"
    templates = tmp_path / "templates.csv"

    p = MetadataDrainParser(log_format=None, structured_csv=str(out), templates_csv=str(templates), save_every=1000, mode="fresh")

    # First flush: columns ['a','b']
    p.buffer = [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]
    p.flush_to_csv()

    # Second flush: different keys, includes 'c' instead of 'a'
    p.buffer = [
        {"b": 5, "c": 6},
    ]
    p.flush_to_csv()

    # Read back and assert header is the canonical one from first flush
    df = pd.read_csv(out)
    assert list(df.columns) == ["a", "b"], "CSV header should preserve canonical ordering"

    # First flush produced two rows; second flush appended one more row
    assert len(df) == 3

    # First row values preserved
    assert int(df.iloc[0]["a"]) == 1
    assert int(df.iloc[0]["b"]) == 2

    # Last row (appended) should have 'b' == 5 and no 'c' column present
    assert int(df.iloc[-1]["b"]) == 5
    assert "c" not in df.columns
