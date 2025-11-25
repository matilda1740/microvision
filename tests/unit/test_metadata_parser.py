import pandas as pd
from src.parsing.metadata_drain_parser import MetadataDrainParser


def test_metadata_drain_parser_flush(tmp_path):
    out_csv = tmp_path / "parsed.csv"
    templates_csv = tmp_path / "templates.csv"

    parser = MetadataDrainParser(log_format=None, structured_csv=str(out_csv), templates_csv=str(templates_csv), save_every=1, mode="fresh")
    lines = ["first line", "second line", "third line"]
    for i, l in enumerate(lines, start=1):
        parser.process_line(l, i)
    parser.finalize()

    df = pd.read_csv(out_csv)
    assert len(df) == len(lines)
    assert "content" in df.columns or "raw" in df.columns
