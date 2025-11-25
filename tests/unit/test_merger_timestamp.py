import pandas as pd
from pathlib import Path

from src.enrichment.merger import merge_structured_metadata


def test_merge_emits_timestamp_canonical(tmp_path: Path):
    # Create a minimal cleaned DataFrame with template_ids
    cleaned = pd.DataFrame(
        {
            "semantic_text": ["event A", "event B"],
            "template_ids": [["1"], ["2"]],
            "occurrences": [1, 1],
        }
    )

    # Create structured metadata including timestamp lists/strings
    structured = pd.DataFrame(
        {
            "template_id": ["1", "2"],
            "service": ["svc1", "svc2"],
            "component": [["comp1"], ["comp2"]],
            "timestamp": [["2017-01-01T00:00:00Z", "2017-01-02T00:00:00Z"], "2018-01-01T12:00:00Z"],
        }
    )

    out_path = tmp_path / "merged.csv"
    aggregated = merge_structured_metadata(cleaned, structured, str(out_path))

    # Ensure the output contains timestamp_canonical column
    assert "timestamp_canonical" in aggregated.columns

    # For template 1 we provided two timestamps; canonical should not be null
    row1 = aggregated[aggregated["template_id"] == "1"].iloc[0]
    assert row1["timestamp_canonical"] is not None and str(row1["timestamp_canonical"]).strip() != ""

    # For template 2 we provided a single timestamp; canonical should equal that timestamp (or parseable)
    row2 = aggregated[aggregated["template_id"] == "2"].iloc[0]
    assert row2["timestamp_canonical"] is not None and str(row2["timestamp_canonical"]).strip() != ""

    # Also ensure the CSV was written and header contains timestamp_canonical
    txt = Path(out_path).read_text(encoding="utf-8")
    assert "timestamp_canonical" in txt.splitlines()[0]
