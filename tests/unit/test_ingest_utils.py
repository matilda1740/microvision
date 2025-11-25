import pandas as pd

from src.encode_chroma.encode_utils import df_to_metadatas


def test_df_to_metadatas_serializes_timestamp():
    df = pd.DataFrame(
        {
            "doc_id": ["a", "b"],
            "component": ["namenode", "datanode"],
            "timestamp": ["2025-11-11 12:00:00", pd.NaT],
            "line_no": [1, 2],
        }
    )

    metas = df_to_metadatas(df)
    assert isinstance(metas, list) and len(metas) == 2

    assert metas[0]["doc_id"] == "a"
    # timestamp should be ISO-8601 Z string
    assert metas[0]["timestamp"].endswith("Z") and "T" in metas[0]["timestamp"]

    # second row had NaT -> None
    assert metas[1]["timestamp"] is None
