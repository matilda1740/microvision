import pandas as pd
import numpy as np

from src.encode_chroma.encode_utils import df_to_metadatas


def test_df_to_metadatas_handles_list_like_timestamps_and_values():
    # Build a DataFrame with various list-like and scalar metadata fields
    df = pd.DataFrame(
        {
            "template": ["t1", "t2"],
            "timestamp": [pd.Index(["2017-05-17T12:02:19Z"]), ["2017-05-17T12:02:35Z"]],
            "component": [pd.Series(["svcA"]), np.array(["svcB"])],
            "tags": [["a", "b"], []],
            "id": [1, 2],
        }
    )

    metas = df_to_metadatas(df, meta_cols=["template", "timestamp", "component", "tags", "id"])

    assert isinstance(metas, list)
    assert len(metas) == 2

    # timestamps should be ISO-8601 strings (or None)
    assert isinstance(metas[0]["timestamp"], str)
    assert metas[0]["timestamp"].startswith("2017-05-17T12:")

    # component should be string coerced
    assert isinstance(metas[0]["component"], str)

    # tags: first row has joined string, second row should be None (empty list -> None)
    assert isinstance(metas[0]["tags"], str)
    assert metas[1]["tags"] is None
