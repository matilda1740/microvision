import pandas as pd
from src.enrichment.merger import merge_structured_metadata


def test_merge_simple_remap(tmp_path):
    # cleaned has a template that references an id '123.0' while structured has '123'
    cleaned = pd.DataFrame({
        "semantic_text": ["error happened"],
        "template_ids": [["123.0"]],
        "occurrences": [1],
    })
    structured = pd.DataFrame({
        "template_id": ["123"],
        "service": ["nova.compute"],
        "component": ["nova.compute.manager"],
        "level": ["ERROR"],
    })
    out_path = tmp_path / "out.csv"
    df = merge_structured_metadata(cleaned, structured, str(out_path))
    # ensure the service was filled and template_ids normalized
    assert "nova.compute" in df["service"].iloc[0]
    assert isinstance(df["template_ids"].iloc[0], list)
    assert any("123" in str(x) for x in df["template_ids"].iloc[0])


def test_merge_majority_fill(tmp_path):
    # two cleaned rows share same semantic_text; one has metadata, other missing
    cleaned = pd.DataFrame(
        {
            "semantic_text": ["s1", "s1"],
            "template_ids": [["1"], ["2"]],
            "occurrences": [2, 3],
        }
    )
    structured = pd.DataFrame(
        {
            "template_id": ["1"],
            "service": ["svc.a"],
            "component": ["svc.a.comp"],
            "level": ["INFO"],
        }
    )
    out_path = tmp_path / "out2.csv"
    df = merge_structured_metadata(cleaned, structured, str(out_path))
    assert df.shape[0] == 1
    assert "svc.a" in df["service"].iloc[0]
