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
    
    # Expect 2 rows (one per template_id), not 1
    assert df.shape[0] == 2
    
    # Both should have the service filled (one from structured, one from majority fill)
    assert all(df["service"] == "svc.a")


def test_merge_produces_scalars(tmp_path):
    """Ensure that metadata columns like service, component, level are aggregated as scalars (mode), not lists."""
    cleaned = pd.DataFrame({
        "semantic_text": ["msg1", "msg1", "msg1"],
        "template_ids": [["1"], ["2"], ["3"]],
        "occurrences": [1, 1, 1],
    })
    # structured data has multiple entries for the same semantic group
    structured = pd.DataFrame({
        "template_id": ["1", "2", "3"],
        "service": ["nova-api", "nova-api", "nova-compute"], 
        "component": ["comp1", "comp1", "comp2"],
        "level": ["INFO", "INFO", "WARN"],
    })
    
    out_path = tmp_path / "out_scalars.csv"
    df = merge_structured_metadata(cleaned, structured, str(out_path))
    
    # Expect 3 rows (one per template_id)
    assert len(df) == 3
    
    # Check that all service values are strings (scalars), not lists
    for val in df["service"]:
        assert isinstance(val, str)
    
    # Check specific values
    # id 1 -> nova-api
    assert df.loc[df["template_id"] == "1", "service"].iloc[0] == "nova-api"
    # id 3 -> nova-compute
    assert df.loc[df["template_id"] == "3", "service"].iloc[0] == "nova-compute"

