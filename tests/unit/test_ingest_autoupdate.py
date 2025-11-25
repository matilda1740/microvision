import pandas as pd
from pathlib import Path

from src.encode_chroma import encode_utils as ingest_utils
from src.enrichment import labels as labels_mod


def test_df_to_metadatas_updates_map(tmp_path, monkeypatch):
    df = pd.DataFrame(
        [
            {"template": "t1", "component": "nova.compute.manager", "timestamp": "2020-01-01T00:00:00Z"},
            {"template": "t2", "component": "neutron.agent", "timestamp": "2020-01-01T00:01:00Z"},
        ]
    )

    # redirect map writes to a temp map path by monkeypatching the updater
    map_path = tmp_path / "service_name_map.json"

    # preserve original and redirect underlying writes to tmp map
    original_updater = labels_mod.update_service_map_with_token

    def updater(token, map_path_arg=None):
        return original_updater(token, map_path=str(map_path))

    monkeypatch.setattr(labels_mod, "update_service_map_with_token", updater)

    metas = ingest_utils.df_to_metadatas(df, update_service_map=True)
    assert isinstance(metas, list) and len(metas) == 2
    # check service_token and service_display present
    for m in metas:
        assert "service_token" in m and m["service_token"]
        assert "service_display" in m and m["service_display"]
    # map file should exist now
    assert map_path.exists()
