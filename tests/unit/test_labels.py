import json
from pathlib import Path

from src.enrichment.labels import (
    extract_service_name_from_component,
    extract_service_label_from_metadata,
    update_service_map_with_token,
    _pretty_display_from_token,
)


def test_extract_tokens():
    assert extract_service_name_from_component("nova.compute.manager") == "nova"
    assert extract_service_name_from_component("nova.compute.manager", prefer_two_level=True) == "nova.compute"
    assert extract_service_name_from_component("neutron.agent.linux") == "neutron"


def test_json_metadata_parsing():
    meta = '{"component": "cinder.volume.manager", "other": "x"}'
    assert extract_service_label_from_metadata(meta, prefer_two_level=True) == "cinder.volume"


def test_update_service_map_atomic_and_audit(tmp_path):
    # ensure we write to a temp map path
    map_path = tmp_path / "service_name_map.json"
    audit_path = tmp_path / "service_name_map.audit.log"

    token = "foo.bar"
    # call update
    disp = update_service_map_with_token(token, map_path=str(map_path))
    assert disp is not None
    # map file exists and contains the token
    data = json.loads(map_path.read_text(encoding="utf-8"))
    assert token in data and data[token] == disp
    # audit log created
    assert audit_path.exists()
    contents = audit_path.read_text(encoding="utf-8").strip()
    assert token in contents


def test_pretty_display():
    assert _pretty_display_from_token("nova.compute") == "Nova Compute"
    assert _pretty_display_from_token("keystone") == "Keystone"
