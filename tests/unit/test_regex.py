import pytest
from src.parsing.regex_utils import extract_fields, validate_examples


def test_extract_fields_basic():
    content = "[req-12345] user_id=alice status: 200 GET /api/v1/items len:123 time:0.23"
    out = extract_fields(content)
    assert out.get("reqid") in ("12345", "12345")
    assert out.get("userid") == "alice"
    assert out.get("status") == "200"
    assert out.get("method").upper() == "GET"
    assert out.get("url") is not None


def test_extract_fields_no_match():
    content = "this has no structured metadata"
    out = extract_fields(content)
    # all keys should be present but None
    assert all(k in out for k in ["reqid", "userid", "ip", "status", "method", "url"])
