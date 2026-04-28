from pathlib import Path
import tempfile
import pytest

from src.visualization import edges_to_networkx, write_pyvis_html


def test_write_inlines_utils_js(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    local_utils = repo_root / "lib" / "bindings" / "utils.js"
    if not local_utils.exists():
        pytest.skip("utils.js not present in repository; skipping inline test")

    edges = [{"source": "a", "target": "b", "weight": 1}, {"source": "b", "target": "c", "weight": 2}]
    G = edges_to_networkx(edges)
    out = tmp_path / "test_graph.html"
    write_pyvis_html(G, str(out))
    assert out.exists()
    txt = out.read_text(encoding="utf-8")
    # Expect the namespaced module to be present so the inlining worked
    assert "_microvision_utils" in txt or "utils-inlined: yes" in txt
