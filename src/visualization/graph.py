"""Small visualization helpers for service dependency graphs.

This module provides utility functions that convert edge streams (dicts)
into networkx graphs and serializable JSON structures. Heavy UI libs like
pyvis are imported lazily so unit tests don't require them.
"""
from __future__ import annotations

from typing import Iterable, Dict, Any
import json
import re
from pathlib import Path

try:
    import networkx as nx
except Exception:  # pragma: no cover - networkx required only for graph operations
    nx = None  # type: ignore

try:
    # optional service label helpers
    from src.enrichment.labels import extract_service_label_from_metadata, _pretty_display_from_token
except Exception:
    extract_service_label_from_metadata = None  # type: ignore
    _pretty_display_from_token = None  # type: ignore


def edges_to_networkx(edges: Iterable[Dict[str, Any]]):
    """Convert an iterable of edge dicts into a NetworkX DiGraph.

    Each edge dict should contain at least 'source' and 'target'. Other keys
    (e.g., 'weight', 'score', 'metadata') will be added as edge attributes.

    Returns a NetworkX DiGraph. If networkx is not installed, raises RuntimeError.
    """
    if nx is None:
        raise RuntimeError("networkx is required for graph construction")

    G = nx.DiGraph()
    # We enumerate edges so we can generate stable fallback ids when none
    # of the expected fields are present. This makes the visualizer robust
    # against upstream schema variants (source_id, source_index, etc.).
    for idx, e in enumerate(edges):
        # helper to pick first available id among candidates and coerce to
        # a trimmed string. Treat None as missing; accept numeric 0 as valid.
        def _pick_id(keys):
            for k in keys:
                v = e.get(k)
                if v is None:
                    continue
                try:
                    s = str(v).strip()
                except Exception:
                    continue
                if s == "":
                    continue
                return s
            return None

        src = _pick_id(("source", "source_id", "source_index"))
        dst = _pick_id(("target", "target_id", "target_index"))

        # If either side is still missing, generate a stable fallback id so
        # the graph remains connected and renderable.
        if src is None:
            src = f"node:{idx}:s"
        if dst is None:
            dst = f"node:{idx}:t"
        # keep other attributes (including source_id/source_index if present)
        attrs = {k: v for k, v in e.items() if k not in ("source", "target")}
        # coerce numpy scalars/lists to python primitives for attributes
        for k, v in list(attrs.items()):
            try:
                if hasattr(v, "tolist"):
                    attrs[k] = v.tolist()
            except Exception:
                pass
        # add nodes and optionally set labels using metadata heuristics
        if not G.has_node(src):
            label = None
            # prefer any token/display present in attrs
            if "service_display" in attrs:
                label = attrs.get("service_display")
            elif "service_token" in attrs and _pretty_display_from_token is not None:
                label = _pretty_display_from_token(attrs.get("service_token"))
            # try extracting from source-related metadata if available
            if label is None and extract_service_label_from_metadata is not None:
                label = extract_service_label_from_metadata(attrs.get("source_metadata") or attrs.get("metadata") or attrs.get("service"))
            # Normalize label to trimmed string
            try:
                label = str(label).strip() if label is not None else None
            except Exception:
                pass
            G.add_node(src, label=label if label is not None else src)
        if not G.has_node(dst):
            label = None
            if "service_display" in attrs:
                label = attrs.get("service_display")
            elif "service_token" in attrs and _pretty_display_from_token is not None:
                label = _pretty_display_from_token(attrs.get("service_token"))
            if label is None and extract_service_label_from_metadata is not None:
                label = extract_service_label_from_metadata(attrs.get("target_metadata") or attrs.get("metadata") or attrs.get("service"))
            try:
                label = str(label).strip() if label is not None else None
            except Exception:
                pass
            G.add_node(dst, label=label if label is not None else dst)
        G.add_edge(src, dst, **attrs)
    return G


def networkx_to_dict(G):
    """Serialize a NetworkX graph to a JSON-serializable dict with nodes and edges.

    Format:
      {"nodes": [{"id": id, "label": label, ...}], "edges": [{"source": s, "target": t, ...}]}

    If networkx is not available, raises RuntimeError.
    """
    if nx is None:
        raise RuntimeError("networkx is required for graph serialization")
    nodes = []
    for n, data in G.nodes(data=True):
        node = {"id": n}
        node.update({k: v for k, v in data.items()})
        nodes.append(node)

    edges = []
    for s, t, data in G.edges(data=True):
        edge = {"source": s, "target": t}
        edge.update({k: v for k, v in data.items()})
        edges.append(edge)

    return {"nodes": nodes, "edges": edges}


def write_pyvis_html(G, output_path: str, notebook: bool = False):
    """Render graph to an interactive HTML using pyvis (lazy import).

    This is a convenience wrapper; if `pyvis` is missing a RuntimeError is raised.
    """
    try:
        from pyvis.network import Network
    except Exception as e:
        raise RuntimeError("pyvis is required to write HTML visualizations") from e

    net = Network(notebook=notebook, directed=True)
    for n, data in G.nodes(data=True):
        net.add_node(n, label=str(data.get("label", n)))
    for s, t, data in G.edges(data=True):
        net.add_edge(s, t, **{k: v for k, v in data.items() if k not in ("metadata",)})

    # Try the pyvis writer first; if it fails (or generates problematic
    # relative script refs), fall back to a minimal vis-network HTML.
    try:
        net.write_html(output_path)

        # Post-process to inline any relative utils.js reference written by
        # pyvis so embedding in Streamlit's iframe doesn't accidentally load
        # another file from the server that conflicts with Streamlit globals.
        try:
            outp = Path(output_path)
            html_text = outp.read_text(encoding="utf-8")
            marker = '<script src="lib/bindings/utils.js"></script>'
            inlined = False
            if marker in html_text:
                repo_root = Path(__file__).resolve().parents[2]
                local_utils = repo_root / "lib" / "bindings" / "utils.js"
                if local_utils.exists():
                    utils_js = local_utils.read_text(encoding="utf-8")
                    # Wrap utils.js in a small namespaced module to avoid
                    # referencing global Streamlit symbols and to prevent
                    # leaking variables into the page global scope.
                    wrapper = (
                        '<script>'
                        '(function(ns){\n'
                        '  try {\n'
                        '    var Streamlit = window.Streamlit || {};\n'
                        '    var module = {exports: {}};\n'
                        '    (function(module, exports){\n'
                        f"{utils_js}\n"
                        '    })(module, module.exports);\n'
                        '    window[ns] = module.exports;\n'
                        '  } catch(e) { console.error("microvision utils inline error", e); }\n'
                        '})("_microvision_utils");'
                        '</script>'
                    )
                    html_text = html_text.replace(marker, wrapper)
                    inlined = True

            # Always add a small HTML comment indicating which renderer path was used
            # and whether utils.js was inlined. This helps tests and debugging.
            comment = f"<!-- graph-renderer: pyvis; utils-inlined: {'yes' if inlined else 'no'} -->"
            if comment not in html_text:
                # insert comment after the opening <head> tag if present
                if '<head>' in html_text:
                    html_text = html_text.replace('<head>', '<head>\n' + comment, 1)
                else:
                    html_text = comment + '\n' + html_text

            # Ensure the pyvis HTML contains the graph data. Some pyvis versions
            # render empty DataSet initializers and populate them elsewhere; when
            # embedding in an iframe those population steps may not run as
            # expected. Strategy:
            #  - perform regex replacements that match var/let/const/no-decl
            #  - append a runtime injector that populates empty DataSets on DOM ready
            try:
                data = networkx_to_dict(G)
                nodes_json = json.dumps(data["nodes"])  # type: ignore[arg-type]
                edges_json = json.dumps(data["edges"])  # type: ignore[arg-type]

                did_replace = False
                # regex patterns to match different declaration styles
                node_patterns = [
                    r"\\b(?:var|let|const)\\s+nodes\\s*=\\s*new\\s+vis\\.DataSet\\s*\\(\\s*\\[\\s*\\]\\s*\\)\\s*;",
                    r"\\bnodes\\s*=\\s*new\\s+vis\\.DataSet\\s*\\(\\s*\\[\\s*\\]\\s*\\)\\s*;",
                ]
                edge_patterns = [
                    r"\\b(?:var|let|const)\\s+edges\\s*=\\s*new\\s+vis\\.DataSet\\s*\\(\\s*\\[\\s*\\]\\s*\\)\\s*;",
                    r"\\bedges\\s*=\\s*new\\s+vis\\.DataSet\\s*\\(\\s*\\[\\s*\\]\\s*\\)\\s*;",
                ]

                for pat in node_patterns:
                    new_html = re.sub(pat, f"nodes = new vis.DataSet({nodes_json});", html_text, flags=re.MULTILINE)
                    if new_html != html_text:
                        did_replace = True
                        html_text = new_html

                for pat in edge_patterns:
                    new_html = re.sub(pat, f"edges = new vis.DataSet({edges_json});", html_text, flags=re.MULTILINE)
                    if new_html != html_text:
                        did_replace = True
                        html_text = new_html

                # Append runtime injector if not present
                inject_marker = '<!-- microvision-inject-graph-data -->'
                did_inject = False
                if inject_marker not in html_text:
                    populate_script = (
                        f"\n{inject_marker}\n<script>\n"
                        "(function(){\n"
                        f"  try {{ window._microvision_graph_data = {{nodes: {nodes_json}, edges: {edges_json}}}; }} catch(e) {{ console.error('microvision: failed to set graph data', e); }}\n"
                        "  try {\n"
                        "    function populateIfEmpty(){\n"
                        "      try {\n"
                        "        if(typeof nodes !== 'undefined' && nodes && typeof nodes.getIds === 'function' && nodes.getIds().length === 0){\n"
                        "          nodes.add(window._microvision_graph_data.nodes);\n"
                        "        }\n"
                        "        if(typeof edges !== 'undefined' && edges && typeof edges.getIds === 'function' && edges.getIds().length === 0){\n"
                        "          edges.add(window._microvision_graph_data.edges);\n"
                        "        }\n"
                        "        // attempt to refresh/redraw the network in case the library did not pick up dataset changes automatically\n"
                        "        try {\n"
                        "          if(typeof network !== 'undefined' && network){\n"
                        "            try {\n"
                        "              if(typeof network.setData === 'function'){\n"
                        "                network.setData({nodes: nodes, edges: edges});\n"
                        "              }\n"
                        "            } catch(e) { console.warn('microvision: network.setData failed', e); }\n"
                        "            try {\n"
                        "              if(typeof network.redraw === 'function'){ network.redraw(); }\n"
                        "            } catch(e) { console.warn('microvision: network.redraw failed', e); }\n"
                        "          }\n"
                        "        } catch(e) { console.warn('microvision: refresh network failed', e); }\n"
                        "      } catch(e) { console.error('microvision: populateIfEmpty failed', e); }\n"
                        "    }\n"
                        "    if(document.readyState === 'complete' || document.readyState === 'interactive'){\n"
                        "      setTimeout(populateIfEmpty, 50);\n"
                        "    } else {\n"
                        "      document.addEventListener('DOMContentLoaded', function(){ setTimeout(populateIfEmpty, 50); });\n"
                        "    }\n"
                        "  } catch(e) { console.error('microvision: runtime populate error', e); }\n"
                        "})();\n</script>\n"
                    )
                    if '</body>' in html_text:
                        html_text = html_text.replace('</body>', populate_script + '</body>', 1)
                    else:
                        html_text = html_text + populate_script
                    did_inject = True

                injected = did_replace or did_inject

                # Add a visible banner in the body to help debugging without devtools
                banner = (
                    f"<div style=\"width:100%;padding:6px 12px;background:#f7f7f7;border-bottom:1px solid #ddd;font-size:12px;color:#333;\">"
                    f"renderer: pyvis (utils-inlined: {'yes' if inlined else 'no'}) | nodes-injected: {'yes' if injected else 'no'}"
                    "</div>"
                )
                if '<body>' in html_text:
                    html_text = html_text.replace('<body>', '<body>\n' + banner, 1)
                else:
                    html_text = banner + html_text
            except Exception:
                # non-fatal; leave HTML as-is
                pass

            outp.write_text(html_text, encoding="utf-8")
        except Exception:
            # Non-fatal; return what we have
            pass

        return output_path
    except Exception:
        # Fallback: create a minimal vis-network HTML without pyvis.
        try:
            data = networkx_to_dict(G)
            nodes_json = json.dumps(data["nodes"])  # type: ignore[arg-type]
            edges_json = json.dumps(data["edges"])  # type: ignore[arg-type]
        except Exception as e:
            raise RuntimeError("Failed to serialize graph for fallback visualization") from e

        repo_root = Path(__file__).resolve().parents[2]
        vis_local = repo_root / "lib" / "vis-9.1.2" / "vis-network.min.js"
        if vis_local.exists():
            vis_js = vis_local.read_text(encoding="utf-8")
            vis_script_tag = f"<script>{vis_js}</script>"
        else:
            vis_script_tag = '<script src="https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.js"></script>'

        # Try to inline the same utils.js into the fallback HTML as a namespaced
        # module so it never depends on global Streamlit symbols.
        inlined_utils = False
        try:
            local_utils = repo_root / "lib" / "bindings" / "utils.js"
            if local_utils.exists():
                utils_js = local_utils.read_text(encoding="utf-8")
                utils_wrapper = (
                    '<script>'
                    '(function(ns){\n'
                    '  try {\n'
                    '    var Streamlit = window.Streamlit || {};\n'
                    '    var module = {exports: {}};\n'
                    '    (function(module, exports){\n'
                    f"{utils_js}\n"
                    '    })(module, module.exports);\n'
                    '    window[ns] = module.exports;\n'
                    '  } catch(e) { console.error("microvision utils inline error", e); }\n'
                    '})("_microvision_utils");'
                    '</script>'
                )
                inlined_utils = True
            else:
                utils_wrapper = ''
        except Exception:
            utils_wrapper = ''

            html = f"""
                    <!doctype html>
                        <html>
                        <head>
                            <meta charset="utf-8">
                            <style>
                                #mynetwork {{ width: 100%; height: 800px; border: 1px solid lightgray; }}
                            </style>
                            {vis_script_tag}
                            {utils_wrapper}
                            <!-- graph-renderer: fallback; utils-inlined: {'yes' if inlined_utils else 'no'} -->
                        </head>
                        <body>
                        <div style="width:100%;padding:6px 12px;background:#f7f7f7;border-bottom:1px solid #ddd;font-size:12px;color:#333;">renderer: fallback (utils-inlined: {'yes' if inlined_utils else 'no'}) | nodes-injected: yes</div>
                        <div id="mynetwork"></div>
                        <script>
                            const nodes = new vis.DataSet({nodes_json});
                            const edges = new vis.DataSet({edges_json});
                            const container = document.getElementById('mynetwork');
                            const data = {{ nodes: nodes, edges: edges }};
                            const options = {{ physics: {{ stabilization: true }}, edges: {{ smooth: true }} }};
                            const network = new vis.Network(container, data, options);
                        </script>
                        </body>
                        </html>
                    """

    Path(output_path).write_text(html, encoding="utf-8")
    return output_path
