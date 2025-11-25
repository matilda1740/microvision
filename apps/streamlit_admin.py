"""Streamlit admin UI for reviewing and approving service name mappings.

Run with:
    streamlit run apps/streamlit_admin.py

This admin allows operators to inspect the audit log of candidate tokens,
edit display names, and persist approved mappings atomically.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

try:
    import streamlit as st
except Exception:
    st = None

from src.enrichment.labels import update_service_map_with_token


def _load_map(map_path: Path) -> Dict[str, str]:
    try:
        if not map_path.exists():
            return {}
        import json

        return json.loads(map_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_audit(audit_path: Path):
    if not audit_path.exists():
        return []
    lines = []
    for line in audit_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            lines.append({"ts": parts[0], "token": parts[1], "display": parts[2]})
    return lines


def main():
    if st is None:
        print("Streamlit not installed. Install streamlit to run the admin UI.")
        return

    st.title("Service Name Map — Admin Review")

    # mapping path override
    map_path_input = st.text_input("Mapping file path (optional override)", value=os.environ.get("SERVICE_NAME_MAP_PATH", "src/enrichment/service_name_map.json"))
    map_path = Path(map_path_input)
    audit_path = map_path.with_suffix(".audit.log")

    st.markdown(f"**Map path:** `{map_path}`  \n**Audit path:** `{audit_path}`")

    st.markdown("### Current map")
    mapping = _load_map(map_path)
    if mapping:
        for token, disp in mapping.items():
            st.text_input(f"Token: {token}", value=disp, key=f"map_{token}")
    else:
        st.info("No mappings yet")

    st.markdown("### Audit candidates")
    audit = _load_audit(audit_path)
    if not audit:
        st.info("No audit records found")
    else:
        # show full audit list
        st.markdown("#### Audit entries (most recent first)")
        import pandas as pd

        df = pd.DataFrame(audit)
        st.dataframe(df)

        # unique tokens for bulk approval
        tokens = []
        for row in reversed(audit):
            if row["token"] not in tokens:
                tokens.append(row["token"])

        selected = st.multiselect("Select tokens to approve", options=tokens, default=[])
        if selected:
            st.markdown("### Edit display names for selected tokens")
            edits = {}
            for t in selected:
                suggested = next((r["display"] for r in audit if r["token"] == t), t)
                edits[t] = st.text_input(f"Display for {t}", value=suggested or t, key=f"edit_{t}")

            if st.button("Bulk approve selected"):
                for t, disp in edits.items():
                    # persist approval (atomic)
                    _ = update_service_map_with_token(t, map_path=str(map_path))
                st.success(f"Approved {len(edits)} tokens")

    if st.button("Reload"):
        st.experimental_rerun()


if __name__ == "__main__":
    main()
