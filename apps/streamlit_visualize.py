"""Streamlit app for MicroVision Results Presentation.

This dashboard focuses on "presentation mode" - visualizing pre-computed artifacts
(Graphs, Metrics, Logs) rather than running the heavy pipeline.

Structure:
- 📊 Executive Dashboard: High-level metrics + The Main Graph
- 🔍 Forensics: Deep dive into specific edge verifications (LLM reasoning)
- 📈 Sensitivity Analysis: Review the hyperparameter tuning study
"""
from __future__ import annotations

import sys
import json
import sqlite3
from pathlib import Path
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Ensure repo root on path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from config.settings import settings

# --- Layout & Config ---

st.set_page_config(
    layout="wide", 
    page_title="MicroVision Dashboard", 
    initial_sidebar_state="expanded"
)

# Custom CSS to inject into the Streamlit app
st.markdown("""
    <style>
    /* FORCED SIDEBAR VISIBILITY: This will prevent it from remaining hidden when the page re-renders */
    section[data-testid="stSidebar"] {
        width: 280px !important;
        visibility: visible !important;
        transform: translate3d(0px, 0px, 0px) !important;
        transition: none !important;
        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    }
    
    /* Hide the close button on the sidebar to ensure the user doesn't accidentally collapse it again */
    button[data-testid="sidebar-close-button"] {
        display: none !important;
    }

    /* Adjust the main content area to follow the forced sidebar */
    .stApp {
        margin-left: initial !important;
    }
    
    /* Compact header and padding to fit graph on screen */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        max-width: 98%;
    }
    h1 {
        margin-top: -1rem;
        font-size: 2.2rem !important;
        color: #1E88E5;
    }
    h3 {
        font-size: 1.1rem !important;
        margin-top: -2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
    }
    /* Hide the top decoration bar and reduce vertical margins between widgets */
    [data-testid="stHeader"] {
        display: none;
    }
    .stMainBlockContainer {
        padding-top: 0rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Add Main Title to the toolbar area (Top of the page)
st.title("🛡️ MicroVision")
st.markdown("### Context-Aware Microservice Dependency Discovery")

st.sidebar.title("🛡️ MicroVision")
st.sidebar.markdown("---")

# 1. Project Context Section
st.sidebar.subheader("Project Context")
st.sidebar.markdown("""
**Dataset**: OpenStack Baseline  
**Logs Processed**: 183,895  
**Core Model**: `all-mpnet-base-v2`  
**Validation Engine**: Llama 3.1
""")

st.sidebar.markdown("---")

# 2. Tech Stack Branding (Small & Professional)
st.sidebar.subheader("Pipeline Stats")
st.sidebar.success("✅ **Artifact Mode Active**")
st.sidebar.info("Retrieval-Augmented Semantic Log Analysis framework is engaged.")

# 4. Feedback Loop for Interactive Demo
if st.sidebar.button("💡 Refresh Artifacts"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("MicroVision")


# --- Helper Functions ---
def load_edges_data(db_path):
    if not Path(db_path).exists():
        return None
    
    conn = sqlite3.connect(db_path)
    # Fetch typical columns + LLM verification
    # Try/except block to handle schemas that might not have llm columns yet
    try:
        df = pd.read_sql_query("SELECT source_service, target_service, hybrid_score, llm_verification, llm_confidence FROM edges", conn)
    except:
        df = pd.read_sql_query("SELECT source_service, target_service, hybrid_score FROM edges", conn)
    conn.close()
    return df

def parse_llm_reasoning(row):
    """Safe extraction and beautification of LLM JSON reasoning."""
    if "llm_verification" not in row or not row["llm_verification"]:
        return "Not Verified"
    try:
        data = json.loads(row["llm_verification"])
        reason = data.get("reasoning", "No valid reasoning")
        
        # Replace generic "Log A" and "Log B" with actual service names for better forensics
        src = row.get("source_service", "Source")
        tgt = row.get("target_service", "Target")
        
        # Note: st.dataframe does not support Markdown emboldening inside cells for the 'reasoning' column.
        # We will capitalize them to make them stand out visually instead.
        reason = reason.replace("Log A", src.upper()).replace("log a", src.upper())
        reason = reason.replace("Log B", tgt.upper()).replace("log b", tgt.upper())
        
        return reason
    except:
        return "Parse Error"

def get_verification_status(row):
    """Categorize edge status."""
    if "llm_verification" not in row or not row["llm_verification"]:
        return "Unverified"
    try:
        data = json.loads(row["llm_verification"])
        if data.get("is_causal", False):
            return "Verified (Causal)"
        return "Rejected"
    except:
        return "Error"


# --- Page Logic ---
tab1, tab2, tab3 = st.tabs(["📊 Executive Dashboard", "🔍 Forensics", "📈 Sensitivity Analysis"])

# === TAB 1: DASHBOARD ===
with tab1:
    # Set the graph to take more space on the left
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("<h4 style='margin-top: -10px; margin-bottom: 0px;'>Service Dependency Map</h4>", unsafe_allow_html=True)
        # Load the pre-generated HTML graph
        graph_path = Path("data/service_dependency_graph_llm_verified.html")
        if not graph_path.exists():
            # Fallback
            graph_path = Path("data/service_dependency_graph.html")
        
        if graph_path.exists():
            with open(graph_path, 'r', encoding='utf-8') as f:
                html_data = f.read()
            
            # Update the style of the inner PyVis container before rendering
            # We look for the 'mynetwork' id which PyVis uses by default and force its height
            centered_html = f"""
            <style>
                #mynetwork {{
                    height: 450px !important;
                    width: 100% !important;
                    border: 1px solid #eee;
                    border-radius: 8px;
                    margin: 0 auto !important;
                }}
            </style>
            <div style="width: 100%; display: flex; justify-content: center;">
                <div style="width: 100%; max-width: 1200px;">
                    {html_data}
                </div>
            </div>
            """
            
            st.components.v1.html(
                centered_html,
                height=480, 
                scrolling=False
            )
            st.caption(f"Rendering: `{graph_path.name}`. Green = Verified, Red = Rejected, Blue = Unverified.")
        else:
            st.warning("No graph artifact found. Please run `visualize_graph.py` first.")

    with col2:
        st.subheader("System Metrics")
        
        # Quick stats from DB
        db_path = "data/openstack/edges/edges.db"
        if not Path(db_path).exists():
            # fallback path
            db_path = "data/edges/edges.db"
            
        df = load_edges_data(db_path)
        
        if df is not None:
            total_edges = len(df)
            col2.metric("Total Edges", total_edges)
            
            if "llm_verification" in df.columns:
                df["status"] = df.apply(get_verification_status, axis=1)
                counts = df["status"].value_counts()
                
                # Metrics in a single row
                m1, m2 = col2.columns(2)
                verified = counts.get("Verified (Causal)", 0)
                rejected = counts.get("Rejected", 0)
                
                m1.metric("Verified", verified)
                if (verified + rejected) > 0:
                    prec = (verified / (verified + rejected)) * 100
                    m2.metric("LLM Precision", f"{prec:.0f}%")
                
                col2.write("### Status Breakdown")
                col2.dataframe(counts, use_container_width=True)
                
                if (verified + rejected) > 0:
                    noise_red = (rejected / (verified + rejected)) * 100
                    col2.success(f"**{noise_red:.1f}%** Noise Filtered via LLM")
            else:
                col2.info("LLM Verification data not found in DB.")
        else:
            col2.error("Database not found.")


# === TAB 2: FORENSICS ===
with tab2:
    st.markdown("<h4 style='margin-top: -10px; margin-bottom: 0px;'>Deep Dive: Edge Inspection</h4>", unsafe_allow_html=True)
    
    if df is not None and "llm_verification" in df.columns:
        # Filters
        all_statuses = df["status"].unique().tolist() if "status" in df else ["Verified (Causal)", "Rejected", "Unverified"]
        status_filter = st.multiselect("Filter by Status", all_statuses, default=[x for x in all_statuses if "Verified" in x or "Rejected" in x])
        
        # Apply Logic
        df["reasoning"] = df.apply(parse_llm_reasoning, axis=1)
        sub_df = df[df["status"].isin(status_filter)] if "status" in df else df
        
        st.dataframe(
            sub_df[["source_service", "target_service", "status", "hybrid_score", "reasoning"]],
            use_container_width=True,
            height=420
        )
    else:
        st.warning("Forensics mode requires a database with LLM validation columns.")

# === TAB 3: SENSITIVITY ===
with tab3:
    st.markdown("<h4 style='margin-top: -10px; margin-bottom: 0px;'>Hyperparameter Sensitivity</h4>", unsafe_allow_html=True)
    
    # Left Column: Controls and Data Preview | Right Column: Plot and Analysis
    t3_col1, t3_col2 = st.columns([1, 1], gap="large")
    
    with t3_col1:
        st.markdown("**Dynamic Threshold Control**")
        threshold = st.slider("Hybrid Similarity Threshold (α)", 0.0, 1.0, 0.70, step=0.05, label_visibility="collapsed")
        st.write(f"Filtering at: **{threshold:.2f}**")

        if df is not None:
            filtered_df = df[df['hybrid_score'] >= threshold].copy()
            
            # Metrics in a two-column split for better alignment
            m1, m2 = st.columns(2)
            m1.metric("Survivors", len(filtered_df))
            
            # Precision Calculation
            if "status" in filtered_df.columns:
                counts = filtered_df["status"].value_counts()
                verified = counts.get("Verified (Causal)", 0)
                rejected = counts.get("Rejected", 0)
                if (verified + rejected) > 0:
                    m2.metric("LLM Precision", f"{(verified / (verified + rejected)) * 100:.0f}%")

            # Compact Preview
            st.dataframe(
                filtered_df[['source_service', 'target_service', 'hybrid_score']].head(8), 
                use_container_width=True,
                height=280
            )

    with t3_col2:
        img_path = Path("docs/images/sensitivity_plot.png")
        if img_path.exists():
            # Force refresh with raw bytes
            with open(img_path, "rb") as f:
                st.image(f.read(), caption="Thesis Evaluation Strategy: Precision Protection", use_container_width=True)
        else:
            st.info("Sensitivity plot not found. Run scripts/generate_research_plot.py")
            

    st.divider()
