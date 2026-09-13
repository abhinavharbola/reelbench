"""
Main Streamlit entrypoint. Three screens per project spec: viewer
selection, recommendations comparison, model performance dashboard. All
three read only cached local artifacts -- no live external calls, no
recomputation, no auth, no write-back.

Lives in a top-level ui/ package (not under src/), so the run command is
`streamlit run ui/app.py` from the repo root.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from ui.screens import dashboard, persona_selector, recommendations
from ui.styles import inject_custom_css

st.set_page_config(page_title="ReelBench: MovieLens Recommender", page_icon="\U0001F3AC", layout="wide")
st.markdown(inject_custom_css(), unsafe_allow_html=True)

if "selected_user_id" not in st.session_state:
    st.session_state["selected_user_id"] = None

with st.sidebar:
    st.markdown('<div class="mc-wordmark">Reel<span>Bench</span></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="mc-sidebar-tagline">A two-stage recommender benchmark on MovieLens 25M. '
        '5 approaches, one shared evaluation harness.</div>',
        unsafe_allow_html=True,
    )

    screen = st.radio(
        "Navigate",
        options=["Select Viewer", "Recommendations", "Model Metrics"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="mc-sprocket"></div>', unsafe_allow_html=True)
    if st.session_state["selected_user_id"] is not None:
        st.caption(f"Current viewer: {st.session_state.get('selected_label')}")
        st.markdown('<div class="mc-sprocket"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="mc-sidebar-footer">
            <span class="mc-stack-chip">Polars</span>
            <span class="mc-stack-chip">FAISS</span>
            <span class="mc-stack-chip">LightGBM</span>
            <span class="mc-stack-chip">PyTorch</span>
            <span class="mc-stack-chip">Streamlit</span>
            <div style="margin-top:0.6rem;">Served entirely from cached local artifacts.<br>No live model calls at request time.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if screen == "Select Viewer":
    persona_selector.render()
elif screen == "Recommendations":
    recommendations.render()
else:
    dashboard.render()
