from html import escape

import streamlit as st

from src.config import MODELS_DIR
from ui.components import render_empty_state, render_page_header
from ui.data_access import get_recommendations, load_model_registry
from ui.styles import MODEL_COLORS, MODEL_LABELS


def _render_movie_card(item: dict | None, accent: str, rank: int) -> str:
    if item is None:
        return '<div class="mc-card mc-card-placeholder"></div>'

    genres = [g for g in item["genres"].split("|") if g][:3]
    genre_tags = "".join(f'<span class="mc-genre-tag">{escape(g)}</span>' for g in genres)
    score_line = f"score {item['score']:.3f}" if item["score"] is not None else f"rank #{rank}"
    return f"""
    <div class="mc-card" style="border-left: 3px solid {accent};">
        <div style="display:flex; align-items:flex-start;">
            <span class="mc-rank-badge">{rank}</span>
            <div style="flex:1; display:flex; flex-direction:column; min-height:100%;">
                <div class="mc-card-title">{escape(item['title'])}</div>
                <div class="mc-card-tags">{genre_tags}</div>
                <div class="mc-score">{score_line}</div>
            </div>
        </div>
    </div>
    """


def _render_comparison_grid(available_models: list[str], user_id: int, top_n: int) -> None:
    header_cols = st.columns(len(available_models))
    for col, model_name in zip(header_cols, available_models):
        accent = MODEL_COLORS[model_name]
        with col:
            st.markdown(
                f'<div class="mc-model-header" style="color:{accent}; border-color:{accent};">'
                f'{MODEL_LABELS[model_name]}</div>',
                unsafe_allow_html=True,
            )

    recs_by_model = {m: get_recommendations(m, user_id, k=top_n) for m in available_models}
    if all(not recs for recs in recs_by_model.values()):
        st.caption("No recommendations available for this user.")
        return

    max_len = max(len(recs) for recs in recs_by_model.values())
    for rank in range(max_len):
        row_cols = st.columns(len(available_models))
        for col, model_name in zip(row_cols, available_models):
            accent = MODEL_COLORS[model_name]
            items = recs_by_model[model_name]
            item = items[rank] if rank < len(items) else None
            with col:
                st.markdown(_render_movie_card(item, accent, rank + 1), unsafe_allow_html=True)


def render():
    render_page_header(
        "Screen 02",
        "Recommendations",
        "The same viewer, run through every trained approach, so you can compare what each one surfaces.",
    )

    user_id = st.session_state.get("selected_user_id")
    if user_id is None:
        render_empty_state(
            "No viewer selected yet.",
            action_hint="Go to the Select Viewer screen first.",
        )
        return

    registry = load_model_registry()
    available_models = [m for m in MODEL_LABELS if m in registry]
    for name, message in registry.get("_errors", {}).items():
        st.warning(f"{MODEL_LABELS.get(name, name)} artifacts were skipped: {message}")

    if not available_models:
        render_empty_state(
            f"No trained models found in {MODELS_DIR}.",
            action_hint="Run the Phase 1-3 training scripts first.",
        )
        return

    col_a, col_b = st.columns([3, 1])
    with col_a:
        top_n = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)
    with col_b:
        compare_all = st.toggle(f"Compare all {len(available_models)} approaches", value=len(available_models) > 1)

    st.markdown('<div class="mc-sprocket"></div>', unsafe_allow_html=True)

    if compare_all:
        _render_comparison_grid(available_models, user_id, top_n)
    else:
        model_name = st.selectbox(
            "Approach", options=available_models, format_func=lambda m: MODEL_LABELS[m]
        )
        accent = MODEL_COLORS[model_name]
        recs = get_recommendations(model_name, user_id, k=top_n)
        if not recs:
            st.caption("No recommendations available for this user.")
        for rank, item in enumerate(recs, start=1):
            st.markdown(_render_movie_card(item, accent, rank), unsafe_allow_html=True)
