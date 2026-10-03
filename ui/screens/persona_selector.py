from html import escape

import streamlit as st

from ui.components import genre_icon, render_empty_state, render_page_header
from ui.data_access import load_personas, load_user_id_index


def _select_viewer(user_id: int, label: str) -> None:
    st.session_state["selected_user_id"] = user_id
    st.session_state["selected_label"] = label


def render():
    render_page_header(
        "Screen 01",
        "Choose a viewer",
        "Pick a curated persona below, or browse recommendations for any real MovieLens user ID.",
    )

    personas = load_personas()

    if personas:
        st.markdown('<div class="mc-eyebrow" style="margin-bottom:0.7rem;">Curated personas</div>', unsafe_allow_html=True)
        cols = st.columns(min(len(personas), 4))
        for i, persona in enumerate(personas):
            top_genres = persona.get("top_genres", [])
            icon = genre_icon(top_genres[0]) if top_genres else genre_icon("")
            tags = "".join(f'<span class="mc-genre-tag">{escape(g)}</span>' for g in top_genres)
            with cols[i % len(cols)]:
                st.markdown(
                    f"""
                    <div class="mc-persona-card">
                        <div class="mc-persona-icon">{escape(icon)}</div>
                        <div class="mc-persona-name">{escape(persona['name'])}</div>
                        <div class="mc-persona-desc">{escape(persona['description'])}</div>
                        <div class="mc-persona-tags">{tags}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.button(
                    "View recommendations",
                    key=f"persona_{i}",
                    use_container_width=True,
                    on_click=_select_viewer,
                    args=(persona["user_id"], persona["name"]),
                )
    else:
        render_empty_state(
            "No curated personas cached yet.",
            action_hint="Run: python scripts/curate_personas.py, or browse by user ID below.",
        )

    st.markdown('<div class="mc-sprocket"></div>', unsafe_allow_html=True)
    st.markdown('<div class="mc-eyebrow" style="margin-bottom:0.7rem;">Browse by user ID</div>', unsafe_allow_html=True)

    user_index = load_user_id_index()
    if user_index:
        entered = int(st.number_input(
            "MovieLens user ID",
            min_value=user_index["min"],
            max_value=user_index["max"],
            value=user_index["min"],
            step=1,
            key="user_id_input",
            help=f"{user_index['count']:,} users available in the cached training split.",
        ))
        is_valid = entered in user_index["ids"]
        if not is_valid:
            st.caption("No user with that ID exists in the training split.")
        button_col, _ = st.columns([1, 2])
        with button_col:
            st.button(
                "View recommendations for this user",
                use_container_width=True,
                disabled=not is_valid,
                on_click=_select_viewer,
                args=(entered, f"User {entered}"),
            )
    else:
        render_empty_state(
            "No processed training data found.",
            action_hint="Run: python -m src.data.ingest &nbsp;then&nbsp; python scripts/run_phase1.py",
        )

    if st.session_state.get("selected_user_id") is not None:
        st.success(f"Selected: {st.session_state.get('selected_label')}. Open the Recommendations screen from the sidebar.")
