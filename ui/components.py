import json
from html import escape

import streamlit as st
import streamlit.components.v1 as components

from ui.styles import COLORS, MODEL_COLORS

GENRE_ICONS = {
    "Action": "Ac",
    "Drama": "Dr",
    "Comedy": "Co",
    "Sci-Fi": "Sf",
    "Romance": "Ro",
    "Horror": "Ho",
    "Thriller": "Th",
    "Documentary": "Dc",
    "Animation": "An",
}
DEFAULT_GENRE_ICON = "Mv"


def genre_icon(genre: str) -> str:
    return GENRE_ICONS.get(genre, DEFAULT_GENRE_ICON)


def _scroll_to_top(screen_key: str) -> None:
    components.html(
        f"""
        <script>
        (function() {{
            var screenKey = {json.dumps(screen_key)};
            try {{
                var doc = window.parent.document;
                var candidates = [
                    doc.querySelector('section[data-testid="stMain"]'),
                    doc.querySelector('[data-testid="stAppViewContainer"]'),
                    doc.querySelector('section.main'),
                ];
                candidates.forEach(function(el) {{
                    if (el) {{ el.scrollTo({{top: 0, behavior: 'instant'}}); }}
                }});
                window.parent.scrollTo({{top: 0, behavior: 'instant'}});
            }} catch (e) {{}}
        }})();
        </script>
        """,
        height=0,
    )


def render_page_header(eyebrow: str, title: str, subtitle: str | None = None) -> None:
    _scroll_to_top(eyebrow)
    st.markdown(f'<div class="mc-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f'<h2 class="mc-page-title">{title}</h2>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="mc-page-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    st.markdown('<div class="mc-sprocket"></div>', unsafe_allow_html=True)


def render_kpi_row(items: list[dict]) -> None:
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        accent = item.get("accent") or COLORS["accent_marquee"]
        caption = item.get("caption")
        caption_html = f'<div class="mc-kpi-caption">{caption}</div>' if caption else ""
        with col:
            st.markdown(
                f"""
                <div class="mc-kpi-card" style="border-top-color:{accent};">
                    <div class="mc-kpi-label">{item['label']}</div>
                    <div class="mc-kpi-value" style="color:{accent};">{item['value']}</div>
                    {caption_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_empty_state(message: str, icon: str = "", action_hint: str | None = None) -> None:
    hint_html = f'<div class="mc-empty-hint">{action_hint}</div>' if action_hint else ""
    icon_html = f'<div class="mc-empty-icon">{escape(icon)}</div>' if icon else ""
    st.markdown(
        f"""
        <div class="mc-empty-state">
            {icon_html}
            <div class="mc-empty-message">{message}</div>
            {hint_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_chip(model_name: str, label: str) -> str:
    color = MODEL_COLORS.get(model_name, COLORS["text_muted"])
    return (
        f'<span style="display:inline-flex; align-items:center; margin-right:1rem;">'
        f'<span style="width:9px; height:9px; border-radius:50%; background:{color}; '
        f'display:inline-block; margin-right:0.4rem;"></span>'
        f'<span style="font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:0.74rem; color:{COLORS["text_muted"]};">{escape(label)}</span>'
        f'</span>'
    )