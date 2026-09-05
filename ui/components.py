"""
Small shared UI building blocks used by every screen, so page headers, KPI
cards, and empty states look and behave identically everywhere instead of
each screen hand-rolling its own markdown. Kept deliberately minimal --
these are thin wrappers over st.markdown with the design tokens in
ui/styles.py, not a component framework.
"""

import streamlit as st
import streamlit.components.v1 as components

from ui.styles import COLORS, MODEL_COLORS

# Icons drawn from the Miscellaneous Symbols / Dingbats / Geometric Shapes
# Unicode blocks (the same family crossed-swords and comet already came
# from), not the newer full-color emoji blocks. These default to plain
# monochrome glyph presentation without a variation selector, so they read
# as typographic symbols rather than cartoon pictures -- consistent with
# the app's editorial mono/serif type pairing elsewhere. Not a literal
# icon per genre in every case (there's no equivalent-weight symbol for
# "drama masks" in this family), prioritizing visual consistency across
# the full set over a literal pictogram for each one.
GENRE_ICONS = {
    "Action": "\u2694",       # crossed swords
    "Drama": "\u2021",        # double dagger (typesetting mark, editorial)
    "Comedy": "\u263A",       # white smiling face
    "Sci-Fi": "\u2604",       # comet
    "Romance": "\u2764",      # heavy black heart
    "Horror": "\u2620",       # skull and crossbones
    "Thriller": "\u25CE",     # bullseye
    "Documentary": "\u25C9",  # fisheye (lens)
    "Animation": "\u2726",    # black four pointed star
}
DEFAULT_GENRE_ICON = "\u25B6"  # black right-pointing triangle (play)


def genre_icon(genre: str) -> str:
    return GENRE_ICONS.get(genre, DEFAULT_GENRE_ICON)


def _scroll_to_top() -> None:
    """Fixes the "page title half-cut by the top bar" bug: this app is a
    single script with `if screen == X: render()` (see ui/app.py), not a
    true multi-page st.Page app, so switching the sidebar radio does NOT
    reset the browser's scroll position -- it only swaps which screen's
    content renders into the same scrollable container. If someone
    scrolled down on a tall screen (Recommendations, with 5 columns of
    movie cards) and then switches to a shorter one, the new screen's
    title renders where the old scroll position happens to be, which can
    land it under Streamlit's floating top header bar.

    st.markdown can't fix this -- it strips <script> tags entirely.
    st.components.v1.html renders in a real iframe where scripts do run,
    and can reach into the parent app document to scroll it back to top.
    Called once at the start of every screen via render_page_header, so
    every screen switch lands back at the top regardless of where the
    previous screen left the scroll position.

    Uses components.v1.html rather than the newer st.iframe, since that's
    what's available on this project's pinned streamlit==1.39.0; if you
    upgrade streamlit past the version where st.iframe replaces it, swap
    this call accordingly.
    """
    components.html(
        """
        <script>
        (function() {
            try {
                var doc = window.parent.document;
                var candidates = [
                    doc.querySelector('section[data-testid="stMain"]'),
                    doc.querySelector('[data-testid="stAppViewContainer"]'),
                    doc.querySelector('section.main'),
                ];
                candidates.forEach(function(el) {
                    if (el) { el.scrollTo({top: 0, behavior: 'instant'}); }
                });
                window.parent.scrollTo({top: 0, behavior: 'instant'});
            } catch (e) {}
        })();
        </script>
        """,
        height=0,
    )


def render_page_header(eyebrow: str, title: str, subtitle: str | None = None) -> None:
    """Consistent top-of-screen header: monospace eyebrow label, serif
    title, optional muted subtitle. Every screen opens with this instead
    of ad hoc st.markdown calls, so the visual rhythm going from screen to
    screen is identical -- and so the scroll-to-top fix above only needs
    to live in one place."""
    _scroll_to_top()
    st.markdown(f'<div class="mc-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f'<h2 class="mc-page-title">{title}</h2>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="mc-page-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    st.markdown('<div class="mc-sprocket"></div>', unsafe_allow_html=True)


def render_kpi_row(items: list[dict]) -> None:
    """items: [{"label": str, "value": str, "caption": str|None, "accent": str|None}, ...]
    Renders a row of compact metric cards -- used on the dashboard to
    surface the headline numbers (best model per metric, dataset size)
    before the person has to read a bar chart to find them."""
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


def render_empty_state(message: str, icon: str = "\U0001F3AC", action_hint: str | None = None) -> None:
    """A single, visually consistent placeholder for every "nothing here
    yet" case (no personas cached, no results table, no artifacts built)
    instead of a plain st.warning -- keeps the empty states from feeling
    like an afterthought."""
    hint_html = f'<div class="mc-empty-hint">{action_hint}</div>' if action_hint else ""
    st.markdown(
        f"""
        <div class="mc-empty-state">
            <div class="mc-empty-icon">{icon}</div>
            <div class="mc-empty-message">{message}</div>
            {hint_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_chip(model_name: str, label: str) -> str:
    """Small inline colored dot + label, matching a model's fixed color
    identity. Returns HTML (caller embeds it), doesn't render directly."""
    color = MODEL_COLORS.get(model_name, COLORS["text_muted"])
    return (
        f'<span style="display:inline-flex; align-items:center; margin-right:1rem;">'
        f'<span style="width:9px; height:9px; border-radius:50%; background:{color}; '
        f'display:inline-block; margin-right:0.4rem;"></span>'
        f'<span style="font-family:\'IBM Plex Mono\',monospace; font-size:0.74rem; color:{COLORS["text_muted"]};">{label}</span>'
        f'</span>'
    )

