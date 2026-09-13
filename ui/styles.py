"""
Design tokens and custom CSS for the Streamlit UI.

Palette: "marquee" -- a single confident cinema red (theater curtains,
"now showing" signage) against near-black ink and true neutral paper,
rather than a warm cream-and-gold combination or a cool indigo
SaaS-dashboard combination -- both read as generic templates for
different reasons; a single bold identity color reads as a deliberate
choice instead. Paper background is genuinely neutral grey, not warm-
leaning or cool-leaning, so the red carries the color story on its own
rather than competing with a tinted background.

A sprocket-rail divider (a row of film-perforation dots) is the signature
structural element, and every model has a fixed color identity used
consistently across all three screens.

Token scale: colors, spacing, and radii are named and reused everywhere
below rather than hardcoded per rule, so the theme can be retuned in one
place -- this file -- without hunting through every screen's markdown.

Pairs with .streamlit/config.toml, which pins Streamlit's own [theme] base
to these same values. Without that file, Streamlit falls back to the
browser/OS's preferred color scheme for every native widget this file
doesn't explicitly override (dataframe, alerts, expanders, disabled
states, tooltips...) -- on a dark-mode system that makes large parts of
the UI render dark-on-dark or light-on-light regardless of what's set
here. The config.toml is not optional decoration, it's the fix for that.
"""

COLORS = {
    "bg_primary": "#F2F1ED",
    "bg_surface": "#FFFFFF",
    "bg_surface_raised": "#E6E3DA",
    "accent_marquee": "#B3242E",
    "accent_marquee_soft": "#D65C63",
    "accent_velvet": "#2B1B24",
    "accent_ink": "#17140F",
    "text_primary": "#1A1814",
    "text_muted": "#5C574C",
    "text_faint": "#8B8578",
    "border_subtle": "#DAD5C7",
    "border_strong": "#C6BFAC",
    "positive": "#2E7D4F",
    "warning": "#B8791E",
}

RADIUS = {"sm": "6px", "md": "10px", "lg": "16px"}
SHADOW = {
    "card": "0 1px 2px rgba(23, 20, 15, 0.06)",
    "card_hover": "0 4px 14px rgba(23, 20, 15, 0.10)",
    "raised": "0 2px 8px rgba(23, 20, 15, 0.07)",
}

# Fixed per-model identity, used consistently across all three screens so a
# user learns to recognize "red = two-tower" the same way they'd recognize
# a labeled film reel. Marquee red is reserved for the hero (two-tower)
# model; the other four sit in a cooler, quieter supporting range so the
# red keeps its role as the one color that means "the production model",
# not just one of five arbitrary chart colors.
MODEL_COLORS = {
    "popularity": "#6E685C",
    "item_item_cf": "#3E6E86",
    "als": "#3F7A4F",
    "bpr": "#5B8C63",
    "two_tower": "#B3242E",
    "sasrec": "#6B3F82",
}

MODEL_LABELS = {
    "popularity": "Popularity",
    "item_item_cf": "Item-Item CF",
    "als": "ALS",
    "bpr": "BPR",
    "two_tower": "Two-Tower",
    "sasrec": "SASRec",
}


def inject_custom_css() -> str:
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background-color: {COLORS['bg_primary']};
    color: {COLORS['text_primary']};
}}

.block-container {{
    max-width: 1180px;
    padding-top: 3.4rem;
}}

section[data-testid="stSidebar"] {{
    background-color: {COLORS['bg_surface']};
    border-right: 1px solid {COLORS['border_subtle']};
}}

h1, h2, h3 {{
    font-family: 'Fraunces', serif;
    font-weight: 600;
    color: {COLORS['text_primary']};
    letter-spacing: -0.01em;
}}

.mc-wordmark {{
    font-family: 'Fraunces', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: {COLORS['text_primary']};
    padding: 1.2rem 0 0.15rem 0;
    letter-spacing: -0.01em;
}}

.mc-wordmark span {{
    color: {COLORS['accent_marquee']};
}}

.mc-sidebar-tagline {{
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    color: {COLORS['text_muted']};
    line-height: 1.5;
    padding-bottom: 1.1rem;
    border-bottom: 1px solid {COLORS['border_subtle']};
    margin-bottom: 1rem;
}}

.mc-sidebar-footer {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    color: {COLORS['text_faint']};
    line-height: 1.7;
}}

.mc-stack-chip {{
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: {COLORS['text_muted']};
    background-color: {COLORS['bg_surface_raised']};
    border: 1px solid {COLORS['border_subtle']};
    border-radius: 4px;
    padding: 0.12rem 0.4rem;
    margin: 0.15rem 0.25rem 0.15rem 0;
}}

.mc-eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: {COLORS['accent_marquee']};
    font-weight: 500;
}}

.mc-page-title {{
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2rem;
    color: {COLORS['text_primary']};
    margin: 0.25rem 0 0.4rem 0;
    letter-spacing: -0.015em;
}}

.mc-page-subtitle {{
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    color: {COLORS['text_muted']};
    line-height: 1.5;
}}

/* Signature element: sprocket-rail divider, a row of film-perforation dots */
.mc-sprocket {{
    height: 14px;
    margin: 1.4rem 0;
    background-image: radial-gradient({COLORS['border_subtle']} 35%, transparent 36%);
    background-size: 18px 14px;
    background-repeat: repeat-x;
    opacity: 0.9;
}}

.mc-card {{
    background-color: {COLORS['bg_surface']};
    border: 1px solid {COLORS['border_subtle']};
    border-radius: {RADIUS['md']};
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.9rem;
    min-height: 148px;
    box-shadow: {SHADOW['card']};
    transition: box-shadow 0.15s ease, border-color 0.15s ease;
}}

.mc-card:hover {{
    border-color: {COLORS['accent_marquee']}77;
    box-shadow: {SHADOW['card_hover']};
}}

.mc-card-placeholder {{
    background-color: transparent;
    border: 1px dashed {COLORS['border_subtle']};
    box-shadow: none;
}}

/* Clamped to 2 lines with a matching min-height reserved either way, so a
one-line title ("Alien") and a four-line one ("Star Wars: Episode V - The
Empire Strikes Back") end up the same height instead of making their row
in the comparison grid uneven. */
.mc-card-title {{
    font-weight: 600;
    line-height: 1.3;
    min-height: 2.6rem;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}}

.mc-card-tags {{
    margin-top: 0.3rem;
    min-height: 1.6rem;
}}

.mc-rank-badge {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background-color: {COLORS['bg_surface_raised']};
    color: {COLORS['text_muted']};
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    margin-right: 0.55rem;
    flex-shrink: 0;
}}

.mc-score {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: {COLORS['accent_marquee']};
    margin-top: auto;
    padding-top: 0.4rem;
}}

.mc-genre-tag {{
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: {COLORS['text_muted']};
    border: 1px solid {COLORS['border_subtle']};
    border-radius: 4px;
    padding: 0.1rem 0.45rem;
    margin: 0.15rem 0.25rem 0 0;
}}

.mc-model-header {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding-bottom: 0.5rem;
    margin-bottom: 0.6rem;
    border-bottom: 2px solid;
}}

/* Fixed height + flex column, not height:100% -- height:100% only matches
whatever the immediate wrapper happens to be, which sizes to its own
content, so cards with a longer description grew taller than their
neighbors and threw off row alignment. A fixed height with the tag row
pinned to the bottom via margin-top:auto keeps every card in a row
identical regardless of description length. */
.mc-persona-card {{
    background-color: {COLORS['bg_surface']};
    border: 1px solid {COLORS['border_subtle']};
    border-radius: {RADIUS['md']};
    padding: 1.1rem 1.2rem;
    height: 232px;
    display: flex;
    flex-direction: column;
    box-shadow: {SHADOW['card']};
    transition: box-shadow 0.15s ease, transform 0.15s ease;
}}

.mc-persona-card:hover {{
    box-shadow: {SHADOW['card_hover']};
    transform: translateY(-1px);
}}

.mc-persona-icon {{
    font-size: 1.5rem;
    line-height: 1;
    margin-bottom: 0.5rem;
}}

.mc-persona-name {{
    font-family: 'Fraunces', serif;
    font-size: 1.08rem;
    font-weight: 600;
    color: {COLORS['text_primary']};
}}

.mc-persona-desc {{
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: {COLORS['text_muted']};
    margin-top: 0.3rem;
    line-height: 1.45;
    /* clamp to 3 lines instead of letting description length drive card
    height -- the fixed-height card above is what actually guarantees
    alignment, this just keeps long descriptions from overflowing it */
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}}

.mc-persona-tags {{
    margin-top: auto;
    padding-top: 0.6rem;
}}

/* KPI summary cards -- dashboard headline numbers. Fixed min-height +
flex column, same reasoning as .mc-card above: a 1-line caption
("Popularity") and a 2-line one ("Popularity, Item-Item CF, ALS / BPR")
would otherwise leave that card taller than its neighbors in the row. */
.mc-kpi-card {{
    background-color: {COLORS['bg_surface']};
    border: 1px solid {COLORS['border_subtle']};
    border-top: 3px solid {COLORS['accent_marquee']};
    border-radius: {RADIUS['sm']};
    padding: 0.85rem 1rem 0.95rem 1rem;
    min-height: 128px;
    display: flex;
    flex-direction: column;
    box-shadow: {SHADOW['card']};
}}

.mc-kpi-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {COLORS['text_muted']};
    margin-bottom: 0.3rem;
}}

.mc-kpi-value {{
    font-family: 'Fraunces', serif;
    font-size: 1.5rem;
    font-weight: 600;
    line-height: 1.15;
}}

.mc-kpi-caption {{
    font-family: 'Inter', sans-serif;
    font-size: 0.76rem;
    color: {COLORS['text_faint']};
    margin-top: 0.25rem;
    line-height: 1.4;
    min-height: 2.1rem;
}}

/* Empty / placeholder states, used wherever an artifact hasn't been built yet */
.mc-empty-state {{
    background-color: {COLORS['bg_surface']};
    border: 1px dashed {COLORS['border_strong']};
    border-radius: {RADIUS['md']};
    padding: 2.4rem 1.5rem;
    text-align: center;
}}

.mc-empty-icon {{
    font-size: 2rem;
    margin-bottom: 0.6rem;
}}

.mc-empty-message {{
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    color: {COLORS['text_primary']};
    font-weight: 500;
}}

.mc-empty-hint {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.76rem;
    color: {COLORS['text_muted']};
    margin-top: 0.5rem;
}}

.stButton > button {{
    background-color: {COLORS['bg_surface_raised']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border_subtle']};
    border-radius: {RADIUS['sm']};
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    transition: border-color 0.15s ease, color 0.15s ease;
}}

.stButton > button:hover {{
    border-color: {COLORS['accent_marquee']};
    color: {COLORS['accent_marquee']};
}}

.stButton > button:focus-visible {{
    outline: 2px solid {COLORS['accent_marquee_soft']};
    outline-offset: 1px;
}}

div[data-testid="stMetricValue"] {{
    font-family: 'IBM Plex Mono', monospace;
    color: {COLORS['accent_marquee']};
}}

hr {{
    border-color: {COLORS['border_subtle']};
}}

/* Streamlit's default header bar is bright and clashes with the theme --
this is a standalone demo app, not embedded, so blend it in and hide the
toolbar controls. */
header[data-testid="stHeader"] {{
    background-color: {COLORS['bg_primary']} !important;
}}

[data-testid="stToolbar"] {{
    visibility: hidden;
}}

/* Select/multiselect internals vary between Streamlit versions (BaseWeb vs
react-aria), so target the stable data-testid wrapper rather than
version-specific internal classes. */
div[data-testid="stSelectbox"] input,
div[data-testid="stMultiSelect"] input,
div[data-testid="stSelectbox"] [role="group"],
div[data-testid="stMultiSelect"] [role="group"],
div[data-testid="stSelectbox"] button,
div[data-testid="stMultiSelect"] button {{
    background-color: {COLORS['bg_surface_raised']} !important;
    border-color: {COLORS['border_subtle']} !important;
    color: {COLORS['text_primary']} !important;
}}

div[data-baseweb="popover"],
ul[role="listbox"] {{
    background-color: {COLORS['bg_surface_raised']} !important;
}}

li[role="option"] {{
    color: {COLORS['text_primary']} !important;
}}

span[data-baseweb="tag"] {{
    background-color: {COLORS['accent_velvet']} !important;
}}

/* Slider and toggle default to Streamlit's red accent -- recolor to the
clay accent so every interactive control matches the token system. */
div[data-testid="stSlider"] div[role="slider"] {{
    background-color: {COLORS['accent_marquee']} !important;
    border-color: {COLORS['accent_marquee']} !important;
}}

div[data-testid="stSlider"] > div > div > div > div {{
    background-color: {COLORS['accent_marquee']} !important;
}}

/* st.toggle renders as an input[role=switch] with the visible track as a
sibling div -- target via the label's data-selected state for the "on" color. */
label:has(input[role="switch"]:checked) > div:nth-of-type(1) {{
    background-color: {COLORS['accent_marquee']} !important;
}}

/* Sidebar nav radio defaults to Streamlit's red accent for the selected
dot -- recolor to match the rest of the token system. The visible dot is
actually two layered divs: an outer "ring" div with the red background,
and a smaller empty leaf div centered inside it. Both need recoloring, or
the red ring shows through around the edges. Auto-generated Streamlit
class names aren't stable across versions, so both are targeted
structurally instead of by class. */
label[data-testid="stRadioOption"][data-selected="true"] div:has(> div:not(:has(*))) {{
    background-color: {COLORS['accent_marquee']} !important;
}}

label[data-testid="stRadioOption"][data-selected="true"] div:not(:has(*)) {{
    background-color: {COLORS['accent_marquee']} !important;
    border-color: {COLORS['accent_marquee']} !important;
}}

label[data-testid="stRadioOption"][data-selected="true"] svg circle {{
    fill: {COLORS['accent_marquee']} !important;
    stroke: {COLORS['accent_marquee']} !important;
}}

/* Dataframe (the comparison table) -- explicit surface/text colors rather
than trusting the browser theme, since the grid/header/cell shading is
otherwise one of the widgets most likely to inherit a dark-mode palette. */
div[data-testid="stDataFrame"] {{
    background-color: {COLORS['bg_surface']};
    border: 1px solid {COLORS['border_subtle']};
    border-radius: {RADIUS['sm']};
}}

div[data-testid="stDataFrame"] [role="columnheader"] {{
    background-color: {COLORS['bg_surface_raised']} !important;
    color: {COLORS['text_primary']} !important;
}}

div[data-testid="stDataFrame"] [role="gridcell"] {{
    background-color: {COLORS['bg_surface']} !important;
    color: {COLORS['text_primary']} !important;
}}

/* Alerts (st.success / st.info / st.warning / st.error) and captions --
Streamlit's own semantic colors are fine, but the surrounding text/icon
color and border can otherwise default to the OS theme. */
div[data-testid="stAlert"] {{
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border_subtle']};
}}

div[data-testid="stCaptionContainer"], .stCaption {{
    color: {COLORS['text_muted']} !important;
}}

/* Expanders and popovers (help tooltips) -- same reasoning as alerts. */
div[data-testid="stExpander"] {{
    background-color: {COLORS['bg_surface']};
    border: 1px solid {COLORS['border_subtle']};
}}

div[data-baseweb="tooltip"] {{
    background-color: {COLORS['accent_ink']} !important;
    color: {COLORS['bg_primary']} !important;
}}

/* Multiselect chips (dashboard's "Metrics to chart" picker) render via a
BaseWeb tag component that ships its own default palette -- pin text
color explicitly since it's easy to end up with dark text on the dark
tag background above. */
span[data-baseweb="tag"] span {{
    color: {COLORS['bg_primary']} !important;
}}
</style>
"""
