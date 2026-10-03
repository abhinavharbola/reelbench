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
html, body, [class*="css"] {{
    font-family: system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
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
    font-family: Georgia, 'Times New Roman', serif;
    font-weight: 600;
    color: {COLORS['text_primary']};
    letter-spacing: -0.01em;
}}

.mc-wordmark {{
    font-family: Georgia, 'Times New Roman', serif;
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
    font-family: system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
    font-size: 0.78rem;
    color: {COLORS['text_muted']};
    line-height: 1.5;
    padding-bottom: 1.1rem;
    border-bottom: 1px solid {COLORS['border_subtle']};
    margin-bottom: 1rem;
}}

.mc-sidebar-footer {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.66rem;
    color: {COLORS['text_faint']};
    line-height: 1.7;
}}

.mc-stack-chip {{
    display: inline-block;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.62rem;
    color: {COLORS['text_muted']};
    background-color: {COLORS['bg_surface_raised']};
    border: 1px solid {COLORS['border_subtle']};
    border-radius: 4px;
    padding: 0.12rem 0.4rem;
    margin: 0.15rem 0.25rem 0.15rem 0;
}}

.mc-eyebrow {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: {COLORS['accent_marquee']};
    font-weight: 500;
}}

.mc-page-title {{
    font-family: Georgia, 'Times New Roman', serif;
    font-weight: 600;
    font-size: 2rem;
    color: {COLORS['text_primary']};
    margin: 0.25rem 0 0.4rem 0;
    letter-spacing: -0.015em;
}}

.mc-page-subtitle {{
    font-family: system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
    font-size: 0.95rem;
    color: {COLORS['text_muted']};
    line-height: 1.5;
}}

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
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.68rem;
    font-weight: 500;
    margin-right: 0.55rem;
    flex-shrink: 0;
}}

.mc-score {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.85rem;
    color: {COLORS['accent_marquee']};
    margin-top: auto;
    padding-top: 0.4rem;
}}

.mc-genre-tag {{
    display: inline-block;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.68rem;
    color: {COLORS['text_muted']};
    border: 1px solid {COLORS['border_subtle']};
    border-radius: 4px;
    padding: 0.1rem 0.45rem;
    margin: 0.15rem 0.25rem 0 0;
}}

.mc-model-header {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.78rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding-bottom: 0.5rem;
    margin-bottom: 0.6rem;
    border-bottom: 2px solid;
}}

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
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    line-height: 1;
    margin-bottom: 0.6rem;
    color: {COLORS['accent_marquee']};
}}

.mc-persona-name {{
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 1.08rem;
    font-weight: 600;
    color: {COLORS['text_primary']};
}}

.mc-persona-desc {{
    font-family: system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
    font-size: 0.85rem;
    color: {COLORS['text_muted']};
    margin-top: 0.3rem;
    line-height: 1.45;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}}

.mc-persona-tags {{
    margin-top: auto;
    padding-top: 0.6rem;
}}

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
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {COLORS['text_muted']};
    margin-bottom: 0.3rem;
}}

.mc-kpi-value {{
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 1.5rem;
    font-weight: 600;
    line-height: 1.15;
}}

.mc-kpi-caption {{
    font-family: system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
    font-size: 0.76rem;
    color: {COLORS['text_faint']};
    margin-top: 0.25rem;
    line-height: 1.4;
    min-height: 2.1rem;
}}

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
    font-family: system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
    font-size: 0.95rem;
    color: {COLORS['text_primary']};
    font-weight: 500;
}}

.mc-empty-hint {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.76rem;
    color: {COLORS['text_muted']};
    margin-top: 0.5rem;
}}

.stButton > button {{
    background-color: {COLORS['bg_surface_raised']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border_subtle']};
    border-radius: {RADIUS['sm']};
    font-family: system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
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
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    color: {COLORS['accent_marquee']};
}}

hr {{
    border-color: {COLORS['border_subtle']};
}}

header[data-testid="stHeader"] {{
    background-color: {COLORS['bg_primary']} !important;
}}

[data-testid="stToolbar"] {{
    visibility: hidden;
}}

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

div[data-testid="stSlider"] div[role="slider"] {{
    background-color: {COLORS['accent_marquee']} !important;
    border-color: {COLORS['accent_marquee']} !important;
}}

div[data-testid="stSlider"] > div > div > div > div {{
    background-color: {COLORS['accent_marquee']} !important;
}}

label:has(input[role="switch"]:checked) > div:nth-of-type(1) {{
    background-color: {COLORS['accent_marquee']} !important;
}}

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

div[data-testid="stAlert"] {{
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border_subtle']};
}}

div[data-testid="stCaptionContainer"], .stCaption {{
    color: {COLORS['text_muted']} !important;
}}

div[data-testid="stExpander"] {{
    background-color: {COLORS['bg_surface']};
    border: 1px solid {COLORS['border_subtle']};
}}

div[data-baseweb="tooltip"] {{
    background-color: {COLORS['accent_ink']} !important;
    color: {COLORS['bg_primary']} !important;
}}

span[data-baseweb="tag"] span {{
    color: {COLORS['bg_primary']} !important;
}}
</style>
"""
