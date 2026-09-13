"""Screen 3: model performance dashboard. Read-only, reads directly from
results/comparison_table.csv -- no recomputation in the UI."""

import math

import plotly.graph_objects as go
import polars as pl
import streamlit as st
from plotly.subplots import make_subplots

from ui.components import render_empty_state, render_kpi_row, render_model_chip, render_page_header
from ui.data_access import load_comparison_table
from ui.styles import COLORS, MODEL_COLORS, MODEL_LABELS


def _build_metrics_figure(table, metrics: list[str]):
    """One cohesive figure for all selected metrics, not N independent
    charts -- consistent bar width and spacing, per-subplot headroom so
    value labels never clip the panel edge, and a shared color legend
    instead of relying on color-learning from other screens."""
    models = table["model"].to_list()
    labels = [MODEL_LABELS.get(m, m) for m in models]
    colors = [MODEL_COLORS.get(m, COLORS["text_muted"]) for m in models]

    n = len(metrics)
    cols = min(n, 3)
    rows = math.ceil(n / cols)

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=metrics,
        horizontal_spacing=0.09,
        vertical_spacing=0.24 if rows > 1 else 0.15,
    )

    for i, metric in enumerate(metrics):
        row, col = i // cols + 1, i % cols + 1
        values = table[metric].to_list()
        headroom_max = max(values) * 1.22 if max(values) > 0 else 1.0

        fig.add_trace(
            go.Bar(
                x=labels, y=values, marker_color=colors,
                text=[f"{v:.3f}" for v in values], textposition="outside",
                textfont=dict(size=11, family="IBM Plex Mono, monospace"),
                showlegend=False,
            ),
            row=row, col=col,
        )
        fig.update_yaxes(
            range=[0, headroom_max], gridcolor=COLORS["border_subtle"],
            zerolinecolor=COLORS["border_subtle"], row=row, col=col,
        )
        fig.update_xaxes(tickangle=-20, row=row, col=col)

    fig.update_annotations(font=dict(family="Fraunces, serif", size=15, color=COLORS["text_primary"]))
    fig.update_layout(
        plot_bgcolor=COLORS["bg_surface"],
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLORS["text_primary"]),
        margin=dict(t=55, b=40, l=40, r=30),
        height=310 * rows,
        showlegend=False,
    )
    return fig


def _render_color_legend(models: list[str]):
    """Compact, self-contained color key so this screen reads correctly
    even for someone who lands here first, without having seen the
    color-coding established on the recommendations screen."""
    chips = "".join(render_model_chip(m, MODEL_LABELS.get(m, m)) for m in models)
    st.markdown(f'<div style="margin: 0.2rem 0 1.2rem 0;">{chips}</div>', unsafe_allow_html=True)


def _leaderboard_kpis(table: pl.DataFrame) -> list[dict]:
    """Headline numbers pulled from the table so the person doesn't have
    to read a bar chart just to find out which model currently wins on
    each metric."""
    kpis = [{
        "label": "Approaches compared",
        "value": str(table.height),
        "caption": ", ".join(MODEL_LABELS.get(m, m) for m in table["model"].to_list()),
        "accent": COLORS["accent_ink"],
    }]

    for metric, title in [("ndcg@10", "Best NDCG@10"), ("recall@10", "Best Recall@10"), ("diversity", "Best Diversity")]:
        if metric not in table.columns:
            continue
        best_row = table.sort(metric, descending=True).row(0, named=True)
        best_model = best_row["model"]
        kpis.append({
            "label": title,
            "value": f"{best_row[metric]:.3f}",
            "caption": MODEL_LABELS.get(best_model, best_model),
            "accent": MODEL_COLORS.get(best_model, COLORS["accent_marquee"]),
        })
    return kpis


def render():
    render_page_header(
        "Screen 03",
        "Model performance",
        "Every approach evaluated through the identical harness, on identical held-out data.",
    )

    table = load_comparison_table()
    if table is None:
        render_empty_state(
            "No results found.",
            icon="\U0001F4CA",
            action_hint="Run: python scripts/run_phase1.py &mdash; to populate results/comparison_table.csv.",
        )
        return

    render_kpi_row(_leaderboard_kpis(table))
    st.markdown('<div class="mc-sprocket"></div>', unsafe_allow_html=True)

    metric_cols = [c for c in table.columns if c != "model"]
    default_metrics = [c for c in ("recall@10", "ndcg@10") if c in metric_cols] or metric_cols[:2]

    selected_metrics = st.multiselect("Metrics to chart", options=metric_cols, default=default_metrics)

    if selected_metrics:
        _render_color_legend(table["model"].to_list())
        st.plotly_chart(_build_metrics_figure(table, selected_metrics), use_container_width=True)

    st.markdown('<div class="mc-sprocket"></div>', unsafe_allow_html=True)
    st.markdown('<div class="mc-eyebrow" style="margin-bottom:0.6rem;">Full comparison table</div>', unsafe_allow_html=True)

    display_table = table.with_columns(
        pl.col("model").map_elements(lambda m: MODEL_LABELS.get(m, m), return_dtype=pl.Utf8)
    ) if "model" in table.columns else table
    st.dataframe(display_table.to_pandas(), use_container_width=True, hide_index=True)
