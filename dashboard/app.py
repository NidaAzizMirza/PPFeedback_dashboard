"""
============================================================================
 DASHBOARD/APP.PY — Planning Portal Feedback Dashboard (Streamlit)
============================================================================
Launch with:
    streamlit run dashboard/app.py

Reads ONLY from SQLite (data/metrics.db) via store.py — never touches raw
survey files or the SurveyMonkey API directly. Run run_pipeline.py (or the
scheduled GitHub Action) first to populate data.

Page structure:
    Tab 1 — Monthly view      : single-month "at a glance" snapshot (KPI
                                 cards, ratings, this-month sentiment split),
                                 styled to match the earlier GitHub Pages
                                 design.
    Tab 2 — Aggregate view    : high-level multi-month trends (Overview
                                 overlay chart + Sentiment split).
    Tab 3 — Feedback comments : everything derived from free-text comments
                                 (tag groups, features, errors, browse
                                 reviews), with its own Month by month /
                                 Aggregate sub-tabs.

Edit THIS file if you want to:
  - Change a page's layout or charts (Section 4 — PAGE FUNCTIONS)
  - Add a brand new page (Section 4 to write it, Section 6 to register it)
  - Change chart colours globally — edit dashboard_config.py instead
============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard_config import PALETTE, SERIES_COLORS, SENTIMENT_COLORS
from store import (
    db_exists,
    load_available_months,
    load_latest_run_date,
    load_monthly_summary,
    load_tag_group_trends,
    load_feature_trends,
    load_error_trends,
    load_review_detail,
    filter_by_months,
)

matplotlib.use("Agg")

# Emoji used in place of the raw positive/negative/neutral text labels
# (Browse Reviews) — matches the SurveyMonkey-style face icons.
SENTIMENT_EMOJI = {
    "positive": "🙂",
    "neutral": "😐",
    "negative": "🙁",
}


# ============================================================================
# SECTION 1 — PAGE CONFIG + CSS
# ============================================================================
st.set_page_config(
    page_title="Planning Portal Feedback",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_css():
    """Card + pill styling to match the earlier GitHub Pages design."""
    st.markdown(
        f"""
        <style>
        .kpi-card {{
            background: {PALETTE["paper"]};
            border: 1px solid {PALETTE["grid"]};
            border-radius: 10px;
            padding: 16px 20px;
            height: 100%;
        }}
        .kpi-label {{
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: {PALETTE["muted"]};
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 2.0rem;
            font-weight: 700;
            color: {PALETTE["ink"]};
            line-height: 1.1;
        }}
        .kpi-delta-up {{
            font-size: 0.85rem;
            color: {PALETTE["positive"]};
            margin-top: 4px;
        }}
        .kpi-delta-down {{
            font-size: 0.85rem;
            color: {PALETTE["negative"]};
            margin-top: 4px;
        }}
        .kpi-delta-flat {{
            font-size: 0.85rem;
            color: {PALETTE["muted"]};
            margin-top: 4px;
        }}
        div[data-testid="stButton"] > button {{
            border-radius: 999px;
            border: 1px solid {PALETTE["grid"]};
            padding: 2px 16px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _kpi_card(label: str, value: str, delta: float | None = None,
              delta_suffix: str = "", higher_is_better: bool = True,
              sublabel: str | None = None):
    delta_html = ""
    if delta is not None:
        if abs(delta) < 1e-9:
            cls, arrow = "kpi-delta-flat", "→"
        elif (delta > 0) == higher_is_better:
            cls, arrow = "kpi-delta-up", "↑"
        else:
            cls, arrow = "kpi-delta-down", "↓"
        delta_html = f'<div class="{cls}">{arrow} {abs(delta):.1f}{delta_suffix} vs prior month</div>'
    elif sublabel:
        # Same visual slot as the delta line, just neutral-colored —
        # keeps the card's height consistent whether it shows a trend
        # delta or a plain sublabel like "40% of respondents".
        delta_html = f'<div class="kpi-delta-flat">{sublabel}</div>'

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# SECTION 2 — CHART HELPERS (shared across pages)
# ============================================================================

def _line_trend(df: pd.DataFrame, x: str, y: str, title: str,
                 ylabel: str = "", color: str = None) -> plt.Figure:
    color = color or PALETTE["primary"]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor(PALETTE["paper"])
    ax.set_facecolor(PALETTE["paper"])
    ax.plot(df[x], df[y], marker="o", color=color, linewidth=2)
    ax.set_title(title, fontsize=12, loc="left", pad=10, color=PALETTE["ink"])
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30, colors=PALETTE["ink"])
    ax.tick_params(axis="y", colors=PALETTE["ink"])
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    fig.tight_layout()
    return fig


def _bar_trend(df: pd.DataFrame, x: str, y: str, title: str,
               ylabel: str = "") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor(PALETTE["paper"])
    ax.set_facecolor(PALETTE["paper"])
    ax.bar(df[x], df[y], color=PALETTE["primary"], width=0.5)
    ax.set_title(title, fontsize=12, loc="left", pad=10, color=PALETTE["ink"])
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30, colors=PALETTE["ink"])
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    fig.tight_layout()
    return fig


def _sentiment_stacked_bar(df: pd.DataFrame, group_col: str, title: str,
                            pct: bool = True) -> plt.Figure:
    """Stacked bar of positive/negative/neutral counts (or %) per group_col value."""
    table = df.groupby(group_col)[["positive_count", "negative_count", "neutral_count"]].sum()
    table = table.rename(columns={
        "positive_count": "positive", "negative_count": "negative", "neutral_count": "neutral"
    })
    if pct:
        totals = table.sum(axis=1)
        table = table.div(totals, axis=0).fillna(0) * 100
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]

    n_rows = len(table)
    fig_h = max(4, n_rows * 0.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    fig.patch.set_facecolor(PALETTE["paper"])
    ax.set_facecolor(PALETTE["paper"])

    categories = table.index.tolist()
    y = np.arange(len(categories))
    lefts = np.zeros(len(categories))

    for col in ["negative", "neutral", "positive"]:
        if col not in table.columns:
            continue
        vals = table[col].values.astype(float)
        color = SENTIMENT_COLORS[col]
        bars = ax.barh(y, vals, left=lefts, label=col.title(), color=color, height=0.6)
        for rect, v in zip(bars, vals):
            if v > (2 if pct else 0.5):
                label = f"{v:.0f}%" if pct else str(int(v))
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_y() + rect.get_height() / 2,
                        label, ha="center", va="center", fontsize=8, color=PALETTE["paper"])
        lefts += vals

    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=9)
    ax.set_xlabel("% of reviews" if pct else "Reviews")
    ax.set_title(title, fontsize=12, loc="left", pad=12, color=PALETTE["ink"])
    ax.legend(frameon=True, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_xlim(0, lefts.max() * 1.05 if lefts.max() > 0 else 1)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.5)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    return fig


def _single_sentiment_bar(pos_pct: float, neg_pct: float, neu_pct: float,
                           title: str) -> plt.Figure:
    """One horizontal 100%-stacked bar — matches the 'Overall comment sentiment' card."""
    fig, ax = plt.subplots(figsize=(10, 1.4))
    fig.patch.set_facecolor(PALETTE["paper"])
    ax.set_facecolor(PALETTE["paper"])
    segments = [("negative", neg_pct), ("neutral", neu_pct), ("positive", pos_pct)]
    left = 0
    for name, val in segments:
        ax.barh([0], [val], left=left, color=SENTIMENT_COLORS[name], height=0.6)
        if val > 3:
            ax.text(left + val / 2, 0, f"{val:.1f}%", ha="center", va="center",
                     fontsize=10, color=PALETTE["paper"], fontweight="bold")
        left += val
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, loc="left", pad=8, color=PALETTE["ink"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def _with_alpha(hex_color: str, alpha: float) -> str:
    """
    Blend a hex color with the white paper background at the given
    alpha, returning a solid hex color. Used to get lighter ★4/★2
    shades from the existing positive/negative palette entries without
    needing new colors in dashboard_config.py, and without matplotlib's
    bar alpha also fading gridlines/labels behind the bar.
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r2 = round(r * alpha + 255 * (1 - alpha))
    g2 = round(g * alpha + 255 * (1 - alpha))
    b2 = round(b * alpha + 255 * (1 - alpha))
    return f"#{r2:02x}{g2:02x}{b2:02x}"


def _rating_distribution_bar(counts: dict, title: str) -> plt.Figure:
    """
    Horizontal ★5..★1 bar chart, green-to-red, matching the earlier
    GitHub Pages design. counts: {5: n, 4: n, 3: n, 2: n, 1: n}.
    """
    stars = [5, 4, 3, 2, 1]
    values = [counts.get(s) or 0 for s in stars]
    colors = [
        PALETTE["positive"],
        _with_alpha(PALETTE["positive"], 0.55),
        PALETTE["neutral"],
        _with_alpha(PALETTE["negative"], 0.55),
        PALETTE["negative"],
    ]
    labels = [f"★{s}" for s in stars]

    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor(PALETTE["paper"])
    ax.set_facecolor(PALETTE["paper"])
    bars = ax.barh(labels, values, color=colors, height=0.6)
    max_val = max(values) if max(values) > 0 else 1
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + max_val * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{int(v)}", va="center", fontsize=9, color=PALETTE["ink"])
    ax.invert_yaxis()  # ★5 on top, matching the earlier design
    if title:
        ax.set_title(title, fontsize=12, loc="left", pad=10, color=PALETTE["ink"])
    ax.set_xlabel("Responses")
    ax.set_xlim(0, max_val * 1.15)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.5)
    fig.tight_layout()
    return fig


def _diverging_sentiment_bar(df: pd.DataFrame, group_col: str, score_col: str,
                              title: str) -> plt.Figure:
    """Horizontal bar from -1 (all negative) to +1 (all positive) per category."""
    table = df.groupby(group_col)[score_col].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, max(4, len(table) * 0.4)))
    fig.patch.set_facecolor(PALETTE["paper"])
    ax.set_facecolor(PALETTE["paper"])
    colors = [SENTIMENT_COLORS["positive"] if v >= 0 else SENTIMENT_COLORS["negative"]
              for v in table.values]
    ax.barh(table.index, table.values, color=colors, height=0.6)
    ax.axvline(0, color=PALETTE["ink"], linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Sentiment score (-1 = all negative, +1 = all positive)")
    ax.set_title(title, fontsize=12, loc="left", pad=10, color=PALETTE["ink"])
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    fig.tight_layout()
    return fig


def _overlay_trend_chart(trend: pd.DataFrame) -> plt.Figure:
    """
    Single combined chart replacing the old separate volume / avg rating /
    NSAT graphs on the Overview page (change #2):
      - grey bars   : total respondents (own visual scale, real counts labeled)
      - blue line   : average rating (left axis, 1-5 scale)
      - green line  : NSAT % (right axis, 0-100 scale — genuinely to scale)
      - purple line : respondents who left feedback (own visual scale,
                      real counts labeled)

    Bars and the "comments" line don't share a real numeric axis with the
    rating line — there's no single scale that fits 1-5, 0-100%, and raw
    counts in the thousands on the same plot without one of them going
    flat. Instead (matching the reference design) they're normalised to
    sit visually within the rating axis's range, with their real values
    printed as data labels so nothing is misleading.
    """
    fig, ax_rating = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(PALETTE["paper"])
    ax_rating.set_facecolor(PALETTE["paper"])

    months = trend["month"].tolist()
    x = np.arange(len(months))

    rating_top = float(trend["avg_rating"].max()) * 1.35 if trend["avg_rating"].notna().any() else 5
    ax_rating.set_ylim(0, rating_top)

    # --- background bars: response volume (visual scale only) -----------
    respondents = trend["total_respondents"].astype(float)
    bar_scale_max = rating_top * 0.72
    bar_heights = respondents / respondents.max() * bar_scale_max if respondents.max() else respondents * 0
    ax_rating.bar(x, bar_heights, color=_with_alpha(PALETTE["muted"], 0.28),
                  width=0.55, zorder=1, label="Total responses")
    for xi, (h, real) in enumerate(zip(bar_heights, respondents)):
        ax_rating.text(xi, h + rating_top * 0.015, f"{int(real):,}", ha="center",
                        va="bottom", fontsize=7.5, color=PALETTE["muted"])

    # --- comments line: respondents who left feedback (visual scale only)
    comments = trend["total_with_feedback"].astype(float) if "total_with_feedback" in trend else None
    if comments is not None and comments.notna().any():
        comments_scale_max = rating_top * 0.28
        comments_norm = comments / comments.max() * comments_scale_max if comments.max() else comments * 0
        ax_rating.plot(x, comments_norm, marker="^", linestyle="--", linewidth=1.6,
                        color=PALETTE["accent"], zorder=3, label="Comments")
        for xi, (v, real) in enumerate(zip(comments_norm, comments)):
            if pd.notna(real):
                ax_rating.text(xi, v + rating_top * 0.02, f"{int(real):,}", ha="center",
                                va="bottom", fontsize=7.5, color=PALETTE["accent"])

    # --- average rating: real values, primary axis -----------------------
    ax_rating.plot(x, trend["avg_rating"], marker="o", linewidth=2.2,
                    color=PALETTE["primary"], zorder=4, label="Avg rating")
    for xi, v in zip(x, trend["avg_rating"]):
        if pd.notna(v):
            ax_rating.text(xi, v + rating_top * 0.02, f"{v:.2f}", ha="center",
                            va="bottom", fontsize=8, color=PALETTE["primary"], fontweight="bold")

    ax_rating.set_ylabel("Average rating (1-5) / responses / comments", fontsize=9)
    ax_rating.set_xticks(x)
    ax_rating.set_xticklabels(months, rotation=30)
    ax_rating.grid(axis="y", color=PALETTE["grid"], linewidth=0.5)
    for spine in ax_rating.spines.values():
        spine.set_color(PALETTE["grid"])

    # --- NSAT: real values, secondary axis --------------------------------
    ax_nsat = ax_rating.twinx()
    ax_nsat.set_facecolor("none")
    ax_nsat.plot(x, trend["nsat"], marker="s", linestyle="--", linewidth=1.8,
                 color=PALETTE["positive"], zorder=5, label="NSAT %")
    for xi, v in zip(x, trend["nsat"]):
        if pd.notna(v):
            ax_nsat.text(xi, v + 0.8, f"{v:.1f}%", ha="center", va="bottom",
                         fontsize=8, color=PALETTE["positive"], fontweight="bold")
    ax_nsat.set_ylabel("NSAT %", fontsize=9, color=PALETTE["positive"])
    ax_nsat.tick_params(axis="y", colors=PALETTE["positive"])
    nsat_vals = trend["nsat"].dropna()
    if not nsat_vals.empty:
        ax_nsat.set_ylim(max(0, nsat_vals.min() - 15), nsat_vals.max() + 15)
    for spine in ax_nsat.spines.values():
        spine.set_visible(False)

    lines1, labels1 = ax_rating.get_legend_handles_labels()
    lines2, labels2 = ax_nsat.get_legend_handles_labels()
    ax_rating.legend(lines1 + lines2, labels1 + labels2, frameon=True, fontsize=8,
                      loc="upper left", bbox_to_anchor=(0, 1.15), ncol=4)

    ax_rating.set_title("Ratings, NSAT, responses & comments", fontsize=12, loc="left",
                         pad=32, color=PALETTE["ink"])
    fig.tight_layout()
    return fig


def _feature_negative_heatmap(feat_view: pd.DataFrame, top_n: int = 12) -> plt.Figure | None:
    """
    Feature x month heatmap, cell color = % negative mentions.
    Replaces the "features to compare" line chart (change #5) — the line
    chart got unreadable once more than 2-3 features were selected;
    a heatmap scales to many features/months at a glance instead.
    """
    if feat_view.empty:
        return None

    grouped = feat_view.groupby(["month", "feature"]).agg(
        total_mentions=("total_mentions", "sum"),
        negative_mentions=("negative_mentions", "sum"),
    ).reset_index()
    grouped["negative_pct"] = (grouped["negative_mentions"] / grouped["total_mentions"] * 100).round(1)

    top_features = (
        grouped.groupby("feature")["total_mentions"].sum()
        .sort_values(ascending=False).head(top_n).index.tolist()
    )
    grouped = grouped[grouped["feature"].isin(top_features)]
    if grouped.empty:
        return None

    pivot = grouped.pivot_table(index="feature", columns="month", values="negative_pct")
    pivot = pivot.loc[top_features]  # keep ranked order, most-mentioned feature on top

    cmap = LinearSegmentedColormap.from_list(
        "neg_pct", [PALETTE["positive"], PALETTE["paper"], PALETTE["negative"]]
    )

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.5)))
    fig.patch.set_facecolor(PALETTE["paper"])
    ax.set_facecolor(PALETTE["paper"])
    im = ax.imshow(pivot.values, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if pd.notna(v):
                text_color = PALETTE["paper"] if v > 65 or v < 20 else PALETTE["ink"]
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                         fontsize=8, color=text_color)

    ax.set_title("Feature comparison across months — % negative mentions",
                  fontsize=12, loc="left", pad=10, color=PALETTE["ink"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("% negative", fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


# ============================================================================
# SECTION 3 — SIDEBAR
# ============================================================================

def _sidebar():
    st.sidebar.title("🗺️ Planning Portal")
    st.sidebar.caption("Feedback Pipeline Dashboard")
    st.sidebar.divider()

    months = load_available_months()
    if not months:
        st.sidebar.warning(
            "No data found. Run the pipeline first:\n\n`python run_pipeline.py`"
        )
        return

    latest_run = load_latest_run_date()
    if latest_run:
        st.sidebar.caption(f"Last pipeline run: **{latest_run}**")
    st.sidebar.caption(f"{len(months)} month(s) available: {months[0]} to {months[-1]}")
    st.sidebar.caption("DB: `data/metrics.db`")


def _aggregate_month_picker(months: list[str], key_prefix: str = "agg") -> list[str]:
    """Month selector used inside any 'Aggregate' sub-tab. key_prefix keeps
    widget keys unique when this is used in more than one tab at once."""
    mode = st.radio("View", ["All time", "Select months"], index=0, horizontal=True,
                     key=f"{key_prefix}_mode")
    if mode == "All time":
        return []
    selected = st.multiselect("Select months", options=months, default=months,
                               key=f"{key_prefix}_months")
    if not selected:
        st.warning("Select at least one month.")
    return selected


# ============================================================================
# SECTION 4 — PAGE FUNCTIONS (Aggregate view)
# ============================================================================

def page_overview(months: list[str]):
    st.header("📊 Overview")

    summary = load_monthly_summary()
    if summary.empty:
        st.warning("No data yet.")
        return

    view = filter_by_months(summary, months) if months else summary
    if view.empty:
        st.warning("No data for the selected period.")
        return

    total_respondents = int(view["total_respondents"].sum())
    total_feedback = int(view["total_with_feedback"].sum())
    avg_rating = round(
        (view["avg_rating"] * view["total_respondents"]).sum() / max(total_respondents, 1), 2
    ) if view["avg_rating"].notna().any() else None
    avg_nsat = round(view["nsat"].dropna().mean(), 1) if view["nsat"].notna().any() else None

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total respondents", f"{total_respondents:,}")
    with col2:
        st.metric("Left written feedback", f"{total_feedback:,}")
    with col3:
        st.metric("Average rating", f"{avg_rating}" if avg_rating else "—", help="1–5 scale")
    with col4:
        st.metric("NSAT", f"{avg_nsat}%" if avg_nsat is not None else "—",
                   help="% rating 4-5 minus % rating 1-2")

    st.divider()

    trend = view.sort_values("month")
    if len(trend) > 1:
        st.subheader("Ratings, NSAT, responses & comments over time")
        st.pyplot(_overlay_trend_chart(trend))
    else:
        st.info("Only one month of data so far — trend charts will appear "
               "once more months are collected.")


def page_sentiment(months: list[str]):
    st.header("💬 Sentiment")

    summary = load_monthly_summary()
    if summary.empty:
        st.warning("No data yet.")
        return

    view = filter_by_months(summary, months) if months else summary
    if view.empty or view["positive_count"].isna().all():
        st.info("No NLP sentiment data yet for this period. "
               "Run the pipeline without SKIP_NLP to populate this.")
        return

    total_pos = int(view["positive_count"].sum())
    total_neg = int(view["negative_count"].sum())
    total_neu = int(view["neutral_count"].sum())
    total = total_pos + total_neg + total_neu

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Positive", f"{round(total_pos/total*100,1)}%" if total else "—")
    with col2:
        st.metric("Negative", f"{round(total_neg/total*100,1)}%" if total else "—")
    with col3:
        st.metric("Neutral", f"{round(total_neu/total*100,1)}%" if total else "—")

    st.divider()

    trend = summary.sort_values("month").copy()
    trend = trend[trend["positive_count"].notna()]
    if len(trend) > 1:
        st.subheader("Sentiment split over time")
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor(PALETTE["paper"])
        ax.set_facecolor(PALETTE["paper"])
        totals = trend["positive_count"] + trend["negative_count"] + trend["neutral_count"]
        pos_pct = (trend["positive_count"] / totals * 100).fillna(0)
        neg_pct = (trend["negative_count"] / totals * 100).fillna(0)
        neu_pct = (100 - pos_pct - neg_pct).clip(lower=0)

        ax.bar(trend["month"], neg_pct, color=SENTIMENT_COLORS["negative"], label="Negative")
        ax.bar(trend["month"], neu_pct, bottom=neg_pct, color=SENTIMENT_COLORS["neutral"], label="Neutral")
        ax.bar(trend["month"], pos_pct, bottom=neg_pct + neu_pct, color=SENTIMENT_COLORS["positive"], label="Positive")
        ax.set_ylabel("% of reviews")
        ax.legend(frameon=True, fontsize=8)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Only one month of sentiment data so far.")


def render_aggregate_view():
    months_all = load_available_months()
    months = _aggregate_month_picker(months_all, key_prefix="agg")
    st.divider()
    page_overview(months)
    st.divider()
    page_sentiment(months)


# ============================================================================
# SECTION 4B — PAGE FUNCTIONS (Feedback comments tab — tag groups / features
# / errors / browse reviews; everything derived from free-text comments)
# ============================================================================

def page_tag_groups(months: list[str]):
    st.header("🏷️ Tag Groups")

    df = load_tag_group_trends()
    if df.empty:
        st.info("No tag data yet. Run the pipeline without SKIP_NLP.")
        return

    view = filter_by_months(df, months) if months else df
    if view.empty:
        st.warning("No data for the selected period.")
        return

    st.subheader("Volume + sentiment by tag group")
    st.pyplot(_sentiment_stacked_bar(view, "tag_group", "", pct=True))

    st.divider()
    st.subheader("Raw counts")
    display = view.groupby("tag_group").agg(
        total_reviews=("total_reviews", "sum"),
        positive=("positive_count", "sum"),
        negative=("negative_count", "sum"),
        neutral=("neutral_count", "sum"),
        avg_sentiment_score=("avg_sentiment_score", "mean"),
    ).sort_values("total_reviews", ascending=False).reset_index()
    st.dataframe(display, use_container_width=True)


def page_features(months: list[str]):
    st.header("🧩 Features")

    df = load_feature_trends()
    if df.empty:
        st.info("No feature data yet. Run the pipeline without SKIP_NLP.")
        return

    view = filter_by_months(df, months) if months else df
    if view.empty:
        st.warning("No data for the selected period.")
        return

    agg = view.groupby("feature").agg(
        total_mentions=("total_mentions", "sum"),
        negative_mentions=("negative_mentions", "sum"),
        positive_mentions=("positive_mentions", "sum"),
    ).reset_index()
    agg["negative_pct"] = round(agg["negative_mentions"] / agg["total_mentions"] * 100, 1)
    agg = agg.sort_values("negative_pct", ascending=False)

    st.subheader("Most negatively-mentioned features")
    top = agg.head(15).set_index("feature")
    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.4)))
    fig.patch.set_facecolor(PALETTE["paper"])
    ax.set_facecolor(PALETTE["paper"])
    ax.barh(top.index[::-1], top["negative_pct"][::-1], color=SENTIMENT_COLORS["negative"])
    ax.set_xlabel("% negative mentions")
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.5)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.subheader("All features")
    st.dataframe(agg, use_container_width=True)

    # ---- Feature comparison across months (heatmap — was the "features
    #      to compare" line chart, replaced per change #5) ----------------
    st.divider()
    st.subheader("Feature comparison across months")
    heatmap_fig = _feature_negative_heatmap(view)
    if heatmap_fig is None:
        st.info("Not enough data across months to build the comparison heatmap yet.")
    else:
        st.pyplot(heatmap_fig)
        plt.close(heatmap_fig)


def page_errors(months: list[str]):
    st.header("⚠️ Errors")

    df = load_error_trends()
    if df.empty:
        st.info("No error pattern data yet. Run the pipeline without SKIP_NLP.")
        return

    view = filter_by_months(df, months) if months else df
    if view.empty:
        st.warning("No data for the selected period.")
        return

    agg = view.groupby("error_pattern")["count"].sum().sort_values(ascending=False).head(15)

    st.subheader("Most frequent error patterns")
    fig, ax = plt.subplots(figsize=(10, max(4, len(agg) * 0.4)))
    fig.patch.set_facecolor(PALETTE["paper"])
    ax.set_facecolor(PALETTE["paper"])
    ax.barh(agg.index[::-1], agg.values[::-1], color=PALETTE["primary"])
    ax.set_xlabel("Mentions")
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.5)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def page_browse_reviews(months: list[str]):
    st.header("🔎 Browse Reviews")

    df = load_review_detail(months if months else None)
    if df.empty:
        st.info("No tagged reviews yet. Run the pipeline without SKIP_NLP.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        tag_options = ["All"] + sorted(df["primary_tag_group"].dropna().unique().tolist())
        tag_filter = st.selectbox("Tag group", tag_options)
    with col2:
        sent_values = sorted(df["grouping_sentiment"].dropna().unique().tolist())
        sent_options = ["All"] + sent_values
        sent_filter = st.selectbox(
            "Sentiment", sent_options,
            format_func=lambda v: v if v == "All" else f"{SENTIMENT_EMOJI.get(v, '')} {v.title()}",
        )
    with col3:
        search = st.text_input("Search feedback text")

    filtered = df.copy()
    if tag_filter != "All":
        filtered = filtered[filtered["primary_tag_group"] == tag_filter]
    if sent_filter != "All":
        filtered = filtered[filtered["grouping_sentiment"] == sent_filter]
    if search:
        filtered = filtered[filtered["feedback_clean"].str.contains(search, case=False, na=False)]

    st.caption(f"{len(filtered):,} reviews match")
    show_cols = [
        "respondent_id", "month", "rating", "grouping_sentiment",
        "primary_tag_group", "primary_tag", "feedback_clean",
    ]
    show_cols = [c for c in show_cols if c in filtered.columns]
    display_df = filtered[show_cols].copy()
    if "grouping_sentiment" in display_df.columns:
        display_df["grouping_sentiment"] = display_df["grouping_sentiment"].map(
            lambda v: f"{SENTIMENT_EMOJI.get(v, '')} {str(v).title()}" if pd.notna(v) else v
        )
    display_df = display_df.rename(columns={
        "respondent_id": "Respondent", "month": "Month", "rating": "Rating",
        "grouping_sentiment": "Sentiment", "primary_tag_group": "Tag group",
        "primary_tag": "Tag", "feedback_clean": "Feedback",
    })
    st.dataframe(display_df, use_container_width=True, height=600)


def _render_comment_themes_month(selected_month: str):
    """Thematic analysis + Feature & error analysis for one month — used by
    the Feedback comments tab's Month by month sub-tab."""
    tag_df = load_tag_group_trends()
    tag_month = tag_df[tag_df["month"] == selected_month] if not tag_df.empty else tag_df

    st.subheader("Thematic analysis")
    if tag_month.empty:
        st.info("No tag data yet for this month. Run the pipeline without SKIP_NLP.")
    else:
        tc1, tc2 = st.columns(2)
        with tc1:
            vol = tag_month.groupby("tag_group")["total_reviews"].sum().sort_values(ascending=False)
            sentiment_by_group = tag_month.groupby("tag_group")["avg_sentiment_score"].mean()
            colors = [SENTIMENT_COLORS["positive"] if sentiment_by_group.get(g, 0) >= 0
                      else SENTIMENT_COLORS["negative"] for g in vol.index]
            fig, ax = plt.subplots(figsize=(10, max(4, len(vol) * 0.4)))
            fig.patch.set_facecolor(PALETTE["paper"])
            ax.set_facecolor(PALETTE["paper"])
            ax.barh(vol.index[::-1], vol.values[::-1], color=colors[::-1])
            ax.set_title("Tag group mentions — this month", fontsize=12, loc="left", color=PALETTE["ink"])
            ax.set_xlabel("Volume of comments per theme")
            ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.5)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with tc2:
            st.pyplot(_diverging_sentiment_bar(tag_month, "tag_group", "avg_sentiment_score",
                                                "Tag group sentiment — this month"))

    st.divider()

    st.subheader("Feature & error analysis")
    fc1, fc2 = st.columns(2)
    with fc1:
        feat_df = load_feature_trends()
        feat_month = feat_df[feat_df["month"] == selected_month] if not feat_df.empty else feat_df
        if feat_month.empty:
            st.info("No feature data yet for this month.")
        else:
            agg = feat_month.groupby("feature").agg(
                total_mentions=("total_mentions", "sum"),
                negative_mentions=("negative_mentions", "sum"),
            )
            agg["negative_pct"] = round(agg["negative_mentions"] / agg["total_mentions"] * 100, 1)
            top = agg.sort_values("negative_pct", ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(6, max(3, len(top) * 0.4)))
            fig.patch.set_facecolor(PALETTE["paper"])
            ax.set_facecolor(PALETTE["paper"])
            ax.barh(top.index[::-1], top["negative_pct"][::-1], color=SENTIMENT_COLORS["negative"])
            ax.set_title("Most negatively mentioned features", fontsize=11, loc="left", color=PALETTE["ink"])
            ax.set_xlabel("% of mentions that are negative")
            ax.set_xlim(0, 100)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    with fc2:
        err_df = load_error_trends()
        err_month = err_df[err_df["month"] == selected_month] if not err_df.empty else err_df
        if err_month.empty:
            st.info("No error pattern data yet for this month.")
        else:
            top_err = err_month.groupby("error_pattern")["count"].sum().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(6, max(3, len(top_err) * 0.4)))
            fig.patch.set_facecolor(PALETTE["paper"])
            ax.set_facecolor(PALETTE["paper"])
            ax.barh(top_err.index[::-1], top_err.values[::-1], color=PALETTE["neutral"])
            ax.set_title("Top error patterns", fontsize=11, loc="left", color=PALETTE["ink"])
            ax.set_xlabel("Most frequently reported issues")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


def render_feedback_comments_view():
    months = load_available_months()
    if not months:
        st.info("No data yet.")
        return

    sub_month, sub_agg = st.tabs(["Month by month", "Aggregate"])

    with sub_month:
        default_month = st.session_state.get("selected_month", months[-1])
        selected = st.selectbox(
            "Month", months, index=months.index(default_month) if default_month in months else len(months) - 1,
            key="comments_month_select",
        )
        st.divider()
        _render_comment_themes_month(selected)

    with sub_agg:
        agg_months = _aggregate_month_picker(months, key_prefix="comments_agg")
        st.divider()
        page_tag_groups(agg_months)
        st.divider()
        page_features(agg_months)
        st.divider()
        page_errors(agg_months)
        st.divider()
        page_browse_reviews(agg_months)


# ============================================================================
# SECTION 5 — MONTHLY SNAPSHOT VIEW (matches the preferred design)
# ============================================================================

def render_monthly_view():
    months = load_available_months()
    summary = load_monthly_summary().sort_values("month")

    # Month pill row
    default_idx = len(months) - 1
    n_pills = min(len(months), 8)
    recent_months = months[-n_pills:]
    if "selected_month" not in st.session_state:
        st.session_state.selected_month = months[default_idx]

    cols = st.columns(len(recent_months))
    for i, m in enumerate(recent_months):
        with cols[i]:
            is_selected = m == st.session_state.selected_month
            if st.button(m, key=f"pill_{m}", type="primary" if is_selected else "secondary",
                         use_container_width=True):
                st.session_state.selected_month = m
                st.rerun()

    selected_month = st.session_state.selected_month
    row = summary[summary["month"] == selected_month]
    if row.empty:
        st.warning("No data for the selected month.")
        return
    row = row.iloc[0]

    prior_months = summary[summary["month"] < selected_month]
    prior = prior_months.iloc[-1] if not prior_months.empty else None

    st.caption("THIS MONTH AT A GLANCE")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        delta = (row["total_respondents"] - prior["total_respondents"]) if prior is not None else None
        _kpi_card("Total responses", f"{int(row['total_respondents']):,}",
                  delta=delta, delta_suffix="")
    with c2:
        delta = (row["avg_rating"] - prior["avg_rating"]) if prior is not None and pd.notna(row["avg_rating"]) else None
        _kpi_card("Average rating", f"{row['avg_rating']:.1f}" if pd.notna(row["avg_rating"]) else "—",
                  delta=delta, delta_suffix="")
    with c3:
        delta = (row["nsat"] - prior["nsat"]) if prior is not None and pd.notna(row["nsat"]) else None
        _kpi_card("NSAT score", f"{row['nsat']:.1f}" if pd.notna(row["nsat"]) else "—",
                  delta=delta, delta_suffix="")
    with c4:
        left_fb = int(row["total_with_feedback"]) if pd.notna(row["total_with_feedback"]) else 0
        pct = round(left_fb / row["total_respondents"] * 100, 0) if row["total_respondents"] else 0
        _kpi_card("Left feedback", f"{left_fb:,}", sublabel=f"{pct:.0f}% of respondents")

    st.divider()

    # ---- Ratings section ------------------------------------------------
    st.subheader("Ratings")
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("**Rating distribution — this month**")
        rating_cols = ["rating_5", "rating_4", "rating_3", "rating_2", "rating_1"]
        if all(c in row.index for c in rating_cols) and row[rating_cols].notna().any():
            counts = {s: row[f"rating_{s}"] for s in [5, 4, 3, 2, 1]}
            st.pyplot(_rating_distribution_bar(counts, ""))
        else:
            st.info("No rating breakdown recorded for this month yet.")
    with rc2:
        if len(summary) > 1:
            st.pyplot(_line_trend(summary, "month", "avg_rating",
                                  "Average rating — all months", ylabel="Avg rating (1-5)"))

    st.divider()

    # ---- Sentiment section ------------------------------------------------
    # Note (change #1): the "NSAT — all months" and "Comment sentiment —
    # all months" trend charts that used to live here have been removed;
    # this section now only shows this month's sentiment split.
    st.subheader("Sentiment — comments only")
    if pd.notna(row.get("positive_count")):
        total = row["positive_count"] + row["negative_count"] + row["neutral_count"]
        if total:
            pos_pct = row["positive_count"] / total * 100
            neg_pct = row["negative_count"] / total * 100
            neu_pct = 100 - pos_pct - neg_pct
            st.pyplot(_single_sentiment_bar(pos_pct, neg_pct, neu_pct,
                                             "Overall comment sentiment — this month"))
    else:
        st.info("No NLP sentiment data yet for this month. Run the pipeline without SKIP_NLP.")


# ============================================================================
# SECTION 6 — PAGE ROUTER
# ============================================================================

def main():
    _inject_css()

    if not db_exists():
        st.title("🗺️ Planning Portal Feedback")
        st.warning(
            "No database found yet. Run the pipeline first:\n\n"
            "`python run_pipeline.py`\n\n"
            "or wait for the scheduled Monday-morning GitHub Action to run."
        )
        return

    _sidebar()

    st.title("🗺️ Planning Portal Feedback Dashboard")

    tab1, tab2, tab3 = st.tabs(["Monthly view", "Aggregate view", "Feedback comments"])
    with tab1:
        render_monthly_view()
    with tab2:
        render_aggregate_view()
    with tab3:
        render_feedback_comments_view()


if __name__ == "__main__":
    main()