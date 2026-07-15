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
    Tab 1 — Monthly view   : single-month "at a glance" snapshot (KPI cards,
                              this-month sentiment/tag/feature/error charts),
                              styled to match the earlier GitHub Pages design.
    Tab 2 — Aggregate view : the original multi-month trends dashboard
                              (Overview / Sentiment / Tag Groups / Features /
                              Errors / Monthly Comparison / Browse Reviews).

Edit THIS file if you want to:
  - Change a page's layout or charts (Section 4 — PAGE FUNCTIONS)
  - Add a brand new page (Section 4 to write it, Section 6 to register it)
  - Change chart colours globally — edit dashboard_config.py instead

KNOWN GAP (see "Rating distribution" card in Monthly view):
  The per-star breakdown (★5: N, ★4: N, ...) shown in the earlier design
  needs rating_1_count..rating_5_count columns that monthly_summary does
  not currently have. That chart is left as a flagged placeholder rather
  than faked — add those columns in aggregation_db.py / run_pipeline.py
  and a loader in store.py to light it up.
============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
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
              delta_suffix: str = "", higher_is_better: bool = True):
    delta_html = ""
    if delta is not None:
        if abs(delta) < 1e-9:
            cls, arrow = "kpi-delta-flat", "→"
        elif (delta > 0) == higher_is_better:
            cls, arrow = "kpi-delta-up", "↑"
        else:
            cls, arrow = "kpi-delta-down", "↓"
        delta_html = f'<div class="{cls}">{arrow} {abs(delta):.1f}{delta_suffix} vs prior month</div>'

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


def _aggregate_month_picker(months: list[str]) -> list[str]:
    """Month selector used only inside the Aggregate view tab."""
    mode = st.radio("View", ["All time", "Select months"], index=0, horizontal=True,
                     key="agg_mode")
    if mode == "All time":
        return []
    selected = st.multiselect("Select months", options=months, default=months, key="agg_months")
    if not selected:
        st.warning("Select at least one month.")
    return selected


# ============================================================================
# SECTION 4 — PAGE FUNCTIONS (Aggregate view — unchanged behaviour)
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

    trend = summary.sort_values("month")
    if len(trend) > 1:
        st.subheader("Respondent volume over time")
        st.pyplot(_bar_trend(trend, "month", "total_respondents",
                             "", ylabel="Respondents"))

        st.subheader("Average rating over time")
        st.pyplot(_line_trend(trend, "month", "avg_rating",
                              "", ylabel="Avg rating (1-5)"))

        st.subheader("NSAT over time")
        st.pyplot(_line_trend(trend, "month", "nsat", "",
                              ylabel="NSAT %", color=PALETTE["accent"]))
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
        sent_options = ["All"] + sorted(df["grouping_sentiment"].dropna().unique().tolist())
        sent_filter = st.selectbox("Sentiment", sent_options)
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
    st.dataframe(filtered[show_cols], use_container_width=True, height=600)


def page_monthly_comparison(months: list[str]):
    st.header("📅 Monthly Comparison")

    summary = load_monthly_summary()
    if summary.empty:
        st.warning("No data yet.")
        return

    view = filter_by_months(summary, months) if months else summary
    view = view.sort_values("month")
    if view.empty:
        st.warning("No data for the selected period.")
        return

    st.subheader("Month-by-month summary")
    table_cols = ["month", "total_respondents", "avg_rating", "nsat"]
    table_cols = [c for c in table_cols if c in view.columns]
    display_table = view[table_cols].rename(columns={
        "month": "Month", "total_respondents": "Responses",
        "avg_rating": "Avg Rating", "nsat": "NSAT %",
    })
    st.dataframe(display_table, use_container_width=True)

    if len(view) > 1:
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(_bar_trend(view, "month", "total_respondents",
                                 "Responses per month", ylabel="Responses"))
        with col2:
            st.pyplot(_line_trend(view, "month", "avg_rating",
                                  "Average rating per month", ylabel="Avg rating (1-5)"))

        st.pyplot(_line_trend(view, "month", "nsat", "NSAT per month",
                              ylabel="NSAT %", color=PALETTE["accent"]))

    st.divider()
    st.subheader("Feature comparison across months")

    feat_df = load_feature_trends()
    if feat_df.empty:
        st.info("No feature data yet. Run the pipeline without SKIP_NLP.")
        return

    feat_view = filter_by_months(feat_df, months) if months else feat_df
    if feat_view.empty:
        st.info("No feature data for the selected period.")
        return

    totals_by_feature = feat_view.groupby("feature")["total_mentions"].sum().sort_values(ascending=False)
    default_features = totals_by_feature.head(6).index.tolist()
    all_features = totals_by_feature.index.tolist()

    selected_features = st.multiselect(
        "Features to compare (defaults to top 6 by mention volume)",
        options=all_features, default=default_features,
    )
    if not selected_features:
        st.info("Select at least one feature to compare.")
        return

    metric = st.radio("Compare by", ["% negative mentions", "Total mentions"], horizontal=True)
    value_col = "negative_pct" if metric == "% negative mentions" else "total_mentions"

    pivot = feat_view[feat_view["feature"].isin(selected_features)].pivot_table(
        index="month", columns="feature", values=value_col, aggfunc="mean"
    ).sort_index()

    if pivot.empty:
        st.info("Not enough data to compare these features over time.")
        return

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor(PALETTE["paper"])
    ax.set_facecolor(PALETTE["paper"])
    for i, feature in enumerate(pivot.columns):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        ax.plot(pivot.index, pivot[feature], marker="o", label=feature, color=color, linewidth=2)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(frameon=True, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(pivot.round(1), use_container_width=True)


def render_aggregate_view():
    months_all = load_available_months()
    months = _aggregate_month_picker(months_all)
    st.divider()
    page_overview(months)
    st.divider()
    page_sentiment(months)
    st.divider()
    page_tag_groups(months)
    st.divider()
    page_features(months)
    st.divider()
    page_errors(months)
    st.divider()
    page_monthly_comparison(months)
    st.divider()
    page_browse_reviews(months)


# ============================================================================
# SECTION 5 — MONTHLY SNAPSHOT VIEW (new — matches the preferred design)
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
        _kpi_card("Left feedback", f"{left_fb:,}", )
        st.caption(f"{pct:.0f}% of respondents")

    st.divider()

    # ---- Ratings section ------------------------------------------------
    st.subheader("Ratings")
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("**Rating distribution — this month**")
        st.info(
            "Per-star breakdown isn't available yet — `monthly_summary` doesn't "
            "currently store rating_1_count..rating_5_count. Add those columns "
            "in aggregation_db.py to light this up."
        )
    with rc2:
        if len(summary) > 1:
            st.pyplot(_line_trend(summary, "month", "avg_rating",
                                  "Average rating — all months", ylabel="Avg rating (1-5)"))

    st.divider()

    # ---- Sentiment section ------------------------------------------------
    st.subheader("Sentiment — comments only")
    if pd.notna(row.get("positive_count")):
        total = row["positive_count"] + row["negative_count"] + row["neutral_count"]
        if total:
            pos_pct = row["positive_count"] / total * 100
            neg_pct = row["negative_count"] / total * 100
            neu_pct = 100 - pos_pct - neg_pct
            st.pyplot(_single_sentiment_bar(pos_pct, neg_pct, neu_pct,
                                             "Overall comment sentiment — this month"))
        sc1, sc2 = st.columns(2)
        with sc1:
            if len(summary) > 1:
                st.pyplot(_line_trend(summary, "month", "nsat", "NSAT score — all months",
                                      ylabel="NSAT %", color=PALETTE["accent"]))
        with sc2:
            trend = summary[summary["positive_count"].notna()]
            if len(trend) > 1:
                fig, ax = plt.subplots(figsize=(10, 3.5))
                fig.patch.set_facecolor(PALETTE["paper"])
                ax.set_facecolor(PALETTE["paper"])
                totals = trend["positive_count"] + trend["negative_count"] + trend["neutral_count"]
                pos_pct_t = (trend["positive_count"] / totals * 100).fillna(0)
                neg_pct_t = (trend["negative_count"] / totals * 100).fillna(0)
                neu_pct_t = (100 - pos_pct_t - neg_pct_t).clip(lower=0)
                ax.bar(trend["month"], neg_pct_t, color=SENTIMENT_COLORS["negative"], label="Negative")
                ax.bar(trend["month"], neu_pct_t, bottom=neg_pct_t, color=SENTIMENT_COLORS["neutral"], label="Neutral")
                ax.bar(trend["month"], pos_pct_t, bottom=neg_pct_t + neu_pct_t, color=SENTIMENT_COLORS["positive"], label="Positive")
                ax.set_title("Comment sentiment — all months", fontsize=12, loc="left", color=PALETTE["ink"])
                ax.set_ylabel("%")
                ax.tick_params(axis="x", rotation=30)
                ax.legend(frameon=True, fontsize=7)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
    else:
        st.info("No NLP sentiment data yet for this month. Run the pipeline without SKIP_NLP.")

    st.divider()

    # ---- Thematic analysis section ----------------------------------------
    st.subheader("Thematic analysis")
    tag_df = load_tag_group_trends()
    tag_month = tag_df[tag_df["month"] == selected_month] if not tag_df.empty else tag_df
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

    # ---- Feature & error analysis -----------------------------------------
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

    tab1, tab2 = st.tabs(["Monthly view", "Aggregate view"])
    with tab1:
        render_monthly_view()
    with tab2:
        render_aggregate_view()


if __name__ == "__main__":
    main()