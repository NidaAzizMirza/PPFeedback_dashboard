"""
about_page.py  --  "About this dashboard" page for the Streamlit feedback dashboard.

How to use
----------
1. Save this file next to dashboard/app.py (or paste the contents into app.py).
2. In app.py:

       from about_page import page_about
       ...
       with tab_about:            # or add "About" to your page/tab list
           page_about(last_updated=latest_response_ts)

3. Edit ABOUT_CONTENT below. All text lives in one place, so you can change
   wording without touching the layout code.

Items marked CHECK are things I could not confirm from your files. Please
verify them before publishing.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
from graphviz import Digraph

UK_TZ = ZoneInfo("Europe/London")


# --------------------------------------------------------------------------
# 1. CONTENT  (edit this, not the layout code)
# --------------------------------------------------------------------------
ABOUT_CONTENT = {
    "title": "About this dashboard",
    "purpose": {
        "summary": (
            "This dashboard shows what users of the Planning Portal tell us "
            "about the service, in one place. It turns survey ratings and "
            "written comments into trends the product team can act on."
        ),
        "questions": [
            "How satisfied are users, and is that changing month to month?",
            "What are users talking about in their comments?",
            "Which topics are mostly positive, and which are mostly negative?",
        ],
        "source": "Feedback is collected through the SurveyMonkey survey on the service.",  # CHECK wording
    },
    # (name, what you find there)  -- match these to your real tab names
    "where_to_look": [
        ("Feedback ratings > Aggregate", "Overall NSAT and rating spread for the period you pick."),
        ("Feedback ratings > Month by month", "How ratings move over time."),
        ("Feedback comments > Aggregate", "Topics and sentiment in comments across the period."),
        ("Feedback comments > Month by month", "How topics and sentiment change by month."),
        ("Browse Reviews", "Read individual comments. Click a row to see the full comment and its tags."),
    ],
    "how_to_read": [
        ("Month picker (sidebar)", "Sets the month shown on the page. Use it first."),
        ("NSAT", "(Rating 4 + Rating 5 - Rating 1 - Rating 2) / total rated x 100. "
                 "Ranges from -100 to +100. Above 0 means more happy than unhappy users."),
        ("Rating", "The 1 to 5 score the user gave. 5 is best."),
        ("Topic", "What the comment is about, for example Payments or Forms. "
                  "One comment can have more than one topic."),
        ("Sentiment", "Whether the user is positive, neutral or negative about that topic. "
                      "Each topic in a comment gets its own sentiment."),
        ("Overall experience", "The general feeling in the comment, when it is not about one topic."),
        ("Heatmap", "Darker cells mean a higher share of negative comments for that topic."),
        ("Small numbers", "Months or topics with few comments move a lot from small changes. "
                          "Check the count before drawing conclusions."),
    ],
    "limits": [
        "Topics and sentiment are tagged by an automated model, so some tags will be wrong. "
        "Treat them as a guide and read the comments for detail.",
        "Only people who chose to answer the survey are included. They may not represent all users.",
    ],
    "privacy": [
        "Respondent names and contact details are not collected or shown. Comments are written by users and may occasionally contain personal details.",
        "The dashboard is password protected. Only people who have been given the password can open it.",
        "Please do not share the password or screenshots outside the team.",  # CHECK: remove if not your policy
    ],
    "contact": "Questions or corrections: add your team contact here.",  # CHECK: fill in
}


# --------------------------------------------------------------------------
# 2. HELPERS
# --------------------------------------------------------------------------
def format_uk(ts: datetime | str | None) -> str:
    """Return e.g. '09 Nov 2026, 14:05' in UK time, or a dash if missing."""
    if ts is None:
        return "-"
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=ZoneInfo("UTC"))
    return ts.astimezone(UK_TZ).strftime("%d %b %Y, %H:%M")


# def _term_table(rows: list[tuple[str, str]]) -> None:
#     """Two-column list: term in bold, meaning beside it."""
#     for term, meaning in rows:
#         left, right = st.columns([1, 3])
#         left.markdown(f"**{term}**")
#         right.markdown(meaning)

def _term_table(rows: list[tuple[str, str]]) -> None:
    """Display terms and meanings in individual visual cards."""

    for term, meaning in rows:
        with st.container(border=True):
            left, right = st.columns([1, 3], vertical_alignment="center")

            with left:
                st.markdown(f"**{term}**")

            with right:
                st.markdown(meaning)


# --------------------------------------------------------------------------
# 3. PAGE
# --------------------------------------------------------------------------
def page_about(last_updated: datetime | str | None = None,
               content: dict = ABOUT_CONTENT) -> None:
    c = content

    st.header(c["title"])
    if last_updated is not None:
        st.caption(f"Latest data: {format_uk(last_updated)}")

    st.subheader("Purpose")
    st.write(c["purpose"]["summary"])
    st.markdown("It helps answer:")
    for q in c["purpose"]["questions"]:
        st.markdown(f"- {q}")
    st.caption(c["purpose"]["source"])

    st.subheader("Where to find things")

    _navigation_tree(
        "Feedback dashboard",
        c["where_to_look"]
    )

    st.subheader("How to read it")
    _term_table(c["how_to_read"])

    st.subheader("Things to keep in mind")
    for item in c["limits"]:
        st.markdown(f"- {item}")

    st.subheader("Privacy and access")
    for item in c["privacy"]:
        st.markdown(f"- {item}")

    st.divider()
    st.caption(c["contact"])


if __name__ == "__main__":
    # Run alone to preview:  streamlit run about_page.py
    page_about(last_updated=datetime.now(UK_TZ))


# --------------------------------------------------------------------------
# Navigation tree
# --------------------------------------------------------------------------
def _navigation_tree(section: str, pages: list[tuple[str, str]]) -> None:
    """Display dashboard navigation as a hierarchical tree."""

    dot = Digraph()

    dot.attr(
        rankdir="TB",
        bgcolor="transparent",
        nodesep="0.5",
        ranksep="0.7"
    )

    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fontname="Arial",
        fontsize="11",
        margin="0.2,0.15"
    )

    dot.attr(
        "edge",
        color="#9CA3AF",
        penwidth="1.5",
        arrowsize="0.7"
    )

    # Main dashboard node
    dot.node(
        "root",
        section,
        fillcolor="#DBEAFE",
        color="#60A5FA",
        penwidth="1.5"
    )

    # Group pages by their parent section
    groups = {}

    for path, description in pages:
        if " > " in path:
            parent, child = path.split(" > ", 1)
        else:
            parent, child = path, None

        groups.setdefault(parent, []).append((child, description))

    # Create parent and child branches
    for i, (parent, children) in enumerate(groups.items()):

        parent_id = f"section_{i}"

        dot.node(
            parent_id,
            parent,
            fillcolor="#EDE9FE",
            color="#A78BFA"
        )

        dot.edge("root", parent_id)

        for j, (child, description) in enumerate(children):

            if child is None:
                continue

            child_id = f"page_{i}_{j}"

            dot.node(
                child_id,
                f"{child}\n{description}",
                fillcolor="#F0FDF4",
                color="#86EFAC"
            )

            dot.edge(parent_id, child_id)

    st.graphviz_chart(dot, use_container_width=True)