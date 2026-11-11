"""
fix_sidebar.py: rewrites the sidebar section of the dashboard app with a known-good version
(title, caption, "About this dashboard" button, then the data info lines).

    python fix_sidebar.py dashboard/app.py

It replaces everything between the "SECTION 3 - SIDEBAR" header and `def _aggregate_month_picker`,
saves a backup next to the file first, and does NOT write anything if the result has a syntax error.
"""
import ast
import shutil
import sys
from datetime import datetime
from pathlib import Path

NEW_BLOCK = '''@st.dialog("About", width="large")
def _about_dialog():
    page_about()


def _sidebar():
    st.sidebar.title("Planning Control Portal")
    st.sidebar.caption("Feedback Pipeline Dashboard")
    if st.sidebar.button("ℹ️ About this dashboard", key="open_about"):
        _about_dialog()
    st.sidebar.divider()

    months = load_available_months()
    if not months:
        st.sidebar.warning(
            "No data found. Run the pipeline first:\\n\\n`python run_pipeline.py`"
        )
        return

    latest_run = load_latest_run_date()
    if latest_run:
        st.sidebar.caption(f"Data last refreshed: **{latest_run}**")
    st.sidebar.caption(f"{len(months)} month(s) available: {months[0]} to {months[-1]}")
    st.sidebar.caption("DB: `data/metrics.db`")
'''

path = Path(sys.argv[1] if len(sys.argv) > 1 else "dashboard/app.py")
lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

hdr = next((i for i, l in enumerate(lines) if "SECTION 3" in l and "SIDEBAR" in l), None)
if hdr is None:
    sys.exit("Could not find the 'SECTION 3 — SIDEBAR' header. Nothing changed.")
close = next((i for i in range(hdr + 1, len(lines)) if lines[i].lstrip().startswith("# ====")), None)
end = next((i for i in range(close + 1, len(lines)) if lines[i].startswith("def _aggregate_month_picker")), None) if close else None
if close is None or end is None:
    sys.exit("Could not find the end of the sidebar section (def _aggregate_month_picker). Nothing changed.")

old = "".join(lines[close + 1:end])
if "_sidebar" not in old:
    sys.exit("The section between the header and _aggregate_month_picker has no _sidebar function. Nothing changed.")

new_lines = lines[:close + 1] + ["\n"] + [NEW_BLOCK] + ["\n", "\n"] + lines[end:]
new_src = "".join(new_lines)
try:
    ast.parse(new_src)
except SyntaxError as e:
    sys.exit(f"The result would have a syntax error ({e}). Nothing changed.")

backup = path.with_name(f"{path.name}.bak_sidebar_{datetime.now():%Y%m%d_%H%M%S}")
shutil.copy2(path, backup)
path.write_text(new_src, encoding="utf-8")
print(f"Replaced {end - close - 1} lines of the sidebar section.  Backup: {backup}  (do NOT commit it)")

for name in ("def _sidebar", "def _about_dialog", "open_about"):
    hits = [i + 1 for i, l in enumerate(new_src.splitlines()) if name in l]
    print(f"  {name!r}: lines {hits}" + ("" if len(hits) == 1 else "   <-- expected exactly 1, check these"))
calls = [i + 1 for i, l in enumerate(new_src.splitlines()) if l.strip() == "_sidebar()"]
print(f"  call to _sidebar(): lines {calls}" + ("" if calls else "   <-- NOT FOUND: main() must call _sidebar()"))