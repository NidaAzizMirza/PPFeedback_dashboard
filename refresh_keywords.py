"""
refresh_keywords.py — apply your CURRENT keyword lists to the older comments, without any models.

Run it after you edit keywords in run_pipeline.py or sentiment_overrides.py, so older months follow
the new lists. It recomputes, for every comment in master.xlsx:

  user_type   from USER_TYPE_KEYWORDS (run_pipeline.py) and the stated-age rule. Replaces the old value,
              so a word you REMOVED from the list stops tagging older comments too.
  topics      the always-add topics (DOCUMENT_KEYWORDS, guidance, competitor names in
              sentiment_overrides.py). These are ADDED to a comment's tags, never removed.

Human tags from the dashboard are applied last and always win.

    python refresh_keywords.py              # DRY RUN: shows what would change, writes nothing
    python refresh_keywords.py --apply      # backs up master.xlsx, writes it, rebuilds metrics.db
    python refresh_keywords.py --apply --yes --no-backup      # for the GitHub Action

AFTER --apply, run:  python retag_history.py --months <months>  so the topic sentiment for newly added
topics is scored (that step reads the tag groups this script just updated).

Not covered: HARD_FALLBACK topic keywords. They only matter when the classifier is unsure, so using a new
HARD_FALLBACK word on old comments means re-running the classifier on them, which can change their tags.
"""
import argparse
import shutil
import sys
from datetime import datetime

import pandas as pd

import manual_tags as mt
import pipeline_config as cfg
import sentiment_overrides as so

ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true")
ap.add_argument("--yes", action="store_true")
ap.add_argument("--no-backup", action="store_true")
ap.add_argument("--months", default=None, help="only these months, e.g. 2026-01,2026-02")
args = ap.parse_args()

ID, CLEAN = cfg.COL_RESPONDENT_ID, cfg.COL_FEEDBACK_CLEAN
master = pd.read_excel(cfg.MASTER_FILE, dtype={ID: str})

print("Loading keyword lists...")
from run_pipeline import extract_user_type   # noqa: E402  (heavy import, comes after the argument checks)

month = pd.to_datetime(master["Start Date"], errors="coerce").dt.strftime("%Y-%m")
rows = master[CLEAN].fillna("").astype(str).str.strip() != ""
if args.months:
    rows &= month.isin({m.strip() for m in args.months.split(",")})
print(f"Comments to check: {int(rows.sum()):,} of {len(master):,}")

empty = lambda v: "" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v)
new_ut, new_tags, new_groups = {}, {}, {}
for idx in master.index[rows]:
    text = master.at[idx, CLEAN]
    new_ut[idx] = extract_user_type(text)
    t, g = so.add_always_topics(text, empty(master.at[idx, "primary_tag_group"]),
                                empty(master.at[idx, "secondary_tags"]),
                                empty(master.at[idx, "secondary_tag_groups"]))
    new_tags[idx], new_groups[idx] = t, g

before = master[["user_type", "secondary_tags", "secondary_tag_groups"]].copy()
for col, vals in (("user_type", new_ut), ("secondary_tags", new_tags), ("secondary_tag_groups", new_groups)):
    master[col] = pd.Series([vals.get(i, v) for i, v in zip(master.index, master[col])],
                            index=master.index, dtype=object)
master = mt.apply_to_df(master, id_col=ID)          # human tags always win

norm = lambda d: d.astype(object).where(d.notna(), "").astype(str).replace({"nan": "", "None": ""})
b, a = norm(before), norm(master[before.columns])
ut_changed, tag_changed = b["user_type"] != a["user_type"], b["secondary_tag_groups"] != a["secondary_tag_groups"]
print(f"\nuser_type changes: {int(ut_changed.sum()):,}   comments getting new topics: {int(tag_changed.sum()):,}")

if ut_changed.any():
    print("\nuser_type: old -> new (first 8)")
    for i in master.index[ut_changed][:8]:
        print(f"  [{b.at[i, 'user_type'] or '-'}] -> [{a.at[i, 'user_type'] or '-'}]  {str(master.at[i, CLEAN])[:80]}")
if tag_changed.any():
    added = pd.Series([g for i in master.index[tag_changed]
                       for g in a.at[i, "secondary_tag_groups"].split(", ")
                       if g and g not in b.at[i, "secondary_tag_groups"]]).value_counts()
    print("\nTopics added most often:\n" + added.head(8).to_string())

if not args.apply:
    print("\nDRY RUN: nothing written. Add --apply to write master.xlsx and rebuild the database.")
    sys.exit(0)
if not (ut_changed | tag_changed).any():
    sys.exit("Nothing to change.")
if not args.yes and input("\nHave you run `git pull`, and is no GitHub workflow running? Type yes to continue: ").strip().lower() != "yes":
    sys.exit("Stopped.")
if not args.no_backup:
    backup = f"{cfg.MASTER_FILE}.bak_{datetime.now():%Y%m%d_%H%M%S}"
    shutil.copy2(cfg.MASTER_FILE, backup)
    print(f"Backup saved: {backup}  (do NOT commit this file)")

master.to_excel(cfg.MASTER_FILE, index=False)
import aggregation_db as adb   # noqa: E402
adb.rebuild_from_master()
adb.export_for_powerbi()
print("Done. Next: python retag_history.py --months <months>  to score topic sentiment for the new topics.")