"""
apply_manual_tags.py — put the human tags from the dashboard into master.xlsx and rebuild metrics.db.

The weekly pipeline only scores NEW responses, so a tag added to an older comment needs this step.
Human tags win over the algorithm and the rules. Safe to run repeatedly (nothing changes the second time).

    python apply_manual_tags.py                 # DRY RUN: shows what would change, writes nothing
    python apply_manual_tags.py --apply         # backs up master.xlsx, writes it, rebuilds metrics.db
    python apply_manual_tags.py --apply --yes   # same, no question asked (for the GitHub Action)
    python apply_manual_tags.py --apply --yes --no-backup    # Action: don't leave a .bak file behind

Reads the tags from the `manual-tags` branch (GitHub API if GITHUB_TOKEN is set, otherwise
`git fetch` + `git show`, otherwise data/manual_tags.csv).
"""
import argparse
import logging
import shutil
import sys
from datetime import datetime

import pandas as pd

import manual_tags as mt
import pipeline_config as cfg

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("apply_manual_tags")

ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true")
ap.add_argument("--yes", action="store_true")
ap.add_argument("--no-backup", action="store_true")
args = ap.parse_args()

ID = cfg.COL_RESPONDENT_ID
TRACKED = ["primary_tag", "primary_tag_group", "secondary_tags", "secondary_tag_groups",
           "topic_sentiment_pairs", "overall_experience_sentiment", "grouping_sentiment", "user_type"]

manual = mt.load_manual(log=log)
if manual.empty:
    sys.exit("No manual tags found. Nothing to do.")

master = pd.read_excel(cfg.MASTER_FILE, dtype={ID: str})
def _norm(d):
    """Empty, NaN and None all compare equal, so Excel's empty-cell round trip is not a 'change'."""
    return d.astype(object).where(d.notna(), "").astype(str).replace({"nan": "", "None": ""})

before = _norm(master[[c for c in TRACKED if c in master.columns]])
master = mt.apply_to_df(master, manual, id_col=ID, log=log)
after = _norm(master[before.columns])
changed = (before != after).any(axis=1)

ids_in_master = set(master[ID].map(mt.norm_id))
missing = [r for r in manual["respondent_id"].map(mt.norm_id) if r not in ids_in_master]
print(f"\nManual rows: {len(manual):,}   found in master: {len(manual) - len(missing):,}   "
      f"not in master: {len(missing):,}")
print(f"Rows in master that would change: {int(changed.sum()):,}")
if missing:
    print(f"  Not in master (wrong ID, or too new for the pipeline yet): {missing[:10]}")

if not args.apply:
    print("\nDRY RUN: nothing written. Add --apply to write master.xlsx and rebuild the database.")
    sys.exit(0)
if not changed.any():
    sys.exit("Nothing to change, master.xlsx is already up to date.")

if not args.yes:
    if input("\nHave you run `git pull`, and is no GitHub workflow running? Type yes to continue: ").strip().lower() != "yes":
        sys.exit("Stopped.")
if not args.no_backup:
    backup = f"{cfg.MASTER_FILE}.bak_{datetime.now():%Y%m%d_%H%M%S}"
    shutil.copy2(cfg.MASTER_FILE, backup)
    print(f"Backup saved: {backup}  (do NOT commit this file)")

master.to_excel(cfg.MASTER_FILE, index=False)
print(f"Updated {int(changed.sum()):,} rows in {cfg.MASTER_FILE}")

import aggregation_db as adb   # noqa: E402
adb.rebuild_from_master()
adb.export_for_powerbi()
print("Done. metrics.db rebuilt.")