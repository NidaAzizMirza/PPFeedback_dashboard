"""
export_manual_to_training.py — add the human tags to the training file (manuallytagged_complete.csv).

Each manual tag becomes one row in the same layout as your Dovetail export. If a comment is already in
the training file, its old rows are replaced by the human tags, so the correction wins.

    python export_manual_to_training.py            # DRY RUN: counts only, writes nothing
    python export_manual_to_training.py --apply    # backs up the training file, then writes it

This does NOT retrain the model. Retrain with your usual training script afterwards.
Keep some rows out of training to test on (for example every 5th respondent), otherwise the accuracy
you measure will look better than it is.
"""
import argparse
import shutil
import sys
from datetime import datetime

import pandas as pd

import manual_tags as mt
import pipeline_config as cfg

ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true")
args = ap.parse_args()

ID, CLEAN = cfg.COL_RESPONDENT_ID, cfg.COL_FEEDBACK_CLEAN
cat = mt.load_catalogue()
manual = mt.load_manual()
manual = manual[manual["tags"].str.strip() != ""]
if manual.empty:
    sys.exit("No manual tags to export.")

master = pd.read_excel(cfg.MASTER_FILE, dtype={ID: str})
master["_id"] = master[ID].map(mt.norm_id)
master = master.drop_duplicates("_id").set_index("_id")

train = pd.read_csv(cfg.MANUAL_TAGS_FILE, encoding="utf-8-sig")
norm = lambda s: str(s).strip().lower()

rows, skipped = [], 0
for _, r in manual.iterrows():
    rid = mt.norm_id(r["respondent_id"])
    if rid not in master.index or not str(master.at[rid, CLEAN]).strip():
        skipped += 1
        continue
    text = str(master.at[rid, CLEAN]).strip()
    start = pd.to_datetime(master.at[rid, "Start Date"], errors="coerce")
    for tag in [t for t in r["tags"].split(mt.SEP) if t in cat]:
        rows.append({"Text": text, "Tag": tag, "Tag - Group": cat[tag], "Tag - Board": "Tags",
                     "Note - Title": "Manual correction",
                     "Note - Month": start.strftime("%b") if pd.notna(start) else "",
                     "Note - Start date": start.strftime("%d/%m/%Y %H:%M") if pd.notna(start) else "",
                     "Note - Respondent ID": rid,
                     "Created": datetime.now().strftime("%d/%m/%Y %H:%M")})
new = pd.DataFrame(rows)
if new.empty:
    sys.exit(f"Nothing to export ({skipped} manual rows not found in master).")

texts = set(new["Text"].map(norm))
replaced = int(train["Text"].map(norm).isin(texts).sum())
out = pd.concat([train[~train["Text"].map(norm).isin(texts)], new], ignore_index=True)

print(f"Training file now: {len(train):,} rows")
print(f"Human tags to add: {len(new):,} rows for {new['Text'].nunique():,} comments")
print(f"Old training rows replaced by the human version: {replaced:,}")
print(f"Manual rows skipped (not in master): {skipped:,}")
print(f"Training file after: {len(out):,} rows")
if not args.apply:
    sys.exit("\nDRY RUN: nothing written. Add --apply to write the training file.")

backup = f"{cfg.MANUAL_TAGS_FILE}.bak_{datetime.now():%Y%m%d_%H%M%S}"
shutil.copy2(cfg.MANUAL_TAGS_FILE, backup)
out.to_csv(cfg.MANUAL_TAGS_FILE, index=False, encoding="utf-8-sig")
print(f"\nBackup: {backup}\nWritten: {cfg.MANUAL_TAGS_FILE}")