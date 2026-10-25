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
for _c in ("mode", "remove_groups"):
    if _c not in manual.columns:
        manual[_c] = ""
manual = manual[(manual["tags"].str.strip() != "") | (manual["remove_groups"].str.strip() != "")]
if manual.empty:
    sys.exit("No manual tags to export.")

master = pd.read_excel(cfg.MASTER_FILE, dtype={ID: str})
master["_id"] = master[ID].map(mt.norm_id)
master = master.drop_duplicates("_id").set_index("_id")

train = pd.read_csv(cfg.MANUAL_TAGS_FILE, encoding="utf-8-sig")
norm = lambda s: str(s).strip().lower()

rows, skipped = [], 0
replace_texts, drop_pairs = set(), []          # texts whose old rows go; (text, group) rows to drop
for _, r in manual.iterrows():
    rid = mt.norm_id(r["respondent_id"])
    if rid not in master.index or not str(master.at[rid, CLEAN]).strip():
        skipped += 1
        continue
    text = str(master.at[rid, CLEAN]).strip()
    start = pd.to_datetime(master.at[rid, "Start Date"], errors="coerce")
    if r["mode"] == "replace":
        replace_texts.add(norm(text))
    for g in [x for x in r["remove_groups"].split(mt.SEP) if x]:
        drop_pairs.append((norm(text), g))
    for tag in [t for t in r["tags"].split(mt.SEP) if t in cat]:
        rows.append({"Text": text, "Tag": tag, "Tag - Group": cat[tag], "Tag - Board": "Tags",
                     "Note - Title": "Manual correction",
                     "Note - Month": start.strftime("%b") if pd.notna(start) else "",
                     "Note - Start date": start.strftime("%d/%m/%Y %H:%M") if pd.notna(start) else "",
                     "Note - Respondent ID": rid,
                     "Created": datetime.now().strftime("%d/%m/%Y %H:%M")})
new = pd.DataFrame(rows)
if new.empty and not drop_pairs:
    sys.exit(f"Nothing to export ({skipped} manual rows not found in master).")

tnorm = train["Text"].map(norm)
drop = tnorm.isin(replace_texts)                                   # "replace": the human version wins
for text, g in drop_pairs:                                         # "add" + removed topic: drop just those rows
    drop |= (tnorm == text) & (train["Tag - Group"] == g)
replaced = int(drop.sum())
if len(new):                                                       # don't write a tag twice for the same comment
    have = set(zip(tnorm[~drop], train.loc[~drop, "Tag"]))
    new = new[[(norm(t), g) not in have for t, g in zip(new["Text"], new["Tag"])]]
out = pd.concat([train[~drop], new], ignore_index=True)

print(f"Training file now: {len(train):,} rows")
print(f"Human tags to add: {len(new):,} rows for {new['Text'].nunique() if len(new) else 0:,} comments")
print(f"Old training rows removed (replaced or removed topics): {replaced:,}")
print(f"Manual rows skipped (not in master): {skipped:,}")
print(f"Training file after: {len(out):,} rows")
if not args.apply:
    sys.exit("\nDRY RUN: nothing written. Add --apply to write the training file.")

backup = f"{cfg.MANUAL_TAGS_FILE}.bak_{datetime.now():%Y%m%d_%H%M%S}"
shutil.copy2(cfg.MANUAL_TAGS_FILE, backup)
out.to_csv(cfg.MANUAL_TAGS_FILE, index=False, encoding="utf-8-sig")
print(f"\nBackup: {backup}\nWritten: {cfg.MANUAL_TAGS_FILE}")