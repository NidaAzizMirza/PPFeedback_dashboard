"""
retag_history.py — give the older comments (before September 2026) the same topic and
sentiment scoring as the live pipeline, so every month has topic data.

It reuses step_absa from run_pipeline.py on the tag groups each older row already has
(primary_tag_group / secondary_tag_groups), so the topic pairs are built the same way
as for September and October. It does NOT re-classify anything and does NOT change an
older row's tags.

Run from the project root, with `git pull` done and no GitHub workflow running:

    python retag_history.py                  # TRIAL: scores 40 older rows, prints them, writes nothing
    python retag_history.py --limit 300      # trial on more rows
    python retag_history.py --apply          # scores every older row, then writes master.xlsx and rebuilds metrics.db

Extras:
    --overwrite-sentiment   also replace grouping_sentiment on the scored rows with the new scoring
                            (changes those months' positive/negative/neutral split — off by default)
    --months 2026-09        re-score these months (comma-separated) even if they were scored before,
                            instead of the older months. Use it when a month's topic pairs were made
                            by an older version of the step.

--apply saves progress after every chunk to data/retag_progress.jsonl, so if it stops you can
run the same command again and it carries on. Don't commit that file.
"""
import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime

import pandas as pd

import pipeline_config as cfg

CUTOFF = "2026-09"                      # months before this have no topic pairs
CHUNK = 250                             # rows scored per batch
PROGRESS = "data/retag_progress.jsonl"
ID = cfg.COL_RESPONDENT_ID
CLEAN = cfg.COL_FEEDBACK_CLEAN
NEW_COLS = ["topic_sentiment_pairs", "absa_aspects", "absa_aspect_sentiments",
            "overall_experience_sentiment"]

ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--overwrite-sentiment", action="store_true")
ap.add_argument("--months", default=None)
args = ap.parse_args()
limit = args.limit if args.limit else (None if args.apply else 40)


def clean_value(x):
    """NaN -> None so the progress file stays valid JSON."""
    return None if (x is None or (isinstance(x, float) and pd.isna(x))) else x


# ── Which rows need scoring? ──────────────────────────────────────────────────
master = pd.read_excel(cfg.MASTER_FILE, dtype={ID: str})
month = pd.to_datetime(master["Start Date"], errors="coerce").dt.strftime("%Y-%m")
is_old = month.fillna("9999") < CUTOFF
has_text = master[CLEAN].fillna("").astype(str).str.strip() != ""
already_scored = master["overall_experience_sentiment"].fillna("").astype(str).str.strip() != ""

done_ids = set()
if args.apply and os.path.exists(PROGRESS):
    with open(PROGRESS, encoding="utf-8") as f:
        done_ids = {json.loads(line)["id"] for line in f if line.strip()}

if args.months:
    wanted = {m.strip() for m in args.months.split(",") if m.strip()}
    todo_mask = month.isin(wanted) & has_text            # re-score these months, scored before or not
else:
    todo_mask = is_old & has_text & ~already_scored
todo = master[todo_mask & ~master[ID].isin(done_ids)]
if limit:
    todo = todo.head(limit)

if args.months:
    print(f"Re-scoring months: {sorted(wanted)}")
else:
    print(f"Older rows (before {CUTOFF}): {int(is_old.sum()):,}")
print(f"Already scored: {int((is_old & already_scored).sum()):,}   resumed from progress file: {len(done_ids):,}")
print(f"To score now: {len(todo):,}" + ("   (TRIAL, nothing will be written)" if not args.apply else ""))
if todo.empty and not (args.apply and done_ids):
    sys.exit("Nothing to do.")

if args.apply:
    answer = input("\nHave you run `git pull`, and is no GitHub workflow running? Type yes to continue: ")
    if answer.strip().lower() != "yes":
        sys.exit("Stopped. Run git pull and check the Actions tab first.")

# ── Score in chunks with the live step ────────────────────────────────────────
print("\nLoading models (this takes a minute)...")
from run_pipeline import step_absa   # noqa: E402  (heavy import, so it comes after the checks above)

cols = [ID, CLEAN, "primary_tag_group", "secondary_tag_groups"]
if "Feedback" in master.columns:
    cols.append("Feedback")

trial_rows = []
started = time.time()
for start in range(0, len(todo), CHUNK):
    part = todo.iloc[start:start + CHUNK][cols].copy()
    part["primary_tag_group"] = part["primary_tag_group"].fillna("")
    part["secondary_tag_groups"] = part["secondary_tag_groups"].fillna("")   # NaN would break the group splitter
    if "Feedback" in part.columns:
        part["Feedback"] = part["Feedback"].fillna(part[CLEAN])

    scored = step_absa(part)

    records = [{
        "id": r[ID],
        "topic_sentiment_pairs": json.dumps(r["topic_sentiment_pairs"] if isinstance(r["topic_sentiment_pairs"], list) else []),
        "absa_aspects": clean_value(r["absa_aspects"]),
        "absa_aspect_sentiments": clean_value(r["absa_aspect_sentiments"]),
        "overall_experience_sentiment": clean_value(r["overall_experience_sentiment"]),
        "grouping_sentiment": clean_value(r["grouping_sentiment"]),
    } for _, r in scored.iterrows()]

    if args.apply:
        with open(PROGRESS, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
    else:
        trial_rows += records

    done = min(start + CHUNK, len(todo))
    elapsed = time.time() - started
    eta = elapsed / done * (len(todo) - done)
    print(f"  scored {done:,}/{len(todo):,}   elapsed {elapsed/60:.1f} min   about {eta/60:.0f} min left")

# ── Trial: show what it produced and stop ─────────────────────────────────────
if not args.apply:
    trial = pd.DataFrame(trial_rows).merge(master[[ID, CLEAN, "primary_tag_group"]], left_on="id", right_on=ID)
    n_with = sum(len(json.loads(p)) > 0 for p in trial["topic_sentiment_pairs"])
    print(f"\n{n_with} of {len(trial)} trial rows got at least one topic.\n")
    for _, r in trial.head(10).iterrows():
        pairs = json.loads(r["topic_sentiment_pairs"])
        print(f"- {str(r[CLEAN])[:110]}")
        print(f"    old primary group: {r['primary_tag_group']}   overall: {r['overall_experience_sentiment']}")
        print(f"    topics: {[(p['topic'], p['sentiment']) for p in pairs] or 'none'}")
    print("\nTRIAL ONLY — nothing was written. Run with --apply to score every older row.")
    sys.exit(0)

# ── Apply: back up, merge results into master, rebuild the database ───────────
with open(PROGRESS, encoding="utf-8") as f:
    results = pd.DataFrame([json.loads(line) for line in f if line.strip()])
results = results.drop_duplicates("id", keep="last").set_index("id")

backup = f"{cfg.MASTER_FILE}.bak_{datetime.now():%Y%m%d_%H%M%S}"
shutil.copy2(cfg.MASTER_FILE, backup)
print(f"\nBackup saved: {backup}  (do NOT commit this file)")

rows = master[ID].isin(results.index)
for col in NEW_COLS:
    master[col] = master[col].astype("object")
    master.loc[rows, col] = master.loc[rows, ID].map(results[col]).values
if args.overwrite_sentiment:
    new_sent = master.loc[rows, ID].map(results["grouping_sentiment"])
    master.loc[rows, "grouping_sentiment"] = new_sent.fillna(master.loc[rows, "grouping_sentiment"]).values

master.to_excel(cfg.MASTER_FILE, index=False)
print(f"Updated {int(rows.sum()):,} rows in {cfg.MASTER_FILE}")

import aggregation_db as adb   # noqa: E402
adb.rebuild_from_master()
adb.export_for_powerbi()

os.remove(PROGRESS)
print("\nDone ✓  Next: python check_topics.py to see topics for every month.")