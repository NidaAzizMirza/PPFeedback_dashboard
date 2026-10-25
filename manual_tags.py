"""
manual_tag_report.py — how often did the algorithm and a human disagree?

Compares the algorithm's topics and sentiment, as saved when each comment was manually tagged,
with the human tags. Run it after you have tagged a batch.

    python manual_tag_report.py

Writes outputs/latest/manual_vs_algo.csv (one row per manually tagged comment) and prints a summary.
Group level: tag titles are mapped to their group in tags.csv, because the dashboard only knows the
algorithm's groups. Only comments tagged once the dashboard saves the algorithm's values are included.
"""
import os

import pandas as pd

import manual_tags as mt

manual = mt.load_manual()
cat = mt.load_catalogue()
changed = (manual["tags"].str.strip() != "") | (manual["remove_groups"].str.strip() != "")
m = manual[changed & (manual["algo_groups"].str.strip() != "")].copy()
if m.empty:
    raise SystemExit("No manually tagged comments with the algorithm's values saved yet.")

sets = lambda s: {x for x in str(s).split(mt.SEP) if x}


def final_groups(r):
    """The human's final set of topics for this comment."""
    added = {cat[t] for t in sets(r["tags"]) if t in cat}
    if r["mode"] == "replace":
        return added
    return (sets(r["algo_groups"]) | added) - sets(r["remove_groups"])


m["human_groups"] = m.apply(final_groups, axis=1)
m["algo_groups_set"] = m["algo_groups"].map(sets)
m["missed_by_algo"] = [h - a for h, a in zip(m["human_groups"], m["algo_groups_set"])]
m["wrongly_added_by_algo"] = [a - h for h, a in zip(m["human_groups"], m["algo_groups_set"])]
m["groups_match"] = m["missed_by_algo"].map(len).eq(0) & m["wrongly_added_by_algo"].map(len).eq(0)
sent = m[m["sentiment"].isin(mt.SENTIMENTS)]

print(f"Comments compared: {len(m):,}")
print(f"Same set of topics as the human: {m['groups_match'].mean():.0%}")
if len(sent):
    print(f"Same overall sentiment as the human (where the human set one): "
          f"{(sent['sentiment'] == sent['algo_sentiment']).mean():.0%}  ({len(sent):,} comments)")

def tally(col):
    return pd.Series([g for s in m[col] for g in s]).value_counts()

print("\nTopics the algorithm MISSED most often:")
print(tally("missed_by_algo").head(10).to_string() if m["missed_by_algo"].map(len).sum() else "  none")
print("\nTopics the algorithm ADDED wrongly most often:")
print(tally("wrongly_added_by_algo").head(10).to_string() if m["wrongly_added_by_algo"].map(len).sum() else "  none")

out = m.assign(**{c: m[c].map(lambda s: mt.SEP.join(sorted(s)))
                  for c in ("human_groups", "algo_groups_set", "missed_by_algo", "wrongly_added_by_algo")})
os.makedirs("outputs/latest", exist_ok=True)
out.drop(columns=["algo_groups_set"]).to_csv("outputs/latest/manual_vs_algo.csv", index=False)
print("\nFull list: outputs/latest/manual_vs_algo.csv")