"""
sentiment_overrides.py  — rule layer that runs around the SVM/ABSA steps.
Imported by run_pipeline.py. Every rule is a row in a list: add a case = add a line.

Three jobs:
  1. ALWAYS_TOPICS   topics that must be added whenever a keyword appears,
                     even if the SVM was confident about something else
                     (HARD_FALLBACK only runs when SVM confidence < 0.3).
  2. BEGINNER_EXTRA  extra beginner/non-professional wording for user_type.
  3. apply_sentiment_overrides(df)  fixes sentiment the ABSA model reads wrongly
                     (comparison with the old version, sarcasm, effort words).
"""
from __future__ import annotations
import re

COMPETITOR_GROUP = "Competitor mention"
COMPETITOR_NAMES = ["planda"]            # add more competitor names here
MIXED_AS = "neutral"                     # label for comments with good AND bad parts

# ── 1. topics that must always be added ─────────────────────────────────────
# group -> (granular tag, keywords). Group names must match tags.csv exactly,
# except COMPETITOR_GROUP which is new and not in tags.csv.
DOCUMENT_GROUP = "Document upload and handling"
# Regex (not plain keywords) so "plan" does not fire on "planning", "I plan to" or "to plan".
DOCUMENT_PATTERNS = [
    r"\bdocuments?\b", r"\bdocumentation\b", r"\bdocs?\b", r"\bdrawings?\b",
    r"(?<!\bi )(?<!\bto )(?<!\bwe )\bplans\b", r"(?<!\bi )(?<!\bto )(?<!\bwe )\bplan\b(?! to\b)(?! for\b)(?! ahead\b)",
    r"\bfloor ?plans?\b", r"\bsite plans?\b", r"\blocation plans?\b",
    r"\bblueprints?\b", r"\battachments?\b", r"\bpdfs?\b", r"\bphotos?\b",
]

ALWAYS_TOPICS = {
    "Guidance, clarity & jargon": ("Lack of guidance", ["guidance"]),
    DOCUMENT_GROUP:               ("Document uploading", ["re:" + p for p in DOCUMENT_PATTERNS]),
    COMPETITOR_GROUP:             ("Competitor mention", COMPETITOR_NAMES),
}

def add_always_topics(text, primary_group, secondary_tags, secondary_groups):
    """secondary_* are comma-joined strings. Group names contain commas
    ("Guidance, clarity & jargon"), so they are checked as substrings, never split."""
    tags = secondary_tags if isinstance(secondary_tags, str) else ""
    groups = secondary_groups if isinstance(secondary_groups, str) else ""
    low = str(text or "").lower()
    for group, (tag, kws) in ALWAYS_TOPICS.items():
        if group == primary_group or group in groups:
            continue
        if any(re.search(k[3:], low) if k.startswith("re:")
               else re.search(r"\b" + re.escape(k) + r"\b", low) for k in kws):
            tags = f"{tags}, {tag}" if tags else tag
            groups = f"{groups}, {group}" if groups else group
    return tags, groups

# ── 2. user_type: merged into USER_TYPE_KEYWORDS["beginner_or_one_time"] ────
BEGINNER_EXTRA = ["general public", "uninitiated", "not done this before",
                  "not used this before", "new to this", "new to planning",
                  "member of the public", "non-professional", "non professional"]

# ── 3. sentiment rules ──────────────────────────────────────────────────────
# STRONG cues are always negative, even if a nearby word sounds positive
STRONG_NEG = [
    r"\b(previous|old|earlier|former|original)\b.{0,80}\b(easier|better|superior|simpler|clearer)\b",
    r"\b(easier|better|simpler)\b.{0,25}\b(before|previously)\b",
    r"\bwhy (do|did|have) you (have to )?(change|changed)\b",
    r"\bdisappointing\b",
    r"\bamazing (amount|level) of \w+ for a (small|simple|minor|basic)\b",
]
# SOFT cues are negative unless the same sentence also has a positive
SOFT_NEG = [
    r"\b(intimidating|daunting|exhausting|overwhelming|stressful|frustrating)\b",
    r"\bnot (at all )?(simple|easy|straightforward|clear|intuitive)\b",
    r"\b(complex|confusing|unclear|difficult|complicated)\b",
    r"\bno (guidance|help|instructions?)\b",
]
POS = r"\b(great|excellent|helpful|good|works?|working|easy|fine|thanks?|brilliant|love)\b"
COMPARATIVE = r"\b(cheaper|better|easier|faster|simpler)\b"   # competitor comparison
NEGATORS = {"not", "no", "never", "without", "isn't", "wasn't", "aren't", "don't", "doesn't", "didn't", "n't"}

def _cue(patterns, sentence):
    """True if a pattern matches and is not itself negated ('not confusing')."""
    low = sentence.lower()
    for p in patterns:
        for m in re.finditer(p, low):
            before = low[:m.start()].split()[-2:]
            if not any(w in NEGATORS for w in before):
                return True
    return False

def _pos(sentence):
    """True if a positive word appears and is not negated ('not easy', 'not as good')."""
    low = sentence.lower()
    for m in re.finditer(POS, low):
        before = low[:m.start()].split()[-2:]
        if not any(w in NEGATORS for w in before):
            return True
    return False

def _sentences(t):
    return [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]

def override_row(text, rating, pairs, overall):
    """Returns (pairs, overall, flag). pairs = list of topic_sentiment_pairs dicts."""
    t = str(text or "")
    pairs = [dict(p) for p in (pairs or [])]
    sents = _sentences(t)
    strong  = any(_cue(STRONG_NEG, s) for s in sents)
    soft_neg_only = any(_cue(SOFT_NEG, s) and not _pos(s) for s in sents)
    soft_mixed    = any(_cue(SOFT_NEG, s) and _pos(s) for s in sents)
    pos_only      = any(_pos(s) and not _cue(SOFT_NEG + STRONG_NEG, s) for s in sents)

    new_overall, new_pair = overall, None
    if strong:
        new_overall = new_pair = "negative"
    elif soft_neg_only or soft_mixed:
        new_pair = "negative" if soft_neg_only else MIXED_AS
        new_overall = MIXED_AS if (soft_mixed or pos_only) else "negative"

    # ONE-WAY RULE: rules only make sentiment worse. An existing negative is never
    # softened, and a neutral can only become negative (never "mixed").
    def _worse(old, new):
        rank = {"positive": 2, "neutral": 1, MIXED_AS: 1, "negative": 0}
        return new if rank.get(new, 1) < rank.get(old, 1) else old
    new_overall = _worse(overall, new_overall) if overall else new_overall
    for p in pairs:
        if p.get("topic") == COMPETITOR_GROUP and re.search(COMPARATIVE, t, re.I):
            target = "negative"                                     # competitor is better/cheaper
        else:
            target = new_pair
        if target and p.get("sentiment") in (None, "positive", "neutral"):
            better = _worse(p.get("sentiment") or "positive", target)
            if better != p.get("sentiment"):
                p["sentiment"] = better; p["source"] = "rule"
    if any(p.get("topic") == COMPETITOR_GROUP and re.search(COMPARATIVE, t, re.I) for p in pairs):
        new_overall = "negative"

    flag = None
    try:
        if new_overall == "positive" and rating is not None and int(float(rating)) <= 2:
            flag = "positive text but rating<=2"
    except (TypeError, ValueError):
        pass
    return pairs, new_overall, flag

def _group_sentiment(pairs, overall):
    if not pairs:
        return overall or "neutral"
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for p in pairs:
        counts[p["sentiment"]] = counts.get(p["sentiment"], 0) + 1
    return max(counts, key=counts.get)

def apply_sentiment_overrides(df, text_col, rating_col="Rating", log=None):
    """Updates topic_sentiment_pairs, overall_experience_sentiment, grouping_sentiment."""
    changed = flagged = 0
    out_p, out_o, out_g = [], [], []
    for _, r in df.iterrows():
        pairs0 = r["topic_sentiment_pairs"] if isinstance(r["topic_sentiment_pairs"], list) else []
        o0 = r["overall_experience_sentiment"]
        pairs, o, flag = override_row(r[text_col], r.get(rating_col), pairs0, o0)
        g = _group_sentiment(pairs, o) if (pairs != pairs0 or o != o0) else r["grouping_sentiment"]
        changed += (pairs != pairs0 or o != o0)
        flagged += flag is not None
        out_p.append(pairs); out_o.append(o); out_g.append(g)
    df["topic_sentiment_pairs"], df["overall_experience_sentiment"], df["grouping_sentiment"] = out_p, out_o, out_g
    if log:
        log.info(f"Sentiment overrides changed {changed} rows; {flagged} flagged (positive text, rating<=2)")
    return df

if __name__ == "__main__":
    P = lambda *t: [{"topic": x, "sentiment": "positive"} for x in t]
    cases = [  # text, rating, topics, expected overall
        ("guidance online for someone who has not done this before is complex and unclear. feedback once application was recieved has been great", 5, P("Guidance, clarity & jargon"), "neutral"),
        ("Planda Portal is cheaper!", 3, P(COMPETITOR_GROUP), "negative"),
        ("Amazing amount of detail for a small greenhouse. The only reason I need planning permission", 2, [], "negative"),
        ("The previous version was easier", 3, [], "negative"),
        ("Process covers so many bases it's profoundly intimidating.", 3, [], "negative"),
        ("The previous way of uploading and describing images was a far superior way of doing things.", 1, P("Document upload and handling"), "negative"),
        ("For someone who is general public this is quite daunting but the help features are helpful. Thank you.", 4, [], "neutral"),
        ("Exhausting process for a simple application.", 2, [], "negative"),
        ("the previous website layout was much easier to follow. Why do you have to change the layout", 2, [], "negative"),
        ("Not simple for the uninitiated but all seems to work.", 4, [], "neutral"),
        ("It is disappointing that the methology has been changed for uploading when the other was working more than adequately", 3, P("Document upload and handling"), "negative"),
        ("Easy to use format", 5, [], "positive"),
        ("Excellent site and easy to use", 5, [], "positive"),
        ("Easy to use, not confusing at all", 5, [], "positive"),
        ("provide a link to the CIL form would be helpful", 5, P("Missing features & feature requests"), "positive"),
    ]
    # existing negatives must stay negative (these were softened by mistake before)
    for text, r in [("not easy or intuitive no help line not clear what to select", 1),
                    ("didnt find it easy to start the application via the online planning portal", 3),
                    ("not easy to navigate", 3),
                    ("it is too complicated and not easy to navigate at all", 1),
                    ("the supporting documents part was confusing i would expect to see a place to upload plans", 4),
                    ("unclear where you put photos not geared up for tree applications not as good as the old one waste of money", 1)]:
        p, o, _ = override_row(text, r, [{"topic": "x", "sentiment": "negative"}], "negative")
        ok = o == "negative" and p[0]["sentiment"] == "negative"
        print("OK  " if ok else "FAIL", "kept negative |", text[:55]); bad_neg = 0 if ok else 1
        if not ok: raise SystemExit("negative was softened")
    bad = 0
    for text, r, pairs, exp in cases:
        p, o, _ = override_row(text, r, pairs, "positive")
        ok = o == exp; bad += not ok
        print("OK  " if ok else "FAIL", o, [x["sentiment"] for x in p], "|", text[:55])
    print("sentiment failures:", bad)
    # topics + user_type
    s, g = add_always_topics("guidance online is unclear", "Forms & application details", "", "")
    s2, g2 = add_always_topics("Planda Portal is cheaper!", "Missing features & feature requests", "", "")
    for t, want in [("The plans would not upload", True), ("could not attach my drawing", True), ("Documents were confusing", True),
                    ("a site plan is needed", True), ("The planning process was slow", False), ("I plan to apply again", False),
                    ("hard to plan ahead", False), ("Easy to use format", False)]:
        _, g = add_always_topics(t, "Overall Positive Experience", "", "")
        assert (DOCUMENT_GROUP in g) == want, (t, g)
    print("document rule: all checks passed")
    print("always topics:", g, "|", g2)