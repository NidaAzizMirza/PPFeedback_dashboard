"""
manual_tags.py — human corrections to the algorithm's tags.

Where the tags live
    data/manual_tags.csv on a SEPARATE git branch called `manual-tags` (not main).
    Saving from the dashboard commits to that branch, so Streamlit Cloud does not
    redeploy (and log everyone out) on every save. One row per respondent.

How it is used
    dashboard  -> save_record() / delete_record()   writes the CSV through the GitHub API
    pipeline   -> apply_to_df()                      human tags win over the algorithm and the rules
    scripts    -> apply_manual_tags.py, manual_tag_report.py, export_manual_to_training.py

This module has no Streamlit and no pipeline_config import, so the dashboard can use it.
"""
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
LOCAL_FILE = ROOT / "data" / "manual_tags.csv"
REPO_PATH = "data/manual_tags.csv"
TAGS_FILE = ROOT / "data" / "tags.csv"
DEFAULT_BRANCH = "manual-tags"
DEFAULT_BASE = "main"
SEP = "|"                      # between tags inside the CSV cell (tag titles can contain commas)

COLUMNS = ["respondent_id", "tags", "user_type", "sentiment", "note",
           "tagged_by", "tagged_at", "algo_groups", "algo_sentiment", "algo_user_type",
           "mode", "remove_groups"]
# mode "add" (default): my tags are ADDED to the algorithm's; remove_groups drops topics the algorithm got wrong
# mode "replace":       my tags REPLACE all the algorithm's tags
SENTIMENTS = ["positive", "neutral", "negative"]

# Keep in step with USER_TYPE_KEYWORDS in run_pipeline.py
USER_TYPES = [
    "beginner_or_one_time", "frequent", "infrequent", "occasional", "homeowner", "agent",
    "project_manager", "designer", "architect", "tree_surgeon", "charity", "community_champion",
    "non_owner", "professional_regular", "non_tech", "tech_user", "disabled",
    "accessibility_issues", "older_user",
]


# ── tag catalogue ───────────────────────────────────────────────────────────
# Tags the pipeline creates itself (sentiment_overrides.py) that are not in tags.csv
EXTRA_TAGS = {"Competitor mention": "Competitor mention"}


def load_catalogue(path: Path | str = TAGS_FILE) -> dict[str, str]:
    """{tag title: group} from tags.csv, plus EXTRA_TAGS."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.dropna(subset=["Title", "Group"])
    cat = dict(zip(df["Title"].astype(str).str.strip(), df["Group"].astype(str).str.strip()))
    return {**cat, **EXTRA_TAGS}


# ── CSV <-> DataFrame ───────────────────────────────────────────────────────
def parse_csv(text: str) -> pd.DataFrame:
    if not text or not text.strip():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[COLUMNS]


def norm_id(x) -> str:
    """Respondent IDs arrive as str, int or float (1.19e+11); compare them as plain digit strings."""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


# ── GitHub storage ──────────────────────────────────────────────────────────
class ConflictError(Exception):
    """Someone else saved in between; read again and retry."""


class GitHubStore:
    def __init__(self, token: str, repo: str, branch: str = DEFAULT_BRANCH,
                 base: str = DEFAULT_BASE, path: str = REPO_PATH):
        self.token, self.repo, self.branch, self.base, self.path = token, repo, branch, base, path
        self.api = f"https://api.github.com/repos/{repo}"

    def _req(self, method, url, **kw):
        import requests
        headers = {"Authorization": f"Bearer {self.token}",
                   "Accept": "application/vnd.github+json",
                   "X-GitHub-Api-Version": "2022-11-28"}
        return requests.request(method, url, headers=headers, timeout=20, **kw)

    def branch_exists(self) -> bool:
        return self._req("GET", f"{self.api}/git/ref/heads/{self.branch}").status_code == 200

    def ensure_branch(self):
        if self.branch_exists():
            return
        r = self._req("GET", f"{self.api}/git/ref/heads/{self.base}")
        r.raise_for_status()
        sha = r.json()["object"]["sha"]
        r = self._req("POST", f"{self.api}/git/refs",
                      json={"ref": f"refs/heads/{self.branch}", "sha": sha})
        if r.status_code not in (200, 201, 422):       # 422 = already exists (race)
            r.raise_for_status()

    def read(self) -> tuple[str, str | None]:
        """(csv text, file sha). ('', None) when the file or branch does not exist yet."""
        r = self._req("GET", f"{self.api}/contents/{self.path}", params={"ref": self.branch})
        if r.status_code == 404:
            return "", None
        r.raise_for_status()
        j = r.json()
        return base64.b64decode(j["content"]).decode("utf-8"), j["sha"]

    def write(self, text: str, sha: str | None, message: str):
        self.ensure_branch()
        body = {"message": message, "branch": self.branch,
                "content": base64.b64encode(text.encode("utf-8")).decode("ascii")}
        if sha:
            body["sha"] = sha
        r = self._req("PUT", f"{self.api}/contents/{self.path}", json=body)
        if r.status_code in (409, 422):
            raise ConflictError(r.text[:200])
        r.raise_for_status()


def make_store(token=None, repo=None, branch=None) -> GitHubStore | None:
    token = token or os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    repo = repo or os.getenv("GITHUB_REPOSITORY")
    if not token or not repo:
        return None
    return GitHubStore(token, repo, branch or os.getenv("MANUAL_TAGS_BRANCH") or DEFAULT_BRANCH)


def _modify(store: GitHubStore, change, message: str, retries: int = 4):
    """Read-modify-write with retry, so two people saving at once do not lose each other's tags."""
    last = None
    for attempt in range(retries):
        text, sha = store.read()
        df = change(parse_csv(text))
        try:
            store.write(df.to_csv(index=False), sha, message)
            return df
        except ConflictError as e:
            last = e
            time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"Could not save after {retries} tries ({last})")


def save_record(store: GitHubStore, record: dict):
    rid = norm_id(record["respondent_id"])
    row = {c: str(record.get(c, "") or "") for c in COLUMNS}
    row["respondent_id"] = rid
    row["tagged_at"] = row["tagged_at"] or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def change(df):
        df = df[df["respondent_id"].map(norm_id) != rid]
        return pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return _modify(store, change, f"Manual tags for respondent {rid}")


def delete_record(store: GitHubStore, respondent_id):
    rid = norm_id(respondent_id)
    return _modify(store, lambda df: df[df["respondent_id"].map(norm_id) != rid],
                   f"Remove manual tags for respondent {rid}")


# ── loading (dashboard + pipeline) ──────────────────────────────────────────
_CACHE: dict = {"t": 0.0, "df": None}


def load_manual(token=None, repo=None, branch=None, allow_git: bool = True,
                cache_seconds: int = 0, log=None) -> pd.DataFrame:
    """GitHub API (if a token is available) -> `git show origin/manual-tags` -> local file -> empty."""
    if cache_seconds and _CACHE["df"] is not None and time.time() - _CACHE["t"] < cache_seconds:
        return _CACHE["df"]
    df, source = None, None
    store = make_store(token, repo, branch)
    if store is not None:
        try:
            df, source = parse_csv(store.read()[0]), "GitHub API"
        except Exception as e:                                   # noqa: BLE001
            if log: log.warning(f"Manual tags: GitHub API read failed ({e})")
    if df is None and allow_git:
        b = branch or os.getenv("MANUAL_TAGS_BRANCH") or DEFAULT_BRANCH
        try:
            subprocess.run(["git", "fetch", "origin", b], cwd=ROOT, check=True, timeout=60,
                           capture_output=True)
            out = subprocess.run(["git", "show", f"origin/{b}:{REPO_PATH}"], cwd=ROOT, check=True,
                                 timeout=30, capture_output=True, text=True)
            df, source = parse_csv(out.stdout), f"git branch {b}"
        except Exception:                                        # noqa: BLE001  (no branch yet, offline, no git)
            pass
    if df is None and LOCAL_FILE.exists():
        df, source = parse_csv(LOCAL_FILE.read_text(encoding="utf-8")), "local file"
    if df is None:
        df, source = pd.DataFrame(columns=COLUMNS), "none (no manual tags yet)"
    if log:
        log.info(f"Manual tags: {len(df)} rows from {source}")
    _CACHE.update(t=time.time(), df=df)
    return df


# ── applying to a dataframe (master or a pipeline batch) ────────────────────
def _split(s) -> list[str]:
    return [t.strip() for t in str(s or "").split(SEP) if t.strip()]


def _pairs_in(v) -> tuple[list, bool]:
    """(list of pair dicts, was it already a list?)"""
    if isinstance(v, list):
        return [dict(p) for p in v], True
    if isinstance(v, str) and v.strip().startswith("["):
        try:
            return [dict(p) for p in json.loads(v)], False
        except (ValueError, TypeError):
            pass
    return [], False


def split_known(s, known) -> list[str]:
    """Split a ", "-joined string whose items may themselves contain ", " (e.g. the group
    "Guidance, clarity & jargon"), using the list of known names."""
    s = s if isinstance(s, str) else ""
    names = sorted(known, key=len, reverse=True)
    out, rest = [], s.strip()
    while rest:
        for k in names:
            if rest == k or rest.startswith(k + ", "):
                out.append(k)
                rest = rest[len(k) + 2:].lstrip() if rest != k else ""
                break
        else:
            head, _, rest = rest.partition(", ")
            if head.strip():
                out.append(head.strip())
    return out


def apply_to_df(df: pd.DataFrame, manual: pd.DataFrame | None = None,
                id_col: str = "Respondent ID", catalogue: dict | None = None,
                log=None) -> pd.DataFrame:
    """Human tags win over the algorithm and the rules. Idempotent: running twice changes nothing more.

    mode "add" (default)  my tags are added to the algorithm's; topics in remove_groups are dropped;
                          the algorithm's main topic stays the main topic unless it was removed
    mode "replace"        my tags replace all the algorithm's tags; the first one is the main topic
    Sentiment  -> overall_experience_sentiment, grouping_sentiment and every topic pair
    User type  -> user_type
    Blank fields in the manual row leave the algorithm's value alone.
    """
    if manual is None:
        manual = load_manual(cache_seconds=300, log=log)
    if manual.empty or id_col not in df.columns:
        return df
    try:
        cat = catalogue if catalogue is not None else load_catalogue()
    except Exception as e:                                       # noqa: BLE001
        if log: log.warning(f"Manual tags: tags.csv unreadable ({e}); tags skipped, sentiment/user type still applied")
        cat = {}
    known_groups = set(cat.values())

    m = manual.copy()
    for c in COLUMNS:
        if c not in m.columns:
            m[c] = ""
    m["_id"] = m["respondent_id"].map(norm_id)
    m = m.drop_duplicates("_id", keep="last").set_index("_id")
    ids = df[id_col].map(norm_id)
    hit_idx = df.index[ids.isin(m.index)]
    if len(hit_idx) == 0:
        return df

    updates: dict[str, dict] = {}
    unknown = 0

    def put(col, idx, val):
        if col in df.columns:
            updates.setdefault(col, {})[idx] = val

    def cell(idx, col):
        v = df.at[idx, col] if col in df.columns else ""
        return v if isinstance(v, str) else ""

    for idx in hit_idx:
        rec = m.loc[ids[idx]]
        add_tags = [t for t in _split(rec["tags"]) if t in cat]
        unknown += len(_split(rec["tags"])) - len(add_tags)
        remove = set(_split(rec["remove_groups"]))
        replace = rec["mode"] == "replace"
        sent = rec["sentiment"] if rec["sentiment"] in SENTIMENTS else None
        groups_final = None

        if replace and add_tags:
            tags_now = list(dict.fromkeys(add_tags))
            groups_now = list(dict.fromkeys(cat[t] for t in tags_now))
            primary_tag, primary_group = tags_now[0], groups_now[0]
            groups_final = groups_now
        elif not replace and (add_tags or remove):
            cur_pt, cur_pg = cell(idx, "primary_tag"), cell(idx, "primary_tag_group")
            tags_now = ([cur_pt] if cur_pt else []) + split_known(cell(idx, "secondary_tags"), cat.keys())
            groups_now = ([cur_pg] if cur_pg else []) + split_known(cell(idx, "secondary_tag_groups"), known_groups)
            tags_now = [t for t in dict.fromkeys(tags_now) if cat.get(t) not in remove]
            groups_now = [g for g in dict.fromkeys(groups_now) if g not in remove]
            for t in add_tags:
                if t not in tags_now:
                    tags_now.append(t)
                if cat[t] not in groups_now:
                    groups_now.append(cat[t])
            keep_primary = cur_pg and cur_pg not in remove
            primary_group = cur_pg if keep_primary else (groups_now[0] if groups_now else "")
            primary_tag = cur_pt if (keep_primary and cur_pt) else (tags_now[0] if tags_now else "")
            groups_final = groups_now
        if groups_final is not None:
            put("primary_tag", idx, primary_tag)
            put("primary_tag_group", idx, primary_group)
            put("secondary_tags", idx, ", ".join(t for t in tags_now if t != primary_tag))
            put("secondary_tag_groups", idx, ", ".join(g for g in groups_final if g != primary_group))

        if "topic_sentiment_pairs" in df.columns and (groups_final is not None or sent):
            old, was_list = _pairs_in(df.at[idx, "topic_sentiment_pairs"])
            old_by_topic = {p.get("topic"): p for p in old}
            overall_old = df.at[idx, "overall_experience_sentiment"] \
                if "overall_experience_sentiment" in df.columns else None
            fallback = sent or (overall_old if isinstance(overall_old, str) and overall_old else "neutral")
            topics = groups_final if groups_final is not None else [p.get("topic") for p in old]
            new = []
            for g in topics:
                p = dict(old_by_topic.get(g, {"topic": g, "source": "manual"}))
                p["sentiment"] = sent or p.get("sentiment") or fallback
                new.append(p)
            put("topic_sentiment_pairs", idx, new if was_list else json.dumps(new))

        if sent:
            put("overall_experience_sentiment", idx, sent)
            put("grouping_sentiment", idx, sent)

        if rec["user_type"].strip():
            put("user_type", idx, ", ".join(_split(rec["user_type"].replace(",", SEP))))

    for col, vals in updates.items():
        df[col] = pd.Series([vals.get(i, v) for i, v in zip(df.index, df[col])],
                            index=df.index, dtype=object)
    if log:
        log.info(f"Manual tags applied to {len(hit_idx)} rows"
                 + (f" ({unknown} unknown tag names ignored)" if unknown else ""))
    return df