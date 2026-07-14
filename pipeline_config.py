# pipeline_config.py
# ── All pipeline settings in one place ────────────────────────────────────
# Edit this file to configure the pipeline without touching run_pipeline.py

import os
from dotenv import load_dotenv

# Load variables from a local .env file (SURVEYMONKEY_ACCESS_TOKEN, etc.)
# In production (e.g. Streamlit Cloud), set these as secrets/env vars instead
# of shipping a .env file — load_dotenv() is a no-op if no .env is present.
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE        = os.path.join(BASE_DIR, "inputs", "new_export.xlsx")
INPUT_SHEET       = "Sheet"

MASTER_FILE       = os.path.join(BASE_DIR, "data", "master.xlsx")
PROCESSED_IDS     = os.path.join(BASE_DIR, "data", "processed_ids.csv")
# Separate dedup list for the lightweight daily raw-metrics job — kept
# apart from PROCESSED_IDS (which tracks what's gone through full NLP)
# so the daily and weekly jobs never double-count the same respondent.
PROCESSED_IDS_RAW = os.path.join(BASE_DIR, "data", "processed_ids_raw.csv")
TAGS_FILE         = os.path.join(BASE_DIR, "data", "tags.csv")
MANUAL_TAGS_FILE  = os.path.join(BASE_DIR, "data", "manuallytagged_complete.csv")

MODELS_DIR        = os.path.join(BASE_DIR, "models")
SVM_MODEL         = os.path.join(MODELS_DIR, "svm_final_model.pkl")
SVM_ENCODER       = os.path.join(MODELS_DIR, "svm_final_label_encoder.pkl")

OUTPUTS_LATEST    = os.path.join(BASE_DIR, "outputs", "latest")
OUTPUTS_ARCHIVE   = os.path.join(BASE_DIR, "outputs", "archive")
LOGS_DIR          = os.path.join(BASE_DIR, "logs")

# ── SurveyMonkey API settings ──────────────────────────────────────────────
# Set these in a .env file (local) or as environment variables / secrets
# (Streamlit Cloud). Never commit real values — see .env.example.
SM_ACCESS_TOKEN   = os.getenv("SURVEYMONKEY_ACCESS_TOKEN")
SM_SURVEY_ID      = os.getenv("SURVEYMONKEY_SURVEY_ID")
SM_PER_PAGE       = 100   # responses per page when paginating the API

# ── Testing ─────────────────────────────────────────────────────────────────
# Set SKIP_NLP=true (env var) to bypass classify/ABSA/entity extraction —
# useful for testing ingest → preprocess → save → dashboard plumbing fast,
# without loading the SVM/ABSA/spaCy models. NEVER use this output for
# real analysis — tag/sentiment/entity columns are placeholders.
SKIP_NLP = os.getenv("SKIP_NLP", "false").lower() == "true"

# Set SKIP_API=true (env var) to bypass the SurveyMonkey API entirely and
# read straight from the manual export at INPUT_FILE instead. Useful while
# the API's daily quota is exhausted, or to avoid burning quota during
# heavy local testing — export responses from SurveyMonkey yourself and
# drop the file at INPUT_FILE. NLP/metrics still run normally on whatever
# rows the file contains.
SKIP_API = os.getenv("SKIP_API", "false").lower() == "true"

# ── Column names ───────────────────────────────────────────────────────────
COL_RESPONDENT_ID = "Respondent ID"
COL_FEEDBACK      = "Feedback"
COL_FEEDBACK_CLEAN = "Feedback_clean"
COL_RATING        = "Rating"
COL_DATE          = "Start Date"
COL_FEEDBACK = "Is there anything you would like to tell us about your experience?"
COL_FEEDBACK_CLEAN = "Feedback_clean"

# ── SVM settings ───────────────────────────────────────────────────────────
EMBEDDING_MODEL   = "all-mpnet-base-v2"
MIN_SAMPLES       = 8
SVM_C             = 5
PRIMARY_THRESHOLD   = 0.3
SECONDARY_THRESHOLD = 0.15

# ── ABSA settings ──────────────────────────────────────────────────────────
ABSA_MODEL        = "yangheng/deberta-v3-base-absa-v1.1"
ABSA_MIN_WORDS    = 2       # skip reviews shorter than this
ABSA_MAX_ASPECTS  = 3       # max aspects to check per review

# ── Summarisation settings ─────────────────────────────────────────────────
SUMMARY_MODEL     = "facebook/bart-large-cnn"
SUMMARY_MIN_REVIEWS = 5     # minimum reviews per group to summarise
SUMMARY_MAX_REVIEWS = 80    # sample size for large groups

# ── Entity extraction settings ─────────────────────────────────────────────
SPACY_MODEL       = "en_core_web_sm"
MIN_ENTITY_COUNT  = 3       # minimum mentions to include in aggregations

# ── Output settings ────────────────────────────────────────────────────────
AUDIENCE_OUTPUTS = {
    "research":    ["master", "entities", "absa_detail"],
    "product":     ["summaries", "entity_analysis", "priority_matrix"],
    "leadership":  ["executive_summary"],
}