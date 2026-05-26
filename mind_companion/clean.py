"""
clean.py
--------
Cleans and processes raw datasets into model-ready CSVs.

Inputs:
    - data/raw/crisis_raw.csv          (downloaded by dataset.py)
    - Counsel Chat from HuggingFace    (nbertagnolli/counsel-chat)

Outputs:
    - data/processed/counsel_chat_clean.csv   (used for RAG retrieval)
    - data/processed/crisis_train.csv         (~70% — classifier training)
    - data/processed/crisis_val.csv           (~15% — validation)
    - data/processed/crisis_test.csv          (~15% — held-out test)

Run:
    python clean.py
"""

import re
import sys

import pandas as pd
from datasets import load_dataset
from loguru import logger
from sklearn.model_selection import train_test_split

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# ---------------------------------------------------------------------------
# File paths & constants
# ---------------------------------------------------------------------------
CRISIS_RAW_PATH    = RAW_DATA_DIR / "crisis_raw.csv"
COUNSEL_RAW_PATH   = RAW_DATA_DIR / "counsel_chat_raw.csv"

COUNSEL_CLEAN_PATH = PROCESSED_DATA_DIR / "counsel_chat_clean.csv"
CRISIS_TRAIN_PATH  = PROCESSED_DATA_DIR / "crisis_train.csv"
CRISIS_VAL_PATH    = PROCESSED_DATA_DIR / "crisis_val.csv"
CRISIS_TEST_PATH   = PROCESSED_DATA_DIR / "crisis_test.csv"

RANDOM_SEED        = 42
MIN_ANSWER_LEN     = 50    # Counsel Chat: drop very short therapist answers
MIN_TEXT_LEN       = 20    # Crisis: drop very short posts


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Normalize whitespace, strip URLs, drop non-ASCII characters."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", " ", text)          # strip URLs
    text = re.sub(r"[^\x00-\x7F]+", " ", text)    # ASCII only
    text = re.sub(r"\s+", " ", text)              # collapse whitespace
    return text.strip()


# ---------------------------------------------------------------------------
# Counsel Chat — RAG corpus
# ---------------------------------------------------------------------------
def process_counsel_chat(force: bool = False) -> pd.DataFrame:
    """
    Pull Counsel Chat from HuggingFace, clean, and save the RAG-ready CSV.
    Each row gets a `document` field combining question + answer for embedding.
    """
    if COUNSEL_CLEAN_PATH.exists() and not force:
        logger.info(
            f"{COUNSEL_CLEAN_PATH.name} already exists, skipping. "
            "Pass force=True to regenerate."
        )
        return pd.read_csv(COUNSEL_CLEAN_PATH)

    logger.info("Loading Counsel Chat from HuggingFace (nbertagnolli/counsel-chat) ...")
    df = load_dataset("nbertagnolli/counsel-chat", split="train").to_pandas()

    # Save a raw snapshot for reproducibility / inspection
    df.to_csv(COUNSEL_RAW_PATH, index=False)
    logger.info(f"Raw snapshot saved to {COUNSEL_RAW_PATH}")

    df["questionText"] = df["questionText"].apply(clean_text)
    df["answerText"]   = df["answerText"].apply(clean_text)

    df = df.dropna(subset=["questionText", "answerText"])
    df = df[df["answerText"].str.len() >= MIN_ANSWER_LEN]

    # Build the RAG document — Q + A combined for embedding & retrieval
    df["document"] = "Question: " + df["questionText"] + "\nAnswer: " + df["answerText"]

    keep_cols = ["questionText", "answerText", "topic", "document"]
    df[keep_cols].to_csv(COUNSEL_CLEAN_PATH, index=False)

    logger.success(f"Counsel Chat cleaned: {len(df):,} rows -> {COUNSEL_CLEAN_PATH}")
    return df


# ---------------------------------------------------------------------------
# Crisis dataset — classifier train/val/test
# ---------------------------------------------------------------------------
def process_crisis_data(force: bool = False):
    """
    Clean the Kaggle suicide-watch CSV and split into stratified
    train (70%) / val (15%) / test (15%) sets.
    """
    if not CRISIS_RAW_PATH.exists():
        raise FileNotFoundError(
            f"Crisis raw data not found at {CRISIS_RAW_PATH}.\n"
            "Run `python dataset.py` first to download it from Kaggle."
        )

    splits = (CRISIS_TRAIN_PATH, CRISIS_VAL_PATH, CRISIS_TEST_PATH)
    if all(p.exists() for p in splits) and not force:
        logger.info(
            "Crisis train/val/test splits already exist, skipping. "
            "Pass force=True to regenerate."
        )
        return tuple(pd.read_csv(p) for p in splits)

    logger.info(f"Loading crisis data from {CRISIS_RAW_PATH} ...")
    df = pd.read_csv(CRISIS_RAW_PATH)

    # Drop Kaggle's unnamed index column if it slipped through
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df[["text", "class"]].copy()
    df.columns = ["text", "label"]
    df["text"] = df["text"].apply(clean_text)

    df = df[df["text"].str.len() >= MIN_TEXT_LEN].dropna()

    # Map textual label -> binary; drop anything that didn't map (defensive)
    df["label_binary"] = df["label"].map({"suicide": 1, "non-suicide": 0})
    df = df.dropna(subset=["label_binary"])
    df["label_binary"] = df["label_binary"].astype(int)

    # Stratified 70 / 15 / 15 split
    train, temp = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=df["label_binary"],
    )
    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp["label_binary"],
    )

    train.to_csv(CRISIS_TRAIN_PATH, index=False)
    val.to_csv(CRISIS_VAL_PATH, index=False)
    test.to_csv(CRISIS_TEST_PATH, index=False)

    logger.success(
        f"Crisis splits saved | Train: {len(train):,} | "
        f"Val: {len(val):,} | Test: {len(test):,}"
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    # Defensive: dataset.py should already have made this, but if someone runs
    # clean.py first we don't want a confusing FileNotFoundError on write.
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DATA_DIR.exists() or not any(RAW_DATA_DIR.iterdir()):
        logger.error(
            f"{RAW_DATA_DIR} is missing or empty. "
            "Run `python dataset.py` first to download raw data."
        )
        sys.exit(1)

    logger.info("=== Cleaning & processing datasets ===")
    process_counsel_chat()
    process_crisis_data()
    logger.success("All processed datasets ready.")


if __name__ == "__main__":
    main()