"""
build_index.py
--------------
Builds the ChromaDB vector store from on-disk data sources. Replaces the
notebook cells that originally did the same job, so the index can be
rebuilt reproducibly from the command line.

Two collections in one persistent DB at MODELS_DIR / "chroma_db":

    - counsel_chat     ~863 unique questions from data/processed/counsel_chat_clean.csv
                        (grouped by questionText, all answers concatenated in metadata)

    - knowledge_base   ~30 curated entries from data/external/knowledge_base.json
                        (CBT / mindfulness / psychoeducation)

Both use sentence-transformers `all-MiniLM-L6-v2` with cosine similarity.

Run:
    python build_index.py            # build both, skip if already exists
    python build_index.py --force    # wipe both collections and rebuild
    python build_index.py --counsel  # only rebuild the counsel_chat collection
    python build_index.py --kb       # only rebuild the knowledge_base collection

Prerequisites:
    - data/processed/counsel_chat_clean.csv must exist (run clean.py first).
    - data/external/knowledge_base.json — auto-downloaded from Google Drive
      if missing (see KB_DRIVE_FILE_ID below).
"""

import argparse
import json
import sys

import chromadb
import gdown
import pandas as pd
from chromadb.config import Settings
from loguru import logger
from sentence_transformers import SentenceTransformer

from config import EXTERNAL_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

CHROMA_DIR         = MODELS_DIR / "chroma_db"
COUNSEL_CLEAN_PATH = PROCESSED_DATA_DIR / "counsel_chat_clean.csv"
KB_JSON_PATH       = EXTERNAL_DATA_DIR / "knowledge_base.json"

# Google Drive file ID for the curated knowledge_base.json.
# If the local file is missing, _ensure_kb_json() downloads from here.
KB_DRIVE_FILE_ID   = "17iipXI5osVQq-g060qQJinPWVF9fSVG0"

COUNSEL_COLLECTION = "counsel_chat"
KB_COLLECTION      = "knowledge_base"

BATCH_SIZE         = 64
ANSWER_SEPARATOR   = "\n\n---\n\n"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _make_client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def _collection_exists(client, name: str) -> bool:
    return name in [c.name for c in client.list_collections()]


def _load_embed_model() -> SentenceTransformer:
    logger.info(f"Loading embedding model '{EMBED_MODEL_NAME}' ...")
    return SentenceTransformer(EMBED_MODEL_NAME)


def _ensure_kb_json() -> None:
    """
    Make sure data/external/knowledge_base.json exists locally.
    If it doesn't, download it from Google Drive.

    The KB is hand-curated content (CBT / mindfulness / psychoed entries).
    Hosting it on Drive lets a fresh clone fetch it without committing
    the file to the repo.
    """
    if KB_JSON_PATH.exists():
        return

    KB_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"{KB_JSON_PATH.name} not found locally; "
        "downloading from Google Drive ..."
    )
    try:
        result = gdown.download(
            id=KB_DRIVE_FILE_ID,
            output=str(KB_JSON_PATH),
            quiet=False,
        )
        if result is None:
            raise RuntimeError(
                "gdown returned no path. Check that the Drive file is shared "
                "as 'Anyone with the link' and that the file ID is correct."
            )
    except Exception as exc:
        # Clean up a partial / empty file so reruns don't think it succeeded
        if KB_JSON_PATH.exists() and KB_JSON_PATH.stat().st_size == 0:
            KB_JSON_PATH.unlink()
        raise RuntimeError(
            f"Failed to download knowledge_base.json from Google Drive: {exc}"
        ) from exc

    logger.success(f"Downloaded knowledge_base.json to {KB_JSON_PATH}")


# ---------------------------------------------------------------------------
# Counsel Chat
# ---------------------------------------------------------------------------
def build_counsel_collection(client, embed_model, force: bool = False) -> None:
    if not COUNSEL_CLEAN_PATH.exists():
        raise FileNotFoundError(
            f"{COUNSEL_CLEAN_PATH} not found. Run `python clean.py` first."
        )

    if _collection_exists(client, COUNSEL_COLLECTION) and not force:
        coll = client.get_collection(COUNSEL_COLLECTION)
        logger.info(
            f"Collection '{COUNSEL_COLLECTION}' already exists with "
            f"{coll.count():,} items — skipping. Use --force to rebuild."
        )
        return

    if _collection_exists(client, COUNSEL_COLLECTION):
        logger.warning(f"Deleting existing '{COUNSEL_COLLECTION}' collection ...")
        client.delete_collection(COUNSEL_COLLECTION)

    # --- Load + group --------------------------------------------------------
    logger.info(f"Loading {COUNSEL_CLEAN_PATH} ...")
    df = pd.read_csv(COUNSEL_CLEAN_PATH)
    df = df.dropna(subset=["questionText", "answerText"]).reset_index(drop=True)
    logger.info(f"Loaded {len(df):,} raw Q&A rows")

    grouped = (
        df.groupby("questionText", as_index=False)
          .agg(
              answers=("answerText", lambda s: ANSWER_SEPARATOR.join(s.astype(str))),
              n_answers=("answerText", "count"),
              topic=("topic", lambda s: s.mode().iloc[0]
                                          if not s.mode().empty else "unknown"),
          )
    )
    logger.info(
        f"After grouping: {len(grouped):,} unique questions "
        f"(avg {grouped['n_answers'].mean():.1f} answers each)"
    )

    # --- Create collection ---------------------------------------------------
    coll = client.create_collection(
        name=COUNSEL_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # --- Embed ---------------------------------------------------------------
    questions = grouped["questionText"].tolist()
    logger.info(f"Embedding {len(questions):,} unique questions ...")
    embeddings = embed_model.encode(
        questions,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # --- Build metadata ------------------------------------------------------
    ids = [f"counsel_{i}" for i in range(len(grouped))]
    documents = questions
    metadatas = [
        {
            "source":       "counsel_chat",
            "topic":        str(row["topic"]),
            "questionText": str(row["questionText"]),
            "answerText":   str(row["answers"]),
            "n_answers":    int(row["n_answers"]),
        }
        for _, row in grouped.iterrows()
    ]

    # --- Insert in batches --------------------------------------------------
    logger.info(f"Inserting into Chroma in batches of {BATCH_SIZE} ...")
    for start in range(0, len(ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        coll.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=metadatas[start:end],
        )

    logger.success(
        f"Indexed {coll.count():,} unique questions into '{COUNSEL_COLLECTION}'."
    )


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
def _kb_to_search_text(entry: dict) -> str:
    """Compose the embedded text per KB entry: title + description + tags."""
    tags = ", ".join(entry.get("tags", []))
    return f"{entry['title']}. {entry['description']} Tags: {tags}"


def build_kb_collection(client, embed_model, force: bool = False) -> None:
    # Auto-fetch the JSON from Google Drive if not on disk
    _ensure_kb_json()

    if not KB_JSON_PATH.exists():
        raise FileNotFoundError(
            f"{KB_JSON_PATH} not found and could not be downloaded."
        )

    if _collection_exists(client, KB_COLLECTION) and not force:
        coll = client.get_collection(KB_COLLECTION)
        logger.info(
            f"Collection '{KB_COLLECTION}' already exists with "
            f"{coll.count()} items — skipping. Use --force to rebuild."
        )
        return

    if _collection_exists(client, KB_COLLECTION):
        logger.warning(f"Deleting existing '{KB_COLLECTION}' collection ...")
        client.delete_collection(KB_COLLECTION)

    # --- Load JSON -----------------------------------------------------------
    logger.info(f"Loading {KB_JSON_PATH} ...")
    with open(KB_JSON_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)
    logger.info(f"Loaded {len(entries)} entries from JSON")

    counts = {}
    for e in entries:
        counts[e["category"]] = counts.get(e["category"], 0) + 1
    for cat, n in counts.items():
        logger.info(f"  {cat:18s} {n}")

    # --- Create collection ---------------------------------------------------
    coll = client.create_collection(
        name=KB_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # --- Embed --------------------------------------------------------------
    search_texts = [_kb_to_search_text(e) for e in entries]
    logger.info(f"Embedding {len(search_texts)} entries ...")
    embeddings = embed_model.encode(
        search_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # --- Build metadata -----------------------------------------------------
    ids = [e["id"] for e in entries]
    documents = search_texts
    metadatas = [
        {
            "source":           "knowledge_base",
            "category":         e["category"],
            "title":            e["title"],
            "tags":             ", ".join(e.get("tags", [])),
            "description":      e["description"],
            "how_to":           e["how_to"],
            "when_useful":      e["when_useful"],
            "citation":         e["source"],
            "disclaimer_level": e["disclaimer_level"],
        }
        for e in entries
    ]

    # --- Insert in batches --------------------------------------------------
    logger.info(f"Inserting into Chroma in batches of {BATCH_SIZE} ...")
    for start in range(0, len(ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        coll.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=metadatas[start:end],
        )

    logger.success(
        f"Indexed {coll.count()} entries into '{KB_COLLECTION}'."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the ChromaDB vector index for Mind Companion."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Wipe and rebuild collections even if they already exist.",
    )
    parser.add_argument(
        "--counsel", action="store_true",
        help="Build only the counsel_chat collection.",
    )
    parser.add_argument(
        "--kb", action="store_true",
        help="Build only the knowledge_base collection.",
    )
    args = parser.parse_args()

    # If neither --counsel nor --kb is given, build both.
    build_counsel = args.counsel or not args.kb
    build_kb      = args.kb      or not args.counsel

    logger.info(f"Chroma DB directory: {CHROMA_DIR}")
    logger.info(f"Embedding model:     {EMBED_MODEL_NAME}")

    client = _make_client()
    embed_model = _load_embed_model()

    if build_counsel:
        logger.info("=== Building counsel_chat collection ===")
        build_counsel_collection(client, embed_model, force=args.force)

    if build_kb:
        logger.info("=== Building knowledge_base collection ===")
        build_kb_collection(client, embed_model, force=args.force)

    # --- Final summary ------------------------------------------------------
    logger.info("Final state of the Chroma DB:")
    for c in client.list_collections():
        info = client.get_collection(c.name)
        logger.info(f"  - {c.name:20s} {info.count():,} items")
    logger.success(f"Index ready at {CHROMA_DIR}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)