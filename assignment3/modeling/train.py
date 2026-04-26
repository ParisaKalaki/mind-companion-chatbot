"""
modelling/train.py
------------------
Fine-tunes DistilBERT as a binary crisis classifier on the
`crisis_train.csv` / `crisis_val.csv` splits produced by clean.py.

Output: models/crisis_classifier/
            ├── config.json
            ├── model.safetensors
            ├── tokenizer.json
            ├── tokenizer_config.json
            └── training_args.bin

Run:
    python modelling/train.py
"""

# ---------------------------------------------------------------------------
# Path shim — train.py lives one folder deeper than config.py, so add the
# parent directory (where config.py lives) to sys.path before importing it.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_PARENT_PKG = Path(__file__).resolve().parent.parent
if str(_PARENT_PKG) not in sys.path:
    sys.path.insert(0, str(_PARENT_PKG))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from config import MODELS_DIR, PROCESSED_DATA_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_PATH = PROCESSED_DATA_DIR / "crisis_train.csv"
VAL_PATH   = PROCESSED_DATA_DIR / "crisis_val.csv"
MODEL_DIR  = MODELS_DIR / "crisis_classifier"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
MODEL_NAME  = "distilbert-base-uncased"
MAX_LENGTH  = 128
SEED        = 42

SAMPLE      = False       # True  = quick smoke run on 10k rows
                         # False = full dataset
SAMPLE_SIZE = 10_000
VAL_SAMPLE  = 2_000

NUM_EPOCHS       = 3
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE  = 32


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
def get_device() -> str:
    if torch.cuda.is_available():
        logger.info("Using CUDA GPU")
        return "cuda"
    if torch.backends.mps.is_available():
        logger.info("Using Apple Silicon MPS")
        return "mps"
    logger.warning("Using CPU — full-dataset training will be very slow. "
                   "Consider Colab/Kaggle for a free GPU.")
    return "cpu"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data():
    """Load train/val CSVs, optionally sample, and rename label column."""
    if not TRAIN_PATH.exists() or not VAL_PATH.exists():
        raise FileNotFoundError(
            f"Crisis splits not found in {PROCESSED_DATA_DIR}. "
            "Run `python clean.py` first."
        )

    train = pd.read_csv(TRAIN_PATH)[["text", "label_binary"]].dropna()
    val   = pd.read_csv(VAL_PATH)[["text", "label_binary"]].dropna()

    if SAMPLE:
        train = train.sample(n=min(SAMPLE_SIZE, len(train)), random_state=SEED)
        val   = val.sample(n=min(VAL_SAMPLE, len(val)),       random_state=SEED)
        logger.warning(
            f"SAMPLE MODE — Train: {len(train):,} | Val: {len(val):,}"
        )
    else:
        logger.info(f"FULL MODE — Train: {len(train):,} | Val: {len(val):,}")

    # HF Trainer expects the label column to be called 'labels'
    train = train.rename(columns={"label_binary": "labels"})
    val   = val.rename(columns={"label_binary": "labels"})
    return train, val


def tokenise(df: pd.DataFrame, tokenizer):
    """Convert a DataFrame into a tokenised HuggingFace Dataset."""
    dataset = Dataset.from_pandas(df, preserve_index=False)

    def tokenise_batch(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    return dataset.map(tokenise_batch, batched=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "f1":        f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall":    recall_score(labels, preds),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train() -> None:
    get_device()  # logs the choice; HF Trainer manages placement itself

    logger.info("Loading data ...")
    train_df, val_df = load_data()

    logger.info(f"Loading tokenizer for '{MODEL_NAME}' ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    logger.info("Tokenising ...")
    train_ds = tokenise(train_df, tokenizer)
    val_ds   = tokenise(val_df,   tokenizer)

    logger.info(f"Loading base model '{MODEL_NAME}' ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        seed=SEED,
        report_to="none",   # disable wandb / tensorboard auto-init
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    logger.info("Training started ...")
    trainer.train()

    logger.info(f"Saving final model to {MODEL_DIR} ...")
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    logger.success("Training complete.")


if __name__ == "__main__":
    train()