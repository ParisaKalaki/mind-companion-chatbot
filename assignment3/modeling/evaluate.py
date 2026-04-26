"""
modelling/evaluate.py
---------------------
Evaluates the fine-tuned crisis classifier on the held-out test set.

Inputs:
    - models/crisis_classifier/         (trained weights)
    - data/processed/crisis_test.csv    (held-out split from clean.py)

Outputs:
    - reports/crisis_classifier_results.csv     (accuracy / F1 / P / R table)
    - reports/figures/confusion_matrix.png      (heatmap)

Run:
    python modelling/evaluate.py
"""

# ---------------------------------------------------------------------------
# Path shim — evaluate.py lives one folder deeper than config.py
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_PARENT_PKG = Path(__file__).resolve().parent.parent
if str(_PARENT_PKG) not in sys.path:
    sys.path.insert(0, str(_PARENT_PKG))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
MODEL_DIR    = MODELS_DIR / "crisis_classifier"
TEST_PATH    = PROCESSED_DATA_DIR / "crisis_test.csv"

CM_FIG_PATH  = FIGURES_DIR / "confusion_matrix.png"
RESULTS_PATH = REPORTS_DIR / "crisis_classifier_results.csv"

MAX_LENGTH   = 128
BATCH_SIZE   = 32
THRESHOLD    = 0.7
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CrisisDataset(Dataset):
    """Wraps tokenised crisis text + binary labels for DataLoader iteration."""

    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label":          self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate() -> dict:
    """Run end-to-end evaluation and return the metrics dict."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load data -----------------------------------------------------
    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test split not found at {TEST_PATH}. "
            "Run `python clean.py` first to generate the splits."
        )

    logger.info(f"Loading test data from {TEST_PATH} ...")
    df = pd.read_csv(TEST_PATH)[["text", "label_binary"]].dropna()
    logger.info(f"Test set: {len(df):,} rows")

    # --- Load model ----------------------------------------------------
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_DIR}. "
            "Run `python download_model.py` or `python modelling/train.py` first."
        )

    logger.info(f"Loading model from {MODEL_DIR} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(DEVICE).eval()

    # --- Tokenise & build loader ---------------------------------------
    logger.info("Tokenising ...")
    dataset = CrisisDataset(
        df["text"].tolist(),
        df["label_binary"].tolist(),
        tokenizer,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # --- Predict -------------------------------------------------------
    logger.info("Running predictions ...")
    all_preds, all_labels = [], []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
            probs = torch.softmax(logits, dim=1)
            preds = (probs[:, 1] >= THRESHOLD).int().cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(batch["label"].tolist())

            if i % 10 == 0:
                logger.info(f"  Batch {i}/{len(loader)} ...")

    # --- Metrics -------------------------------------------------------
    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec  = recall_score(all_labels, all_preds)
    cm   = confusion_matrix(all_labels, all_preds)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {acc:.4f} ({acc * 100:.2f}%)")
    print(f"F1 Score:  {f1:.4f} ({f1 * 100:.2f}%)")
    print(f"Precision: {prec:.4f} ({prec * 100:.2f}%)")
    print(f"Recall:    {rec:.4f} ({rec * 100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=["Non-Crisis", "Crisis"],
    ))

    # --- Confusion matrix plot -----------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Non-Crisis", "Crisis"],
        yticklabels=["Non-Crisis", "Crisis"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix — Crisis Classifier", fontsize=13)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(CM_FIG_PATH, dpi=150)
    plt.close(fig)
    logger.success(f"Confusion matrix saved to {CM_FIG_PATH}")

    # --- Save metrics CSV ---------------------------------------------
    results = pd.DataFrame({
        "Metric": ["Accuracy", "F1 Score", "Precision", "Recall"],
        "Score":  [acc, f1, prec, rec],
    })
    results.to_csv(RESULTS_PATH, index=False)
    logger.success(f"Results saved to {RESULTS_PATH}")

    return {
        "accuracy":  acc,
        "f1":        f1,
        "precision": prec,
        "recall":    rec,
    }


if __name__ == "__main__":
    evaluate()