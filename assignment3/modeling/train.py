import pandas as pd
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch

# ── Config ─────────────────────────────────────────────────────────────
TRAIN_PATH  = Path('data/processed/crisis_train.csv')
VAL_PATH    = Path('data/processed/crisis_val.csv')
MODEL_DIR   = Path('models/crisis_classifier')
MODEL_NAME  = 'distilbert-base-uncased'
SAMPLE      = True   # ← True = 10k rows for testing, False = full dataset
SAMPLE_SIZE = 10_000

# ── Device detection ───────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        print('✅ Using CUDA GPU')
        return 'cuda'
    elif torch.backends.mps.is_available():
        print('✅ Using Apple M2 MPS')
        return 'mps'
    else:
        print('⚠️  Using CPU — consider Colab for faster training')
        return 'cpu'

# ── Load data ──────────────────────────────────────────────────────────
def load_data():
    train = pd.read_csv(TRAIN_PATH)[['text', 'label_binary']].dropna()
    val   = pd.read_csv(VAL_PATH)[['text', 'label_binary']].dropna()

    if SAMPLE:
        train = train.sample(n=SAMPLE_SIZE, random_state=42)
        val   = val.sample(n=2_000, random_state=42)
        print(f'⚠️  SAMPLE MODE — Train: {len(train):,} | Val: {len(val):,}')
    else:
        print(f'✅ FULL MODE — Train: {len(train):,} | Val: {len(val):,}')

    train = train.rename(columns={'label_binary': 'labels'})
    val   = val.rename(columns={'label_binary': 'labels'})
    return train, val

# ── Tokenise ───────────────────────────────────────────────────────────
def tokenise(df, tokenizer):
    dataset = Dataset.from_pandas(df, preserve_index=False)
    def tokenise_batch(batch):
        return tokenizer(
            batch['text'],
            padding='max_length',
            truncation=True,
            max_length=128
        )
    return dataset.map(tokenise_batch, batched=True)

# ── Metrics ────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        'accuracy':  accuracy_score(labels, preds),
        'f1':        f1_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall':    recall_score(labels, preds),
    }

# ── Train ──────────────────────────────────────────────────────────────
def train():
    device = get_device()

    print('Loading data...')
    train_df, val_df = load_data()

    print('Loading tokenizer...')
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    print('Tokenising...')
    train_dataset = tokenise(train_df, tokenizer)
    val_dataset   = tokenise(val_df,   tokenizer)

    print('Loading model...')
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        logging_steps=50,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print('Training started...')
    trainer.train()

    print(f'Saving model to {MODEL_DIR}...')
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print('✅ Training complete')

if __name__ == '__main__':
    train()