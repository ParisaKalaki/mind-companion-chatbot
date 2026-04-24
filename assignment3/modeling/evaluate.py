import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, Dataset

# ── Config ─────────────────────────────────────────────────────────────
MODEL_DIR  = Path('models/crisis_classifier')
TEST_PATH  = Path('data/processed/crisis_test.csv')
FIGURES    = Path('reports/figures')
MAX_LENGTH = 128
BATCH_SIZE = 32


# ── Dataset class ──────────────────────────────────────────────────────
class CrisisDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'label':          self.labels[idx]
        }


# ── Evaluate ───────────────────────────────────────────────────────────
def evaluate():
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Load test data
    print('Loading test data...')
    df = pd.read_csv(TEST_PATH)[['text', 'label_binary']].dropna()
    print(f'✅ Test set: {len(df):,} rows')

    # Load model
    print('Loading model...')
    tokenizer = DistilBertTokenizer.from_pretrained(str(MODEL_DIR))
    model     = DistilBertForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()

    # Tokenise
    print('Tokenising...')
    dataset = CrisisDataset(df['text'].tolist(), 
                             df['label_binary'].tolist(), 
                             tokenizer)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE)

    # Run predictions
    print('Running predictions...')
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            logits = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            ).logits
            probs  = torch.softmax(logits, dim=1)
            preds  = (probs[:, 1] >= 0.7).int().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch['label'].tolist())

            if i % 10 == 0:
                print(f'  Batch {i}/{len(loader)}...')



    # ── Metrics ────────────────────────────────────────────────────────
    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec  = recall_score(all_labels, all_preds)
    cm   = confusion_matrix(all_labels, all_preds)

    print('\n' + '=' * 50)
    print('EVALUATION RESULTS')
    print('=' * 50)
    print(f'Accuracy:  {acc:.4f} ({acc*100:.2f}%)')
    print(f'F1 Score:  {f1:.4f} ({f1*100:.2f}%)')
    print(f'Precision: {prec:.4f} ({prec*100:.2f}%)')
    print(f'Recall:    {rec:.4f} ({rec*100:.2f}%)')
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds,
                                 target_names=['Non-Crisis', 'Crisis']))

    # ── Confusion matrix plot ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Non-Crisis', 'Crisis'],
        yticklabels=['Non-Crisis', 'Crisis'],
        ax=ax
    )
    ax.set_title('Confusion Matrix — Crisis Classifier', fontsize=13)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(FIGURES / 'confusion_matrix.png', dpi=150)
    plt.show()
    print(f'\n✅ Confusion matrix saved to reports/figures/confusion_matrix.png')

    # ── Save results to CSV for report ────────────────────────────────
    results = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall'],
        'Score':  [acc, f1, prec, rec]
    })
    results.to_csv('reports/crisis_classifier_results.csv', index=False)
    print('✅ Results saved to reports/crisis_classifier_results.csv')

if __name__ == '__main__':
    evaluate()