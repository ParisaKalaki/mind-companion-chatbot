import re
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from pathlib import Path

def setup_dirs():
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    Path('reports/figures').mkdir(parents=True, exist_ok=True)

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

def load_counsel_chat():
    df = load_dataset('nbertagnolli/counsel-chat', split='train').to_pandas()
    df.to_csv('data/raw/counsel_chat_raw.csv', index=False)
    df['questionText'] = df['questionText'].apply(clean_text)
    df['answerText']   = df['answerText'].apply(clean_text)
    df = df[df['answerText'].str.len() >= 50]
    df = df.dropna(subset=['questionText', 'answerText'])
    df['document'] = 'Question: ' + df['questionText'] + '\nAnswer: ' + df['answerText']
    df[['questionText', 'answerText', 'topic', 'document']].to_csv(
        'data/processed/counsel_chat_clean.csv', index=False)
    print(f'✅ Counsel Chat: {len(df):,} rows')
    return df

def load_crisis_data():
    local_path = Path('data/raw/crisis_raw.csv')
    
    if not local_path.exists():
        raise FileNotFoundError(
            'crisis_raw.csv not found. Download it from Kaggle:\n'
            'https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch\n'
            'and place it in data/raw/crisis_raw.csv'
        )
    
    print('Loading crisis data from data/raw/crisis_raw.csv...')
    df = pd.read_csv(local_path)

    # Drop unnamed index column if present (Kaggle CSV artifact)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    df = df[['text', 'class']].copy()
    df.columns = ['text', 'label']
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.len() >= 20].dropna()
    df['label_binary'] = df['label'].map({'suicide': 1, 'non-suicide': 0})

    train, temp = train_test_split(df, test_size=0.30, random_state=42,
                                    stratify=df['label_binary'])
    val, test   = train_test_split(temp, test_size=0.50, random_state=42,
                                    stratify=temp['label_binary'])

    train.to_csv('data/processed/crisis_train.csv', index=False)
    val.to_csv(  'data/processed/crisis_val.csv',   index=False)
    test.to_csv( 'data/processed/crisis_test.csv',  index=False)
    print(f'✅ Crisis — Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}')
    return train, val, test

if __name__ == '__main__':
    setup_dirs()
    load_counsel_chat()
    load_crisis_data()