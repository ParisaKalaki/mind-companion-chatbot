# Setup Guide — MindBridge AI

_Written by Parisa — last updated April 2026_

---

## What's been built so far

- ✅ Data pipeline — loads and cleans both datasets
- ✅ Crisis classifier — DistilBERT fine-tuned, 97% accuracy
- 🔨 RAG pipeline — in progress
- ⏳ LLM integration — not started
- ⏳ App/demo UI — not started

---

## Step 1 — Clone and setup

```bash
git clone https://github.com/Doombuoyz/assignment3.git
cd assignment3
git checkout feature/crisis-classifier
poetry install
```

---

## Step 2 — Get the crisis dataset

Download from Kaggle:
https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch

Place the file here:
data/raw/crisis_raw.csv

---

## Step 3 — Generate processed data

```bash
poetry run python assignment3/dataset.py 
```

This creates:
data/processed/counsel_chat_clean.csv ← used for RAG
data/processed/crisis_train.csv
data/processed/crisis_val.csv
data/processed/crisis_test.csv

---

## Step 4 — Get the trained model

Download the `crisis_classifier` folder from Google Drive:
https://drive.google.com/drive/folders/19kEbdl_d39XfjqHA5Z0jYZRA7t2PlEZL?usp=sharing

Place it here:

```bash
models/crisis_classifier/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── training_args.bin
```

---

## Step 5 — Test the classifier works

```bash
poetry run python assignment3/modeling/predict.py
```

Expected output:

✅ [NON-CRISIS] I have been feeling really anxious lately...

🚨 [CRISIS] I want to kill myself, I can't take this anymore

✅ [NON-CRISIS] Can you help me with some breathing exercises?

✅ [NON-CRISIS] I've been having dark thoughts...

✅ [NON-CRISIS] I'm struggling with my relationship...

---

## Project structure

```bash
── assignment3
│ ├── **init**.py
│ ├── dataset.py
│ └── modeling
│ ├── **init**.py
│ ├── evaluate.py
│ ├── predict.py
│ └── train.py
├── data
│ ├── processed
│ │ ├── counsel_chat_clean.csv
│ │ ├── crisis_test.csv
│ │ ├── crisis_train.csv
│ │ └── crisis_val.csv
│ └── raw
│ ├── counsel_chat_raw.csv
│ └── crisis_raw.csv
├── docs
│ ├── docs
│ │ ├── getting-started.md
│ │ └── index.md
│ ├── mkdocs.yml
│ └── README.md
├── LICENSE
├── Makefile
├── models
│ ├── crisis_classifier
│ │ ├── config.json
│ │ ├── model.safetensors
│ │ ├── tokenizer_config.json
│ │ ├── tokenizer.json
│ │ └── training_args.bin
│ └── predict.py
├── notebooks
│ ├── 0.1-data-exploration.ipynb
│ ├── 0.2-crisis-classifier.ipynb
│ └── exp.ipynb
├── pipeline.py
├── poetry.lock
├── pyproject.toml
├── README.md
├── references
├── reports
│ ├── crisis_classifier_results.csv
│ └── figures
│ └── confusion_matrix.png
├── requirements.txt
├── setup.cfg
├── SETUP.md
└── Tasks.md
```

---

## Branch structure

master ← stable, do not push directly

feature/data-pipeline ← Parisa — done ✅

feature/crisis-classifier ← Parisa — done ✅

feature/rag-pipeline ← your job 🔨

feature/llm-integration ← not started ⏳

feature/app-and-demo ← not started ⏳

---

## For RAG pipeline — your job

Create your branch:

```bash
git checkout -b feature/rag-pipeline
```

Install dependencies:

```bash
poetry add chromadb sentence-transformers
```

Build these files:

```bash
assignment3/
└── rag/
├── init.py
├── build_index.py ← index counsel_chat_clean.csv into ChromaDB
├── embedder.py ← convert text to vectors
└── retriever.py ← search ChromaDB for relevant chunks
```

    Input file: `data/processed/counsel_chat_clean.csv`

Key column: `document` (Question + Answer combined)

Message Parisa if you get stuck.

## What's left (in order)

### Step 1 — RAG Pipeline (`feature/rag-pipeline`)

- `build_index.py` → Index `counsel_chat_clean.csv` into ChromaDB
- `embedder.py` → Convert user message into a vector
- `retriever.py` → Search ChromaDB and return top 3 relevant chunks

---

### Step 2 — LLM Integration (`feature/llm-integration`)

- `generator.py` → Send user message + RAG chunks to Groq API
  - Returns an empathetic response

---

### Step 3 — Connect Everything (`pipeline.py`)

User message
↓
Crisis classifier → if crisis → return Lifeline number
↓ (if not crisis)
RAG retriever → get relevant chunks
↓
LLM generator → generate response using chunks
↓
Return response to user

---

### Step 4 — App / Demo UI (`feature/app-and-demo`)

- `app.py` → Simple Gradio chat interface for the demo

---

### Step 5 — Evaluation (`feature/evaluation`)

- Evaluate RAG quality → Are retrieved chunks relevant?
- Evaluate full pipeline → End-to-end response quality
- Generate all figures → For the report

---

### Step 6 — Report + Slides

- Write report sections
- Build PowerPoint slides
- Record demo video
