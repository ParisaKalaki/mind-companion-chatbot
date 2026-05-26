# Mind Companion — Mental Health Support Chatbot

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A retrieval-augmented mental health support chatbot built for AI Assignment 3.

The system combines a fine-tuned **DistilBERT crisis classifier** that routes urgent messages to safety resources, a **ChromaDB** vector store over the Counsel Chat dataset and a curated CBT/mindfulness/psychoeducation knowledge base, and **Google Gemini 2.5 Flash** for empathic, grounded replies. The frontend is a **Streamlit** chat UI with a dark theme, live tuning controls, and a three-tier API key resolution system (user-paste → Streamlit secrets → `.env`).

> **Disclaimer.** This is a student project. It is not a substitute for professional mental health care. If you or someone you know is in crisis, contact your local emergency services or a crisis helpline. Australian resources are surfaced in the app sidebar.

---

## How it works

```
user message
     │
     ▼
clean_text()  ─────────────────────────────────┐
     │                                          │
     ▼                                          │
DistilBERT crisis classifier                    │
     │                                          │
     ├── crisis ──► empathic ack (Gemini)       │
     │              + hardcoded resources       │
     │              + log to crisis_log.jsonl   │
     │                                          │
     └── non-crisis                             │
            │                                   │
            ▼                                   │
       Chroma retrieval                         │
            │  (counsel_chat + knowledge_base)  │
            ▼                                   │
       prompt assembly  ◄───── conversation history
            │
            ▼
       Gemini 2.5 Flash
            │
            ▼
       reply + cited sources
```

Everything routes through a single `respond(message, history)` entry point in `rag.py`. The Streamlit UI is a thin presentation layer; all business logic lives in the pipeline module.

---

## Project Organization

```
mind_companion/
├── README.md                       <- This file
├── TEAM_BRIEFING.md                <- In-depth design decisions for teammates / report
├── LICENSE
├── Makefile                        <- (template — optional)
├── pyproject.toml                  <- Poetry project & dependency definition
├── poetry.lock                     <- Resolved dependency tree (Poetry)
├── requirements.txt                <- Pinned deps (exported from Poetry)
├── setup.cfg                       <- flake8 config
├── .env                            <- API keys (NEVER commit)
│
├── data/
│   ├── external/
│   │   └── knowledge_base.json     <- Curated CBT / mindfulness / psychoed
│   │                                  (auto-downloaded from Google Drive
│   │                                  by build_index.py if missing)
│   ├── interim/
│   ├── processed/
│   │   ├── counsel_chat_clean.csv  <- Cleaned RAG corpus
│   │   ├── crisis_train.csv        <- Classifier training split (~70%)
│   │   ├── crisis_val.csv          <- Validation (~15%)
│   │   ├── crisis_test.csv         <- Held-out test (~15%)
│   │   └── crisis_log.jsonl        <- Append-only log of crisis-flagged events
│   └── raw/
│       ├── counsel_chat_raw.csv    <- HuggingFace snapshot (downloaded by clean.py)
│       └── crisis_raw.csv          <- Kaggle snapshot (downloaded by dataset.py)
│
├── docs/                           <- mkdocs scaffolding
│
├── models/
│   ├── chroma_db/                  <- Persistent ChromaDB (counsel_chat + knowledge_base)
│   └── crisis_classifier/          <- Fine-tuned DistilBERT weights
│       ├── config.json             <- (auto-downloaded from Google Drive
│       ├── model.safetensors          by download_model.py if missing)
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── training_args.bin
│
├── notebooks/
│   ├── 1_setup.ipynb               <- One-shot setup notebook (runs every step in order)
│   └── 2_main.ipynb                <- End-to-end walk-through + interactive examples
│
├── references/                     <- (empty placeholder)
│
├── reports/
│   ├── crisis_classifier_results.csv
│   └── figures/
│       ├── confusion_matrix.png    <- Crisis classifier eval
│       ├── fig1_class_distribution.png
│       ├── fig2_text_length.png
│       ├── fig3_wordcloud_crisis.png
│       ├── fig4_wordcloud_non_crisis.png
│       ├── fig5_counsel_topics.png
│       ├── fig6_answer_length.png
│       └── fig7_knowledge_base.png
│
└── mind_companion/                    <- Source package (importable)
    ├── __init__.py
    ├── .streamlit/
    │   └── config.toml             <- Locks dark theme + teal accents
    ├── config.py                   <- Project paths + .env loader
    ├── dataset.py                  <- Folder setup + Kaggle download
    ├── clean.py                    <- Text cleaning + train/val/test splits
    ├── eda.py                      <- Exploratory data analysis figures (7 PNGs → reports/figures/)
    ├── download_model.py           <- Pulls trained classifier from Google Drive
    ├── build_index.py              <- Builds the Chroma vector index (both collections)
    ├── rag.py                      <- The pipeline: clean → classify → route → retrieve → generate → log
    ├── app.py                      <- Streamlit UI
    └── modeling/
        ├── __init__.py
        ├── train.py                <- Fine-tune DistilBERT on crisis_train.csv
        ├── predict.py              <- Standalone classifier inference (rag.py inlines this)
        └── evaluate.py             <- Test-set metrics + confusion matrix
```

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/Doombuoyz/mind_companion
cd mind_companion
poetry install         # or: pip install -r requirements.txt
```

### 2. Configure secrets

Create a `.env` file in the project root:

```
KAGGLE_USERNAME=<your_kaggle_username>
KAGGLE_KEY=<your_kaggle_api_key>
GEMINI_API_KEY=<your_gemini_api_key>
```

- **Kaggle credentials** — get them at https://www.kaggle.com/settings → "API" → "Create New Token"
- **Gemini key** — free key at https://aistudio.google.com/apikey

`.env` should be in `.gitignore`. Never commit it.

### 3. One-shot setup (recommended)

The simplest path: open `notebooks/1_setup.ipynb` and run all cells top-to-bottom. The notebook walks through dataset download, cleaning, classifier download, evaluation, vector index build, and a smoke test of `rag.py`.

### 3b. (alternative) Run each step from the CLI

If you'd rather skip the notebook, every step is also a standalone script:

```bash
poetry run python mind_companion/dataset.py         # downloads suicide-watch dataset from Kaggle
poetry run python mind_companion/clean.py           # cleans data + builds train/val/test splits
poetry run python mind_companion/download_model.py  # downloads pre-trained classifier from Google Drive
poetry run python mind_companion/build_index.py     # builds the Chroma vector index
                                                  #   (auto-downloads knowledge_base.json
                                                  #    from Drive if not present locally)
```

> If you'd rather train the classifier yourself, run `poetry run python mind_companion/modeling/train.py` instead of `download_model.py`. Takes ~10 min on GPU, longer on CPU.

`build_index.py` is idempotent — re-runs are safe and cheap. Useful flags:

```bash
poetry run python mind_companion/build_index.py --force      # wipe and rebuild both collections
poetry run python mind_companion/build_index.py --counsel    # only rebuild counsel_chat
poetry run python mind_companion/build_index.py --kb         # only rebuild knowledge_base
```

<!-- README for a minimal cleaned project -->

# Mind Companion

Mind Companion is an educational retrieval-augmented chatbot prototype for mental-health support. It wires a lightweight crisis classifier with a retrieval pipeline and an LLM-backed responder to produce grounded, empathetic replies.

Important: this is a student project and not a substitute for professional care. For emergencies, contact local emergency services or your national crisis hotline.

Quick links

- Source package: `mind_companion/`
- Notebooks: `notebooks/`
- Data: `data/` (raw and processed)

Features

- Crisis classification (binary routing)
- Retrieval from a counseling dataset + curated knowledge base
- Streamlit demo UI (`mind_companion/app.py`)

Requirements

- Python 3.11+
- See `pyproject.toml` / `requirements.txt` for exact dependencies

Setup

1. Install dependencies:

```bash
poetry install
# or: pip install -r requirements.txt
```

2. Create a `.env` file at the repo root with any API keys you need (examples):

```env
KAGGLE_USERNAME=...
KAGGLE_KEY=...
GEMINI_API_KEY=...
```

Running the pipeline (typical)

```bash
# download dataset (if needed)
poetry run python mind_companion/dataset.py
# clean & prepare splits
poetry run python mind_companion/clean.py
# download pretrained classifier weights (or train your own)
poetry run python mind_companion/download_model.py
# build vector index
poetry run python mind_companion/build_index.py
```

Run the demo UI

```bash
poetry run streamlit run mind_companion/app.py
# then open http://localhost:8501
```

Project layout (trimmed)

```
.
├── data/
├── mind_companion/
│   ├── app.py
+   ├── rag.py
+   ├── dataset.py
+   └── modeling/
├── notebooks/
├── reports/
├── pyproject.toml
└── requirements.txt
```

Notes & maintenance

- Some large/generated artifacts (model weights, Chroma DB) are intentionally ignored by git. If you removed them, re-run the corresponding script (e.g. `download_model.py` or `build_index.py`) to recreate them.
- I cleaned and renamed the package from `assignment3` → `mind_companion`. Search the repo for `mind_companion` to confirm.

Contributing

- This repo is a personal/educational project. If you want to contribute, open an issue or submit a PR with a concise description of your change.

Contact

- parisa.kalaki@example.com
