# Mind Companion — Mental Health Support Chatbot

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A retrieval-augmented mental health support chatbot built for AI Assignment 3.

The system combines a fine-tuned **DistilBERT crisis classifier** that routes urgent messages to safety resources, a **ChromaDB** vector store over the Counsel Chat dataset and a curated CBT/mindfulness/psychoeducation knowledge base, and **Google Gemini 2.5 Flash** for empathic, grounded replies. The frontend is a **Streamlit** chat UI with a dark theme and live tuning controls.

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
assignment3/
├── README.md                       <- This file
├── LICENSE
├── Makefile                        <- (template — optional)
├── pyproject.toml                  <- Poetry project & dependency definition
├── requirements.txt                <- Pinned deps (exported from Poetry)
├── setup.cfg                       <- flake8 config
├── .env                            <- API keys (NEVER commit)
├── .streamlit/
│   └── config.toml                 <- Locks dark theme + teal accents
│
├── data/
│   ├── external/
│   │   └── knowledge_base.json     <- Curated CBT / mindfulness / psychoed entries
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
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── training_args.bin
│
├── notebooks/
│   ├── 0_main.ipynb                <- End-to-end walk-through
│   └── 1_setup.ipynb               <- One-off setup cells (incl. Chroma index builds)
│
├── reports/
│   ├── crisis_classifier_results.csv
│   └── figures/
│       └── confusion_matrix.png
│
└── assignment3/                    <- Source package (importable)
    ├── __init__.py
    ├── config.py                   <- Project paths + .env loader
    ├── dataset.py                  <- Folder setup + Kaggle download
    ├── clean.py                    <- Text cleaning + train/val/test splits
    ├── download_model.py           <- Pulls trained classifier from Google Drive
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
git clone <your-repo-url>
cd assignment3
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

### 3. Get the data and the model

```bash
poetry run python dataset.py          # downloads suicide-watch dataset from Kaggle
poetry run python clean.py            # cleans and splits the data
poetry run python download_model.py   # downloads pre-trained classifier from Google Drive
```

> If you'd rather train the classifier yourself, run `poetry run python modeling/train.py` instead. Takes ~10 min on GPU, longer on CPU.

### 4. Build the vector index

The Chroma index is built from notebook cells (one-time setup). Open `notebooks/1_setup.ipynb` and run the cells in order. This creates `models/chroma_db/` with two collections (`counsel_chat`, `knowledge_base`).

### 5. Run the app

```bash
poetry run streamlit run app.py
```

Opens at http://localhost:8501. First message takes ~10 seconds (model warmup); subsequent messages are ~1-2 seconds.

---

## Live tuning controls

The Streamlit sidebar exposes three sliders that affect retrieval and routing in real time:

- **Counsel Chat passages retrieved** (0–6, default 3) — empathic context from real counselor conversations
- **Knowledge base entries retrieved** (0–5, default 2) — curated CBT / mindfulness / psychoed
- **Crisis classifier threshold** (0.1–0.9, default 0.5) — lower = more sensitive routing to safety pathway

The sidebar also accepts a user-pasted Gemini API key for per-session use without modifying `.env`.

---

## File-by-file reference

| File | Role |
|---|---|
| `config.py` | Defines `PROJ_ROOT`, `DATA_DIR`, `MODELS_DIR`, etc. Loads `.env`. |
| `dataset.py` | Validates `.env`, creates the data folder tree, downloads the Kaggle suicide-watch dataset. |
| `clean.py` | Pulls Counsel Chat from HuggingFace, cleans both datasets, builds train/val/test splits. |
| `download_model.py` | Pulls fine-tuned DistilBERT artifacts from Google Drive into `models/crisis_classifier/`. |
| `modeling/train.py` | Fine-tunes DistilBERT on `crisis_train.csv` with `AutoTokenizer` / `AutoModel`. |
| `modeling/evaluate.py` | Test-set metrics + confusion matrix → `reports/`. |
| `modeling/predict.py` | Standalone `is_crisis()` for ad-hoc inference. |
| `rag.py` | The pipeline: `respond(msg, history)` orchestrates cleaning, classification, retrieval, and generation. Lazy-loads all heavy components on first call. |
| `app.py` | Streamlit chat UI. Imports `rag.respond`. Handles UI state, sidebar controls, and Gemini API key resolution. |

---

## Deployment to Streamlit Cloud

1. Push the repo to GitHub. Confirm `.env` is in `.gitignore`.
2. Create a new app on https://share.streamlit.io pointing at `app.py`.
3. In the app's **Secrets** page, add:
   ```
   GEMINI_API_KEY = "your_fallback_key_here"
   ```
4. Make sure `requirements.txt` and `.streamlit/config.toml` are in the repo root.
5. The Chroma DB and DistilBERT weights are large binary artifacts. Either commit them via Git LFS or have the app download them on first boot.

---

## Architecture notes

**Why ChromaDB?** Persists to disk in one line, no server, clean Python API, and bundled HNSW indexing. FAISS is faster but bare-bones; not worth the extra plumbing for this scale.

**Why embed only the question (not Q+A)?** User messages look like questions, not therapist answers. Embedding only the `questionText` and storing all therapist responses in metadata gives much higher retrieval relevance. Counsel Chat has multiple therapist responses per question, so we group by `questionText` first — 2,608 raw rows collapse to 863 unique questions with ~3 answers each.

**Why two collections, not one?** Counsel Chat (~863 items) and the curated knowledge base (~30 items) play different roles. Counsel Chat gives "what other people have said about similar problems"; the KB gives "structured techniques." A bucketed retrieve (3 from counsel + 2 from KB) gives the LLM both kinds of context. A unified collection with similarity-sort would let the longer counsel chat passages drown out the shorter KB entries.

**Why disable Gemini "thinking"?** `gemini-2.5-flash` has thinking enabled by default and thinking tokens count against `max_output_tokens`. With our 600-token cap, the model would burn 400+ tokens reasoning internally and truncate the visible reply mid-sentence. `thinking_config=ThinkingConfig(thinking_budget=0)` is the standard fix for chat-style use.

**Why crisis_threshold = 0.5 instead of 0.7?** For a mental-health context, missing a real crisis is worse than over-flagging. Lowering the default threshold makes the classifier more sensitive at the cost of slightly more false positives — but a false positive just shows the user a list of helplines, while a false negative means missing someone in danger.

**Why hardcode the crisis resources block?** The LLM writes the empathic acknowledgement (which adapts to what the user said), but the helpline numbers are concatenated verbatim from a constant. Phone numbers must never be paraphrased, hallucinated, or "improved" by the LLM. Wrong number = real harm.

---

## Generating the requirements file (Poetry)

This project uses Poetry for dependency management. To export the locked dependencies into a `requirements.txt` that Streamlit Cloud and other deployment targets can use:

```bash
# install the export plugin once
poetry self add poetry-plugin-export

# export to requirements.txt (without dev dependencies, without hashes)
poetry export -f requirements.txt --output requirements.txt --without-hashes

# include dev dependencies as well
poetry export -f requirements.txt --output requirements.txt --without-hashes --with dev
```

To install from the lock file:

```bash
poetry install
```

To add a new dependency:

```bash
poetry add <package>           # runtime dep
poetry add --group dev <package>   # dev-only dep
```

To regenerate the lock file after editing `pyproject.toml`:

```bash
poetry lock
```

---

## Acknowledgements

- **Counsel Chat dataset** by [nbertagnolli](https://huggingface.co/datasets/nbertagnolli/counsel-chat) on HuggingFace
- **Suicide and Depression Detection** dataset by [Nikhileswar Komati](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch) on Kaggle
- Crisis support resources (Australia): Lifeline, Beyond Blue, Suicide Call Back Service, 13YARN, Kids Helpline