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
assignment3/
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
└── assignment3/                    <- Source package (importable)
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
git clone https://github.com/Doombuoyz/assignment3
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

### 3. One-shot setup (recommended)

The simplest path: open `notebooks/1_setup.ipynb` and run all cells top-to-bottom. The notebook walks through dataset download, cleaning, classifier download, evaluation, vector index build, and a smoke test of `rag.py`.

### 3b. (alternative) Run each step from the CLI

If you'd rather skip the notebook, every step is also a standalone script:

```bash
poetry run python assignment3/dataset.py         # downloads suicide-watch dataset from Kaggle
poetry run python assignment3/clean.py           # cleans data + builds train/val/test splits
poetry run python assignment3/download_model.py  # downloads pre-trained classifier from Google Drive
poetry run python assignment3/build_index.py     # builds the Chroma vector index
                                                  #   (auto-downloads knowledge_base.json
                                                  #    from Drive if not present locally)
```

> If you'd rather train the classifier yourself, run `poetry run python assignment3/modeling/train.py` instead of `download_model.py`. Takes ~10 min on GPU, longer on CPU.

`build_index.py` is idempotent — re-runs are safe and cheap. Useful flags:

```bash
poetry run python assignment3/build_index.py --force      # wipe and rebuild both collections
poetry run python assignment3/build_index.py --counsel    # only rebuild counsel_chat
poetry run python assignment3/build_index.py --kb         # only rebuild knowledge_base
```

To regenerate the EDA figures used in the report:

```bash
poetry run python assignment3/eda.py             # writes 7 PNGs to reports/figures/
```

### 4. Run the app

```bash
poetry run streamlit run assignment3/app.py
```

Opens at http://localhost:8501. First message takes ~10 seconds (model warmup); subsequent messages are ~1-2 seconds.

---

## Live tuning controls

The Streamlit sidebar exposes three sliders that affect retrieval and routing in real time:

- **Counsel Chat passages retrieved** (0–6, default 3) — empathic context from real counselor conversations
- **Knowledge base entries retrieved** (0–5, default 2) — curated CBT / mindfulness / psychoed
- **Crisis classifier threshold** (0.1–0.9, default 0.5) — lower = more sensitive routing to safety pathway

The sidebar also accepts a user-pasted Gemini API key (with a Submit / Clear form) for per-session use without modifying `.env`. The app falls back gracefully through three tiers: user-pasted key → `st.secrets` → `.env`. When the shared key hits its quota, the app auto-opens the sidebar and prompts the user to paste their own key.

---

## File-by-file reference

| File | Role |
|---|---|
| `config.py` | Defines `PROJ_ROOT`, `DATA_DIR`, `MODELS_DIR`, etc. Loads `.env`. |
| `dataset.py` | Validates `.env`, creates the data folder tree, downloads the Kaggle suicide-watch dataset. |
| `clean.py` | Pulls Counsel Chat from HuggingFace, cleans both datasets, builds train/val/test splits. |
| `eda.py` | Generates 7 EDA figures (class distribution, text-length distributions, word clouds, topic frequencies, answer-length histogram, KB category breakdown) into `reports/figures/`. |
| `download_model.py` | Pulls fine-tuned DistilBERT artifacts from Google Drive into `models/crisis_classifier/`. |
| `build_index.py` | Builds the ChromaDB vector index (`counsel_chat` + `knowledge_base` collections). Auto-downloads `knowledge_base.json` from Google Drive if missing. |
| `modeling/train.py` | Fine-tunes DistilBERT on `crisis_train.csv` with `AutoTokenizer` / `AutoModel`. |
| `modeling/evaluate.py` | Test-set metrics + confusion matrix → `reports/`. |
| `modeling/predict.py` | Standalone `is_crisis()` for ad-hoc inference. |
| `rag.py` | The pipeline: `respond(msg, history)` orchestrates cleaning, classification, retrieval, and generation. Lazy-loads all heavy components on first call. |
| `app.py` | Streamlit chat UI. Imports `rag.respond`. Handles UI state, sidebar controls, the API key form, and Gemini quota recovery. |

---

## Video Demo of Deployment to Streamlit Cloud

The demo video of the deployed instance lives at : https://drive.google.com/file/d/1tegNpx_GSeLBSwVEz05E-8ps0ri80_QT/view?usp=sharing


## Deployment to Streamlit Cloud

The deployed instance lives at: https://ai-mind-companion.streamlit.app

To deploy your own:

1. Push the repo to GitHub. Confirm `.env` is in `.gitignore`. **Do not** commit `models/crisis_classifier/` (the classifier weights are ~265 MB — over GitHub's 100 MB per-file limit).
2. Make sure `requirements.txt` and `assignment3/.streamlit/config.toml` are in the repo.
3. Create a new app on https://share.streamlit.io pointing at `assignment3/app.py`.
4. In the app's **Secrets** page, add:
   ```
   GEMINI_API_KEY = "your_fallback_key_here"
   ```
5. On first boot, the app calls `_bootstrap_artifacts()` which detects the missing classifier and runs `download_crisis_classifier()` to pull it from Google Drive (~30 seconds, one-time per fresh container). The `knowledge_base.json` file is similarly auto-fetched by `build_index.py` if needed.

---

## Architecture notes

**Why ChromaDB?** Persists to disk in one line, no server, clean Python API, and bundled HNSW indexing. FAISS is faster but bare-bones; not worth the extra plumbing for this scale.

**Why embed only the question (not Q+A)?** User messages look like questions, not therapist answers. Embedding only the `questionText` and storing all therapist responses in metadata gives much higher retrieval relevance. Counsel Chat has multiple therapist responses per question, so we group by `questionText` first — 2,608 raw rows collapse to 863 unique questions with ~3 answers each.

**Why two collections, not one?** Counsel Chat (~863 items) and the curated knowledge base (~30 items) play different roles. Counsel Chat gives "what other people have said about similar problems"; the KB gives "structured techniques." A bucketed retrieve (3 from counsel + 2 from KB) gives the LLM both kinds of context. A unified collection with similarity-sort would let the longer counsel chat passages drown out the shorter KB entries.

**Why disable Gemini "thinking"?** `gemini-2.5-flash` has thinking enabled by default and thinking tokens count against `max_output_tokens`. With our 600-token cap, the model would burn 400+ tokens reasoning internally and truncate the visible reply mid-sentence. `thinking_config=ThinkingConfig(thinking_budget=0)` is the standard fix for chat-style use.

**Why crisis_threshold = 0.5 instead of 0.7?** For a mental-health context, missing a real crisis is worse than over-flagging. Lowering the default threshold makes the classifier more sensitive at the cost of slightly more false positives — but a false positive just shows the user a list of helplines, while a false negative means missing someone in danger.

**Why hardcode the crisis resources block?** The LLM writes the empathic acknowledgement (which adapts to what the user said), but the helpline numbers are concatenated verbatim from a constant. Phone numbers must never be paraphrased, hallucinated, or "improved" by the LLM. Wrong number = real harm.

**Why three-tier API key resolution?** User-pasted (sidebar form) wins for the highest priority — keys never touch disk and a power user can dodge shared-key rate limits. Streamlit secrets is the deployment-time fallback. `.env` is the local-dev fallback. When the shared key hits its quota, the app catches the 429, auto-expands the sidebar, and prompts the user to paste their own key.

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
poetry add <package>               # runtime dep
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