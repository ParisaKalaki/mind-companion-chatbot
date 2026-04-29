# Mind Companion — Team Briefing Document

> **Audience:** team members who weren't in the build loop and need to understand the project well enough to write the report and slides.
>
> **Length:** long. Skim the table of contents, read the parts you need.
>
> **Tone:** honest about what we built, why, and what we'd do differently. The marker will respect a project that knows its own limitations more than one that pretends to be perfect.

---

## Table of contents

1. [What we built (in one paragraph)](#1-what-we-built-in-one-paragraph)
2. [The architecture, end to end](#2-the-architecture-end-to-end)
3. [Component-by-component deep dive](#3-component-by-component-deep-dive)
4. [Key design decisions and the reasoning behind each](#4-key-design-decisions-and-the-reasoning-behind-each)
5. [Datasets used and how they were processed](#5-datasets-used-and-how-they-were-processed)
6. [The model: training, evaluation, results](#6-the-model-training-evaluation-results)
7. [The RAG layer: indexing, retrieval, why we made the calls we made](#7-the-rag-layer-indexing-retrieval-why-we-made-the-calls-we-made)
8. [The LLM layer: prompts, generation, citations](#8-the-llm-layer-prompts-generation-citations)
9. [The orchestration layer: how it all comes together](#9-the-orchestration-layer-how-it-all-comes-together)
10. [The Streamlit UI](#10-the-streamlit-ui)
11. [Deployment to Streamlit Cloud](#11-deployment-to-streamlit-cloud)
12. [Bugs, gotchas, and things we learned the hard way](#12-bugs-gotchas-and-things-we-learned-the-hard-way)
13. [Limitations and ethical considerations](#13-limitations-and-ethical-considerations)
14. [What we'd do differently with more time](#14-what-wed-do-differently-with-more-time)
15. [Suggested structure for the report and slides](#15-suggested-structure-for-the-report-and-slides)

---

## 1. What we built (in one paragraph)

A retrieval-augmented mental health support chatbot called **Mind Companion**. The user types a message; we clean it, classify it for crisis indicators using a fine-tuned DistilBERT model, and route it down one of two paths. **Crisis path:** Gemini writes a short empathic acknowledgement, we append a hardcoded list of Australian crisis helplines (never LLM-generated), and log the event for safety review. **RAG path:** we embed the message with `all-MiniLM-L6-v2`, retrieve top-k passages from a ChromaDB vector store containing the Counsel Chat dataset and a curated CBT/mindfulness/psychoeducation knowledge base, assemble a structured prompt with conversation history and retrieved context, and Gemini 2.5 Flash generates a grounded reply with inline source citations. The frontend is a Streamlit chat UI with a custom dark theme (teal accents), live tuning sliders, transparent source citations, and a three-tier API key resolution system. It's deployed on Streamlit Cloud at https://ai-mind-companion.streamlit.app.

---

## 2. The architecture, end to end

```
                        User message
                              │
                              ▼
                    ┌──────────────────┐
                    │  clean_text()    │  strip URLs, ASCII-only, collapse whitespace
                    └────────┬─────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  is_crisis()     │  DistilBERT, returns prob of crisis
                    └────────┬─────────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
          crisis path                   RAG path
                │                           │
                ▼                           ▼
   ┌────────────────────┐         ┌──────────────────────────┐
   │ Gemini writes      │         │ retrieve()               │
   │ empathic ack       │         │   3 from counsel_chat    │
   │ (no resources)     │         │   2 from knowledge_base  │
   └────────┬───────────┘         └─────────┬────────────────┘
            │                                │
            ▼                                ▼
   ┌────────────────────┐         ┌──────────────────────────┐
   │ Append hardcoded   │         │ build_prompt()           │
   │ AU resources       │         │  system + context +      │
   └────────┬───────────┘         │  history + user message  │
            │                     └─────────┬────────────────┘
            ▼                                │
   ┌────────────────────┐                    ▼
   │ Log to             │         ┌──────────────────────────┐
   │ crisis_log.jsonl   │         │ Gemini 2.5 Flash         │
   └────────┬───────────┘         └─────────┬────────────────┘
            │                                │
            └────────────┬───────────────────┘
                         ▼
              Unified return dict
              {text, sources, retrieved, path, classifier, ...}
                         │
                         ▼
                  Streamlit UI
                  renders the message
```

Everything routes through one entry point: `respond(user_message, history)` in `rag.py`. The Streamlit UI is a thin presentation layer — all business logic lives in the pipeline module.

---

## 3. Component-by-component deep dive

| File | Lines of code | What it does |
|---|---|---|
| `config.py` | ~40 | Defines project paths (`PROJ_ROOT`, `DATA_DIR`, `MODELS_DIR`, etc.). Loads `.env`. Sets up loguru. |
| `dataset.py` | ~100 | Validates `.env` has Kaggle credentials, creates the data folder tree, downloads the Kaggle suicide-watch dataset. Idempotent. |
| `clean.py` | ~150 | Pulls Counsel Chat from HuggingFace, cleans both datasets (strips URLs, non-ASCII, normalizes whitespace), builds stratified 70/15/15 train/val/test splits for the crisis classifier. |
| `download_model.py` | ~80 | Downloads the 5 fine-tuned DistilBERT artifacts from Google Drive into `models/crisis_classifier/`. Used both during local setup and during first boot on Streamlit Cloud. |
| `build_index.py` | ~310 | Builds both Chroma collections (`counsel_chat`, `knowledge_base`) end-to-end. CLI flags (`--force`, `--counsel`, `--kb`) for selective rebuild. Auto-downloads `knowledge_base.json` from Google Drive if not present locally. |
| `modelling/train.py` | ~150 | Fine-tunes DistilBERT on `crisis_train.csv` using HuggingFace Trainer. Uses `AutoTokenizer`/`AutoModel` (not `DistilBertTokenizer`) — see decisions section. |
| `modelling/predict.py` | ~120 | Standalone `is_crisis()` for ad-hoc inference. Mostly superseded by `rag.py` which inlines this logic. |
| `modelling/evaluate.py` | ~150 | Evaluates the classifier on the held-out test set. Outputs metrics CSV and confusion matrix PNG to `reports/`. |
| `rag.py` | ~470 | **The pipeline.** Single entry point `respond()`. Lazy-loads all heavy components. Contains the orchestrator, retrieval, prompt assembly, generation (RAG and crisis paths), and crisis logging. |
| `app.py` | ~600 | **The UI.** Streamlit chat interface with custom CSS dark theme, hero header, chat bubbles, sidebar with sliders and API key form, citation expanders, route pills, typing indicator, quota-error handling. |
| `notebooks/1_setup.ipynb` | — | One-shot setup notebook. Runs every script in order: dataset download → clean → classifier download → evaluate → build_index → smoke-test rag. The recommended setup path. |
| `.streamlit/config.toml` | ~10 | Locks dark theme + teal accents at the framework level. |

---

## 4. Key design decisions and the reasoning behind each

This is the section that will save you in viva questions.

### 4.1 Why DistilBERT for the crisis classifier (not zero-shot GPT, not full BERT)

- **Zero-shot GPT** would have worked but cost money per inference and adds a network round-trip on every message. We need crisis classification on every single user turn.
- **Full BERT-base** is ~110M parameters; DistilBERT is ~66M, about 60% the size with ~95% of the performance. Fits comfortably in 1GB of RAM (Streamlit Cloud's free tier).
- DistilBERT-base-uncased is the standard baseline for binary text classification in 2024–2026. Replicable, well-documented, fast on CPU (~50ms inference).

### 4.2 Why we lowered the crisis threshold to 0.5 (not the more conservative 0.7)

- For a mental health context, **missing a real crisis is worse than over-flagging**.
- A false positive shows the user a list of helplines — minor friction.
- A false negative means an at-risk person is treated like a generic conversation. That's the bad outcome.
- We exposed it as a slider so it can be demonstrated either way during the demo.

### 4.3 Why two separate Chroma collections instead of one

- Counsel Chat (~863 unique questions, ~3 answers each) and the curated Knowledge Base (~30 entries) play different roles.
- Counsel Chat = "what other people in similar situations have heard" — emotional resonance.
- Knowledge Base = "structured techniques and psychoeducation" — actionable advice.
- A unified collection sorted by similarity would let the longer Counsel Chat passages dominate top-k because longer text tends to produce more context-rich embeddings; the shorter, tag-anchored KB entries would lose.
- **Bucketed retrieval** (3 from counsel + 2 from KB by default) guarantees the LLM sees both kinds of context.

### 4.4 Why we embed only `questionText`, not the full Q+A document

- User messages look like questions, not therapist answers.
- Embedding the question gives much higher similarity scores when the user's input is conversational.
- Therapist answers go into Chroma metadata so the LLM still sees them — they just aren't part of the search target.
- We tested this empirically against full Q+A embedding; question-only retrieval was visibly better on test queries.

### 4.5 Why we group Counsel Chat by question before indexing

- Raw Counsel Chat has 2,608 rows but only 863 unique questions — multiple therapists answer the same question.
- Without grouping, top-5 retrieval often returns the same question 5 times with 5 different answers. Bad — gives the LLM redundant context.
- After grouping, top-5 returns 5 *different* questions with all answers concatenated in metadata. Result: more diverse retrieval, richer LLM grounding.
- This was an actual bug we found during testing. Worth mentioning in the report.

### 4.6 Why Gemini 2.5 Flash (not GPT-4o, not Claude, not local)

- **Free tier** with generous quotas. Critical for an academic project.
- **Speed**: ~1-2 second response time, faster than GPT-4o.
- **Local models** (LLaMA 3, Mistral via Ollama) would have eaten time on dependency hell, especially for a deployed demo.
- **Anthropic / OpenAI** are commercial APIs; no free quota that scales.
- Trade-off: Gemini's empathy quality is slightly behind Claude in our ad-hoc testing, but well above the bar for an academic demo.

### 4.7 Why we disable Gemini's "thinking" mode for chat

- `gemini-2.5-flash` has thinking enabled by default. Thinking tokens count against `max_output_tokens`.
- With our 600-token cap, the model would burn 400+ tokens reasoning internally and truncate the visible reply mid-sentence.
- `thinking_config=ThinkingConfig(thinking_budget=0)` is the standard fix for chat-style use.
- We discovered this through actual broken responses in testing — talk about it in the report as an example of LLM quirk debugging.

### 4.8 Why the crisis-resources block is hardcoded, not LLM-generated

- The LLM writes the **empathic acknowledgement** because that should adapt to what the person said.
- The LLM **never** writes phone numbers because: paraphrasing, hallucinating, or "improving" a helpline number could cause real harm.
- `CRISIS_RESOURCES_BLOCK` is a single source of truth in `rag.py`. Update once if a number ever changes.
- The crisis-path system prompt explicitly instructs Gemini *not* to include phone numbers in its acknowledgement, then we concatenate the block ourselves.

### 4.9 Why we use `AutoTokenizer` / `AutoModel` instead of `DistilBertTokenizer` / `DistilBertForSequenceClassification`

- The original code used the slow tokenizer class. The trained model on Drive ships `tokenizer.json` (fast tokenizer format) but not `vocab.txt` (slow tokenizer requirement).
- The slow class would have failed to load the trained model.
- `AutoTokenizer` picks the right class automatically.
- Modern HuggingFace idiom anyway.

### 4.10 Why three-tier API key resolution (user → secrets → .env)

| Tier | Purpose |
|---|---|
| User-pasted in sidebar | Best UX: per-session, never written to disk. User has full control. |
| `st.secrets` | Streamlit Cloud deployment fallback. Lets us share a default key. |
| `.env` | Local development fallback. Same default key, in a file. |

This means **anyone can use the deployed app without their own key** (until quota), but power users can paste their own to skip rate limits. Quota errors trigger a UI flow that prompts the user to paste their key.

### 4.11 Why we built `rag.py` as one file instead of splitting into `pipeline.py` / `retrieval.py` / `generation.py` / etc.

- It's ~470 lines. Splitting would create ~6 files of ~80 lines each.
- More files = more import boilerplate, more places for circular imports.
- The Streamlit app only ever imports `respond()` and `set_gemini_client()`. A single file gives one clear public interface.
- Architectural overengineering is a trap on a 1-week deadline.

### 4.12 Why a Streamlit form for the API key, not a free-floating input

- `st.form` batches inputs — the key only registers when the user clicks "Submit" (or hits Enter).
- Without a form, every keystroke would trigger a Streamlit rerun. Annoying and wasteful.
- The form pattern also gives a natural place for the green confirmation message after submit.

### 4.13 Why we extracted index-building from notebook cells into `build_index.py`

- Originally the Chroma index was built inside notebook cells. Worked fine for development, but:
  - A teammate cloning the repo couldn't reproduce it without opening Jupyter and running cells in order.
  - Streamlit Cloud's container can't open notebooks.
  - Any change to the build logic meant editing two places (notebook + the script that started to mirror it).
- `build_index.py` puts both index builds (Counsel Chat + KB) into a single CLI script with `--force`, `--counsel`, `--kb` flags. The notebook now `%run`s the script — single source of truth.
- The script is idempotent and selective, so editing only the KB JSON doesn't force a re-embedding of all 863 counsel chat questions.

### 4.14 Why `knowledge_base.json` is auto-downloaded from Drive (not committed to the repo)

- The KB is hand-curated content with citations and helpline numbers — it's authored data, not derived data.
- We could have committed it to the repo. We chose Drive auto-download in `build_index.py` for two reasons:
  - Keeps the data layer's pattern consistent: classifier weights are on Drive (too big for GitHub), KB is on Drive (same flow). Less special-casing.
  - Allows iterating on the KB without forcing a repo commit + push to update the deployment.
- Trade-off: requires an internet connection and a `gdown`-compatible Drive file (must be shared "Anyone with the link"). For a real-world product, committing the JSON would be cleaner. For an academic project, the Drive approach is fine and demonstrates a useful pattern.

---

## 5. Datasets used and how they were processed

### 5.1 Counsel Chat

- **Source:** [`nbertagnolli/counsel-chat` on HuggingFace](https://huggingface.co/datasets/nbertagnolli/counsel-chat)
- **Content:** ~3,000 anonymized questions from people seeking therapy advice on counselchat.com, with multiple licensed therapist answers per question.
- **Use:** RAG corpus for the empathic-context part of retrieval.
- **Pipeline:** `clean.py` downloads it, drops nulls, drops therapist answers under 50 characters (junk filter), strips URLs and non-ASCII, builds a `document` field combining question + answer.
- **Indexing:** `build_index.py` groups by `questionText` (so 2,608 raw rows become 863 unique questions), embeds the question with `all-MiniLM-L6-v2`, stores all therapist answers concatenated in metadata.

### 5.2 Suicide-watch dataset (crisis classifier training)

- **Source:** [Kaggle: nikhileswarkomati/suicide-watch](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)
- **Content:** ~232,000 Reddit posts labeled as `suicide` or `non-suicide`.
- **Use:** Training data for the binary crisis classifier.
- **Pipeline:** `dataset.py` downloads it via Kaggle API, `clean.py` cleans text, drops posts under 20 characters, maps labels to binary, splits 70/15/15 stratified.

### 5.3 Curated knowledge base

- **Source:** **We wrote it ourselves.** ~30 entries covering CBT techniques (12), mindfulness/grounding (8), and psychoeducation (10). Drafted with LLM assistance, then reviewed and edited.
- **Citations:** Beck, Linehan, Kabat-Zinn, Burns, Greenberger & Padesky, etc. — real authors and books.
- **Schema:** `{id, category, title, tags, description, how_to, when_useful, source, disclaimer_level}`.
- **Hosted on Google Drive** as a single JSON. Auto-downloaded by `build_index.py` to `data/external/knowledge_base.json` if not already present locally.
- **Indexed in Chroma** as a separate collection. We embed `title + description + tags`. The `how_to` stays in metadata — it's instructions, not a search target.

---

## 6. The model: training, evaluation, results

### 6.1 Training setup

- **Base model:** `distilbert-base-uncased`
- **Task head:** binary sequence classification (2 labels: 0 = non-crisis, 1 = crisis)
- **Hyperparameters:** 3 epochs, batch size 16 (train) / 32 (eval), max length 128 tokens, AdamW optimizer (HuggingFace defaults), seed 42
- **Sample mode for development:** 10,000 train / 2,000 val. Full-dataset training also supported via a flag.
- **Hardware:** Trained on a single GPU. Took ~10 minutes on the sample, longer on the full dataset.
- **Evaluation strategy:** evaluate every epoch, save best model by F1.

### 6.2 Evaluation

- **Held-out test set:** 15% of the cleaned data, never seen during training/validation.
- **Metrics computed:** accuracy, F1, precision, recall, full classification report.
- **Confusion matrix** saved as PNG to `reports/figures/confusion_matrix.png`.
- **Threshold:** 0.7 in the original `predict.py` for evaluation; lowered to **0.5** in production (`rag.py`) — see decision 4.2.
- **Production observation:** classifier confidence on clearly non-crisis messages (exam stress, feeling empty) is ~0.000; on clearly crisis messages ("I want to end it all") it's ~0.9–1.0. Well-separated, so threshold tuning has limited effect on obvious cases — the borderline cases are where it matters.

### 6.3 What to put in the report

- The classification report numbers (accuracy / F1 / precision / recall on test).
- The confusion matrix figure.
- A brief note about the precision-recall trade-off and why we chose recall-favoring threshold.
- Confidence histogram if you have time — separation between classes is a strong story.

---

## 7. The RAG layer: indexing, retrieval, why we made the calls we made

### 7.1 Embedding model

- **`all-MiniLM-L6-v2`** from sentence-transformers. 384-dim embeddings, fast on CPU, well-known baseline.
- We **normalize embeddings** at write time and query time so cosine similarity becomes a simple dot product.

### 7.2 Vector store

- **ChromaDB** with HNSW index, cosine distance.
- Persists to disk at `models/chroma_db/`. No server, no docker — just a folder.
- Why not FAISS: faster but bare-bones; no metadata API. Chroma is the right call for this scale.

### 7.3 Two collections

| Collection | Items | Embedded text | Metadata |
|---|---|---|---|
| `counsel_chat` | 863 | the `questionText` only | source, topic, full questionText, all therapist answers concatenated, n_answers |
| `knowledge_base` | 30 | title + description + tags | source, category, title, tags, description, how_to, when_useful, citation, disclaimer_level |

### 7.4 Building the index

`build_index.py` is a single CLI script that builds both collections end-to-end. Default behaviour: skip a collection if it already has entries; pass `--force` to wipe and rebuild. `--counsel` and `--kb` flags let you rebuild only one collection (useful when iterating on the KB JSON without re-embedding 863 counsel chat questions).

The script also handles the KB's auto-download: if `data/external/knowledge_base.json` is missing, it pulls from Google Drive via `gdown` before building the KB collection.

### 7.5 Retrieval

- `retrieve(query, k_counsel=3, k_kb=2)` embeds the query, queries each collection separately, returns a normalized list with consistent keys (`source`, `similarity`, `topic_or_category`, `text`, `content`, `raw_meta`).
- **Bucketed merge** — counsel chat first, then KB. Not globally sorted by similarity. (Decision 4.3.)
- KB similarities tend to be lower than counsel chat (0.3–0.5 vs 0.5–0.7) because KB entries are short. The bucketed merge protects them from being drowned.

### 7.6 Sanity-check results

We tested with 5–6 representative queries:
- "I've been feeling really anxious lately" → anxiety counsel chat + Worry Time KB. ✅
- "How can I stop catastrophizing every small problem?" → off-topic counsel chat (because no Reddit user uses the word "catastrophizing") **but** the KB caught it with the `Decatastrophizing` entry. **This is exactly why we kept two collections.** Mention this in the report.
- "I want to feel less depressed but have no motivation" → depression counsel chat + Behavioral Activation + Activity Scheduling. ✅
- "I just lost my mother" → grief-and-loss counsel chat + Understanding Grief + When to Seek Professional Help. ✅

---

## 8. The LLM layer: prompts, generation, citations

### 8.1 Two system prompts

- **`MENTAL_HEALTH_SYSTEM_PROMPT`** for the RAG path. Defines tone, citation rules, what not to do (no diagnosis, no medication advice, no minimizing), safety fallbacks. ~600 tokens.
- **`CRISIS_SYSTEM_PROMPT`** for the crisis path. Much shorter. Tells Gemini to write 2–4 short sentences acknowledging pain, **not** to write phone numbers (we append those ourselves), and not to say "I understand."

### 8.2 RAG prompt structure

```
SYSTEM: <MENTAL_HEALTH_SYSTEM_PROMPT>

USER:
RETRIEVED CONTEXT (use [S1], [S2]... to cite):
[S1] (Counsel Chat — topic: anxiety) Question: ... Therapist response: ...
[S2] ...
[S5] (Knowledge Base — CBT: Worry Time) ...

---
RECENT CONVERSATION:
User: ...
Assistant: ...

---
CURRENT USER MESSAGE:
<the user's actual message>

Respond as the support companion described in your system instructions.
```

### 8.3 Citations

- The system prompt asks the model to cite inline like `(S1)`, `(S2)`.
- After Gemini responds, we regex-scan for `(S\d+)` patterns and look up the corresponding retrieved chunks.
- The UI shows them in a "📎 Sources cited" expander with similarity scores and topic.

### 8.4 Conversation history

- Capped at last **6 turns** (3 user + 3 assistant pairs).
- Longer histories dilute instruction-following and waste tokens.
- Crisis path **ignores** history — when someone says "I want to end it," what they said 3 turns ago doesn't matter.

### 8.5 Generation config

- `temperature=0.7` for RAG path (warm but not erratic), `0.6` for crisis (slightly cooler).
- `max_output_tokens=600` for RAG (~3-6 short paragraphs), `200` for crisis (intentionally short).
- `thinking_budget=0` — disable thinking entirely. (Decision 4.7.)

---

## 9. The orchestration layer: how it all comes together

`respond()` in `rag.py` is the single public entry point. ~50 lines. The flow:

```python
def respond(user_message, history=None, crisis_threshold=0.5, k_counsel=3, k_kb=2):
    # 1. Clean
    cleaned = clean_text(user_message)
    if not cleaned:
        return fallback_response()  # path = "fallback"

    # 2. Classify
    classifier_result = is_crisis(cleaned, threshold=crisis_threshold)

    # 3. Route
    if classifier_result["is_crisis"]:
        result = build_crisis_response(cleaned, classifier_result)
        _log_crisis_event(cleaned, classifier_result, result["text"])
        return result  # path = "crisis"

    # 4. RAG
    result = generate_response(cleaned, history=history,
                               k_counsel=k_counsel, k_kb=k_kb)
    result["path"] = "rag"
    result["classifier"] = classifier_result
    return result
```

**Three things to highlight:**

- **Unified return shape** across all three paths (`rag`, `crisis`, `fallback`). The UI never has to branch on path internals.
- **Lazy component loading** via `get_components()` — heavy stuff (DistilBERT, Chroma, embedding model, Gemini client) loads on first call, then cached.
- **Crisis logging** writes a JSONL line to `data/processed/crisis_log.jsonl` with timestamp, message, classifier confidence, response excerpt. This gives us a real audit trail for the report.

---

## 10. The Streamlit UI

### 10.1 Visual design

- **Dark theme locked** via `.streamlit/config.toml`. Won't follow system preference.
- **Teal/green accent** (`#10b981`) — calming, common in mental health UIs, avoids the corporate-purple AI cliché.
- **Hero header** with gradient text and 3 feature badges.
- **Custom chat bubbles** via CSS — user (right-aligned, teal gradient), assistant (left-aligned, dark gradient), crisis-route (red-tinted border).
- **Route pill** above each assistant message showing the path and classifier confidence.
- **Typing indicator** — three teal bouncing dots, replaces the boring spinner.
- **Animations:** smooth scroll, fade-in on new messages, hover effects on bubbles and buttons, pulsing teal border on the API key field when quota is hit.

### 10.2 Sidebar features

- **API key form** with Submit and Clear buttons. Three-tier resolution under the hood.
- **Three live tuning sliders:** counsel chat passages retrieved (0–6), knowledge base entries retrieved (0–5), crisis classifier threshold (0.1–0.9). All wired end-to-end.
- **Always-visible crisis resources card** — defense in depth, even if everything else fails the user can still see helplines.
- **Clear conversation button** + message counter.
- **About expander** explaining the architecture briefly.

### 10.3 Quota handling

When the shared Gemini key hits its daily limit:

1. `rag.respond()` raises a 429 error.
2. `_is_quota_error()` detects the quota signature in the exception text.
3. `st.session_state.quota_blocked = True`.
4. `st.rerun()` triggers a fresh page render.
5. `set_page_config` reads `quota_blocked` and forces sidebar open.
6. Sidebar shows pulsing teal "🪫 Shared key quota exhausted" prompt with a bouncing 👇 emoji pointing at the input.
7. Top-of-page warning banner appears.
8. User pastes their key, hits Submit, banner clears.

### 10.4 Citation transparency

- Per assistant message, "📎 Sources cited" expander shows the chunks the LLM actually used (extracted from `(S1)`, `(S2)` patterns in the reply).
- Each source card shows the type (counsel chat or KB), topic, similarity score, and snippet.
- Bonus expander "🔬 Classifier details" shows the raw classifier output JSON. Useful during the demo for showing the audience how the routing works.

---

## 11. Deployment to Streamlit Cloud

- **Live URL:** https://ai-mind-companion.streamlit.app
- **Secret:** `GEMINI_API_KEY` set in the Streamlit Cloud dashboard under Secrets.
- **Files in repo:** all source code, `models/chroma_db/` (small, ~10MB), `data/processed/counsel_chat_clean.csv`.
- **Files NOT in repo:**
  - `models/crisis_classifier/` (~265 MB — exceeds GitHub's 100 MB per-file limit)
  - `data/external/knowledge_base.json` (we host it on Drive instead, see decision 4.14)
- **Bootstrap on first boot:** `app.py` calls `_bootstrap_artifacts()` which detects the missing classifier and runs `download_crisis_classifier()` from `download_model.py` to pull it from Google Drive. The KB JSON is similarly auto-fetched by `build_index.py`. The first cold boot adds ~30s for these downloads.
- **Cold start:** ~5-7 minutes (pip install + first download).
- **Subsequent starts:** ~10s (pip cache warm, classifier still in container).
- **Free tier puts apps to sleep** after 7 days of no traffic. First wake-up is ~1-2 min.

---

## 12. Bugs, gotchas, and things we learned the hard way

These are great stories for the report. Real engineering problems, not pretend ones.

### 12.1 The duplicate-question bug

- First indexing pass: 2,608 documents, top-5 retrieval often returned the same question 5 times.
- Fix: group by `questionText` before indexing. 2,608 → 863. (Decision 4.5.)

### 12.2 The Gemini truncation bug

- First test responses cut off mid-sentence.
- Cause: Gemini 2.5 Flash spends 400+ tokens on internal "thinking" by default; counts against `max_output_tokens`.
- Fix: `thinking_config=ThinkingConfig(thinking_budget=0)`. (Decision 4.7.)

### 12.3 The tokenizer compatibility bug

- Trained model on Drive shipped `tokenizer.json` only, no `vocab.txt`.
- `DistilBertTokenizer` needs `vocab.txt`. `AutoTokenizer` doesn't.
- Fix: switch to `AutoTokenizer` / `AutoModelForSequenceClassification`. (Decision 4.9.)

### 12.4 The Streamlit Cloud loguru bug

- `config.py` calls `logger.remove(0)` to deregister loguru's default handler.
- Streamlit Cloud removes that handler before user code runs.
- `logger.remove(0)` then raises `ValueError: There is no existing handler with id 0`.
- Fix: wrap in `try/except ValueError`.

### 12.5 The `set_gemini_client` ordering bug

- `set_gemini_client()` internally calls `get_components()` (to access the cache).
- `get_components()` requires the classifier to be on disk.
- On first cloud boot, classifier wasn't there yet.
- The error came up as "Failed to initialize Gemini client" — confusing because it had nothing to do with Gemini.
- Fix: in `app.py`, call `_load_components()` (which runs the bootstrap) BEFORE `set_gemini_client()`.

### 12.6 The Chroma DB missing on cloud

- After fixing the classifier bootstrap, deploy still failed: "Chroma DB not found".
- Cause: we'd told Streamlit Cloud the classifier would be auto-downloaded but assumed Chroma was committed to the repo. The `.gitignore` wasn't actually keeping it.
- Fix: confirm `models/chroma_db/` is in the repo (it's only ~10MB, well within GitHub limits). Update `.gitignore` to be selective: exclude `models/crisis_classifier/`, keep `models/chroma_db/`.

### 12.7 The over-broad CSS rule that hid the sidebar toggle

- `header[data-testid="stHeader"] { visibility: hidden; }` hid the entire Streamlit header.
- Including the sidebar collapse button.
- Users couldn't reopen the sidebar after closing it.
- Fix: hide just `#MainMenu` and `footer`, leave the header transparent so the toggle is still clickable.

### 12.8 The broken Gemini package import

- Old SDK: `google-generativeai` (deprecated as of 2025).
- New SDK: `google-genai` (note the hyphen, `from google import genai`).
- Different package, different API entirely.
- Worth noting in dependency discussion.

### 12.9 The notebook-bound index build

- Originally the Chroma index was built from notebook cells. A teammate cloning the repo couldn't reproduce setup without opening Jupyter and running cells in order. Streamlit Cloud's container can't open notebooks at all.
- Fix: extracted the build logic into `build_index.py` (~310 lines, reproducible from CLI). Notebook now `%run`s the script — single source of truth. (Decision 4.13.)

### 12.10 The missing `knowledge_base.json` on a fresh clone

- On the first clean re-run after the build_index extraction, the script crashed because `data/external/knowledge_base.json` didn't exist — the JSON had been generated in a notebook cell during early development and was sitting only on the original developer's machine.
- Fix: hosted the JSON on Google Drive and added an `_ensure_kb_json()` helper to `build_index.py` that auto-downloads via `gdown` if missing. Same pattern as the classifier. (Decision 4.14.)

---

## 13. Limitations and ethical considerations

Worth being upfront about in the report — markers respect intellectual honesty.

### 13.1 Scope limitations

- **Not a real therapist.** Says so everywhere — sidebar, hero header, system prompt, every assistant message has the option for it.
- **English only.** No multilingual support.
- **AU-focused crisis resources.** Lifeline, Beyond Blue, etc. International users would need adapted resources.
- **Adult-focused.** No specific child-safety logic beyond the Kids Helpline reference. We don't validate user age.
- **Not personalized.** No user profiles, no memory across sessions, no symptom tracking.

### 13.2 Model limitations

- Crisis classifier was trained on Reddit text — real-world chatbot input may differ stylistically.
- Counsel Chat is North American in origin — therapeutic norms vary by culture.
- LLMs hallucinate. Even with citation grounding, Gemini can paraphrase a source incorrectly. The `disclaimer_level` field on KB entries was meant to help here but isn't yet wired into the LLM prompt.

### 13.3 Knowledge base origin

- The 30 entries were drafted with LLM assistance, then reviewed.
- Citations point at real source texts, but we did not cross-reference page numbers.
- For a production tool, every entry should be reviewed by a clinical psychologist. For an academic project, "drafted with LLM assistance, then human-reviewed" is a defensible position.

### 13.4 Ethical considerations

- **Privacy:** we don't store conversations. Crisis events are logged with the message for safety review (not user identification), but only on the local container — they don't persist across Streamlit Cloud restarts.
- **Harm reduction:** the design treats false negatives on the crisis path as worse than false positives. Hardcoded resources block ensures helpline numbers can't be hallucinated.
- **Manipulation risk:** users could try to prompt-inject the system prompt away. Out of scope for this project but worth mentioning.
- **Diagnostic boundary:** the system prompt explicitly forbids diagnostic language. Tested in the demo with messages like "do I have anxiety?" — Gemini correctly declines to diagnose.

---

## 14. What we'd do differently with more time

Useful for the "future work" section of the report.

1. **Conversation memory across sessions.** Currently, history dies on tab close.
2. **Real knowledge graph** instead of flat KB — links from symptom → strategy → resource.
3. **Multi-turn evaluation suite.** Right now we evaluate the classifier in isolation. We'd want end-to-end metrics: relevance of retrieved chunks, citation rate, latency distribution, agreement between LLM output and retrieved context.
4. **Per-message user feedback** (👍/👎) logged for retrospective analysis.
5. **Multi-language support** via translation pre/post processing.
6. **Better safety:** combine the ML classifier with a keyword-based rule engine for defense in depth. Currently the rule-based fallback is commented out in `predict.py`.
7. **Embedding fine-tuning** on counseling-specific text — generic `all-MiniLM-L6-v2` is decent but not domain-tuned.
8. **Streaming responses** so the user sees Gemini's reply token-by-token instead of waiting for the full reply.
9. **Persistent crisis log** — currently flushed on container restart. Would need an external sink (Google Sheets / a database) for a production safety-review workflow.

---

## 15. Suggested structure for the report and slides

### Report structure (10–15 pages)

1. **Introduction** (1 page) — what we built, why, the problem motivation.
2. **System architecture** (1–2 pages) — the diagram from section 2, brief description of each component.
3. **Datasets** (1 page) — section 5.
4. **Crisis classifier** (2 pages) — model choice, training setup, evaluation results, confusion matrix figure.
5. **RAG pipeline** (2–3 pages) — embedding, vector store, two-collection design, retrieval examples (use the catastrophizing example from section 7.6 — it's a great story).
6. **LLM integration** (1–2 pages) — Gemini choice, prompt engineering, citation extraction, the thinking-mode bug as a debugging example.
7. **Streamlit UI** (1 page) — screenshots, key features.
8. **Deployment** (0.5 pages) — Streamlit Cloud, bootstrap pattern, the live URL.
9. **Limitations and ethics** (1 page) — section 13.
10. **Future work** (0.5 pages) — section 14.
11. **References** — Counsel Chat dataset, suicide-watch dataset, all source citations from `knowledge_base.json`, HuggingFace models.

### Slide deck structure (15–20 slides)

1. **Title** + team names + project name
2. **The problem** — mental health access gap, why a chatbot
3. **What we built** — one-line description + screenshot of the UI + live URL
4. **Architecture** — the diagram from section 2
5. **Crisis classifier** — model + training + key metric
6. **Crisis classifier evaluation** — confusion matrix + classification report
7. **RAG: the embedding model** — `all-MiniLM-L6-v2`, why
8. **RAG: the two collections** — bucketed merge diagram, sample retrieval
9. **The catastrophizing example** — concrete story showing why the KB matters
10. **LLM: prompt structure** — system + retrieved context + history + user message
11. **Citation grounding** — screenshot of the source expander
12. **Crisis routing** — flowchart, hardcoded resources, screenshot of crisis reply
13. **The Streamlit UI** — screenshot tour of the sliders, dark theme, source cards
14. **Deployment** — cloud architecture + bootstrap pattern (mention the 265MB classifier story and the KB-on-Drive pattern)
15. **Bugs and lessons** — pick 2–3 from section 12 for the live demo / Q&A
16. **Limitations and ethics** — short, honest
17. **Future work** — pick 3–4 from section 14
18. **Demo** — live walkthrough at https://ai-mind-companion.streamlit.app
19. **Q&A**

---

## How to navigate this codebase as a new reader

If you're seeing this codebase for the first time and want to understand it, read in this order:

1. **`README.md`** — overall overview, setup steps
2. **`config.py`** — paths, very short
3. **`rag.py`** — the brain. Read top to bottom. Skim the prompt strings; focus on `respond()`, `get_components()`, `retrieve()`, `generate_response()`, `build_crisis_response()`.
4. **`app.py`** — the face. Skim the CSS block (just know it's there). Focus on `main()` and the sidebar/key-resolution flow.
5. **`build_index.py`** — how the vector store is constructed. Short and well-commented.
6. **`modelling/train.py`** + **`modelling/evaluate.py`** — how the classifier was made.
7. **`notebooks/1_setup.ipynb`** — the canonical end-to-end setup walkthrough.

Most of the project is in `rag.py` (~470 lines) and `app.py` (~600 lines). The rest is glue, scripts, and one-off setup.

---

*Last updated by you, [date], based on the build sessions during the assignment week.*