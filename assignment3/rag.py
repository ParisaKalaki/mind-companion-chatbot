"""
rag.py
------
Runtime pipeline for the mental health chatbot. Single public entry point:

    from rag import respond
    result = respond("I've been feeling anxious", history=[])

Architecture:
    user_message
        ↓
    clean_text()
        ↓
    is_crisis() ──── crisis ──→ build_crisis_response() → log → return
        ↓
       rag
        ↓
    retrieve()  →  build_prompt()  →  Gemini  →  return

All return values share a unified shape:
    {
        "text":        str,            # the chatbot's reply
        "sources":     list[dict],     # RAG chunks the model cited
        "retrieved":   list[dict],     # all retrieved chunks (transparency)
        "prompt_used": str | None,     # assembled user-side prompt (debug)
        "path":        "rag" | "crisis" | "fallback",
        "classifier":  dict | None,    # classifier output for the message
    }

This module assumes the artifacts below are already on disk:
    - models/chroma_db/                 (counsel_chat + knowledge_base)
    - models/crisis_classifier/         (DistilBERT weights)
    - .env with GEMINI_API_KEY

Run directly to smoke-test:
    python rag.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from textwrap import dedent
from typing import Any, Optional

import chromadb
import torch
from chromadb.config import Settings
from dotenv import load_dotenv
from google import genai
from google.genai import types
from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import MODELS_DIR, PROCESSED_DATA_DIR, PROJ_ROOT


# ===========================================================================
# Environment validation — fail fast on missing credentials
# ===========================================================================
load_dotenv(PROJ_ROOT / ".env")


# ===========================================================================
# 1. REPLACE _check_env()  — lenient at import time
# ===========================================================================
def _check_env() -> None:
    """
    Lenient check at import time: warn if GEMINI_API_KEY is missing, but
    don't sys.exit. This lets Streamlit Cloud (where the key lives in
    st.secrets) and the user-paste-key flow both work.
 
    The actual hard failure happens in respond() if no key is available
    by the time we try to call Gemini.
    """
    if not os.getenv("GEMINI_API_KEY"):
        logger.warning(
            "GEMINI_API_KEY not found in environment at import time. "
            "Set it via .env, st.secrets, or set_gemini_client() before "
            "calling respond()."
        )
 
 
 
_check_env()
# ===========================================================================
# 2. ADD set_gemini_client()  — for runtime key swapping from the UI
# ===========================================================================
def set_gemini_client(api_key: str) -> None:
    """
    Swap in a new Gemini client with the given API key. Used by the Streamlit
    UI when the user pastes their own key — heavy components (embedding model,
    crisis classifier, Chroma) are not touched.
 
    Raises ValueError if api_key is empty or whitespace.
    """
    if not api_key or not api_key.strip():
        raise ValueError("api_key must be a non-empty string")
 
    c = get_components()
    c.gemini_client = genai.Client(api_key=api_key.strip())
    logger.info("Gemini client replaced with new API key.")
 

# ===========================================================================
# Configuration
# ===========================================================================
EMBED_MODEL_NAME       = "all-MiniLM-L6-v2"
CHROMA_DIR             = MODELS_DIR / "chroma_db"
COUNSEL_COLLECTION     = "counsel_chat"
KB_COLLECTION          = "knowledge_base"

CRISIS_MODEL_DIR       = MODELS_DIR / "crisis_classifier"
CRISIS_MAX_LENGTH      = 128
CRISIS_THRESHOLD       = 0.5  # lower than predict.py's 0.7 — see respond() docstring

GEMINI_MODEL           = "gemini-2.5-flash"
GEN_TEMPERATURE        = 0.7
GEN_MAX_OUTPUT_TOKENS  = 600
CRISIS_GEN_TEMPERATURE = 0.6
CRISIS_GEN_MAX_TOKENS  = 200

DEFAULT_K_COUNSEL      = 3
DEFAULT_K_KB           = 2
MAX_HISTORY_TURNS      = 6

CRISIS_LOG_PATH        = PROCESSED_DATA_DIR / "crisis_log.jsonl"
DEVICE                 = "cuda" if torch.cuda.is_available() else "cpu"


# ===========================================================================
# Prompts (edit these to tune behavior)
# ===========================================================================
MENTAL_HEALTH_SYSTEM_PROMPT = dedent("""
    You are a warm, careful mental health support companion. You are NOT a
    therapist, doctor, or diagnostician — you are a thoughtful presence that
    helps people feel heard and offers evidence-informed self-help ideas.

    HOW TO RESPOND
    - Lead with acknowledgement. Reflect what the person is feeling in your
      own words before offering anything else. People want to be heard first.
    - Use plain, gentle language. Short sentences. Avoid clinical jargon
      unless explaining it.
    - Offer one or two concrete, doable suggestions — not a list of ten.
      Pick the most relevant from the provided context.
    - Always close by inviting the person to share more, or to consider
      professional support if the situation warrants it.
    - Keep replies to roughly 3 to 6 short paragraphs unless the person
      asks for more detail.

    GROUNDING & CITATIONS
    - You will be given retrieved context labeled [S1], [S2], etc. Draw on
      this context when offering techniques or framing.
    - When you use information from a source, cite it inline like (S1) or
      (S2). Do not invent source IDs that weren't provided.
    - If the context doesn't fit the user's situation, rely on general
      empathic listening rather than forcing a poor match.

    WHAT YOU MUST NOT DO
    - Do not diagnose. Do not say "you have X" or "this sounds like X
      disorder." You can say "what you're describing is something many
      people experience" or "a professional could help you understand
      what's going on."
    - Do not prescribe or recommend specific medications or dosages.
    - Do not promise outcomes ("you will feel better in a week").
    - Do not minimize ("it could be worse," "others have it harder").
    - Do not push religion, politics, or unsolicited life philosophy.
    - Do not pretend to be human. If asked directly, say you're an AI
      support tool.

    SAFETY
    - If the person mentions suicidal thoughts, self-harm, or being in
      immediate danger, gently acknowledge what they shared and provide
      Australian crisis resources:
        - Lifeline: 13 11 14 (24/7)
        - Beyond Blue: 1300 22 4636
        - Suicide Call Back Service: 1300 659 467
        - In immediate danger: call 000
    - Encourage professional support for ongoing or severe distress.

    TONE EXAMPLES
    - Good: "That sounds really heavy. It makes sense you'd feel exhausted
      carrying it."
    - Bad: "I understand. Have you tried mindfulness?" (too quick to advice)
    - Bad: "Based on your symptoms, you may have generalized anxiety
      disorder." (diagnosing)
""").strip()


CRISIS_SYSTEM_PROMPT = (
    "You are responding to someone who has just expressed thoughts of suicide, "
    "self-harm, or being in serious crisis. Your only job right now is to "
    "acknowledge their pain warmly and personally — NOT to offer techniques, "
    "advice, or solutions. Crisis resources will be appended to your reply "
    "automatically; do NOT include any phone numbers or helpline names in your "
    "response.\n\n"
    "Write 2 to 4 short sentences. Be human and gentle. Reflect what they said "
    "in your own words. Tell them their pain matters and that they are not "
    "alone. Do not say 'I understand' (you don't). Do not minimize. Do not "
    "promise things will be okay. Do not diagnose. End by gently letting them "
    "know that help is available right now."
)


# Hardcoded — NEVER let the LLM rewrite phone numbers
CRISIS_RESOURCES_BLOCK = (
    "If you are in immediate danger, please call **000**.\n\n"
    "You don't have to face this alone. These services are free, confidential, "
    "and staffed by trained counsellors right now:\n\n"
    "- **Lifeline** — 13 11 14 (24/7)\n"
    "- **Suicide Call Back Service** — 1300 659 467\n"
    "- **Beyond Blue** — 1300 22 4636\n"
    "- **13YARN** — 13 92 76 (for Aboriginal and Torres Strait Islander people)\n"
    "- **Kids Helpline** — 1800 55 1800 (ages 5–25)\n\n"
    "If you can, please reach out to one of these now. Talking to someone "
    "trained can help you get through this moment."
)

CRISIS_FALLBACK_ACK = (
    "I hear you, and I'm really glad you reached out. What you're carrying "
    "sounds incredibly heavy. You are not alone in this, and your pain "
    "matters. Please don't go through this by yourself."
)


# ===========================================================================
# Components — lazy singleton
# ===========================================================================
@dataclass
class Components:
    embed_model:        SentenceTransformer
    counsel_collection: Any
    kb_collection:      Any
    crisis_tokenizer:   Any
    crisis_model:       Any
    gemini_client:      genai.Client


_components: Optional[Components] = None


def get_components() -> Components:
    """
    Initialize all components on first call; return cached afterward.
    Safe to call repeatedly (e.g. from Streamlit on every rerun).
    """
    global _components
    if _components is not None:
        return _components

    logger.info("Initializing rag.py pipeline components ...")

    # --- Embedding model ----------------------------------------------------
    logger.info(f"Loading embedding model '{EMBED_MODEL_NAME}' ...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # --- Chroma collections -------------------------------------------------
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Chroma DB not found at {CHROMA_DIR}. "
            "Run the index-building notebook cells first."
        )
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    counsel_coll = chroma_client.get_collection(COUNSEL_COLLECTION)
    kb_coll      = chroma_client.get_collection(KB_COLLECTION)
    logger.info(
        f"Connected to Chroma | counsel_chat: {counsel_coll.count():,} | "
        f"knowledge_base: {kb_coll.count()}"
    )

    # --- Crisis classifier --------------------------------------------------
    if not CRISIS_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Crisis classifier not found at {CRISIS_MODEL_DIR}. "
            "Run download_model.py or modelling/train.py first."
        )
    logger.info(f"Loading crisis classifier on {DEVICE} ...")
    crisis_tokenizer = AutoTokenizer.from_pretrained(str(CRISIS_MODEL_DIR))
    crisis_model = AutoModelForSequenceClassification.from_pretrained(
        str(CRISIS_MODEL_DIR)
    )
    crisis_model.to(DEVICE).eval()

    # --- Gemini client ------------------------------------------------------
    logger.info(f"Initializing Gemini client (model: {GEMINI_MODEL}) ...")
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    _components = Components(
        embed_model=embed_model,
        counsel_collection=counsel_coll,
        kb_collection=kb_coll,
        crisis_tokenizer=crisis_tokenizer,
        crisis_model=crisis_model,
        gemini_client=gemini_client,
    )
    logger.success("Pipeline components ready.")
    return _components


# ---------------------------------------------------------------------------
# ADD this new function anywhere after get_components() is defined:
# ---------------------------------------------------------------------------
def set_gemini_client(api_key: str) -> None:
    """
    Swap in a new Gemini client with the given API key. Used by the Streamlit
    UI when the user pastes their own key — heavy components (embedding model,
    crisis classifier, Chroma) are not touched.
 
    Raises ValueError if api_key is empty or whitespace.
    """
    if not api_key or not api_key.strip():
        raise ValueError("api_key must be a non-empty string")
 
    # Force lazy load of everything else first if it hasn't happened yet,
    # so we have a Components instance to mutate.
    c = get_components()
    c.gemini_client = genai.Client(api_key=api_key.strip())
    logger.info("Gemini client replaced with new API key.")
 


# ===========================================================================
# Text cleaning (inlined from clean.py to keep rag.py self-contained)
# ===========================================================================
def clean_text(text: str) -> str:
    """Normalize whitespace, strip URLs, drop non-ASCII characters."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ===========================================================================
# Crisis classifier inference
# ===========================================================================
def is_crisis(text: str, threshold: float = CRISIS_THRESHOLD) -> dict:
    """
    Score a single message for crisis content.

    Returns dict: {is_crisis, confidence, label, method}
    """
    c = get_components()

    inputs = c.crisis_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=CRISIS_MAX_LENGTH,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = c.crisis_model(**inputs).logits
    probs = torch.softmax(logits, dim=1)
    crisis_prob = probs[0][1].item()

    return {
        "is_crisis":  crisis_prob >= threshold,
        "confidence": round(crisis_prob, 4),
        "label":      "crisis" if crisis_prob >= threshold else "non-crisis",
        "method":     "model",
    }


# ===========================================================================
# Retrieval (counsel_chat + knowledge_base, bucketed merge)
# ===========================================================================
def _embed_query(query: str) -> list:
    return get_components().embed_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].tolist()


def _query_counsel(q_emb: list, k: int) -> list[dict]:
    if k <= 0:
        return []
    res = get_components().counsel_collection.query(
        query_embeddings=[q_emb], n_results=k
    )
    out = []
    for doc, meta, dist in zip(
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0],
    ):
        out.append({
            "source":            "counsel_chat",
            "similarity":        round(1.0 - dist, 4),
            "topic_or_category": meta.get("topic", "unknown"),
            "text":              doc,
            "content":           meta.get("answerText", ""),
            "raw_meta":          meta,
        })
    return out


def _query_kb(q_emb: list, k: int) -> list[dict]:
    if k <= 0:
        return []
    res = get_components().kb_collection.query(
        query_embeddings=[q_emb], n_results=k
    )
    out = []
    for doc, meta, dist in zip(
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0],
    ):
        content = (
            f"{meta.get('description', '')}\n\n"
            f"How to use: {meta.get('how_to', '')}\n\n"
            f"When useful: {meta.get('when_useful', '')}"
        )
        out.append({
            "source":            "knowledge_base",
            "similarity":        round(1.0 - dist, 4),
            "topic_or_category": meta.get("category", "unknown"),
            "text":              meta.get("title", doc),
            "content":           content,
            "raw_meta":          meta,
        })
    return out


def retrieve(
    query: str,
    k_counsel: int = DEFAULT_K_COUNSEL,
    k_kb: int = DEFAULT_K_KB,
) -> list[dict]:
    """
    Retrieve from both Chroma collections and return a bucketed-merge list:
    counsel_chat results first (empathic context), then knowledge_base
    (techniques). Within each bucket, ordered by similarity.
    """
    q_emb = _embed_query(query)
    return _query_counsel(q_emb, k_counsel) + _query_kb(q_emb, k_kb)


# ===========================================================================
# Prompt assembly
# ===========================================================================
def _format_chunk(idx: int, chunk: dict) -> str:
    source = chunk["source"]
    topic  = chunk["topic_or_category"]
    text   = chunk["text"]
    body   = chunk["content"]

    # Truncate very long bodies to keep the prompt focused (~200 tokens).
    if len(body) > 800:
        body = body[:800].rstrip() + " ..."

    if source == "counsel_chat":
        header = f"[S{idx}] (Counsel Chat — topic: {topic})"
        return f"{header}\nQuestion: {text}\nTherapist response: {body}"
    else:
        header = f"[S{idx}] (Knowledge Base — {topic}: {text})"
        return f"{header}\n{body}"


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(No prior conversation.)"
    recent = history[-MAX_HISTORY_TURNS:]
    lines = []
    for turn in recent:
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def build_prompt(
    user_message: str,
    retrieved_chunks: list[dict],
    history: Optional[list[dict]] = None,
) -> str:
    """Assemble the user-side prompt that goes to Gemini alongside the
    system instruction."""
    if history is None:
        history = []

    if retrieved_chunks:
        context_blocks = "\n\n".join(
            _format_chunk(i + 1, c) for i, c in enumerate(retrieved_chunks)
        )
    else:
        context_blocks = "(No retrieved context available.)"

    parts = [
        "RETRIEVED CONTEXT (use [S1], [S2]... to cite):",
        context_blocks,
        "",
        "---",
        "RECENT CONVERSATION:",
        _format_history(history),
        "",
        "---",
        "CURRENT USER MESSAGE:",
        user_message,
        "",
        "Respond as the support companion described in your system instructions.",
    ]
    return "\n".join(parts)


# ===========================================================================
# Generation (RAG path)
# ===========================================================================
def _extract_cited_sources(reply_text: str, retrieved: list[dict]) -> list[dict]:
    """Find (S1), (S2)... in the reply and return matching chunks
    (deduped, in citation order)."""
    cited_ids = []
    for m in re.finditer(r"\(S(\d+)\)", reply_text):
        idx = int(m.group(1))
        if 1 <= idx <= len(retrieved) and idx not in cited_ids:
            cited_ids.append(idx)
    return [retrieved[i - 1] for i in cited_ids]


def generate_response(
    user_message: str,
    history: Optional[list[dict]] = None,
    k_counsel: int = DEFAULT_K_COUNSEL,
    k_kb: int = DEFAULT_K_KB,
) -> dict:
    """RAG path: retrieve → build prompt → Gemini → package response."""
    if history is None:
        history = []

    retrieved   = retrieve(user_message, k_counsel=k_counsel, k_kb=k_kb)
    user_prompt = build_prompt(user_message, retrieved, history)

    config = types.GenerateContentConfig(
        system_instruction=MENTAL_HEALTH_SYSTEM_PROMPT,
        temperature=GEN_TEMPERATURE,
        max_output_tokens=GEN_MAX_OUTPUT_TOKENS,
        # Disable internal "thinking" so the full token budget goes to the
        # visible reply, not invisible reasoning. Required for chat-style use.
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    c = get_components()
    response = c.gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=config,
    )
    reply_text = (response.text or "").strip()

    return {
        "text":        reply_text,
        "sources":     _extract_cited_sources(reply_text, retrieved),
        "retrieved":   retrieved,
        "prompt_used": user_prompt,
    }


# ===========================================================================
# Crisis path
# ===========================================================================
def build_crisis_response(
    user_message: str,
    classifier_result: dict,
) -> dict:
    """
    LLM writes a short empathic acknowledgement; resources block is appended
    verbatim. Returns the same dict shape as generate_response().
    """
    config = types.GenerateContentConfig(
        system_instruction=CRISIS_SYSTEM_PROMPT,
        temperature=CRISIS_GEN_TEMPERATURE,
        max_output_tokens=CRISIS_GEN_MAX_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    try:
        c = get_components()
        response = c.gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f'The person just said: "{user_message}"',
            config=config,
        )
        ack = (response.text or "").strip() or CRISIS_FALLBACK_ACK
    except Exception as exc:
        # Network / API failure: never let the crisis path silently die.
        logger.warning(f"Crisis ack LLM call failed ({exc}); using fallback.")
        ack = CRISIS_FALLBACK_ACK

    full_text = f"{ack}\n\n{CRISIS_RESOURCES_BLOCK}"

    return {
        "text":        full_text,
        "sources":     [],
        "retrieved":   [],
        "prompt_used": None,
        "path":        "crisis",
        "classifier":  classifier_result,
    }


def _log_crisis_event(
    user_message: str,
    classifier_result: dict,
    response_text: str,
) -> None:
    """Append one record to data/processed/crisis_log.jsonl."""
    record = {
        "timestamp_utc":     datetime.now(timezone.utc).isoformat(),
        "user_message":      user_message,
        "classifier_label":  classifier_result.get("label"),
        "classifier_conf":   classifier_result.get("confidence"),
        "classifier_method": classifier_result.get("method"),
        "response_excerpt":  response_text[:300],
    }
    try:
        CRISIS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CRISIS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        # Logging must NEVER break the user-facing path.
        logger.warning(f"Failed to log crisis event ({exc}); continuing anyway.")


# ===========================================================================
# Public entry point
# ===========================================================================

# ===========================================================================
# 3. REPLACE respond()  — now accepts k_counsel and k_kb
# ===========================================================================
def respond(
    user_message: str,
    history: Optional[list[dict]] = None,
    crisis_threshold: float = CRISIS_THRESHOLD,
    k_counsel: int = DEFAULT_K_COUNSEL,
    k_kb: int = DEFAULT_K_KB,
) -> dict:
    """
    The orchestrator. Cleans → classifies → routes → returns unified dict.
 
    Parameters
    ----------
    user_message : str
        Raw user input.
    history : list of {'role': 'user'|'assistant', 'content': str}, optional
        Prior conversation. Crisis branch ignores history; RAG branch uses it.
    crisis_threshold : float
        Probability cutoff for the crisis branch. Default 0.5 — more sensitive
        than predict.py's 0.7, because in a mental-health context missing a
        real crisis is worse than over-flagging.
    k_counsel : int
        Top-k from the counsel_chat collection. Default 3.
    k_kb : int
        Top-k from the knowledge_base collection. Default 2.
 
    Returns
    -------
    dict with keys:
        text, sources, retrieved, prompt_used, path, classifier
    where path is 'rag', 'crisis', or 'fallback'.
    """
    if history is None:
        history = []
 
    cleaned = clean_text(user_message)
    if not cleaned:
        return {
            "text":        "Could you tell me a bit more about what's on your mind?",
            "sources":     [],
            "retrieved":   [],
            "prompt_used": None,
            "path":        "fallback",
            "classifier":  None,
        }
 
    classifier_result = is_crisis(cleaned, threshold=crisis_threshold)
 
    if classifier_result["is_crisis"]:
        result = build_crisis_response(cleaned, classifier_result)
        _log_crisis_event(cleaned, classifier_result, result["text"])
        return result
 
    # RAG branch — pass slider-controlled k values through
    result = generate_response(
        cleaned,
        history=history,
        k_counsel=k_counsel,
        k_kb=k_kb,
    )
    result["path"] = "rag"
    result["classifier"] = classifier_result
    return result

# ===========================================================================
# Smoke test (only runs when executed directly)
# ===========================================================================
def _smoke_test() -> None:
    test_cases = [
        ("I've been really stressed about exams and can't focus", "rag"),
        ("I just feel kind of empty lately", "rag"),
        ("I don't want to be here anymore. I can't keep doing this.", "crisis"),
        ("I've been thinking about ending it all", "crisis"),
    ]

    for i, (msg, expected_path) in enumerate(test_cases, 1):
        print("\n" + "=" * 80)
        print(f"TEST {i}: ({expected_path.upper()} expected)  {msg}")
        print("=" * 80)

        result = respond(msg, history=[])
        path_icon = "🚨" if result["path"] == "crisis" else "💬"
        match     = "✅" if result["path"] == expected_path else "❌"
        print(f"\n{match} routed to: {path_icon} {result['path']}")

        if result.get("classifier"):
            c = result["classifier"]
            print(f"   classifier: label={c['label']}  "
                  f"conf={c['confidence']:.3f}  method={c['method']}")

        print(f"\n💬 REPLY:\n{result['text']}")

        if result["sources"]:
            print(f"\n📎 CITED ({len(result['sources'])}):")
            for s in result["sources"]:
                print(f"   - {s['source']:14s} | {s['topic_or_category']:25s} | "
                      f"{s['text'][:80]}")

    print("\n" + "=" * 80)
    print(f"Crisis log: {CRISIS_LOG_PATH}")
    if CRISIS_LOG_PATH.exists():
        n = sum(1 for _ in open(CRISIS_LOG_PATH, encoding="utf-8"))
        print(f"Crisis log now contains {n} entries")


if __name__ == "__main__":
    _smoke_test()