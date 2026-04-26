"""
modelling/predict.py
--------------------
Inference for the fine-tuned crisis classifier (DistilBERT).

Loads the model lazily (and caches it module-wide), exposes a single
`is_crisis(text)` entry point that the chatbot routing layer can call
on every incoming user message.

Run directly to smoke-test on a few example messages:
    python modelling/predict.py
"""

# ---------------------------------------------------------------------------
# Path shim: predict.py lives one level deeper than config.py, so we add
# the parent folder (which contains config.py) to sys.path before the
# local config import.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_PARENT_PKG = Path(__file__).resolve().parent.parent
if str(_PARENT_PKG) not in sys.path:
    sys.path.insert(0, str(_PARENT_PKG))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import MODELS_DIR

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CRISIS_MODEL_DIR  = MODELS_DIR / "crisis_classifier"
MAX_LENGTH        = 128
DEFAULT_THRESHOLD = 0.7
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

# Optional rule-based keyword fallback (currently unused; flip the flag in
# `is_crisis` to enable). Useful for catching obvious cases the model misses.
CRISIS_KEYWORDS = [
    "ending it all", "end it all", "no reason to live",
    "want to die", "better off dead", "can't go on",
    "no point anymore", "goodbye forever", "final goodbye",
    "don't want to be here", "disappear forever",
]

# Module-level cache so the model loads only once per process
_TOKENIZER = None
_MODEL = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(force_reload: bool = False):
    """
    Load the crisis classifier and tokenizer. Cached after the first call.

    Returns
    -------
    (tokenizer, model) tuple, with model already on `DEVICE` and in eval mode.
    """
    global _TOKENIZER, _MODEL

    if _TOKENIZER is not None and _MODEL is not None and not force_reload:
        return _TOKENIZER, _MODEL

    if not CRISIS_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model not found at {CRISIS_MODEL_DIR}. "
            "Run `python download_model.py` first to fetch it from Google Drive."
        )

    logger.info(f"Loading crisis classifier from {CRISIS_MODEL_DIR} on {DEVICE} ...")
    # AutoTokenizer / AutoModel handle the fast-vs-slow tokenizer pick automatically
    # (the model dir ships tokenizer.json, not vocab.txt — so we need the fast one).
    tokenizer = AutoTokenizer.from_pretrained(str(CRISIS_MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(CRISIS_MODEL_DIR))
    model.to(DEVICE).eval()

    _TOKENIZER, _MODEL = tokenizer, model
    logger.success("Crisis classifier loaded.")
    return tokenizer, model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def is_crisis(
    text: str,
    tokenizer=None,
    model=None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """
    Score a single message for crisis content.

    Parameters
    ----------
    text : str
        The user's message.
    tokenizer, model :
        Optional preloaded artifacts. If omitted, the cached model is used
        (loaded on first call).
    threshold : float
        Probability cutoff for flagging crisis. Default 0.7.

    Returns
    -------
    dict with keys: is_crisis (bool), confidence (float),
                    label (str), method (str)
    """
    if tokenizer is None or model is None:
        tokenizer, model = load_model()

    # --- Optional rule-based check (disabled by default) ---
    # text_lower = text.lower()
    # if any(kw in text_lower for kw in CRISIS_KEYWORDS):
    #     return {
    #         "is_crisis":  True,
    #         "confidence": 1.0,
    #         "label":      "crisis",
    #         "method":     "keyword",
    #     }

    # --- ML model check ---
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)
    crisis_prob = probs[0][1].item()

    return {
        "is_crisis":  crisis_prob >= threshold,
        "confidence": round(crisis_prob, 4),
        "label":      "crisis" if crisis_prob >= threshold else "non-crisis",
        "method":     "model",
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("Loading model ...")
    tokenizer, model = load_model()
    logger.success("Model loaded.\n")

    test_messages = [
        "I have been feeling really anxious lately and don't know what to do",
        "I want to kill myself, I can't take this anymore",
        "Can you help me with some breathing exercises?",
        "I've been having dark thoughts and feel like ending it all",
        "I'm struggling with my relationship and feeling lost",
    ]

    print("Testing messages:")
    print("-" * 60)
    for msg in test_messages:
        result = is_crisis(msg, tokenizer, model)
        flag = "🚨" if result["is_crisis"] else "✅"
        print(
            f"{flag} [{result['label'].upper()}] "
            f"confidence: {result['confidence']:.2%}"
        )
        print(f"   {msg[:80]}")
        print()


if __name__ == "__main__":
    main()