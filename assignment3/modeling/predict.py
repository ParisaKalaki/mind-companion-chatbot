import torch
from pathlib import Path
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# ── Config ─────────────────────────────────────────────────────────────
MODEL_DIR  = Path('models/crisis_classifier')
MAX_LENGTH = 128

# ── Load model once ────────────────────────────────────────────────────
def load_model():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f'Model not found at {MODEL_DIR}. '
            'Download it from Google Drive and place it in models/crisis_classifier/'
        )
    tokenizer = DistilBertTokenizer.from_pretrained(str(MODEL_DIR))
    model     = DistilBertForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()
    return tokenizer, model

# ── Predict single message ─────────────────────────────────────────────
CRISIS_KEYWORDS = [
    'ending it all', 'end it all', 'no reason to live',
    'want to die', 'better off dead', 'can\'t go on',
    'no point anymore', 'goodbye forever', 'final goodbye',
    'don\'t want to be here', 'disappear forever'
]

def is_crisis(text, tokenizer, model, threshold=0.7):
    # Rule-based check first
    # text_lower = text.lower()
    # if any(kw in text_lower for kw in CRISIS_KEYWORDS):
    #     return {
    #         'is_crisis':  True,
    #         'confidence': 1.0,
    #         'label':      'crisis',
    #         'method':     'keyword'
    #     }
    
    # ML model check
    inputs = tokenizer(text, return_tensors='pt', truncation=True,
                       padding='max_length', max_length=MAX_LENGTH)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)
    crisis_prob = probs[0][1].item()

    return {
        'is_crisis':  crisis_prob >= threshold,
        'confidence': round(crisis_prob, 4),
        'label':      'crisis' if crisis_prob >= threshold else 'non-crisis',
        'method':     'model'
    }

# ── Test it ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Loading model...')
    tokenizer, model = load_model()
    print('✅ Model loaded\n')

    test_messages = [
        "I have been feeling really anxious lately and don't know what to do",
        "I want to kill myself, I can't take this anymore",
        "Can you help me with some breathing exercises?",
        "I've been having dark thoughts and feel like ending it all",
        "I'm struggling with my relationship and feeling lost",
    ]

    print('Testing messages:')
    print('-' * 60)
    for msg in test_messages:
        result = is_crisis(msg, tokenizer, model)
        flag = '🚨' if result['is_crisis'] else '✅'
        print(f"{flag} [{result['label'].upper()}] confidence: {result['confidence']:.2%}")
        print(f"   {msg[:80]}")
        print()