"""
app.py
------
Streamlit chat UI for the mental health support chatbot.

Run locally:
    streamlit run app.py

Deploy to Streamlit Cloud:
    1. Push this repo to GitHub (without .env in the commit)
    2. Create a new app on https://share.streamlit.io pointing at app.py
    3. In the app's "Secrets" page, add:
            GEMINI_API_KEY = "your_fallback_key_here"
    4. Make sure requirements.txt is in the repo root
    5. Make sure .streamlit/config.toml is in the repo (locks dark theme)

API key resolution order (highest priority first):
    1. User-pasted key in the sidebar (per-session, never written to disk)
    2. st.secrets["GEMINI_API_KEY"]  (Streamlit Cloud deployment)
    3. .env GEMINI_API_KEY            (local dev fallback / instructor key)
"""

import os
import time

import streamlit as st

# Pull GEMINI_API_KEY from Streamlit secrets BEFORE importing rag so its
# import-time .env load picks it up via os.environ.
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ.setdefault("GEMINI_API_KEY", st.secrets["GEMINI_API_KEY"])
except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
    pass

import rag  # noqa: E402


# ===========================================================================
# Page config
# ===========================================================================
# Dynamic sidebar state: if we previously hit a quota error, force the sidebar
# open on this rerun so the user can see the API key field. set_page_config
# only honors initial_sidebar_state at first paint per rerun, which is exactly
# what we want — flipping the flag + st.rerun() triggers a re-paint with the
# sidebar opened.
_initial_sidebar = (
    "expanded"
    if st.session_state.get("quota_blocked", False)
    else "expanded"   # default for desktop; toggle is always visible
)

st.set_page_config(
    page_title="Mind Companion",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state=_initial_sidebar,
    menu_items={
        "About": (
            "Mind Companion — a student RAG project demonstrating empathic "
            "AI support. Not a replacement for professional mental health care."
        ),
    },
)


# ===========================================================================
# Custom CSS — dark theme polish, teal accents, animations
# ===========================================================================
CUSTOM_CSS = """
<style>
    /* ============================================================
       Global resets + smooth scroll
       ============================================================ */
    html { scroll-behavior: smooth; }

    /* Hide Streamlit's hamburger menu and "Made with Streamlit" footer,
       but keep the header itself so the sidebar toggle button remains. */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] {
        background: transparent;
    }
    /* Keep the sidebar collapse/expand toggle clearly visible */
    button[data-testid="stSidebarCollapseButton"],
    button[data-testid="stBaseButton-headerNoPadding"],
    [data-testid="stSidebarCollapsedControl"] {
        visibility: visible !important;
        z-index: 999;
    }

    /* Tighten the main container */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 6rem !important;
        max-width: 900px;
    }

    /* ============================================================
       Hero header
       ============================================================ */
    .hero {
        position: relative;
        padding: 2.4rem 2rem 2rem;
        margin-bottom: 1.5rem;
        border-radius: 24px;
        background:
            radial-gradient(circle at 20% 0%, rgba(16, 185, 129, 0.18) 0%, transparent 55%),
            radial-gradient(circle at 100% 100%, rgba(45, 212, 191, 0.12) 0%, transparent 55%),
            linear-gradient(135deg, #111821 0%, #0d1218 100%);
        border: 1px solid rgba(16, 185, 129, 0.18);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.35);
        overflow: hidden;
    }
    .hero::before {
        content: "";
        position: absolute;
        top: -50%; right: -20%;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(16, 185, 129, 0.12) 0%, transparent 70%);
        filter: blur(40px);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, #10b981 0%, #34d399 50%, #6ee7b7 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        margin: 0.5rem 0 0;
        color: #9ca3af;
        font-size: 1.05rem;
        max-width: 620px;
        line-height: 1.5;
    }
    .hero-badges {
        margin-top: 1.2rem;
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 500;
        color: #6ee7b7;
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 999px;
    }

    /* ============================================================
       Chat bubbles — full custom replacement
       ============================================================ */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        padding: 0.5rem 0 !important;
        border: none !important;
    }

    /* User bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
        [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #064e3b 0%, #047857 100%);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 18px 18px 4px 18px;
        padding: 0.9rem 1.1rem !important;
        margin-left: auto;
        max-width: 75%;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
        transition: all 0.2s ease;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
        [data-testid="stChatMessageContent"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.22);
    }

    /* Assistant bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
        [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #1a2129 0%, #141a23 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px 18px 18px 4px;
        padding: 1rem 1.2rem !important;
        max-width: 85%;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: all 0.2s ease;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
        [data-testid="stChatMessageContent"]:hover {
        border-color: rgba(16, 185, 129, 0.25);
    }

    /* Crisis-route assistant bubble — softer red border */
    .crisis-bubble [data-testid="stChatMessageContent"] {
        border-color: rgba(248, 113, 113, 0.4) !important;
        background: linear-gradient(135deg, #2a1518 0%, #1a0f12 100%) !important;
    }

    /* ============================================================
       Route pill — sits above each assistant message
       ============================================================ */
    .route-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.65rem;
        margin-bottom: 0.5rem;
        font-size: 0.72rem;
        font-weight: 500;
        border-radius: 999px;
        letter-spacing: 0.02em;
    }
    .route-pill.rag {
        color: #6ee7b7;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .route-pill.crisis {
        color: #fca5a5;
        background: rgba(248, 113, 113, 0.1);
        border: 1px solid rgba(248, 113, 113, 0.35);
    }
    .route-pill.fallback {
        color: #9ca3af;
        background: rgba(156, 163, 175, 0.1);
        border: 1px solid rgba(156, 163, 175, 0.25);
    }

    /* ============================================================
       Sidebar polish
       ============================================================ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1218 0%, #0a0e14 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        background: linear-gradient(90deg, #10b981, #34d399);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
    }

    /* Crisis resources block in sidebar — styled card */
    .crisis-card {
        padding: 1rem;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(248, 113, 113, 0.08), rgba(248, 113, 113, 0.03));
        border: 1px solid rgba(248, 113, 113, 0.25);
        margin: 0.5rem 0;
    }
    .crisis-card-title {
        color: #fca5a5;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.6rem;
    }
    .crisis-card-line {
        font-size: 0.85rem;
        color: #d1d5db;
        margin: 0.25rem 0;
    }
    .crisis-card-line strong { color: #fef2f2; }

    /* ============================================================
       Buttons + inputs
       ============================================================ */
    .stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.2rem;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.25);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.4);
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
    }
    .stButton > button:active { transform: translateY(0); }

    /* Chat input */
    [data-testid="stChatInput"] {
        background: #141a23 !important;
        border: 1px solid rgba(16, 185, 129, 0.2) !important;
        border-radius: 16px !important;
        transition: border-color 0.2s ease;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: rgba(16, 185, 129, 0.5) !important;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1) !important;
    }

    /* Sliders */
    [data-baseweb="slider"] [role="slider"] {
        background: #10b981 !important;
        border-color: #10b981 !important;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.25) !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: rgba(20, 26, 35, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        margin-top: 0.5rem;
    }
    [data-testid="stExpander"] summary:hover {
        background: rgba(16, 185, 129, 0.05);
    }

    /* ============================================================
       Typing indicator (three bouncing dots)
       ============================================================ */
    .typing {
        display: inline-flex;
        gap: 5px;
        padding: 0.6rem 0.4rem;
    }
    .typing span {
        width: 8px; height: 8px;
        background: #10b981;
        border-radius: 50%;
        opacity: 0.4;
        animation: typing-bounce 1.4s infinite ease-in-out;
    }
    .typing span:nth-child(2) { animation-delay: 0.2s; }
    .typing span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes typing-bounce {
        0%, 80%, 100% { transform: translateY(0);   opacity: 0.4; }
        40%           { transform: translateY(-6px); opacity: 1.0; }
    }

    /* ============================================================
       Source-card styling inside the citations expander
       ============================================================ */
    .source-card {
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        background: rgba(16, 185, 129, 0.04);
        border-left: 3px solid #10b981;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    .source-card.kb { border-left-color: #6ee7b7; }
    .source-card-meta {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-bottom: 0.3rem;
    }
    .source-card-snippet {
        color: #d1d5db;
        font-style: italic;
        margin-top: 0.4rem;
        padding-left: 0.5rem;
        border-left: 2px solid rgba(16, 185, 129, 0.3);
    }

    /* Fade-in animation for new messages */
    @keyframes message-fade-in {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    [data-testid="stChatMessage"] {
        animation: message-fade-in 0.3s ease-out;
    }

    /* ============================================================
       Quota-blocked highlight — pulsing teal border + arrow
       Applied to the API key block in the sidebar when the shared
       key has been exhausted.
       ============================================================ */
    .api-key-prompt {
        position: relative;
        padding: 0.9rem 1rem;
        margin: 0.5rem 0 1rem;
        border-radius: 12px;
        background: linear-gradient(135deg,
            rgba(16, 185, 129, 0.12),
            rgba(16, 185, 129, 0.04));
        border: 2px solid #10b981;
        color: #d1fae5;
        font-size: 0.85rem;
        line-height: 1.45;
        animation: pulse-glow 2s ease-in-out infinite;
    }
    .api-key-prompt::before {
        content: "👇";
        position: absolute;
        bottom: -22px;
        left: 50%;
        transform: translateX(-50%);
        font-size: 1.4rem;
        animation: bounce-down 1.2s ease-in-out infinite;
    }
    @keyframes pulse-glow {
        0%, 100% {
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.6),
                        0 0 14px rgba(16, 185, 129, 0.25);
        }
        50% {
            box-shadow: 0 0 0 6px rgba(16, 185, 129, 0),
                        0 0 24px rgba(16, 185, 129, 0.45);
        }
    }
    @keyframes bounce-down {
        0%, 100% { transform: translate(-50%, 0); }
        50%      { transform: translate(-50%, 6px); }
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ===========================================================================
# Quota-error detection
# ===========================================================================
def _is_quota_error(exc: Exception) -> bool:
    """
    Heuristic: did Gemini fail because the API key has hit its quota /
    rate limit? Google's SDK raises errors with text like:
        429 RESOURCE_EXHAUSTED
        quota exceeded
        rate limit
    """
    msg = (str(exc) or "").lower()
    needles = ("429", "resource_exhausted", "quota", "rate limit", "rate-limit")
    return any(n in msg for n in needles)


# ===========================================================================
# Bootstrap — download large artifacts on first Streamlit Cloud boot
# ===========================================================================
def _bootstrap_artifacts() -> None:
    """
    Download the crisis classifier (~265 MB) on first Streamlit Cloud boot.
    Streamlit Cloud gives a fresh container per deploy, and the classifier
    weights are too big to commit to GitHub.

    No-op if all classifier files are already on disk (local dev).
    """
    from config import MODELS_DIR

    classifier_dir = MODELS_DIR / "crisis_classifier"
    needed_files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    all_present = (
        classifier_dir.exists()
        and all((classifier_dir / f).exists() for f in needed_files)
    )
    if all_present:
        return

    st.info(
        "📥 First-time setup: downloading the crisis classifier (~265 MB). "
        "This happens once per deployment and takes about 30 seconds."
    )
    try:
        from download_model import download_crisis_classifier
        download_crisis_classifier()
    except Exception as e:
        st.error(
            f"Could not download the crisis classifier: {e}\n\n"
            "Check that the Drive files are still set to 'Anyone with the link'."
        )
        st.stop()


# ===========================================================================
# Component caching
# ===========================================================================
@st.cache_resource(show_spinner=False)
def _load_components():
    """Bootstrap missing artifacts, then trigger lazy load of rag.py components."""
    _bootstrap_artifacts()
    return rag.get_components()


# ===========================================================================
# API key resolution
# ===========================================================================
def _resolve_api_key(user_pasted_key: str) -> tuple[str | None, str]:
    if user_pasted_key and user_pasted_key.strip():
        return user_pasted_key.strip(), "user"
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"], "secrets"
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        pass
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key, "env"
    return None, "missing"


# ===========================================================================
# Hero header
# ===========================================================================
def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1 class="hero-title">🧠 Mind Companion</h1>
            <p class="hero-subtitle">
                A warm, careful AI support presence. Powered by retrieval-augmented
                generation over real counselor conversations and curated mental
                health techniques.
            </p>
            <div class="hero-badges">
                <span class="hero-badge">📚 RAG over Counsel Chat</span>
                <span class="hero-badge">🛡️ Crisis-aware routing</span>
                <span class="hero-badge">🇦🇺 AU resources</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# Sidebar
# ===========================================================================
def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("## Mind Companion")
        st.caption("Empathic support, not therapy.")

        st.divider()

        # --- API key ----------------------------------------------------
        st.markdown("### 🔑 API key")

        # If the shared key just hit quota, show a prominent prompt right
        # above the input field so the user knows where to act.
        if st.session_state.get("quota_blocked", False):
            st.markdown(
                """
                <div class="api-key-prompt">
                    <strong>🪫 Shared key quota exhausted.</strong><br>
                    Paste your own free Gemini key below to keep chatting.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption(
            "Paste your own Gemini key for the best experience. "
            "If left blank, a fallback key is used (rate limits apply)."
        )

        # Use a form so the key only registers when the user clicks Submit
        # (or presses Enter in the field) — not on every keystroke.
        with st.form("api_key_form", clear_on_submit=False):
            key_input = st.text_input(
                "Your Gemini API key",
                type="password",
                placeholder="AIza...",
                label_visibility="collapsed",
                value=st.session_state.get("submitted_api_key", ""),
            )
            col_submit, col_clear = st.columns([2, 1])
            with col_submit:
                submitted = st.form_submit_button(
                    "Use my key", use_container_width=True
                )
            with col_clear:
                cleared = st.form_submit_button(
                    "Clear", use_container_width=True
                )

        if submitted and key_input.strip():
            st.session_state.submitted_api_key = key_input.strip()
            st.session_state.quota_blocked = False
            st.success("✅ Your key is active for this session.")
        elif submitted and not key_input.strip():
            st.warning("Please paste a key before submitting.")
        elif cleared:
            st.session_state.submitted_api_key = ""
            st.info("Cleared. Falling back to the shared key.")

        # The actual key we'll use downstream — populated only after submit.
        user_key = st.session_state.get("submitted_api_key", "")

        st.caption(
            "Get a free key at "
            "[aistudio.google.com/apikey](https://aistudio.google.com/apikey). "
            "Stored only in this browser session."
        )

        st.divider()

        # --- Settings ---------------------------------------------------
        st.markdown("### ⚙️ Settings")
        k_counsel = st.slider(
            "Counsel Chat passages retrieved",
            min_value=0, max_value=6, value=3,
            help="More passages = richer empathic context, longer prompts.",
        )
        k_kb = st.slider(
            "Knowledge base entries retrieved",
            min_value=0, max_value=5, value=2,
            help="Curated CBT / mindfulness / psychoed entries.",
        )
        crisis_threshold = st.slider(
            "Crisis classifier threshold",
            min_value=0.1, max_value=0.9, value=0.5, step=0.05,
            help=(
                "Lower = more sensitive (catches more crisis messages, "
                "but also more false positives)."
            ),
        )

        st.divider()

        # --- Conversation ----------------------------------------------
        st.markdown("### 💬 Conversation")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.history = []
            st.session_state.messages = []
            st.rerun()
        st.caption(f"Messages this session: {len(st.session_state.get('history', []))}")

        st.divider()

        # --- Crisis resources ------------------------------------------
        st.markdown(
            """
            <div class="crisis-card">
                <div class="crisis-card-title">🆘 If you're in crisis (Australia)</div>
                <div class="crisis-card-line"><strong>000</strong> — immediate danger</div>
                <div class="crisis-card-line"><strong>Lifeline</strong> — 13 11 14</div>
                <div class="crisis-card-line"><strong>Beyond Blue</strong> — 1300 22 4636</div>
                <div class="crisis-card-line"><strong>Suicide Call Back</strong> — 1300 659 467</div>
                <div class="crisis-card-line"><strong>13YARN</strong> — 13 92 76</div>
                <div class="crisis-card-line"><strong>Kids Helpline</strong> — 1800 55 1800</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        with st.expander("ℹ️ About this app"):
            st.markdown(
                "Student project demonstrating retrieval-augmented generation "
                "for mental health support. It combines:\n\n"
                "- a fine-tuned **DistilBERT crisis classifier** that "
                "routes urgent messages to safety resources,\n"
                "- a **ChromaDB** vector store over Counsel Chat Q&A and "
                "curated CBT/mindfulness/psychoed entries,\n"
                "- **Google Gemini 2.5 Flash** for empathic, grounded replies.\n\n"
                "**Not a substitute for professional care.** "
                "If you are struggling, please reach out to a qualified "
                "mental health professional."
            )

    return {
        "user_key":         user_key,
        "k_counsel":        k_counsel,
        "k_kb":             k_kb,
        "crisis_threshold": crisis_threshold,
    }


# ===========================================================================
# Message rendering
# ===========================================================================
def _render_route_pill(result: dict) -> None:
    path = result.get("path", "rag")
    classifier = result.get("classifier")

    if path == "crisis":
        conf = classifier.get("confidence", 0) if classifier else 0
        label = f"🚨 Crisis pathway · classifier {conf:.0%}"
        klass = "crisis"
    elif path == "fallback":
        label = "💭 Empty input — gentle fallback"
        klass = "fallback"
    else:
        conf = classifier.get("confidence", 0) if classifier else 0
        n_sources = len(result.get("sources") or [])
        label = f"💬 RAG · classifier {conf:.0%} · {n_sources} sources cited"
        klass = "rag"

    st.markdown(
        f'<span class="route-pill {klass}">{label}</span>',
        unsafe_allow_html=True,
    )


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📎 Sources cited ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            src     = s.get("source", "?")
            topic   = s.get("topic_or_category", "?")
            snippet = (s.get("text", "") or "")[:220]
            sim     = s.get("similarity", 0.0)
            badge   = "💬 Counsel Chat" if src == "counsel_chat" else "📘 Knowledge Base"
            css_class = "source-card" if src == "counsel_chat" else "source-card kb"
            st.markdown(
                f"""
                <div class="{css_class}">
                    <div class="source-card-meta">
                        <strong>{i}.</strong> {badge} · {topic} · similarity {sim:.2f}
                    </div>
                    <div class="source-card-snippet">{snippet}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_assistant_message(result: dict) -> None:
    _render_route_pill(result)

    if result.get("path") == "crisis":
        st.warning(
            "Safety resources are shown below because this message was "
            "routed through our crisis support pathway."
        )

    st.markdown(result["text"])

    _render_sources(result.get("sources") or [])

    classifier = result.get("classifier")
    if classifier:
        with st.expander("🔬 Classifier details"):
            st.json(classifier)


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    render_hero()

    if "history" not in st.session_state:
        st.session_state.history = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "active_key_source" not in st.session_state:
        st.session_state.active_key_source = None
    if "quota_blocked" not in st.session_state:
        st.session_state.quota_blocked = False
    if "submitted_api_key" not in st.session_state:
        st.session_state.submitted_api_key = ""

    # Top-of-page banner when the shared key has hit quota — visible even
    # before the user looks at the sidebar.
    if st.session_state.quota_blocked:
        st.warning(
            "🪫 **The shared API key has hit its daily quota.** "
            "Please paste your own free Gemini key in the sidebar on the "
            "left (under **🔑 API key**) to keep chatting. "
            "Get one in 30 seconds at "
            "[aistudio.google.com/apikey](https://aistudio.google.com/apikey).",
            icon="⚠️",
        )

    settings = render_sidebar()

    # --- Heavy components FIRST (cached) -----------------------------------
    # MUST run before set_gemini_client, because that function indirectly
    # triggers get_components() — which fails if the classifier isn't on
    # disk yet. _load_components() handles bootstrap before that happens.
    try:
        with st.spinner("⚡ Warming up models ..."):
            _load_components()
    except FileNotFoundError as e:
        st.error(
            f"Required artifacts missing: {e}\n\n"
            "Run dataset.py, clean.py, download_model.py, and the indexing "
            "notebook cells before launching the app."
        )
        st.stop()

    # --- Resolve API key (after components are ready) ----------------------
    api_key, key_source = _resolve_api_key(settings["user_key"])
    if api_key is None:
        st.error(
            "No Gemini API key available. Please paste one in the sidebar, "
            "or set GEMINI_API_KEY in .env / Streamlit secrets."
        )
        st.stop()

    if key_source != st.session_state.active_key_source or key_source == "user":
        try:
            rag.set_gemini_client(api_key)
            st.session_state.active_key_source = key_source
            # If we were previously quota-blocked and the user has now pasted
            # their own key, clear the banner state.
            if key_source == "user":
                st.session_state.quota_blocked = False
        except Exception as e:
            st.error(f"Failed to initialize Gemini client: {e}")
            st.stop()

    # --- Empty state --------------------------------------------------------
    if not st.session_state.messages:
        st.markdown(
            """
            <div style="text-align:center; padding: 3rem 1rem; color:#6b7280;">
                <div style="font-size:2.5rem; margin-bottom:0.8rem;">💚</div>
                <div style="font-size:1.05rem; margin-bottom:0.3rem; color:#9ca3af;">
                    What's on your mind?
                </div>
                <div style="font-size:0.85rem;">
                    Anything from a passing thought to something that's been weighing on you.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Render conversation history ----------------------------------------
    for msg in st.session_state.messages:
        role = msg["role"]
        avatar = "🧑" if role == "user" else "🧠"
        is_crisis = (role == "assistant"
                     and msg.get("result", {}).get("path") == "crisis")

        if is_crisis:
            st.markdown('<div class="crisis-bubble">', unsafe_allow_html=True)

        with st.chat_message(role, avatar=avatar):
            if role == "assistant":
                _render_assistant_message(msg["result"])
            else:
                st.markdown(msg["content"])

        if is_crisis:
            st.markdown('</div>', unsafe_allow_html=True)

    # --- Chat input ---------------------------------------------------------
    user_msg = st.chat_input("Share what's on your mind ...")
    if not user_msg:
        return

    # User turn
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_msg)
    st.session_state.messages.append({"role": "user", "content": user_msg})
    st.session_state.history.append({"role": "user", "content": user_msg})

    # Assistant turn — typing indicator + actual call
    with st.chat_message("assistant", avatar="🧠"):
        placeholder = st.empty()
        placeholder.markdown(
            '<div class="typing"><span></span><span></span><span></span></div>',
            unsafe_allow_html=True,
        )

        try:
            result = rag.respond(
                user_msg,
                history=st.session_state.history[:-1],
                crisis_threshold=settings["crisis_threshold"],
                k_counsel=settings["k_counsel"],
                k_kb=settings["k_kb"],
            )
        except Exception as e:
            placeholder.empty()
            if _is_quota_error(e):
                # Friendly UX: don't dump the raw 429. Tell the user how to
                # recover, and remember the state so the banner persists across
                # reruns until they paste a key or refresh.
                st.session_state.quota_blocked = True
                # Roll back the user's turn so they can retry without duplicates
                st.session_state.history.pop()
                st.session_state.messages.pop()
                # Force rerun so the sidebar re-opens with the highlighted
                # API key prompt visible.
                st.rerun()
            else:
                st.error(f"Something went wrong: {e}")
                st.session_state.history.pop()
                st.session_state.messages.pop()
            return

        placeholder.empty()
        _render_assistant_message(result)

    st.session_state.messages.append({"role": "assistant", "result": result})
    st.session_state.history.append({"role": "assistant", "content": result["text"]})

    # Trigger a rerun so the crisis-bubble wrapper is applied uniformly
    # to the just-added message on next render.
    time.sleep(0.05)
    st.rerun()


if __name__ == "__main__":
    main()