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

API key resolution order (highest priority first):
    1. User-pasted key in the sidebar (per-session, never written to disk)
    2. st.secrets["GEMINI_API_KEY"]  (Streamlit Cloud deployment)
    3. .env GEMINI_API_KEY            (local dev fallback / instructor key)
"""

import os

import streamlit as st

# Pull GEMINI_API_KEY from Streamlit secrets BEFORE importing rag so its
# import-time .env load picks it up via os.environ. This matters on
# Streamlit Cloud where there's no .env file.
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ.setdefault("GEMINI_API_KEY", st.secrets["GEMINI_API_KEY"])
except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
    # st.secrets may raise locally if no secrets.toml exists — that's fine,
    # we'll fall back to .env via rag.py's load_dotenv.
    pass

import rag  # noqa: E402  — must come after the secrets/env handling above


# ===========================================================================
# Page config
# ===========================================================================
st.set_page_config(
    page_title="Mind Companion — Mental Health Support",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ===========================================================================
# Component caching — heavy loads happen once per Streamlit session
# ===========================================================================
@st.cache_resource(show_spinner="Loading models (one-time, ~10 seconds) ...")
def _load_components():
    """Trigger lazy load of rag.py components and cache for the session."""
    return rag.get_components()


# ===========================================================================
# API key resolution
# ===========================================================================
def _resolve_api_key(user_pasted_key: str) -> tuple[str | None, str]:
    """
    Returns (api_key, source_label) where source_label is one of:
        "user", "secrets", "env", "missing".
    """
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
# Sidebar
# ===========================================================================
def render_sidebar() -> dict:
    """Render the sidebar and return the current settings dict."""
    with st.sidebar:
        st.markdown("## 🧠 Mind Companion")
        st.caption("An empathic AI support companion — not a therapist.")

        st.divider()

        # --- API key block --------------------------------------------------
        st.markdown("### 🔑 Gemini API Key")
        st.caption(
            "Paste your own key for the best experience. "
            "If left blank, the app will use a fallback key (rate limits apply)."
        )
        user_key = st.text_input(
            "Your Gemini API key",
            type="password",
            placeholder="AIza...",
            label_visibility="collapsed",
            key="user_api_key",
        )
        st.caption(
            "Get a free key at "
            "[aistudio.google.com/apikey](https://aistudio.google.com/apikey). "
            "Your key is kept only for this browser session."
        )

        st.divider()

        # --- Tuning controls ------------------------------------------------
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
                "but also more false positives). 0.5 is the default."
            ),
        )

        st.divider()

        # --- Conversation controls -----------------------------------------
        st.markdown("### 💬 Conversation")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.history = []
            st.session_state.messages = []
            st.rerun()

        st.caption(f"Messages so far: {len(st.session_state.get('history', []))}")

        st.divider()

        # --- Always-visible crisis resources -------------------------------
        st.markdown("### 🆘 If you're in crisis (Australia)")
        st.markdown(
            "**000** — immediate danger\n\n"
            "**Lifeline** — 13 11 14 (24/7)\n\n"
            "**Beyond Blue** — 1300 22 4636\n\n"
            "**Suicide Call Back** — 1300 659 467\n\n"
            "**13YARN** — 13 92 76\n\n"
            "**Kids Helpline** — 1800 55 1800"
        )

        st.divider()

        # --- About ----------------------------------------------------------
        with st.expander("ℹ️ About this app"):
            st.markdown(
                "This is a student project demonstrating retrieval-augmented "
                "generation (RAG) for mental health support. It combines:\n\n"
                "- a fine-tuned **DistilBERT crisis classifier** that "
                "routes urgent messages to safety resources,\n"
                "- a **ChromaDB** vector store over Counsel Chat Q&A and "
                "curated CBT/mindfulness/psychoed entries,\n"
                "- **Google Gemini 2.5 Flash** for empathic, grounded replies.\n\n"
                "**This tool does not provide diagnosis or treatment.** "
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
# Message rendering helpers
# ===========================================================================
def _render_assistant_message(result: dict) -> None:
    """Render a stored assistant turn from session_state, including extras."""
    is_crisis = result.get("path") == "crisis"

    if is_crisis:
        # Subtle indicator that the safety branch fired
        st.warning(
            "Safety resources are shown below because this message was "
            "routed through our crisis support pathway."
        )

    st.markdown(result["text"])

    # Sources (collapsed by default) — only for RAG path with citations
    sources = result.get("sources") or []
    if sources:
        with st.expander(f"📎 Sources cited ({len(sources)})"):
            for i, s in enumerate(sources, 1):
                src     = s.get("source", "?")
                topic   = s.get("topic_or_category", "?")
                snippet = s.get("text", "")[:200]
                sim     = s.get("similarity", 0.0)
                badge   = "💬 Counsel Chat" if src == "counsel_chat" else "📘 Knowledge Base"
                st.markdown(
                    f"**{i}. {badge}** — *{topic}* "
                    f"(similarity {sim:.2f})\n\n"
                    f"> {snippet}"
                )

    # Classifier info (collapsed) — useful for the demo
    classifier = result.get("classifier")
    if classifier:
        with st.expander("🔬 Classifier output"):
            st.json(classifier)


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    # --- Header -------------------------------------------------------------
    st.title("🧠 Mind Companion")
    st.caption(
        "A retrieval-augmented support companion. Not a therapist, not a "
        "diagnosis — a thoughtful presence that listens and shares "
        "evidence-informed ideas."
    )

    # --- Init session state -------------------------------------------------
    # `history` is what we pass into rag.respond (just role+content).
    # `messages` is what we render (full result dicts for assistant turns).
    if "history" not in st.session_state:
        st.session_state.history = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "active_key_source" not in st.session_state:
        st.session_state.active_key_source = None

    # --- Sidebar ------------------------------------------------------------
    settings = render_sidebar()

    # --- Resolve API key ----------------------------------------------------
    api_key, key_source = _resolve_api_key(settings["user_key"])
    if api_key is None:
        st.error(
            "No Gemini API key available. Please paste one in the sidebar, "
            "or set GEMINI_API_KEY in .env / Streamlit secrets."
        )
        st.stop()

    # If the key source changed since last run, swap the Gemini client.
    # (rag.set_gemini_client triggers component load on first call.)
    if key_source != st.session_state.active_key_source or key_source == "user":
        try:
            rag.set_gemini_client(api_key)
            st.session_state.active_key_source = key_source
        except Exception as e:
            st.error(f"Failed to initialize Gemini client: {e}")
            st.stop()

    # --- Trigger heavy component load (cached) ------------------------------
    try:
        _load_components()
    except FileNotFoundError as e:
        st.error(
            f"Required artifacts missing: {e}\n\n"
            "Make sure you've run dataset.py, clean.py, download_model.py, "
            "and the indexing notebook cells before launching the app."
        )
        st.stop()

    # --- Render conversation history ----------------------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                _render_assistant_message(msg["result"])
            else:
                st.markdown(msg["content"])

    # --- Chat input ---------------------------------------------------------
    user_msg = st.chat_input("How are you feeling today?")
    if not user_msg:
        return

    # Show user's turn immediately
    with st.chat_message("user"):
        st.markdown(user_msg)
    st.session_state.messages.append({"role": "user", "content": user_msg})
    st.session_state.history.append({"role": "user", "content": user_msg})

    # Generate assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking ..."):
            try:
                result = rag.respond(
                    user_msg,
                    history=st.session_state.history[:-1],  # exclude the just-added user msg
                    crisis_threshold=settings["crisis_threshold"],
                    k_counsel=settings["k_counsel"],
                    k_kb=settings["k_kb"],
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                # Don't append a broken assistant turn to history
                st.session_state.history.pop()  # remove the user msg too
                return

        _render_assistant_message(result)

    # Persist the assistant turn
    st.session_state.messages.append({"role": "assistant", "result": result})
    st.session_state.history.append({"role": "assistant", "content": result["text"]})


if __name__ == "__main__":
    main()