"""Smart Legal Assistance – Streamlit Entry Point
==================================================
Rerun-prevention fixes applied:
  1. `init_rag_system` is decorated with `@st.cache_resource` – runs ONCE
     per interpreter lifetime, never again on widget interaction or page change.
  2. `st.sidebar.info()` is NOT called inside the cached function (doing so
     inside a cached resource causes Streamlit warnings and may trigger reruns).
     Instead, a `session_state` flag `_rag_init_shown` gates a one-time banner.
  3. `logging.basicConfig` configured at module level so all sub-loggers
     (rag_engine, retriever, ingestion) write to both console and logs/app.log.
  4. `init_session()` is already idempotent (checks before setting), so no
     further guard is needed there.
"""

import sys
import os
import logging


# ─── Logging: configure once at module level ─────────────────────────────────
# All child loggers (rag_engine, retriever, ingestion…) inherit this config.
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join("logs", "app.log"), encoding="utf-8"),
    ],
)
_startup_logger = logging.getLogger("startup")
_startup_logger.info("=== Smart Legal Assistance starting ===")

import streamlit as st

# ─── Load env variables early ────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "config", ".env"))
except ModuleNotFoundError:
    pass  # python-dotenv not installed; continue without .env

# ─── Ensure DB tables exist (runs once at import; idempotent) ────────────────
from database.connection import engine, Base
Base.metadata.create_all(bind=engine)

from auth.session_manager import init_session, is_authenticated, logout_user
from auth.login import render_login
from auth.register import render_register
from ui.dashboard import render_dashboard
from ui.upload import render_upload
from ui.chatbot import render_chatbot
from ui.explorer import render_explorer
from ui.case_summary import render_case_summary
from ui.feedback import render_feedback
from ui.settings import render_settings
from ui.diagnostics import render_diagnostics


# ─── RAG imports ─────────────────────────────────────────────────────────────
from src.vector_store import VectorStore
from src.document_ingestion import DocumentIngestion
from src.retriever import Retriever
from src.enhanced_rag_engine import EnhancedRAGEngine
from src.enhanced_chatbot import EnhancedChatbot

st.set_page_config(
    page_title="Smart Legal Assistance",
    page_icon="⚖️",
    layout="wide",
)


# ─── Heavy resource initialisation (cached for the lifetime of the server) ───
@st.cache_resource(show_spinner=False)
def init_rag_system(version: int = 7):
    """
    Initialise the full RAG pipeline ONCE and cache the result.

    IMPORTANT: Do NOT call any st.* widget functions here – cached resource
    functions run outside the normal render cycle and will cause warnings /
    duplicate widget keys.  Use logging instead.
    """
    logger = logging.getLogger("startup")
    logger.info("[RAG] Initialising RAG system (version=%d)…", version)

    vs = VectorStore(
        index_path="models/faiss_index.bin",
        metadata_path="models/metadata.pkl",
    )

    # Try to load existing persisted index
    try:
        vs.load_index()
        logger.info(
            "[RAG] Loaded existing FAISS index (%d vectors).",
            vs.index.ntotal if vs.index else 0,
        )
    except Exception as exc:
        logger.warning("[RAG] No existing FAISS index found (%s). Will build on upload.", exc)

    ingestion = DocumentIngestion(vs)
    retriever = Retriever(vs, ingestion.embedder)

    # Wire retriever into ingestion so BM25 rebuilds after each new upload
    ingestion.set_retriever(retriever)

    # Process default documents on startup ONLY if index is empty
    ingestion.load_default_documents()

    # Rebuild BM25 after potential new documents
    retriever.rebuild_bm25()
    logger.info("[RAG] BM25 rebuilt. System ready.")

    rag_engine = EnhancedRAGEngine(retriever, model_name="qwen2.5:7b")
    chatbot = EnhancedChatbot(rag_engine)

    logger.info("[RAG] EnhancedRAGEngine + EnhancedChatbot ready.")
    return ingestion, chatbot


# ─── Main application ─────────────────────────────────────────────────────────
def main() -> None:
    # 1. Initialise auth session keys (idempotent – skips if already set)
    init_session()

    # 2. Authentication gate
    if not is_authenticated():
        auth_mode = st.sidebar.radio("Authentication", ["Login", "Register"])
        if auth_mode == "Login":
            render_login()
        else:
            render_register()
        return

    # 3. Authenticated: show app
    st.sidebar.title(f"Welcome, {st.session_state['username']}")

    # 4. Initialise RAG once; show one-time banner to avoid rerun on every page
    ingestion, chatbot = init_rag_system()

    # Show the "RAG ready" banner only once per session
    if not st.session_state.get("_rag_init_shown", False):
        st.sidebar.success("✅ RAG system ready")
        st.session_state["_rag_init_shown"] = True

    # Requirement #4/#7/#8: validate Piper installation once and store
    # whether voice is available. If missing, chatbot continues normally.
    if "voice_available" not in st.session_state:
        try:
            from src.text_to_speech import validate_piper_installation

            ok, _reason = validate_piper_installation()
            st.session_state["voice_available"] = bool(ok)
        except Exception:
            st.session_state["voice_available"] = False


    # 5. Navigation
    pages = {
        "Dashboard":         render_dashboard,
        "Upload PDF":        lambda: render_upload(ingestion),
        "Chatbot":           lambda: render_chatbot(chatbot),
        "Document Explorer": render_explorer,
        "Case Summary":      render_case_summary,
        "Feedback":          render_feedback,
        "Settings":          render_settings,
        "Diagnostics":       render_diagnostics,
    }

    selection = st.sidebar.radio("Navigate", list(pages.keys()))

    # 6. Render selected page
    pages[selection]()

    # 7. Logout button
    st.sidebar.divider()
    if st.sidebar.button("Logout", key="logout_btn"):
        logout_user()
        st.rerun()


if __name__ == "__main__":
    main()
