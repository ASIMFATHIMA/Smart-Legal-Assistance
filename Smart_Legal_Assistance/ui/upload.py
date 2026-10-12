"""Smart Legal Assistance – Upload UI
======================================
Rerun-prevention fixes applied (2026):
  1. _upload_processing flag in session_state prevents Process Documents from
     being invoked twice if a Streamlit rerun fires mid-upload (e.g. from a
     widget interaction while the spinner is visible).
  2. SHA-256 hash of each uploaded file is computed before processing.  The
     hash is stored in session_state["processed_file_hashes"] so that if the
     same file is uploaded again in the same session it is skipped immediately
     without touching the ingestion pipeline.
  3. Document DB record is only written for newly indexed files (num_chunks > 0),
     not for duplicates (process_pdf returns -1).
  4. All errors are logged via the standard logger instead of bare print().
"""

import hashlib
import logging
import os

import streamlit as st

from database.connection import get_db
from database.models import Document

logger = logging.getLogger("rag_engine")


def _compute_file_hash(file_bytes: bytes) -> str:
    """Return the SHA-256 hex digest of the given bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def render_upload(ingestion=None):
    st.title("Upload Legal Documents")
    st.write(
        "Upload PDFs to extract text, generate embeddings, and update the "
        "FAISS vector database automatically."
    )

    # Initialise session-state keys (idempotent)
    if "processed_file_hashes" not in st.session_state:
        # Maps sha256_hex → filename for all files processed this session
        st.session_state["processed_file_hashes"] = {}
    if "_upload_processing" not in st.session_state:
        st.session_state["_upload_processing"] = False

    uploaded_files = st.file_uploader(
        "Choose PDF files", type="pdf", accept_multiple_files=True
    )

    # FIX: If processing is already in progress (unlikely but possible on rapid
    # reruns) show a notice and bail out early.
    if st.session_state["_upload_processing"]:
        st.info("⏳ Processing in progress… please wait.")
        return

    if not st.button("Process Documents", key="process_docs_btn"):
        return

    if not ingestion:
        st.error("Document ingestion pipeline is not initialized.")
        return

    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")
        return

    # ── Lock to prevent duplicate processing ─────────────────────────────────
    st.session_state["_upload_processing"] = True

    upload_dir = os.getenv("UPLOAD_FOLDER", "legal_pdfs/")
    os.makedirs(upload_dir, exist_ok=True)

    db_generator = get_db()
    db = next(db_generator)

    try:
        newly_indexed: list[str] = []
        already_seen: list[str] = []
        failed: list[str] = []

        for file in uploaded_files:
            file_bytes = file.getbuffer()

            # FIX: Compute SHA-256 before any disk I/O.
            file_hash = _compute_file_hash(bytes(file_bytes))

            if file_hash in st.session_state["processed_file_hashes"]:
                # Same binary content already processed in this session.
                orig_name = st.session_state["processed_file_hashes"][file_hash]
                st.info(
                    f"⏭️ **{file.name}** skipped – identical content already indexed "
                    f"as `{orig_name}` this session."
                )
                already_seen.append(file.name)
                logger.info(
                    "[Upload] Skipping %s — SHA-256 %s already in session cache.",
                    file.name,
                    file_hash[:12],
                )
                continue

            file_path = os.path.join(upload_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file_bytes)

            def _progress(pct, msg, _fname=file.name):
                logger.info("[Upload] %s – %.0f%% – %s", _fname, pct * 100, msg)

            with st.spinner(
                f"Processing **{file.name}** "
                "(Extracting text → Chunking → Generating embeddings → FAISS)…"
            ):
                num_chunks = ingestion.process_pdf(file_path, progress_callback=_progress)

            if num_chunks == -1:
                # Duplicate detected by the ingestion pipeline (filename / DB check)
                st.info(f"⏭️ **{file.name}** skipped – already indexed in the vector store.")
                already_seen.append(file.name)
                logger.info("[Upload] Duplicate detected by ingestion pipeline: %s", file.name)
                # Still register in session hash cache to avoid re-checking next time
                st.session_state["processed_file_hashes"][file_hash] = file.name
                continue

            if num_chunks == 0:
                st.warning(f"⚠️ **{file.name}** produced 0 valid chunks. Nothing indexed.")
                failed.append(file.name)
                logger.warning("[Upload] %s produced 0 chunks.", file.name)
                continue

            # New document successfully indexed
            st.session_state["processed_file_hashes"][file_hash] = file.name
            newly_indexed.append(file.name)
            logger.info("[Upload] %s indexed: %d chunks.", file.name, num_chunks)

            # Save to DB
            pages = max(1, num_chunks // 2)  # rough page estimate
            new_doc = Document(filename=file.name, pages=pages, chunk_count=num_chunks)
            db.add(new_doc)

        db.commit()

        # ── Summary ──────────────────────────────────────────────────────────
        if newly_indexed:
            st.success(
                f"✅ Successfully indexed **{len(newly_indexed)}** document(s): "
                + ", ".join(f"`{n}`" for n in newly_indexed)
                + "\n\nThe chatbot can now answer from these documents!"
            )
        if already_seen:
            st.info(
                f"ℹ️ {len(already_seen)} document(s) were already indexed and skipped."
            )
        if failed:
            st.warning(
                f"⚠️ {len(failed)} document(s) could not be indexed: "
                + ", ".join(f"`{n}`" for n in failed)
            )

    except Exception as e:
        db.rollback()

        # Diagnostic block: show real error + FAISS runtime info
        import traceback

        diag_lines: list[str] = []
        try:
            import sys as _sys
            diag_lines.append(f"Python executable: {_sys.executable}")
        except Exception:
            pass
        try:
            import faiss as _faiss
            diag_lines.append(f"FAISS module: {_faiss}")
            diag_lines.append(f"FAISS version: {getattr(_faiss, '__version__', 'Unknown')}")
            diag_lines.append(f"Has IndexFlatL2: {hasattr(_faiss, 'IndexFlatL2')}")
        except Exception as _faiss_e:
            diag_lines.append(f"FAISS import failed in Streamlit: {repr(_faiss_e)}")

        logger.exception("[Upload] Unhandled error during processing: %s", e)
        st.error(
            "🔍 Upload Error Analysis\n\n"
            "Error Type: FAISS/Vector Store Error\n\n"
            f"Real exception: {repr(e)}\n\n"
            + ("\n".join(diag_lines) + "\n\n")
            + "Traceback:\n"
            + traceback.format_exc()
        )

    finally:
        db_generator.close()
        # FIX: Always release the processing lock, even on exception.
        st.session_state["_upload_processing"] = False
