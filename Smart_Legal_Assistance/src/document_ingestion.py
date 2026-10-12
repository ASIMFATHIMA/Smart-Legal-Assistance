"""Smart Legal Assistance – Document Ingestion
===============================================
Fixes applied (2026):
  - Pass chunk_size=700, overlap_size=100 to DocumentChunker (was defaulting to 1000).
  - Log per-page extraction count and text length.
  - Log embedding shape after generation.
  - Log FAISS vector count after indexing.
  - Rebuild BM25 in Retriever after ingestion so new documents are searchable.
  - SHA-256 hash-based duplicate detection: compute hash from raw PDF bytes and
    check against existing FAISS metadata before any extraction/embedding work.
  - file_hash stored in chunk metadata for future hash lookups.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Optional

import numpy as np

from src.knowledge_builder import KnowledgeBuilder
from src.embedding_generator import EmbeddingGenerator
from src.vector_store import VectorStore

logger = logging.getLogger("rag_engine")


class DocumentIngestion:
    def __init__(
        self,
        vector_store: VectorStore,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.vector_store = vector_store
        self.builder = KnowledgeBuilder()
        self.embedder = EmbeddingGenerator(model_name)
        # Retriever reference – set externally after construction so BM25
        # can be rebuilt after every new ingestion.
        self._retriever: Optional[object] = None

    def set_retriever(self, retriever: object) -> None:
        """Attach retriever so BM25 can be rebuilt after ingestion."""
        self._retriever = retriever

    def process_pdf(self, file_path: str, progress_callback=None) -> int:
        """Extract, chunk, embed, and index a PDF file.

        Returns the number of chunks successfully added to the FAISS index,
        or -1 if the file is a duplicate (already indexed).

        Duplicate detection order:
          1. SHA-256 hash of the raw PDF bytes (catches renamed duplicates).
          2. Filename match in FAISS metadata (fast in-memory check).
          3. Filename match in the database (persistent check).
        """
        filename = os.path.basename(file_path)
        logger.info("[Ingestion] Processing: %s", filename)

        if progress_callback:
            progress_callback(0.05, f"Computing SHA-256 hash for: {filename}")

        # ── Compute SHA-256 hash from raw bytes ───────────────────────────────
        try:
            with open(file_path, "rb") as fh:
                file_hash = hashlib.sha256(fh.read()).hexdigest()
            logger.info("[Ingestion] SHA-256 of %s: %s…", filename, file_hash[:12])
        except OSError as exc:
            logger.error("[Ingestion] Could not read %s for hashing: %s", filename, exc)
            file_hash = ""

        if progress_callback:
            progress_callback(0.1, f"Checking for duplicates: {filename}")

        # ── Duplicate detection: SHA-256 hash in FAISS metadata ───────────────
        if file_hash and self.vector_store.metadata:
            for m in self.vector_store.metadata:
                if m.get("file_hash") == file_hash:
                    logger.info(
                        "[Ingestion] Duplicate by SHA-256 hash in FAISS metadata: %s", filename
                    )
                    return -1

        # ── Duplicate detection: filename in FAISS metadata ───────────────────
        if self.vector_store.metadata:
            for m in self.vector_store.metadata:
                if m.get("source_document") == filename:
                    logger.info(
                        "[Ingestion] Duplicate by filename in FAISS metadata: %s", filename
                    )
                    return -1

        # ── Duplicate detection: filename in database ─────────────────────────
        from database.connection import SessionLocal
        from database.models import Document
        db = SessionLocal()
        try:
            existing = db.query(Document).filter(Document.filename == filename).first()
            if existing:
                logger.info("[Ingestion] Duplicate by filename in database: %s", filename)
                return -1
        except Exception as exc:
            logger.warning("[Ingestion] Database duplicate check failed: %s", exc)
        finally:
            db.close()

        # ── Structured extraction for the reference matrix ───────────────────
        if "ipc_bns_reference_matrix" in filename.lower():
            records = self.builder.build_knowledge_base(file_path)
            if not records:
                logger.warning(f"[Ingestion] No records generated from {filename}")
                return 0

            chunks = []
            for r in records:
                text = (
                    f"IPC Section: {r['ipc_section']}\n"
                    f"Crime: {r['crime']}\n"
                    f"Description: {r['description']}\n"
                    f"Ingredients: {r['ingredients']}\n"
                    f"Punishment: {r['punishment']}\n"
                    f"Important Cases: {r['important_cases']}\n"
                    f"BNS Section: {r['bns_section']}"
                )
                chunks.append(
                    {
                        "chunk_id": f"{filename}_{r['ipc_section']}",
                        "source_document": filename,
                        "file_hash": file_hash,
                        "text": text,
                        "section_number": r["ipc_section"],
                        "crime_name": r["crime"],
                        "bns_section": r["bns_section"],
                        "keywords": r.get("keywords", []),
                        "page_number": r.get("page_number"),
                    }
                )

        else:
            # ── Generic PDF extraction (PyMuPDF → pdfplumber → PyPDF2) ───────
            from src.pdf_loader import PDFLoader
            from src.chunking import DocumentChunker

            loader = PDFLoader()
            # FIXED: chunk_size=700 (was defaulting to 1000), overlap=100
            chunker = DocumentChunker(chunk_size=700, overlap_size=100)

            pages_text = loader.extract_text(file_path)
            if not pages_text:
                logger.warning(f"[Ingestion] No text extracted from {filename}")
                return 0

            # Log extraction stats per page
            total_chars = sum(len(p.get("text", "")) for p in pages_text if isinstance(p, dict))
            logger.info(
                "[Ingestion] Extracted %d pages, %s chars from %s",
                len(pages_text),
                f"{total_chars:,}",
                filename,
            )

            # Detect error block from PDFLoader
            first_text = pages_text[0].get("text") if isinstance(pages_text[0], dict) else None
            if isinstance(first_text, str) and first_text.startswith("🔍 Error Detected"):
                chunks = [
                    {
                        "chunk_id": f"{filename}_error",
                        "source_document": filename,
                        "text": first_text,
                        "section_number": "N/A",
                        "crime_name": "N/A",
                        "bns_section": "N/A",
                        "keywords": [],
                        "page_number": 1,
                    }
                ]
            else:
                # Build a page-aware combined text preserving page numbers
                page_chunks: list = []
                for page in pages_text:
                    if not isinstance(page, dict):
                        continue
                    pg_text = page.get("text", "").strip()
                    pg_num = page.get("page_number", 0)
                    if not pg_text:
                        continue  # skip empty pages
                    pg_chunks = chunker.chunk_document(
                        pg_text, doc_id=f"{filename}_p{pg_num}", doc_source=filename
                    )
                    # Annotate each chunk with its page number
                    for ch in pg_chunks:
                        ch["page_number"] = pg_num
                        ch["file_hash"] = file_hash  # store hash in every chunk
                    page_chunks.extend(pg_chunks)

                chunks = page_chunks if page_chunks else chunker.chunk_document(
                    "\n".join(p.get("text", "") for p in pages_text if isinstance(p, dict)),
                    doc_id=filename,
                    doc_source=filename,
                )

        # ── Validate metadata fields ──────────────────────────────────────────
        required_keys = [
            "chunk_id", "source_document", "text",
            "section_number", "crime_name", "bns_section",
        ]
        validated_chunks = []
        for c in chunks:
            if not all(k in c and c[k] is not None for k in required_keys):
                logger.debug(f"[Ingestion] Rejected incomplete chunk: {c.get('chunk_id')}")
                continue
            if not c.get("text", "").strip():
                logger.debug(f"[Ingestion] Rejected empty-text chunk: {c.get('chunk_id')}")
                continue
            if "keywords" not in c:
                c["keywords"] = []
            if "page_number" not in c:
                c["page_number"] = None
            validated_chunks.append(c)

        chunks = validated_chunks
        logger.info("[Ingestion] Validated chunks: %d", len(chunks))

        if not chunks:
            logger.warning(f"[Ingestion] No valid chunks for {filename}")
            return 0

        # ── Generate embeddings ───────────────────────────────────────────────
        if progress_callback:
            progress_callback(0.6, f"Generating vector embeddings for {len(chunks)} chunks...")
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.generate_embeddings(texts)

        if embeddings is None or (hasattr(embeddings, "size") and embeddings.size == 0):
            logger.error(f"[Ingestion] Embedding generation failed for {filename}")
            return 0

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            logger.error(
                f"[Ingestion] Bad embedding shape {embeddings.shape} for {filename}"
            )
            return 0

        logger.info(
            "[Ingestion] Embedding shape: %s (chunks=%d, dim=%d)",
            embeddings.shape,
            embeddings.shape[0],
            embeddings.shape[1],
        )

        # ── Add to FAISS and save ─────────────────────────────────────────────
        if progress_callback:
            progress_callback(0.8, "Writing embeddings to FAISS database...")
        self.vector_store.add_embeddings(embeddings, chunks)
        self.vector_store.save_index()

        vector_count = (
            int(self.vector_store.index.ntotal)
            if self.vector_store.index is not None
            else 0
        )
        logger.info("[Ingestion] FAISS total vectors after ingestion: %d", vector_count)

        # ── Rebuild BM25 so new document is searchable ───────────────────────
        if progress_callback:
            progress_callback(0.9, "Rebuilding BM25 keyword indices...")
        if self._retriever is not None:
            try:
                self._retriever.rebuild_bm25()
                logger.info("[Ingestion] BM25 index rebuilt.")
            except Exception as exc:
                logger.warning(f"[Ingestion] BM25 rebuild failed: {exc}")

        if progress_callback:
            progress_callback(1.0, f"Successfully indexed {len(chunks)} chunks!")
        return len(chunks)

    def load_default_documents(self, default_dir: str = "legal_pdfs") -> None:
        """Ingest all PDFs in default_dir if FAISS index is empty."""
        if self.vector_store.index is not None and self.vector_store.index.ntotal > 0:
            logger.info(
                f"[Ingestion] FAISS index already has "
                f"{self.vector_store.index.ntotal} vectors — skipping default load."
            )
            return

        if not os.path.exists(default_dir):
            logger.warning(f"[Ingestion] Default PDF directory not found: {default_dir}")
            return

        pdf_files = [f for f in os.listdir(default_dir) if f.lower().endswith(".pdf")]
        logger.info(f"[Ingestion] Loading {len(pdf_files)} default PDF(s) from {default_dir}")

        for file in pdf_files:
            file_path = os.path.join(default_dir, file)
            try:
                count = self.process_pdf(file_path)
                logger.info(f"[Ingestion] {file}: {count} chunks ingested.")
            except Exception as exc:
                logger.warning(f"[Ingestion] Failed to load {file}: {exc}")
                print(f"[Ingestion] Warning: Failed to load {file}: {exc}")
