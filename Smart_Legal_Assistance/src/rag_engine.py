"""Smart Legal Assistance - Production RAG Engine (Structured Legal Extractor)

Fixes Wrong Legal Answer Retrieval
-------------------------------------
This project has two RAG implementations:
- `RAGEngine` (this file)
- `EnhancedRAGEngine` (`src/enhanced_rag_engine.py`)

Production requirement: *any* entrypoint using `RAGEngine` must:
- retrieve Top-K chunks via FAISS
- select the chunk that contains an actual legal record
- never parse table headers as values
- validate extracted fields and reject header-placeholder collisions
- format output exactly in the required template

This file provides a production-correct implementation by delegating to the
same deterministic extraction pipeline used in `EnhancedRAGEngine`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.enhanced_rag_engine import (
    detect_query_type,
    select_best_record,
    format_record,
)
from src.retriever import Retriever


class RAGEngine:
    """Compatibility wrapper.

    The Streamlit UI primarily uses `EnhancedRAGEngine`, but other modules may
    still import/instantiate `RAGEngine`. This wrapper ensures consistent,
    production-safe behaviour.
    """

    def __init__(self, retriever: Retriever, top_k: int = 5):
        self.retriever = retriever
        self.top_k = top_k

    def retrieve_context(self, question: str) -> List[Dict[str, Any]]:
        return self.retriever.retrieve_top_k(question, top_k=self.top_k)

    def answer_question(self, question: str) -> Dict[str, Any]:
        retrieved = self.retrieve_context(question)
        if not retrieved:
            return {
                "answer": "No relevant legal information found.",
                "sources": [],
                "similarities": [],
                "retrieved_chunks": [],
                "query_type": detect_query_type(question)[0],
                "detected_section": detect_query_type(question)[1],
            }

        best = select_best_record(question, retrieved)
        query_type, section_number = detect_query_type(question)

        if not best:
            return {
                "answer": "No relevant legal information found.",
                "sources": [],
                "similarities": [c.get("similarity") for c in retrieved],
                "retrieved_chunks": retrieved,
                "query_type": query_type,
                "detected_section": section_number,
            }

        record = best["record"]
        answer = format_record(record)

        used_chunk = best["source_chunk"]
        similarity = float(
            used_chunk.get("similarity", 0.0)
            or used_chunk.get("faiss_similarity", 0.0)
            or 0.0
        )

        sources = [
            {
                "chunk_id": used_chunk.get("chunk_id"),
                "source_document": used_chunk.get("source_document"),
                "page_number": used_chunk.get("page_number"),
                "section_number": used_chunk.get("section_number")
                or used_chunk.get("ipc_section")
                or record.get("ipc_section"),
                "similarity": similarity,
                "crime_name": used_chunk.get("crime_name") or record.get("crime"),
            }
        ]

        return {
            "answer": answer,
            "sources": sources,
            "similarities": [c.get("similarity") for c in retrieved],
            "retrieved_chunks": retrieved,
            "query_type": query_type,
            "detected_section": section_number,
        }


__all__ = ["RAGEngine"]

