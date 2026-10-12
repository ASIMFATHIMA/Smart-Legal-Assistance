"""Smart Legal Assistance - RAG Pipeline

This module centralizes query understanding + retrieval + reranking and
returns a structured, legal-first answer.

NOTE:
- The current Streamlit UI uses `EnhancedRAGEngine` directly.
- This pipeline is provided as a production-ready orchestrator for future
  wiring without breaking existing behavior.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from src.retriever import Retriever


_SECTION_RE = re.compile(r"\b(?:ipc|bns)?\s*(?:section\s*)?(\d{1,4}[A-Z]{0,3})\b", re.IGNORECASE)


def detect_query_type(query: str) -> Tuple[str, Optional[str]]:
    q = query.lower()
    # Prefer explicit prefixes
    if "ipc" in q or "indian penal code" in q:
        m = re.search(r"\bipc\b.*?(?:section\s*)?(\d{1,4}[a-z]{0,3})", q, re.IGNORECASE)
        if m:
            return "ipc_section", m.group(1).upper()
    if "bns" in q or "bharatiya nyaya sanhita" in q:
        m = re.search(r"\bbns\b.*?(?:section\s*)?(\d{1,4}[a-z]{0,3})", q, re.IGNORECASE)
        if m:
            return "bns_section", m.group(1).upper()

    # Generic 'Section 302' / 'IPC 302'
    m = _SECTION_RE.search(q)
    if m:
        return "section", m.group(1).upper()

    # Crime name fallback: not implemented here (metadata handles it)
    return "unknown", None


def _format_from_chunk(chunk: Dict[str, Any]) -> str:
    ipc_section = chunk.get("ipc_section") or chunk.get("section_number") or "N/A"
    bns_section = chunk.get("bns_section") or "N/A"

    # Some ingestion paths store crime under crime or crime_name
    crime = chunk.get("crime") or chunk.get("crime_name") or "N/A"
    description = chunk.get("description") or "N/A"
    ingredients = chunk.get("ingredients") or "N/A"
    punishment = chunk.get("punishment") or "N/A"
    important_cases = chunk.get("important_cases") or "N/A"

    return (
        f"IPC Section: {ipc_section}\n\n"
        f"Crime:\n{crime}\n\n"
        f"Description:\n{description}\n\n"
        f"Ingredients:\n{ingredients}\n\n"
        f"Punishment:\n{punishment}\n\n"
        f"Important Cases:\n{important_cases}\n\n"
        f"BNS Equivalent:\n{bns_section}"
    )


class RAGPipeline:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def answer(self, user_query: str, top_k: int = 20) -> Dict[str, Any]:
        query_type, section_number = detect_query_type(user_query)
        retrieved = self.retriever.retrieve(user_query, top_k=top_k)

        if not retrieved:
            return {
                "answer": "I couldn't find sufficient information in the uploaded legal documents.",
                "sources": [],
                "retrieved_chunks": [],
                "query_type": query_type,
                "detected_section": section_number,
            }

        # Deterministic legal formatting from top chunk metadata
        top = retrieved[0]
        answer = _format_from_chunk(top)

        sources = [
            {
                "chunk_id": c.get("chunk_id"),
                "document": c.get("source_document"),
                "page_number": c.get("page_number"),
                "section_number": c.get("section_number") or c.get("ipc_section"),
                "similarity": float(c.get("similarity", 0.0) or 0.0),
                "crime_name": c.get("crime_name") or c.get("crime"),
            }
            for c in retrieved[:5]
        ]

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved,
            "query_type": query_type,
            "detected_section": section_number,
        }
