"""Smart Legal Assistance - RAG Engine Enhancements

Deterministic extraction of legal records from retrieved chunks.

Requirements implemented:
- Search every retrieved chunk; never auto-use the first chunk.
- Ignore non-record chunks (tables/matrices/headers/OCR garbage/random text).
- Ignore header placeholder values exactly equal to:
  DESCRIPTION, PUNISHMENT, KEYWORDS, BNS EQUIVALENT, CRIME, IPC SECTION.
- Regex parsing with required labels.
- Fallback table-row parsing: never parse first row (header), use next row as data.
- Validate extracted values and reject swapped/header selections.
- Score chunks and return highest-scoring valid record.
- If no valid legal record exists: "No relevant legal information found."
- Sources include only the document actually used for the answer.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from src.retriever import Retriever


# ---------------------------------------------------------------------------
# Query type detection (kept compatible)
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"\b(?:ipc|bns)?\s*(?:section\s*)?(\d{1,4}[A-Z]{0,3})\b",
    re.IGNORECASE,
)


def detect_query_type(query: str) -> Tuple[str, Optional[str]]:
    q = (query or "").lower()

    if "ipc" in q or "indian penal code" in q:
        m = re.search(r"\bipc\b.*?(?:section\s*)?(\d{1,4}[a-z]{0,3})", q, re.IGNORECASE)
        if m:
            return "ipc_section", m.group(1).upper()

    if "bns" in q or "bharatiya nyaya sanhita" in q:
        m = re.search(
            r"\bbns\b.*?(?:section\s*)?(\d{1,4}[a-z]{0,3})",
            q,
            re.IGNORECASE,
        )
        if m:
            return "bns_section", m.group(1).upper()

    m = _SECTION_RE.search(q)
    if m:
        return "section", m.group(1).upper()

    return "unknown", None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

HEADER_PLACEHOLDERS = {
    "DESCRIPTION",
    "PUNISHMENT",
    "KEYWORDS",
    "BNS EQUIVALENT",
    "CRIME",
    "IPC SECTION",
}

FIELD_ORDER = [
    "ipc_section",
    "crime",
    "description",
    "ingredients",
    "punishment",
    "important_cases",
    "bns_section",
]

_LABEL_ALIASES: Dict[str, List[str]] = {
    "ipc_section": ["IPC Section"],
    "crime": ["Crime"],
    "description": ["Description"],
    "ingredients": ["Ingredients"],
    "punishment": ["Punishment"],
    "important_cases": ["Important Cases"],
    "bns_section": ["BNS Section", "BNS Equivalent"],
}

_REGEX_PATTERNS: Dict[str, List[str]] = {
    "ipc_section": [r"IPC Section\s*:\s*(.+)", r"IPC Section\s*:\s*(.+)$"],
    "crime": [r"Crime\s*:\s*(.+)", r"Crime\s*:\s*(.+)$"],
    "description": [r"Description\s*:\s*(.+)", r"Description\s*:\s*(.+)$"],
    "ingredients": [r"Ingredients\s*:\s*(.+)", r"Ingredients\s*:\s*(.+)$"],
    "punishment": [r"Punishment\s*:\s*(.+)", r"Punishment\s*:\s*(.+)$"],
    "important_cases": [r"Important Cases\s*:\s*(.+)", r"Important\s*Cases\s*:\s*(.+)$"],
    "bns_section": [
        r"BNS Section\s*:\s*(.+)",
        r"BNS Equivalent\s*:\s*(.+)",
    ],
}


def _get_detectable_text(chunk: Dict[str, Any]) -> str:
    parts: List[str] = []
    for k in ("text", "page_content", "content"):
        v = chunk.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v)
    doc = chunk.get("document")
    if isinstance(doc, str) and doc.strip():
        parts.append(doc)
    return "\n".join(parts).strip()


def _reject_placeholder_value(v: str) -> str:
    vv = (v or "").strip()
    if not vv:
        return "N/A"
    if vv.upper() in HEADER_PLACEHOLDERS:
        return "N/A"
    return vv


def _chunk_looks_like_non_record(chunk: Dict[str, Any]) -> bool:
    raw = _get_detectable_text(chunk)
    if not raw:
        return True

    t = raw.lower()
    blockers = [
        "property offences",
        "keywords",
        "matrix",
        "matrices",
        "table",
        "tables",
        "header",
        "headers",
        "ocr garbage",
        "ocr",
        "garbage",
        "random text",
    ]
    if any(b in t for b in blockers):
        # allow if it clearly has record labels
        if re.search(r"\bipc\b\s*section\b", raw, flags=re.IGNORECASE) and re.search(
            r"\bcrime\b\s*:\s*", raw, flags=re.IGNORECASE
        ):
            return False
        return True

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) <= 2:
        return True

    expected_label_regexes = [
        r"\bipc\s*section\b",
        r"\bcrime\b\s*:",
        r"\bdescription\b\s*:",
        r"\bingredients\b\s*:",
        r"\bpunishment\b\s*:",
        r"\bimportant\s*cases\b\s*:",
        r"\bbns\s*(section|equivalent)\b\s*:",
    ]

    if not any(re.search(p, raw, flags=re.IGNORECASE) for p in expected_label_regexes):
        return True

    return False


def parse_record_with_regex(raw_text: str) -> Dict[str, str]:
    record: Dict[str, str] = {k: "N/A" for k in FIELD_ORDER}

    for field in FIELD_ORDER:
        for pat in _REGEX_PATTERNS[field]:
            m = re.search(pat, raw_text, flags=re.IGNORECASE)
            if not m:
                continue
            val = _reject_placeholder_value(m.group(1))
            if val != "N/A":
                record[field] = val
                break

    # Special normalize bns_section if needed
    if record["bns_section"] != "N/A":
        record["bns_section"] = record["bns_section"].strip()

    return record


def parse_record_from_table_rows(raw_text: str) -> Dict[str, str]:
    """Fallback parsing by table rows.

    Never parse first row (header). Use next row as data.
    """

    record: Dict[str, str] = {k: "N/A" for k in FIELD_ORDER}

    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return record

    data_lines = lines[1:]  # skip header row

    # 1) Try label:value within data lines
    for ln in data_lines:
        for field in FIELD_ORDER:
            for pat in _REGEX_PATTERNS[field]:
                m = re.search(pat, ln, flags=re.IGNORECASE)
                if not m:
                    continue
                val = _reject_placeholder_value(m.group(1))
                if val != "N/A":
                    record[field] = val

    # If still mostly empty, try column split heuristic.
    if sum(1 for v in record.values() if v != "N/A") < 2:
        for ln in data_lines:
            if "|" in ln:
                parts = [p.strip() for p in ln.split("|") if p.strip()]
            elif "\t" in ln:
                parts = [p.strip() for p in ln.split("\t") if p.strip()]
            else:
                parts = []

            if len(parts) >= len(FIELD_ORDER):
                for i, k in enumerate(FIELD_ORDER):
                    record[k] = _reject_placeholder_value(parts[i])
                return record

    return record


def validate_record(record: Dict[str, str]) -> bool:
    # Exact placeholder rejections
    for k in FIELD_ORDER:
        if (record.get(k) or "").strip().upper() in HEADER_PLACEHOLDERS:
            return False

    crime = (record.get("crime") or "").strip().upper()
    description = (record.get("description") or "").strip().upper()
    punishment = (record.get("punishment") or "").strip().upper()
    bns_section = (record.get("bns_section") or "").strip().upper()
    ipc_section = (record.get("ipc_section") or "").strip().upper()

    # Swapped/header collision rules (required)
    if crime == "DESCRIPTION":
        return False
    if description == "PUNISHMENT":
        return False
    if punishment == "BNS EQUIVALENT":
        return False
    if bns_section == "KEYWORDS":
        return False
    if ipc_section == "IPC SECTION":
        return False

    essentials = sum(
        1
        for k in ("ipc_section", "crime", "description", "punishment")
        if (record.get(k) or "").strip() not in ("", "N/A")
    )
    return essentials >= 2


# ---------------------------------------------------------------------------
# Scoring + selection
# ---------------------------------------------------------------------------


def score_record_for_query(record: Dict[str, str], query: str) -> int:
    q = (query or "").lower()
    score = 0

    if record.get("ipc_section") not in (None, "", "N/A"):
        score += 5
    if record.get("crime") not in (None, "", "N/A"):
        score += 5
    if record.get("punishment") not in (None, "", "N/A"):
        score += 5
    if record.get("description") not in (None, "", "N/A"):
        score += 5

    # +3 if chunk/query contains user keywords (simple intent keywords)
    user_keywords = ["murder", "punishment", "homicide", "culpable homicide", "302", "300", "death"]
    if any(kw in q for kw in user_keywords):
        score += 3

    return score


def extract_record_from_chunk(query: str, chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(chunk, dict):
        return None
    if _chunk_looks_like_non_record(chunk):
        return None

    raw_text = _get_detectable_text(chunk)
    if not raw_text:
        return None

    # Regex first
    record = parse_record_with_regex(raw_text)

    # If regex failed to find meaningful fields, try table-row parsing.
    if sum(1 for v in record.values() if v != "N/A") < 2:
        record = parse_record_from_table_rows(raw_text)

    if not validate_record(record):
        return None

    return {
        "record": record,
        "score": score_record_for_query(record, query),
        "source_chunk": chunk,
    }


def select_best_record(query: str, retrieved_chunks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_score = float("-inf")

    for chunk in retrieved_chunks or []:
        cand = extract_record_from_chunk(query, chunk)
        if not cand:
            continue
        if cand["score"] > best_score:
            best_score = cand["score"]
            best = cand

    return best


def format_record(record: Dict[str, str]) -> str:
    return (
        f"⚖ IPC Section\n{record.get('ipc_section','N/A')}\n\n"
        f"📌 Crime\n{record.get('crime','N/A')}\n\n"
        f"📖 Description\n{record.get('description','N/A')}\n\n"
        f"⚠ Ingredients\n{record.get('ingredients','N/A')}\n\n"
        f"⚖ Punishment\n{record.get('punishment','N/A')}\n\n"
        f"📚 Important Cases\n{record.get('important_cases','N/A')}\n\n"
        f"🔄 BNS Equivalent\n{record.get('bns_section','N/A')}"
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class RAGPipeline:
    def __init__(self, retriever: Retriever | None = None):
        self.retriever = retriever

    def init(self, retriever: Retriever) -> None:
        self.retriever = retriever

    def retrieve_context(self, user_query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        if self.retriever is None:
            return []
        return self.retriever.retrieve(user_query, top_k=top_k)

    def answer(self, user_query: str, top_k: int = 20) -> Dict[str, Any]:
        query_type, section_number = detect_query_type(user_query)

        retrieved: List[Dict[str, Any]] = []
        if self.retriever is not None:
            retrieved = self.retriever.retrieve(user_query, top_k=top_k)

        if not retrieved:
            return {
                "answer": "No relevant legal information found.",
                "sources": [],
                "retrieved_chunks": [],
                "query_type": query_type,
                "detected_section": section_number,
            }

        best = select_best_record(user_query, retrieved)
        if not best:
            return {
                "answer": "No relevant legal information found.",
                "sources": [],
                "retrieved_chunks": retrieved,
                "query_type": query_type,
                "detected_section": section_number,
            }

        record = best["record"]
        answer = format_record(record)

        used_chunk = best["source_chunk"]
        similarity = float(used_chunk.get("similarity", 0.0) or used_chunk.get("faiss_similarity", 0.0) or 0.0)

        # Only the actually used source
        sources = [
            {
                "chunk_id": used_chunk.get("chunk_id"),
                "document": used_chunk.get("source_document"),
                "page_number": used_chunk.get("page_number"),
                "section_number": used_chunk.get("section_number") or record.get("ipc_section"),
                "similarity": similarity,
                "crime_name": used_chunk.get("crime_name") or record.get("crime"),
            }
        ]

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved,
            "query_type": query_type,
            "detected_section": section_number,
        }


class EnhancedRAGEngine(RAGPipeline):
    """Engine used by the chat UI."""

    def __init__(self, retriever: Retriever, model_name: str | None = None):
        super().__init__(retriever)
        self.model_name = model_name

    def generate_response(
        self,
        query: str,
        original_query: str | None = None,
        history: str | None = None,
        top_k: int = 20,
        **_: Any,
    ) -> Dict[str, Any]:
        result = self.answer(user_query=original_query or query, top_k=top_k)

        return {
            "answer": result.get("answer", ""),
            "confidence": 1.0,
            "sources": result.get("sources", []),
            "retrieved_chunks": result.get("retrieved_chunks", []),
        }


__all__ = [
    "RAGPipeline",
    "EnhancedRAGEngine",
    "detect_query_type",
    "select_best_record",
]

