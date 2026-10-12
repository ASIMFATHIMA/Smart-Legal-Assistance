"""Smart Legal Assistance – Hybrid Retriever
============================================
Fixes applied:
  - Declare `merged`, `meta_hits`, `keyword_hits` before use (was NameError → silent [] return).
  - Define SIM_THRESHOLD inside retrieve() (was NameError).
  - Add `retrieve_top_k()` alias (RAGEngine calls this name).
  - Lower SIM_THRESHOLD to 0.10 so L2-derived cosine-like scores are not over-filtered.
  - Add `normalize_query()` with legal entity expansion.
  - Add `detect_legal_entities()` for metadata keyword pre-search.
  - Full DEBUG_MODE block printing every retrieval step.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from src.vector_store import VectorStore
from src.embedding_generator import EmbeddingGenerator

logger = logging.getLogger("rag_engine")

# ---------------------------------------------------------------------------
# Global DEBUG flag – set DEBUG_MODE=1 in environment to enable
# ---------------------------------------------------------------------------
DEBUG_MODE: bool = os.environ.get("DEBUG_MODE", "0") == "1"


# ---------------------------------------------------------------------------
# Legal entity vocabulary for query expansion and metadata pre-search
# ---------------------------------------------------------------------------
LEGAL_ENTITIES: Dict[str, List[str]] = {
    # Gender / person categories
    "adult female": ["woman", "female", "lady", "adult female", "women"],
    "adult male": ["man", "male", "adult male", "men"],
    "child": ["child", "minor", "juvenile", "infant", "boy", "girl"],
    "woman": ["woman", "female", "lady", "women"],
    "minor": ["minor", "juvenile", "child", "infant"],

    # Offences – IPC/BNS keywords
    "murder": ["murder", "homicide", "killing", "death", "culpable", "300", "302"],
    "culpable homicide": ["culpable homicide", "death", "homicide", "299"],
    "rape": ["rape", "sexual assault", "376", "375"],
    "kidnapping": ["kidnapping", "abduction", "361", "362", "363"],
    "theft": ["theft", "stolen", "stealing", "378", "379"],
    "robbery": ["robbery", "dacoity", "392", "395"],
    "dowry": ["dowry", "dowry death", "304B", "498A"],
    "acid attack": ["acid attack", "326A"],
    "cheating": ["cheating", "fraud", "420"],
    "fraud": ["fraud", "cheating", "deceit", "420"],
    "sedition": ["sedition", "124A"],
    "forgery": ["forgery", "fabrication", "467", "468"],
    "assault": ["assault", "hurt", "grievous hurt", "323", "324", "325"],
    "harassment": ["harassment", "stalking", "molestation"],

    # Special acts
    "ndps": ["ndps", "narcotic", "drug", "psychotropic", "narcotics"],
    "wildlife": ["wildlife", "animal", "forest", "poaching", "wildlife protection"],
    "environment": ["environment", "pollution", "air", "water", "environmental"],
    "air pollution": ["air pollution", "pollution", "environment", "air act"],
    "consumer": ["consumer", "consumer protection", "consumer rights"],
    "trade": ["trade", "commerce", "licence", "license", "business"],
    "commerce": ["commerce", "trade", "business", "licence"],
    "financial": ["financial", "money laundering", "banking", "cheque"],

    # Legal concepts
    "punishment": ["punishment", "sentence", "imprisonment", "fine", "penalty"],
    "legal rights": ["legal rights", "rights", "fundamental rights", "constitutional"],
    "procedure": ["procedure", "crpc", "trial", "cognizable", "non-cognizable", "bail"],
    "bail": ["bail", "anticipatory bail", "regular bail"],
    "ipc": ["ipc", "indian penal code", "penal code"],
    "bns": ["bns", "bharatiya nyaya sanhita"],
}

# Flat set of all legal terms for fast membership check
_ALL_LEGAL_TERMS: set = {
    term.lower()
    for terms in LEGAL_ENTITIES.values()
    for term in terms
}


# ---------------------------------------------------------------------------
# BM25 implementation (self-contained, no external dependency)
# ---------------------------------------------------------------------------
class SimpleBM25:
    def __init__(self, corpus: List[List[str]]):
        self.corpus_size = len(corpus)
        self.avgdl = (
            sum(map(len, corpus)) / self.corpus_size if self.corpus_size else 0
        )
        self.doc_freqs: List[Counter] = []
        self.idf: Dict[str, float] = {}
        self.doc_len: List[int] = []

        nd: Dict[str, int] = {}
        for document in corpus:
            self.doc_len.append(len(document))
            frequencies = Counter(document)
            self.doc_freqs.append(frequencies)
            for word in frequencies:
                nd[word] = nd.get(word, 0) + 1

        for word, freq in nd.items():
            self.idf[word] = math.log(
                1 + (self.corpus_size - freq + 0.5) / (freq + 0.5)
            )

    def get_scores(self, query: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
        scores = [0.0] * self.corpus_size
        for q in query:
            if q not in self.idf:
                continue
            idf = self.idf[q]
            for i, doc_freq in enumerate(self.doc_freqs):
                freq = doc_freq.get(q, 0)
                if freq == 0:
                    continue
                dl = self.doc_len[i]
                score = idf * (freq * (k1 + 1)) / (
                    freq + k1 * (1 - b + b * dl / (self.avgdl or 1))
                )
                scores[i] += score
        return scores


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
class Retriever:
    """Hybrid BM25 + FAISS retriever with legal entity awareness."""

    # Similarity threshold for FAISS results (L2-distance → 1/(1+d)).
    # Keep low: useful chunks for legal queries can score 0.10–0.25.
    SIM_THRESHOLD: float = 0.10

    def __init__(self, vector_store: VectorStore, embedder: EmbeddingGenerator):
        self.vector_store = vector_store
        self.embedder = embedder
        self._build_bm25()

    # ------------------------------------------------------------------
    # BM25 helpers
    # ------------------------------------------------------------------
    def _build_bm25(self) -> None:
        chunks = self.vector_store.get_all_chunks()
        if not chunks:
            self.bm25 = None
            self.chunks: List[Dict[str, Any]] = []
            return
        self.chunks = chunks
        tokenized_corpus = [self._tokenize(c.get("text", "")) for c in chunks]
        self.bm25 = SimpleBM25(tokenized_corpus)
        logger.info(f"[Retriever] BM25 built over {len(chunks)} chunks.")

    def rebuild_bm25(self) -> None:
        """Call after new documents are ingested to refresh BM25."""
        self._build_bm25()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _keyword_query(self, query: str) -> List[str]:
        return [t for t in self._tokenize(query) if len(t) >= 3]

    # ------------------------------------------------------------------
    # Query normalization & legal entity detection
    # ------------------------------------------------------------------
    def normalize_query(self, query: str) -> str:
        """Normalize the query: lowercase, strip, collapse whitespace."""
        q = query.lower().strip()
        q = re.sub(r"\s+", " ", q)
        # Normalize common abbreviations
        q = re.sub(r"\bipc\b", "ipc", q)
        q = re.sub(r"\bbns\b", "bns", q)
        q = re.sub(r"\bsec\b\.?\s*(\d)", r"section \1", q)
        q = re.sub(r"\bsection\b\s*(\d)", r"section \1", q)
        return q

    def detect_legal_entities(self, normalized_query: str) -> List[str]:
        """Return a list of legal entity groups matched in the query."""
        matched: List[str] = []
        for entity, synonyms in LEGAL_ENTITIES.items():
            for syn in synonyms:
                if syn.lower() in normalized_query:
                    matched.append(entity)
                    break
        return list(set(matched))

    def _extract_section_number(self, query: str) -> Optional[str]:
        """Extract bare section numbers like '302', '376A', 'Section 420'."""
        m = re.search(
            r"\b(?:section|sec|ipc|bns)?\s*(\d{1,4}[A-Z]{0,3})\b",
            query,
            flags=re.IGNORECASE,
        )
        return m.group(1).upper() if m else None

    def _expand_query_with_entities(
        self, normalized_query: str, entities: List[str]
    ) -> str:
        """Append synonym terms for detected entities to broaden BM25 coverage."""
        extra_terms: List[str] = []
        for entity in entities:
            synonyms = LEGAL_ENTITIES.get(entity, [])
            extra_terms.extend(synonyms)
        if extra_terms:
            return normalized_query + " " + " ".join(extra_terms)
        return normalized_query

    # ------------------------------------------------------------------
    # Metadata keyword pre-search
    # ------------------------------------------------------------------
    def _metadata_keyword_search(
        self, keywords: List[str], entities: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Scan chunk metadata for keyword matches BEFORE FAISS semantic search.
        Returns chunks whose crime_name / section_number / text contains any keyword.
        """
        if not self.chunks:
            return []

        # Build a combined search set: raw keywords + entity synonyms
        search_terms: set = set(keywords)
        for entity in entities:
            for syn in LEGAL_ENTITIES.get(entity, []):
                search_terms.update(self._tokenize(syn))

        hits: List[Dict[str, Any]] = []
        for chunk in self.chunks:
            text_lower = chunk.get("text", "").lower()
            crime_lower = chunk.get("crime_name", "").lower()
            section = str(chunk.get("section_number", "")).lower()
            kws = [str(k).lower() for k in chunk.get("keywords", [])]

            searchable = text_lower + " " + crime_lower + " " + section + " " + " ".join(kws)
            if any(term in searchable for term in search_terms if len(term) >= 3):
                c = chunk.copy()
                c["faiss_similarity"] = 0.20  # neutral metadata-hit score
                c["similarity"] = 0.20
                hits.append(c)

        return hits

    # ------------------------------------------------------------------
    # Main retrieval entry points
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval pipeline:
          1. Metadata keyword pre-search
          2. Section number exact-match
          3. FAISS semantic search
          4. BM25 keyword search
          5. Merge, deduplicate, sort by similarity → top_k
        """
        try:
            if self.vector_store.index is None or self.vector_store.index.ntotal == 0:
                logger.warning("[Retriever] FAISS index is empty.")
                return []

            # ── Initialise accumulators ─────────────────────────────────────────
            merged: Dict[str, Dict[str, Any]] = {}
            meta_hits: List[Dict[str, Any]] = []
            keyword_hits: List[Dict[str, Any]] = []


            # ── Query pre-processing ─────────────────────────────────────────────
            normalized_query = self.normalize_query(query)
            legal_entities = self.detect_legal_entities(normalized_query)
            section_number = self._extract_section_number(normalized_query)
            q_keywords = self._keyword_query(normalized_query)
            expanded_query = self._expand_query_with_entities(normalized_query, legal_entities)
            expanded_keywords = self._keyword_query(expanded_query)

            if DEBUG_MODE:
                print("\n" + "=" * 70)
                print("[DEBUG] RETRIEVER")
                print(f"  User Question   : {query}")
                print(f"  Normalized Query: {normalized_query}")
                print(f"  Legal Entities  : {legal_entities}")
                print(f"  Section Number  : {section_number}")
                print(f"  Keywords        : {q_keywords}")
                print(f"  Expanded Query  : {expanded_query}")
                print("=" * 70)

            # ── Step 0: Metadata keyword pre-search ─────────────────────────────
            meta_results = self._metadata_keyword_search(q_keywords, legal_entities)
            if DEBUG_MODE:
                print(f"[DEBUG] Metadata keyword hits: {len(meta_results)}")

            for r in meta_results:
                cid = r.get("chunk_id")
                if cid and cid not in merged:
                    meta_hits.append(r)
                    merged[cid] = r

            # ── Step 1: Section number exact-match ──────────────────────────────
            if section_number:
                exact_results = self.vector_store.exact_search(section_number)
                if DEBUG_MODE:
                    print(f"[DEBUG] Exact section match for '{section_number}': {len(exact_results)} hit(s)")
                for r in exact_results:
                    cid = r.get("chunk_id")
                    if not cid:
                        continue
                    r = r.copy()
                    r["faiss_similarity"] = float(r.get("similarity", 1.0))
                    r["similarity"] = r["faiss_similarity"]
                    if cid not in merged:
                        merged[cid] = r

            # ── Step 2: FAISS semantic search ────────────────────────────────────
            embeddings = self.embedder.generate_embeddings([expanded_query])
            query_embedding = embeddings[0]
            faiss_results = self.vector_store.search(query_embedding, top_k=15)

            if DEBUG_MODE:
                print(f"[DEBUG] FAISS returned {len(faiss_results)} result(s)")
                for r in faiss_results[:5]:
                    print(
                        f"    chunk={r.get('chunk_id')}  sim={r.get('similarity', 0):.4f}"
                        f"  src={r.get('source_document')}  crime={r.get('crime_name', '')[:40]}"
                    )

            for r in faiss_results:
                cid = r.get("chunk_id")
                if not cid:
                    continue
                r = r.copy()
                r["faiss_similarity"] = float(r.get("similarity", 0.0))
                r["similarity"] = r["faiss_similarity"]

                # Apply threshold – lower is more permissive (good for legal queries)
                if r["similarity"] < self.SIM_THRESHOLD:
                    continue

                if cid not in merged:
                    merged[cid] = r
                else:
                    # Upgrade similarity if FAISS found a better score
                    if r["similarity"] > merged[cid].get("similarity", 0.0):
                        merged[cid]["similarity"] = r["similarity"]
                        merged[cid]["faiss_similarity"] = r["faiss_similarity"]

            # ── Step 3: BM25 keyword search ──────────────────────────────────────
            if self.bm25 and expanded_keywords:
                bm25_scores = self.bm25.get_scores(expanded_keywords)
                top_indices = sorted(
                    range(len(bm25_scores)),
                    key=lambda i: bm25_scores[i],
                    reverse=True,
                )[:15]

                if DEBUG_MODE:
                    print(f"[DEBUG] BM25 top scores: {[round(bm25_scores[i], 3) for i in top_indices[:5]]}")

                for i in top_indices:
                    if bm25_scores[i] <= 0:
                        continue
                    res = self.chunks[i]
                    cid = res.get("chunk_id")
                    if not cid:
                        continue
                    res = res.copy()
                    # Map BM25 score to a pseudo-similarity in [0, 1] for ranking
                    res_sim = min(float(bm25_scores[i]) / 20.0, 1.0)
                    res["faiss_similarity"] = res_sim
                    res["similarity"] = max(
                        float(res.get("similarity", 0.0)), res_sim
                    )
                    keyword_hits.append(res)
                    if cid not in merged:
                        merged[cid] = res
                    else:
                        if res_sim > merged[cid].get("similarity", 0.0):
                            merged[cid]["similarity"] = res_sim

            if DEBUG_MODE:
                print(f"[DEBUG] Keyword/BM25 hits added: {len(keyword_hits)}")

            # ── Section accuracy enforcement (exact IPC/BNS match wins)
            def _norm_sec(v: Any) -> Optional[str]:
                if v is None:
                    return None
                s = str(v).strip().upper()
                if not s or s == "N/A":
                    return None
                return s

            def _is_exact_section_match(chunk: Dict[str, Any], target: str) -> bool:
                if not target:
                    return False
                target = str(target).strip().upper()
                for c in (
                    chunk.get("ipc_section"),
                    chunk.get("section_number"),
                    chunk.get("section"),
                    chunk.get("bns_section"),
                ):
                    nc = _norm_sec(c)
                    if nc and nc == target:
                        return True
                return False

            # Extract detected section and enforce before semantic ranking
            retrieved = list(merged.values())
            exact_matches: List[Dict[str, Any]] = []
            if section_number:
                exact_matches = [c for c in retrieved if _is_exact_section_match(c, section_number)]

            print("\n" + "=" * 90)
            print("[RETRIEVER_LOG]")
            print("Query:", query)
            print("Detected section:", section_number)
            print("Retrieved sections:")
            for c in retrieved[:30]:
                secvals = [c.get("ipc_section"), c.get("section_number"), c.get("section"), c.get("bns_section")]
                secvals = [_norm_sec(x) for x in secvals]
                secvals = sorted({s for s in secvals if s})
                if secvals:
                    print(" -", secvals, "| chunk_id=", c.get("chunk_id"))
            print("Selected section:", section_number if exact_matches else None)
            print("Exact match count:", len(exact_matches))
            print("=" * 90)

            if exact_matches:
                exact_matches.sort(key=lambda x: float(x.get("similarity", 0.0) or 0.0), reverse=True)
                for c in exact_matches[:top_k]:
                    c["selected_reason"] = "exact_section_match"
                return exact_matches[:top_k]

            # ── Step 4: Deduplicate and rank (semantic similarity unless exact section exists)
            merged_list = list(merged.values())
            merged_list.sort(key=lambda x: float(x.get("similarity", 0.0) or 0.0), reverse=True)

            final = merged_list[:top_k]


            if DEBUG_MODE:
                print(f"\n[DEBUG] Final merged chunks: {len(final)}")
                for i, c in enumerate(final[:10]):
                    print(
                        f"  [{i+1}] chunk={c.get('chunk_id')}  "
                        f"sim={c.get('similarity', 0):.4f}  "
                        f"src={c.get('source_document')}  "
                        f"crime={str(c.get('crime_name', ''))[:40]}  "
                        f"page={c.get('page_number')}"
                    )
                    print(f"       text preview: {str(c.get('text', ''))[:120]}")
                print("=" * 70 + "\n")

            return final

        except Exception as exc:
            logger.exception("[Retriever] retrieve() failed: %s", exc)
            return []

    def retrieve_top_k(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Alias used by RAGEngine.retrieve_context()."""
        return self.retrieve(query, top_k=top_k)
