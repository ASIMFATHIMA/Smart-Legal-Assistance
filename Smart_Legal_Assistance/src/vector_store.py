"""Smart Legal Assistance - Phase 2 Module 1

FAISS VectorStore
------------------
Builds and loads a local FAISS index (IndexFlatL2) over Phase-1 embeddings.

CRITICAL:
- This module must NEVER silently turn FAISS into None.
- We also guard against a corrupted Index object.

Rerun-prevention fix (2026):
- _print_faiss_state() now uses logger.debug() instead of print() so it is
  invisible during normal Streamlit runs.  It was called on every build_index
  and add_embeddings invocation, flooding the terminal on every PDF upload.
- verify_index_stats upgraded to logger.info for structured log output.
"""

from __future__ import annotations

import os
import pickle
import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Strict FAISS import: if it fails, raise immediately with the real error.
try:
    import faiss  # type: ignore
except ImportError as e:
    import sys
    msg = (
        f"FAISS import failed: {e}\n"
        f"Python executable used: {sys.executable}\n"
        f"sys.path: {sys.path}\n"
        f"Suggestion: It looks like the application is running outside the active virtual environment (.venv).\n"
        f"Please run: .\\.venv\\Scripts\\streamlit run streamlit_app.py"
    )
    raise RuntimeError(msg) from e


class VectorStore:
    """FAISS vector store for semantic retrieval."""

    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        embedding_dim: Optional[int] = None,
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_dim = embedding_dim

        self.index: Optional[Any] = None
        self.metadata: Optional[List[Dict[str, Any]]] = None

    def verify_index_stats(self) -> None:
        if self.index is None:
            logger.info("[VectorStore] FAISS Index: None")
            return
        index_vectors = int(self.index.ntotal)
        metadata_count = len(self.metadata) if self.metadata is not None else 0
        dimension = int(self.index.d)

        logger.info(
            "[VectorStore] FAISS stats — vectors: %d | metadata: %d | dim: %d",
            index_vectors,
            metadata_count,
            dimension,
        )

        if index_vectors != metadata_count:
            raise ValueError(
                f"FAISS Mismatch: index vectors ({index_vectors}) != "
                f"metadata count ({metadata_count})!"
            )

    def _assert_faiss_ok(self, context: str = "") -> None:
        if faiss is None:  # pragma: no cover
            raise RuntimeError(f"FAISS module is None unexpectedly ({context})")
        if not hasattr(faiss, "IndexFlatL2"):
            raise RuntimeError(f"faiss.IndexFlatL2 missing at runtime ({context})")

    def _print_faiss_state(self, context: str) -> None:
        # FIX: Demoted to logger.debug so output is hidden during normal runs.
        # Previously used print() which flooded the Streamlit terminal on every
        # build_index / add_embeddings call.
        logger.debug(
            "[VectorStore::%s] FAISS object: %s | type: %s | HasIndexFlatL2: %s | version: %s",
            context,
            faiss,
            type(faiss),
            hasattr(faiss, "IndexFlatL2"),
            getattr(faiss, "__version__", None),
        )

    def build_index(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
    ) -> None:
        """Build an IndexFlatL2 FAISS index safely."""
        self._assert_faiss_ok("before_build_index")
        self._print_faiss_state("before_build_index")

        embeddings = np.asarray(embeddings)
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape={embeddings.shape}")

        n, d = embeddings.shape
        if n == 0:
            raise ValueError("No embeddings provided. Cannot build FAISS index.")

        if d is None or int(d) <= 0:
            raise ValueError("Invalid embedding dimension")

        if len(metadata) != n:
            raise ValueError(
                f"metadata length mismatch: embeddings has {n} vectors but metadata has {len(metadata)}"
            )

        required_keys = {"chunk_id", "source_document", "text", "section_number"}
        for i, m in enumerate(metadata[:5]):
            missing = required_keys - set(m.keys())
            if missing:
                raise KeyError(f"metadata[{i}] missing keys: {sorted(missing)}")

        dimension = int(d)
        logger.info(
            "Building FAISS IndexFlatL2 with %d vectors (dim=%d)", n, dimension
        )

        # Create the index and verify it is not None/corrupted.
        index_obj = faiss.IndexFlatL2(dimension)
        if index_obj is None:
            raise RuntimeError("faiss.IndexFlatL2 returned None (corrupted FAISS runtime)")
        self.index = index_obj

        vectors = embeddings.astype(np.float32, copy=False)
        if vectors.shape[0] == 0:
            raise ValueError("Embedding generation failed (empty vectors).")

        # The reported crash points here: self.index.add(...)
        # If index_obj is valid, vectors dtype/shape should be valid too.
        self.index.add(vectors)

        self.metadata = metadata
        self.embedding_dim = dimension

        self._print_faiss_state("after_build_index")
        self.verify_index_stats()

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
    ) -> None:
        """Add vectors to the existing index (or build if missing)."""
        if self.index is None or self.metadata is None:
            self.build_index(embeddings, metadata)
            return

        self._assert_faiss_ok("before_add_embeddings")

        embeddings = np.asarray(embeddings)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D")

        n, d = embeddings.shape
        if n == 0:
            raise ValueError("No embeddings provided to add.")

        if self.embedding_dim is not None and d != self.embedding_dim:
            raise ValueError(f"Dim mismatch. Expected {self.embedding_dim}, got {d}")

        vectors = embeddings.astype(np.float32, copy=False)
        if vectors.shape[0] == 0:
            raise ValueError("Embedding generation failed (empty vectors).")

        if self.index is None:
            # Prevent NoneType issues in .add
            raise RuntimeError("Vector index is None unexpectedly before adding")

        self.index.add(vectors)
        self.metadata.extend(metadata)

        if self.embedding_dim is None:
            self.embedding_dim = int(d)
        
        self.verify_index_stats()

    def save_index(self) -> None:
        if self.index is None or self.metadata is None:
            raise ValueError("Index/metadata not built. Call build_index first.")

        self._assert_faiss_ok("before_save_index")

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)

        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load_index(self) -> None:
        self._assert_faiss_ok("before_load_index")

        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")

        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        if self.metadata is None:
            raise ValueError("Loaded metadata is None")

        if self.embedding_dim is None:
            self.embedding_dim = int(self.index.d)
        
        self.verify_index_stats()

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        # IndexFlatL2 returns squared L2 distances.
        return float(1.0 / (1.0 + float(distance)))

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None or self.metadata is None:
            raise ValueError("Index not loaded/built. Call load_index() or build_index().")

        query_embedding = np.asarray(query_embedding)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
            raise ValueError("query_embedding must have shape (dim,) or (1, dim)")

        if query_embedding.shape[1] != int(self.index.d):
            raise ValueError(
                f"Query dim mismatch: index dim={self.index.d}, query dim={query_embedding.shape[1]}"
            )

        distances, indices = self.index.search(query_embedding.astype(np.float32), int(top_k))
        distances = distances[0]
        indices = indices[0]

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances, indices):
            if idx < 0:
                continue
            m = self.metadata[int(idx)]
            results.append(
                {
                    **m,
                    "distance": float(dist),
                    "similarity": self._distance_to_similarity(dist),
                }
            )

        return results

    def exact_search(self, section_number: str) -> List[Dict[str, Any]]:
        if self.metadata is None:
            return []

        results: List[Dict[str, Any]] = []
        for m in self.metadata:
            if str(m.get("section_number", "")).upper() == str(section_number).upper():
                results.append({**m, "distance": 0.0, "similarity": 1.0})
        return results

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        return self.metadata or []

    def diagnostic(self) -> Dict[str, Any]:
        return {
            "faiss_installed": faiss is not None,
            "faiss_version": getattr(faiss, "__version__", None),
            "embedding_dim": self.embedding_dim,
            "vector_count": int(self.index.ntotal) if self.index is not None else None,
            "index_type": type(self.index).__name__ if self.index is not None else None,
        }

