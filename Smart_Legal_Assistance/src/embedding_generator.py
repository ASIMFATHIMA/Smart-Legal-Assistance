# sentence_transformers is only needed to generate embeddings for RAG.
# Make it optional so Streamlit can start even if the dependency isn't installed.
#
# Rerun-prevention fix (2026):
#   - Removed the noisy STEP 2 print() banners that flooded the Streamlit
#     terminal on every embedding batch (called once per PDF chunk batch).
#   - Replaced with logger.debug() calls so they only appear when the logger
#     level is set to DEBUG.  Standard runs remain clean.

import logging

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    SentenceTransformer = None

import numpy as np

logger = logging.getLogger("rag_engine")


class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise ModuleNotFoundError(
                "sentence_transformers is required for embedding generation. "
                "Install it via requirements.txt"
            )
        logger.info("[EmbeddingGenerator] Loading SentenceTransformer: %s", model_name)
        self.model = SentenceTransformer(model_name)
        logger.info("[EmbeddingGenerator] Model loaded: %s", model_name)

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Generate dense vector embeddings for a list of strings.
        Returns a numpy array of shape (N, D).
        """
        if not texts:
            logger.warning("[EmbeddingGenerator] Received empty text list. Returning empty array.")
            return np.empty((0, 384), dtype=np.float32)

        embeddings = self.model.encode(texts, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        # Verify shape (logged at DEBUG to avoid terminal noise on each call)
        logger.debug(
            "[EmbeddingGenerator] Embedding shape: %s for %d texts",
            embeddings.shape,
            len(texts),
        )

        if embeddings.size == 0:
            raise ValueError("[EmbeddingGenerator] Generated embeddings are empty!")

        if np.isnan(embeddings).any():
            raise ValueError("[EmbeddingGenerator] Generated embeddings contain NaN values!")

        # Log first chunk preview at DEBUG level only
        if texts:
            logger.debug(
                "[EmbeddingGenerator] First chunk preview: %s…",
                texts[0][:100],
            )

        logger.info(
            "[EmbeddingGenerator] Generated %d embeddings (dim=%d).",
            embeddings.shape[0],
            embeddings.shape[1],
        )

        return embeddings
