"""
Buchungs-Anomalie Pre-Filter — Text-Embedding-Modul

Singleton TextEmbedder mit Lazy-Load für sentence-transformers.
Graceful Degradation: Wenn sentence-transformers nicht installiert,
werden alle Methoden zu No-Ops und HAS_EMBEDDINGS = False.

Public API:
    HAS_EMBEDDINGS: bool
    get_embedder() -> TextEmbedder | None
    TextEmbedder.embed_texts(texts) -> np.ndarray
    TextEmbedder.cosine_similarity_pairs(emb, idx_a, idx_b) -> np.ndarray
"""

from __future__ import annotations

import numpy as np

from src.logging_config import get_logger

logger = get_logger("prefilter.embeddings")

# ── Feature-Flag ─────────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    logger.info("sentence-transformers nicht installiert — Embedding-Features deaktiviert")

_MODEL_NAME = "all-MiniLM-L6-v2"
_BATCH_SIZE = 512

# ── Singleton ────────────────────────────────────────────────────────────────
_instance: TextEmbedder | None = None


class TextEmbedder:
    """Lazy-loading Singleton für Text-Embeddings."""

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None  # type: ignore[name-defined]

    def _ensure_model(self) -> None:
        if self._model is None:
            logger.info("Lade Embedding-Modell", model=_MODEL_NAME)
            self._model = SentenceTransformer(_MODEL_NAME)
            logger.info("Embedding-Modell geladen", model=_MODEL_NAME)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Berechnet normalisierte Embeddings für eine Liste von Texten.

        Returns:
            np.ndarray mit Shape (len(texts), embedding_dim), L2-normalisiert.
        """
        self._ensure_model()
        assert self._model is not None
        embeddings = self._model.encode(
            texts,
            batch_size=_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    @staticmethod
    def cosine_similarity_pairs(
        emb: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray
    ) -> np.ndarray:
        """Berechnet Cosine-Similarity für Indexpaare.

        Da Embeddings L2-normalisiert sind, ist Cosine = Dot-Product.

        Args:
            emb: (N, D) normalisierte Embeddings
            idx_a: (K,) Indices der ersten Elemente
            idx_b: (K,) Indices der zweiten Elemente

        Returns:
            (K,) Array mit Cosine-Similarities in [-1, 1]
        """
        a = emb[idx_a]
        b = emb[idx_b]
        return np.einsum("ij,ij->i", a, b)


def get_embedder() -> TextEmbedder | None:
    """Gibt den Singleton-TextEmbedder zurück, oder None wenn nicht verfügbar."""
    global _instance
    if not HAS_EMBEDDINGS:
        return None
    if _instance is None:
        _instance = TextEmbedder()
    return _instance
