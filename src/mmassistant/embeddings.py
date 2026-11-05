"""
Embedding generation abstraction and BGE local embedding implementation.

Implements the Embedder interface using `sentence-transformers` with
`BAAI/bge-small-en-v1.5`, adhering to asymmetric query/document formatting.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from mmassistant.config import AppConfig, get_config

logger = logging.getLogger(__name__)


class Embedder(ABC):
    """Abstract interface for dense text embedding models."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Compute dense vector embeddings for a batch of document texts.

        Args:
            texts: List of document or chunk strings.

        Returns:
            List of float vectors, each of size `dimension`.
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Compute dense vector embedding for a single user query.

        Args:
            query: The user search or question string.

        Returns:
            A single float vector of size `dimension`.
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality of the embedding model."""
        pass


class BGEEmbedder(Embedder):
    """Local SentenceTransformers embedder adhering to BGE formatting conventions."""

    # BGE recommendation: queries require an instruction prefix; documents do not.
    DEFAULT_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
    ) -> None:
        """Initialize the BGE Embedder.

        Args:
            model_name: HuggingFace model identifier (e.g. 'BAAI/bge-small-en-v1.5').
            device: 'cpu', 'cuda', or None for automatic hardware detection.
            query_instruction: Task instruction prepended to search queries.
        """
        cfg: AppConfig = get_config()
        self.model_name = (model_name or cfg.embedding_model_id).strip()
        self.query_instruction = query_instruction
        self._device = device
        self._model: Any = None
        self._dimension: int = 384  # Default for bge-small-en-v1.5

    def _get_model(self) -> Any:
        """Lazy-load the SentenceTransformer model on first usage."""
        if self._model is None:
            logger.info("Loading embedding model '%s'...", self.model_name)
            from sentence_transformers import SentenceTransformer

            try:
                # Fast path: load from local cache without checking remote HF Hub
                self._model = SentenceTransformer(
                    self.model_name, device=self._device, local_files_only=True
                )
            except Exception:
                # Fallback: download if not yet cached
                self._model = SentenceTransformer(self.model_name, device=self._device)

            # Inspect actual hidden dimension
            sample_emb = self._model.encode("warmup", normalize_embeddings=True)
            self._dimension = len(sample_emb)
            logger.info("Loaded '%s' successfully. Dimension: %d", self.model_name, self._dimension)
        return self._model

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality."""
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document passages without query instruction prefix."""
        if not texts:
            return []

        model = self._get_model()
        # Documents are encoded raw with L2 normalization
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query with the required BGE instruction prefix."""
        model = self._get_model()
        formatted_query = f"{self.query_instruction}{query}"
        embedding = model.encode(
            formatted_query,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embedding.tolist()
