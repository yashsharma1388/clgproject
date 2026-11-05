"""
Retrieval orchestration and similarity score thresholding.

Coordinates query embedding, vector similarity search, and result filtering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from mmassistant.config import AppConfig, get_config
from mmassistant.embeddings import BGEEmbedder, Embedder
from mmassistant.vectorstore import ChromaVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Represents a retrieved chunk formatted for LLM context and user citations."""

    chunk_id: str
    text: str
    doc_id: str
    filename: str
    page: int
    chunk_type: str
    similarity: float

    @property
    def citation(self) -> str:
        """Formatted citation string for the LLM and user interface."""
        return f"[Doc: {self.filename}, Page: {self.page}]"


class DocumentRetriever:
    """Manages query embedding, semantic search, and context filtering."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        vector_store: ChromaVectorStore | None = None,
    ) -> None:
        """Initialize the retriever with an embedder and vector store."""
        self.embedder = embedder or BGEEmbedder()
        self.vector_store = vector_store or ChromaVectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the top-K relevant chunks for a user query.

        Args:
            query: The user query string.
            top_k: Maximum chunks to retrieve (defaults to AppConfig.top_k_chunks).
            similarity_threshold: Minimum cosine similarity required.
            doc_ids: Optional document IDs to restrict retrieval to.

        Returns:
            Ranked list of RetrievedChunk objects meeting the threshold.
        """
        cfg: AppConfig = get_config()
        k = top_k if top_k is not None else cfg.top_k_chunks
        min_sim = (
            similarity_threshold
            if similarity_threshold is not None
            else cfg.similarity_threshold
        )

        query_cleaned = query.strip()
        if not query_cleaned or self.vector_store.total_chunks == 0:
            return []

        # 1. Embed query with BGE search instruction prefix
        query_vec = self.embedder.embed_query(query_cleaned)

        # 2. Query ChromaDB
        raw_results = self.vector_store.search(
            query_embedding=query_vec,
            top_k=k,
            doc_ids=doc_ids,
        )

        # 3. Filter by similarity threshold
        chunks: list[RetrievedChunk] = []
        for r in raw_results:
            sim = r["similarity"]
            if sim < min_sim:
                logger.debug(
                    "Filtered out chunk '%s' (similarity %.3f < threshold %.3f)",
                    r["chunk_id"],
                    sim,
                    min_sim,
                )
                continue

            meta = r.get("metadata", {})
            chunks.append(
                RetrievedChunk(
                    chunk_id=r["chunk_id"],
                    text=r["text"],
                    doc_id=meta.get("doc_id", "unknown"),
                    filename=meta.get("filename", "unknown"),
                    page=int(meta.get("page", 1)),
                    chunk_type=meta.get("chunk_type", "text"),
                    similarity=sim,
                )
            )

        logger.info(
            "Retrieved %d relevant chunks for query '%s' (top_k=%d, threshold=%.2f)",
            len(chunks),
            query_cleaned[:40],
            k,
            min_sim,
        )
        return chunks
