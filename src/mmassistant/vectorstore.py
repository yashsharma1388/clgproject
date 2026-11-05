"""
Vector database management using ChromaDB.

Provides persistent storage, chunk indexing, metadata filtering,
document deduplication by SHA-256 hash, and similarity search.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from mmassistant.config import AppConfig, get_config
from mmassistant.ingestion.chunking import Chunk

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Persistent vector store abstraction backed by ChromaDB."""

    COLLECTION_NAME = "multimodal_documents"

    def __init__(self, persist_dir: Path | str | None = None) -> None:
        """Initialize ChromaDB client with disk persistence."""
        cfg: AppConfig = get_config()
        self.persist_dir = Path(persist_dir or cfg.chroma_persist_dir).resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # Use cosine distance space for normalized dense vectors
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Connected to ChromaDB at '%s' (Collection: '%s', existing chunks: %d)",
            self.persist_dir,
            self.COLLECTION_NAME,
            self._collection.count(),
        )

    @property
    def total_chunks(self) -> int:
        """Total number of vector chunks indexed."""
        return self._collection.count()

    def is_document_indexed(self, doc_hash: str) -> bool:
        """Check if a document with the identical SHA-256 hash has already been indexed."""
        results = self._collection.get(
            where={"doc_hash": doc_hash},
            limit=1,
        )
        return bool(results and results["ids"])

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Insert or update chunks with dense vector embeddings into ChromaDB."""
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) does not match embedding count ({len(embeddings)})."
            )

        ids: list[str] = [c.chunk_id for c in chunks]
        documents: list[str] = [c.text for c in chunks]
        metadatas: list[dict[str, Any]] = [c.to_metadata() for c in chunks]

        # Upsert in batches to handle large documents comfortably
        batch_size = 250
        for i in range(0, len(chunks), batch_size):
            batch_end = i + batch_size
            self._collection.upsert(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end],
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )

        logger.info("Successfully indexed %d chunks into ChromaDB.", len(chunks))
        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 4,
        doc_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Perform similarity search with optional document filtering.

        Args:
            query_embedding: Dense embedding vector for the search query.
            top_k: Number of nearest neighbors to retrieve.
            doc_ids: Optional list of doc_ids to restrict the search to.

        Returns:
            List of result dictionaries containing chunk metadata, text, and similarity score.
        """
        if self._collection.count() == 0:
            return []

        where_filter: dict[str, Any] | None = None
        if doc_ids:
            if len(doc_ids) == 1:
                where_filter = {"doc_id": doc_ids[0]}
            else:
                where_filter = {"doc_id": {"$in": doc_ids}}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        formatted_results: list[dict[str, Any]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return formatted_results

        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else []
        metas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for chunk_id, text, meta, dist in zip(ids, docs, metas, distances):
            # In Chroma's cosine space: cosine distance = 1.0 - cosine_similarity
            # Thus similarity = 1.0 - distance
            similarity = max(0.0, 1.0 - float(dist))
            formatted_results.append({
                "chunk_id": chunk_id,
                "text": text,
                "metadata": meta,
                "similarity": similarity,
                "distance": float(dist),
            })

        return formatted_results

    def list_documents(self) -> list[dict[str, Any]]:
        """List all indexed documents with aggregate page and chunk statistics."""
        data = self._collection.get(include=["metadatas"])
        if not data or not data["metadatas"]:
            return []

        docs_map: dict[str, dict[str, Any]] = {}
        for meta in data["metadatas"]:
            if not meta:
                continue
            doc_id = str(meta.get("doc_id", "unknown"))
            filename = str(meta.get("filename", "unknown"))
            page = int(meta.get("page", 1))
            doc_hash = str(meta.get("doc_hash", ""))

            if doc_id not in docs_map:
                docs_map[doc_id] = {
                    "doc_id": doc_id,
                    "filename": filename,
                    "doc_hash": doc_hash,
                    "chunk_count": 0,
                    "pages": set(),
                }

            docs_map[doc_id]["chunk_count"] += 1
            docs_map[doc_id]["pages"].add(page)

        # Format page set as sorted list
        return [
            {
                "doc_id": v["doc_id"],
                "filename": v["filename"],
                "doc_hash": v["doc_hash"],
                "chunk_count": v["chunk_count"],
                "total_pages": len(v["pages"]),
                "pages": sorted(list(v["pages"])),
            }
            for v in docs_map.values()
        ]

    def delete_document(self, doc_id: str) -> int:
        """Remove all chunks associated with a specific document ID."""
        existing = self._collection.get(where={"doc_id": doc_id})
        count = len(existing["ids"]) if existing and existing["ids"] else 0
        if count > 0:
            self._collection.delete(where={"doc_id": doc_id})
            logger.info("Deleted document '%s' (%d chunks removed)", doc_id, count)
        return count

    def reset(self) -> None:
        """Purge and recreate the collection."""
        self._client.delete_collection(self.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
