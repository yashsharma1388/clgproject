"""
End-to-end RAG Pipeline Coordinator.

Connects document ingestion, chunking, embedding generation, vector storage,
dense retrieval, and grounded LLM generation with exact source citations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mmassistant.config import AppConfig, get_config
from mmassistant.embeddings import BGEEmbedder, Embedder
from mmassistant.ingestion.chunking import Chunk, chunk_document
from mmassistant.ingestion.loaders import Document, compute_file_hash, load_document
from mmassistant.llm.base import LLMProvider, LLMResponse
from mmassistant.llm.gemini import GeminiProvider
from mmassistant.prompts import REFUSAL_MESSAGE, SYSTEM_GROUNDING_PROMPT, format_rag_prompt
from mmassistant.retrieval import DocumentRetriever, RetrievedChunk
from mmassistant.vectorstore import ChromaVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Standardized response payload for document Q&A."""

    query: str
    answer: str
    sources: list[RetrievedChunk]
    citations: list[str] = field(default_factory=list)
    model_name: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class RAGPipeline:
    """Orchestrates document ingestion and grounded retrieval-augmented generation."""

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        embedder: Embedder | None = None,
        vector_store: ChromaVectorStore | None = None,
        retriever: DocumentRetriever | None = None,
    ) -> None:
        """Initialize the pipeline components."""
        self.config: AppConfig = get_config()
        self.llm = llm_provider or GeminiProvider()
        self.embedder = embedder or BGEEmbedder()
        self.vector_store = vector_store or ChromaVectorStore()
        self.retriever = retriever or DocumentRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
        )

    def ingest_document(
        self,
        file_path: str | Path,
        force_reindex: bool = False,
    ) -> dict[str, Any]:
        """Ingest, parse, chunk, embed, and index a document into ChromaDB.

        Args:
            file_path: Path to target document (PDF, DOCX, TXT, MD).
            force_reindex: If True, re-indexes even if identical hash exists.

        Returns:
            Dictionary reporting ingestion status, document ID, and chunk count.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_hash = compute_file_hash(path)

        # De-duplication check
        if not force_reindex and self.vector_store.is_document_indexed(file_hash):
            logger.info("Document '%s' already indexed (hash: %s). Skipping.", path.name, file_hash[:16])
            return {
                "status": "skipped",
                "reason": "already_indexed",
                "filename": path.name,
                "doc_id": file_hash[:16],
            }

        # 1. Load document
        doc: Document = load_document(path)

        # 2. Chunk document
        chunks: list[Chunk] = chunk_document(doc)
        if not chunks:
            logger.warning("No chunks generated for document '%s'.", path.name)
            return {
                "status": "empty",
                "filename": path.name,
                "doc_id": doc.doc_id,
                "chunks": 0,
            }

        # 3. Compute vector embeddings for all chunks
        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_documents(chunk_texts)

        # 4. Insert into ChromaDB
        self.vector_store.add_chunks(chunks, embeddings)

        logger.info(
            "Successfully indexed '%s' (%d pages, %d chunks)",
            doc.filename,
            doc.total_pages,
            len(chunks),
        )
        return {
            "status": "indexed",
            "filename": doc.filename,
            "doc_id": doc.doc_id,
            "total_pages": doc.total_pages,
            "chunks": len(chunks),
        }

    def ask(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> RAGResult:
        """Answer a user query strictly grounded in indexed document context.

        Args:
            query: User's question.
            doc_ids: Optional list of document IDs to restrict search to.
            top_k: Max chunks to retrieve.
            similarity_threshold: Minimum cosine similarity.

        Returns:
            RAGResult containing the grounded answer, sources, and citations.
        """
        query_clean = query.strip()
        if not query_clean:
            return RAGResult(
                query=query,
                answer="Please provide a valid question.",
                sources=[],
                citations=[],
            )

        # 1. Retrieve relevant chunks
        chunks = self.retriever.retrieve(
            query=query_clean,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            doc_ids=doc_ids,
        )

        # 2. Guard: If no relevant chunks meet threshold, politely refuse without wasting LLM quota
        if not chunks:
            return RAGResult(
                query=query_clean,
                answer=REFUSAL_MESSAGE,
                sources=[],
                citations=[],
                model_name=self.config.gemini_model_id,
            )

        # 3. Build grounded prompt with XML quarantine
        prompt = format_rag_prompt(query_clean, chunks)

        # 4. Query LLM
        response: LLMResponse = self.llm.generate_text(
            prompt=prompt,
            system_instruction=SYSTEM_GROUNDING_PROMPT,
            temperature=0.1,  # Low temperature for strict factual determinism
        )

        # 5. Extract citation tags from LLM response
        citations = list(set(re.findall(r"\[Doc:\s*[^,]+,\s*Page:\s*\d+\]", response.content)))

        return RAGResult(
            query=query_clean,
            answer=response.content.strip(),
            sources=chunks,
            citations=citations,
            model_name=response.model_name,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )

    def list_documents(self) -> list[dict[str, Any]]:
        """List all indexed documents in the vector store."""
        return self.vector_store.list_documents()

    def delete_document(self, doc_id: str) -> int:
        """Remove a document from the vector store by doc_id."""
        return self.vector_store.delete_document(doc_id)
