"""
End-to-end Multimodal AI Assistant Pipeline Coordinator.

Connects document ingestion, OCR, figure extraction, chunking, embedding generation,
vector storage, dense retrieval, direct vision Q&A, and mixed multimodal queries.
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
from mmassistant.ingestion.figures import extract_and_caption_figures
from mmassistant.ingestion.loaders import Document, compute_file_hash, load_document
from mmassistant.llm.base import LLMProvider, LLMResponse
from mmassistant.llm.factory import get_llm_provider
from mmassistant.llm.gemini import GeminiProvider
from mmassistant.prompts import (
    REFUSAL_MESSAGE,
    SYSTEM_GROUNDING_PROMPT,
    SYSTEM_MIXED_PROMPT,
    format_mixed_prompt,
    format_rag_prompt,
)
from mmassistant.retrieval import DocumentRetriever, RetrievedChunk
from mmassistant.vectorstore import ChromaVectorStore
from mmassistant.vision import VisionEngine, load_image_bytes

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


@dataclass
class VisionResult:
    """Standardized response payload for standalone image analysis."""

    query: str
    answer: str
    mime_type: str
    model_name: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class MixedResult:
    """Standardized response payload for joint image + document queries."""

    query: str
    answer: str
    sources: list[RetrievedChunk]
    citations: list[str] = field(default_factory=list)
    mime_type: str = ""
    model_name: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class MultimodalPipeline:
    """Unified pipeline orchestrating Document RAG, Vision Q&A, and Mixed Multimodal queries."""

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        embedder: Embedder | None = None,
        vector_store: ChromaVectorStore | None = None,
        retriever: DocumentRetriever | None = None,
        vision_engine: VisionEngine | None = None,
    ) -> None:
        """Initialize pipeline subsystems."""
        self.config: AppConfig = get_config()
        self.llm = llm_provider or get_llm_provider()
        self.embedder = embedder or BGEEmbedder()
        self.vector_store = vector_store or ChromaVectorStore()
        self.retriever = retriever or DocumentRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
        )
        self.vision = vision_engine or VisionEngine(llm_provider=self.llm)

    def ingest_document(
        self,
        file_path: str | Path,
        force_reindex: bool = False,
        extract_figures: bool | None = None,
    ) -> dict[str, Any]:
        """Ingest, parse, chunk, optionally caption figures, embed, and index into ChromaDB.

        Args:
            file_path: Target document file path.
            force_reindex: If True, re-indexes even if already indexed.
            extract_figures: Toggle PDF figure captioning (defaults to AppConfig).

        Returns:
            Dictionary reporting ingestion summary.
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

        # 1. Load document (native text, tables, and RapidOCR for scanned pages)
        doc: Document = load_document(path)

        # 2. Chunk document text and tables
        chunks: list[Chunk] = chunk_document(doc)

        # 3. Figure captioning for PDF documents if enabled
        should_extract_figs = (
            extract_figures
            if extract_figures is not None
            else self.config.enable_figure_extraction
        )
        fig_chunks: list[Chunk] = []
        if should_extract_figs and doc.file_type == "pdf":
            logger.info("Extracting figures and generating visual captions for '%s'...", doc.filename)
            fig_chunks = extract_and_caption_figures(
                pdf_path=path,
                doc_id=doc.doc_id,
                llm_provider=self.llm,
                max_pages=self.config.max_figure_pages,
                start_chunk_idx=len(chunks),
            )
            chunks.extend(fig_chunks)

        if not chunks:
            logger.warning("No chunks generated for document '%s'.", path.name)
            return {
                "status": "empty",
                "filename": path.name,
                "doc_id": doc.doc_id,
                "chunks": 0,
            }

        # 4. Compute dense vector embeddings
        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_documents(chunk_texts)

        # 5. Insert into ChromaDB
        self.vector_store.add_chunks(chunks, embeddings)

        logger.info(
            "Successfully indexed '%s' (%d pages, %d chunks [%d figure captions])",
            doc.filename,
            doc.total_pages,
            len(chunks),
            len(fig_chunks),
        )
        return {
            "status": "indexed",
            "filename": doc.filename,
            "doc_id": doc.doc_id,
            "total_pages": doc.total_pages,
            "chunks": len(chunks),
            "figure_chunks": len(fig_chunks),
        }

    def ask(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> RAGResult:
        """Answer a question grounded strictly in indexed document context."""
        query_clean = query.strip()
        if not query_clean:
            return RAGResult(
                query=query,
                answer="Please provide a valid question.",
                sources=[],
                citations=[],
            )

        chunks = self.retriever.retrieve(
            query=query_clean,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            doc_ids=doc_ids,
        )

        if not chunks:
            return RAGResult(
                query=query_clean,
                answer=REFUSAL_MESSAGE,
                sources=[],
                citations=[],
                model_name=self.config.gemini_model_id,
            )

        prompt = format_rag_prompt(query_clean, chunks)

        response: LLMResponse = self.llm.generate_text(
            prompt=prompt,
            system_instruction=SYSTEM_GROUNDING_PROMPT,
            temperature=0.1,
        )

        citations = list(set(re.findall(r"\[Doc:\s*[^,]+,\s*Page:\s*[\d\.]+\]", response.content)))

        return RAGResult(
            query=query_clean,
            answer=response.content.strip(),
            sources=chunks,
            citations=citations,
            model_name=response.model_name,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )

    def ask_image(
        self,
        image_input: bytes | Path | str,
        query: str,
    ) -> VisionResult:
        """Answer a question directly about an image using computer vision."""
        raw_bytes, mime_type = load_image_bytes(image_input)
        answer = self.vision.analyze_image(raw_bytes, query)
        return VisionResult(
            query=query,
            answer=answer,
            mime_type=mime_type,
            model_name=self.config.gemini_model_id,
        )

    def ask_mixed(
        self,
        query: str,
        image_input: bytes | Path | str,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> MixedResult:
        """Execute a mixed query combining an attached image and retrieved document context.

        Sends the image and retrieved document excerpts together in a single multimodal call.
        """
        raw_bytes, mime_type = load_image_bytes(image_input)
        chunks = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            doc_ids=doc_ids,
        )

        prompt = format_mixed_prompt(query, chunks)

        response: LLMResponse = self.llm.generate_multimodal(
            prompt=prompt,
            image_bytes=raw_bytes,
            mime_type=mime_type,
            system_instruction=SYSTEM_MIXED_PROMPT,
            temperature=0.2,
        )

        citations = list(set(re.findall(r"\[Doc:\s*[^,]+,\s*Page:\s*[\d\.]+\]", response.content)))

        return MixedResult(
            query=query,
            answer=response.content.strip(),
            sources=chunks,
            citations=citations,
            mime_type=mime_type,
            model_name=response.model_name,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )

    def list_documents(self) -> list[dict[str, Any]]:
        return self.vector_store.list_documents()

    def delete_document(self, doc_id: str) -> int:
        return self.vector_store.delete_document(doc_id)


# Backward compatibility alias
RAGPipeline = MultimodalPipeline
