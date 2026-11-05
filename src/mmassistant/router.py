"""
Input Processing and Query Routing Module.

Classifies incoming multimodal user inputs into four distinct pipeline pathways:
1. Document Ingestion (PDF, DOCX, TXT, MD)
2. Document RAG (Pure text query against vector index)
3. Direct Vision Q&A (Image analysis and OCR)
4. Mixed Multimodal Query (Image + Document vector retrieval combined)
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_DOC_EXTS = {".pdf", ".docx", ".txt", ".md"}
SUPPORTED_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


class InputType(str, Enum):
    """Categorization of incoming requests for downstream pipeline dispatching."""

    DOCUMENT_INGEST = "document_ingest"
    TEXT_DOC_QUERY = "text_doc_query"
    IMAGE_QUERY = "image_query"
    MIXED_QUERY = "mixed_query"


class InputRouter:
    """Dispatches user inputs to the appropriate subsystem."""

    @staticmethod
    def classify_file(file_path: str | Path) -> str:
        """Identify whether a file path points to a document or an image."""
        ext = Path(file_path).suffix.lower()
        if ext in SUPPORTED_DOC_EXTS:
            return "document"
        elif ext in SUPPORTED_IMG_EXTS:
            return "image"
        return "unsupported"

    @staticmethod
    def route(
        query: str | None = None,
        file_path: str | Path | None = None,
        image_bytes: bytes | None = None,
        has_indexed_docs: bool = True,
        is_mixed_mode: bool = False,
    ) -> InputType:
        """Classify the user action based on attached files, text query, and system state.

        Args:
            query: Natural language query text.
            file_path: Optional path to an uploaded document or image file.
            image_bytes: Optional binary image data.
            has_indexed_docs: Whether documents exist in the vector database.
            is_mixed_mode: User flag requesting document retrieval alongside attached image.

        Returns:
            InputType enum specifying the target processing pipeline.
        """
        # 1. Document file ingestion
        if file_path and not query:
            kind = InputRouter.classify_file(file_path)
            if kind == "document":
                return InputType.DOCUMENT_INGEST

        # 2. Check for attached image
        has_image = bool(image_bytes) or (
            file_path and InputRouter.classify_file(file_path) == "image"
        )

        if has_image:
            # If user explicitly asks for mixed comparison or has indexed docs and flags mixed
            if is_mixed_mode and has_indexed_docs:
                return InputType.MIXED_QUERY
            # If there's an image and a query without document retrieval
            return InputType.IMAGE_QUERY

        # 3. Text query over indexed documents
        return InputType.TEXT_DOC_QUERY
