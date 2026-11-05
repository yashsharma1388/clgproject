"""
Document Chunking and Table Preservation Module.

Splits documents into overlapping chunks using RecursiveCharacterTextSplitter
while preserving structured tables as indivisible semantic units.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from mmassistant.ingestion.loaders import Document

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents an indexable chunk with semantic text and retrieval metadata."""

    chunk_id: str
    doc_id: str
    filename: str
    page: int
    chunk_type: str  # 'text' | 'table' | 'figure_caption' | 'ocr'
    chunk_index: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        """Convert chunk attributes to a ChromaDB-compatible metadata dictionary."""
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "page": int(self.page),
            "chunk_type": self.chunk_type,
            "chunk_index": int(self.chunk_index),
            **{k: v for k, v in self.metadata.items() if isinstance(v, (str, int, float, bool))},
        }


def chunk_document(
    document: Document,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """Segment a Document into indexable chunks, keeping tables intact.

    Args:
        document: The loaded Document instance.
        chunk_size: Target maximum character length for narrative text chunks.
        chunk_overlap: Overlapping character window to preserve context boundaries.

    Returns:
        A list of Chunk instances ready for embedding and vector storage.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        strip_whitespace=True,
    )

    chunks: list[Chunk] = []
    global_chunk_idx = 0

    for page in document.pages:
        # 1. Process and preserve tables intact as dedicated chunks
        for tbl_idx, table_md in enumerate(page.tables):
            table_clean = table_md.strip()
            if not table_clean:
                continue

            chunk_id = f"{document.doc_id}_p{page.page_number}_tbl{tbl_idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    filename=document.filename,
                    page=page.page_number,
                    chunk_type="table",
                    chunk_index=global_chunk_idx,
                    text=f"[TABLE on Page {page.page_number}]:\n{table_clean}",
                    metadata={
                        "doc_hash": document.doc_hash,
                        "table_index": tbl_idx,
                    },
                )
            )
            global_chunk_idx += 1

        # 2. Process narrative text on the page
        page_text = page.text.strip()
        if not page_text:
            continue

        raw_splits = splitter.split_text(page_text)
        for split_idx, split_text in enumerate(raw_splits):
            clean_split = split_text.strip()
            if not clean_split:
                continue

            chunk_id = f"{document.doc_id}_p{page.page_number}_c{split_idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    filename=document.filename,
                    page=page.page_number,
                    chunk_type="text",
                    chunk_index=global_chunk_idx,
                    text=clean_split,
                    metadata={"doc_hash": document.doc_hash},
                )
            )
            global_chunk_idx += 1

    logger.info(
        "Document '%s' chunked into %d chunks (%d text, %d tables)",
        document.filename,
        len(chunks),
        sum(1 for c in chunks if c.chunk_type == "text"),
        sum(1 for c in chunks if c.chunk_type == "table"),
    )
    return chunks
