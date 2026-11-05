"""
Unit tests for text chunking and table preservation.
"""

from pathlib import Path
from mmassistant.ingestion.chunking import Chunk, chunk_document
from mmassistant.ingestion.loaders import Document, DocumentPage


def test_chunk_document_text_and_tables():
    """Verify document chunking breaks text appropriately while keeping tables intact."""
    table_sample = "| Sensor | Accuracy |\n| --- | --- |\n| Navcam | 2mm |"
    long_narrative = "The autonomous rover utilizes stereoscopic cameras. " * 30

    doc = Document(
        doc_id="testdoc12345678",
        filename="test.pdf",
        file_type="pdf",
        file_path=Path("test.pdf"),
        doc_hash="hash12345678",
        pages=[
            DocumentPage(
                page_number=1,
                text=long_narrative,
                tables=[table_sample],
            ),
            DocumentPage(
                page_number=2,
                text="Short second page summary.",
                tables=[],
            ),
        ],
    )

    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1

    # Check table chunk
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) == 1
    assert "Navcam" in table_chunks[0].text
    assert table_chunks[0].page == 1

    # Check text chunks
    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    assert len(text_chunks) >= 2
    for c in text_chunks:
        assert c.doc_id == "testdoc12345678"
        assert c.filename == "test.pdf"
        assert c.chunk_type == "text"


def test_chunk_to_metadata():
    """Verify chunk metadata dictionary conversion for ChromaDB compatibility."""
    chunk = Chunk(
        chunk_id="doc1_p1_c0",
        doc_id="doc1",
        filename="doc.pdf",
        page=1,
        chunk_type="text",
        chunk_index=0,
        text="Sample text",
        metadata={"custom_flag": True, "score": 0.95},
    )

    meta = chunk.to_metadata()
    assert meta["doc_id"] == "doc1"
    assert meta["filename"] == "doc.pdf"
    assert meta["page"] == 1
    assert meta["chunk_type"] == "text"
    assert meta["custom_flag"] is True
