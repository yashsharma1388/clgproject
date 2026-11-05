"""
Unit tests for ChromaVectorStore persistence, indexing, and search.
"""

from pathlib import Path
from mmassistant.ingestion.chunking import Chunk
from mmassistant.vectorstore import ChromaVectorStore


def test_chroma_vector_store_lifecycle(tmp_path: Path):
    """Test full lifecycle: add, duplicate check, search, list, and delete."""
    store = ChromaVectorStore(persist_dir=tmp_path / "chroma")
    assert store.total_chunks == 0

    # 1. Create sample chunks
    chunks = [
        Chunk(
            chunk_id="doc1_p1_c0",
            doc_id="doc1",
            filename="falcon.pdf",
            page=1,
            chunk_type="text",
            chunk_index=0,
            text="Project Falcon is an autonomous exploration rover.",
            metadata={"doc_hash": "hash_falcon_123"},
        ),
        Chunk(
            chunk_id="doc1_p2_c1",
            doc_id="doc1",
            filename="falcon.pdf",
            page=2,
            chunk_type="text",
            chunk_index=1,
            text="The power subsystem consists of dual MMRTG units.",
            metadata={"doc_hash": "hash_falcon_123"},
        ),
        Chunk(
            chunk_id="doc2_p1_c0",
            doc_id="doc2",
            filename="rover_ops.docx",
            page=1,
            chunk_type="table",
            chunk_index=0,
            text="| Subsystem | Mass |\n| MMRTG | 43.5kg |",
            metadata={"doc_hash": "hash_ops_456"},
        ),
    ]

    # Sample normalized 4-dim embeddings
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.8, 0.6, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]

    # 2. Add chunks
    added = store.add_chunks(chunks, embeddings)
    assert added == 3
    assert store.total_chunks == 3

    # 3. Deduplication check
    assert store.is_document_indexed("hash_falcon_123") is True
    assert store.is_document_indexed("non_existent_hash") is False

    # 4. Search query
    # Query vector close to doc1_p1
    results = store.search(query_embedding=[1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0]["chunk_id"] == "doc1_p1_c0"
    assert results[0]["similarity"] > 0.99

    # Filtered search by doc_id
    filtered = store.search(
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        top_k=2,
        doc_ids=["doc2"],
    )
    assert len(filtered) == 1
    assert filtered[0]["chunk_id"] == "doc2_p1_c0"

    # 5. List documents
    docs = store.list_documents()
    assert len(docs) == 2
    filenames = {d["filename"] for d in docs}
    assert filenames == {"falcon.pdf", "rover_ops.docx"}

    # 6. Delete document
    deleted_count = store.delete_document("doc1")
    assert deleted_count == 2
    assert store.total_chunks == 1

    remaining_docs = store.list_documents()
    assert len(remaining_docs) == 1
    assert remaining_docs[0]["doc_id"] == "doc2"
