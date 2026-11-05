"""
Unit tests for Embedder interface and BGE formatting conventions.
"""

from unittest.mock import MagicMock, patch
import numpy as np

from mmassistant.embeddings import BGEEmbedder, Embedder


def test_bge_embedder_query_vs_document_formatting():
    """Verify BGE embedder formats queries with instruction and documents raw."""
    embedder = BGEEmbedder(model_name="mock-bge")

    mock_model = MagicMock()
    # Mock encode returning 384-dim numpy arrays
    mock_model.encode.side_effect = lambda texts, **kwargs: (
        np.ones((len(texts), 384)) if isinstance(texts, list) else np.ones(384)
    )
    embedder._model = mock_model
    embedder._dimension = 384

    # 1. Embed query
    query = "What is the mission slope limit?"
    q_vec = embedder.embed_query(query)
    assert len(q_vec) == 384
    # Check that model.encode was called with the instruction prefix
    called_arg = mock_model.encode.call_args[0][0]
    assert called_arg.startswith(embedder.query_instruction)
    assert query in called_arg

    # 2. Embed documents
    docs = ["Doc chunk 1 text", "Doc chunk 2 text"]
    d_vecs = embedder.embed_documents(docs)
    assert len(d_vecs) == 2
    assert len(d_vecs[0]) == 384
    # Check that documents were passed without query prefix
    doc_arg = mock_model.encode.call_args[0][0]
    assert doc_arg == docs
