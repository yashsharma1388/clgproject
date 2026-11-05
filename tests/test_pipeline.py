"""
Integration tests for RAGPipeline with mocked LLM and synthetic embeddings.
"""

from pathlib import Path
from typing import Any
import numpy as np

from mmassistant.embeddings import Embedder
from mmassistant.pipeline import RAGPipeline, RAGResult
from mmassistant.prompts import REFUSAL_MESSAGE
from mmassistant.vectorstore import ChromaVectorStore
from tests.conftest import MockLLMProvider


class SyntheticEmbedder(Embedder):
    """Deterministic in-memory embedder for fast offline testing."""

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Generate predictable normalized vectors based on string length
        vectors = []
        for t in texts:
            vec = np.zeros(self._dim)
            idx = len(t) % self._dim
            vec[idx] = 1.0
            vectors.append(vec.tolist())
        return vectors

    def embed_query(self, query: str) -> list[float]:
        vec = np.zeros(self._dim)
        idx = len(query) % self._dim
        vec[idx] = 1.0
        return vec.tolist()


def test_rag_pipeline_end_to_end(tmp_path: Path):
    """Test ingestion, retrieval, and grounded Q&A with MockLLMProvider."""
    # 1. Setup sample document
    doc_path = tmp_path / "falcon_specs.md"
    doc_path.write_text(
        "# Project Falcon\n\nThe MMRTG delivers 240 Watts of electrical power.",
        encoding="utf-8",
    )

    store = ChromaVectorStore(persist_dir=tmp_path / "chroma")
    embedder = SyntheticEmbedder(dim=16)
    mock_llm = MockLLMProvider(
        response_text="The MMRTG generates 240 Watts of power [Doc: falcon_specs.md, Page: 1]."
    )

    pipeline = RAGPipeline(
        llm_provider=mock_llm,
        embedder=embedder,
        vector_store=store,
    )

    # 2. Ingest document
    res_ingest = pipeline.ingest_document(doc_path)
    assert res_ingest["status"] == "indexed"
    assert res_ingest["chunks"] >= 1

    # 3. Ingest again (deduplication check)
    res_duplicate = pipeline.ingest_document(doc_path)
    assert res_duplicate["status"] == "skipped"
    assert res_duplicate["reason"] == "already_indexed"

    # 4. Ask grounded query
    res_ask = pipeline.ask("How much power does the MMRTG deliver?", similarity_threshold=0.0)
    assert isinstance(res_ask, RAGResult)
    assert "240 Watts" in res_ask.answer
    assert len(res_ask.citations) >= 1
    assert "[Doc: falcon_specs.md, Page: 1]" in res_ask.citations
    assert len(res_ask.sources) >= 1


def test_rag_pipeline_refusal_when_no_context(tmp_path: Path):
    """Verify pipeline returns strict refusal when vector store is empty."""
    store = ChromaVectorStore(persist_dir=tmp_path / "chroma_empty")
    embedder = SyntheticEmbedder(dim=16)
    mock_llm = MockLLMProvider()

    pipeline = RAGPipeline(
        llm_provider=mock_llm,
        embedder=embedder,
        vector_store=store,
    )

    result = pipeline.ask("What is the capital of Mars?")
    assert result.answer == REFUSAL_MESSAGE
    assert len(result.sources) == 0
    # Verify LLM was NOT invoked, protecting API quota
    assert len(mock_llm.generate_text_calls) == 0
