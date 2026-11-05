"""
Integration tests for mixed multimodal queries (image + document retrieval).
"""

from pathlib import Path
from PIL import Image

from mmassistant.pipeline import MixedResult, MultimodalPipeline, VisionResult
from mmassistant.vectorstore import ChromaVectorStore
from tests.mock_provider import MockLLMProvider
from tests.test_pipeline import SyntheticEmbedder


def test_ask_image_pipeline(sample_image_bytes: bytes):
    """Test direct ask_image flow."""
    mock_llm = MockLLMProvider(response_text="The image contains an architectural schematic.")
    pipeline = MultimodalPipeline(llm_provider=mock_llm)

    res = pipeline.ask_image(sample_image_bytes, "Explain this diagram")
    assert isinstance(res, VisionResult)
    assert "architectural schematic" in res.answer
    assert res.mime_type == "image/png"


def test_ask_mixed_pipeline(tmp_path: Path, sample_image_bytes: bytes):
    """Test mixed query combining document retrieval and attached image."""
    doc_path = tmp_path / "system_specs.txt"
    doc_path.write_text("The radar system has an operating bandwidth of 500 MHz.", encoding="utf-8")

    store = ChromaVectorStore(persist_dir=tmp_path / "chroma_mixed")
    embedder = SyntheticEmbedder(dim=16)
    mock_llm = MockLLMProvider(
        response_text="Comparing the attached chart with [Doc: system_specs.txt, Page: 1], the peak aligns with 500 MHz."
    )

    pipeline = MultimodalPipeline(
        llm_provider=mock_llm,
        embedder=embedder,
        vector_store=store,
    )

    pipeline.ingest_document(doc_path)

    mixed_res = pipeline.ask_mixed(
        query="Compare the radar frequency in the diagram with document specifications",
        image_input=sample_image_bytes,
        top_k=2,
    )

    assert isinstance(mixed_res, MixedResult)
    assert "500 MHz" in mixed_res.answer
    assert len(mixed_res.citations) >= 1
    assert "[Doc: system_specs.txt, Page: 1]" in mixed_res.citations
    assert len(mixed_res.sources) >= 1
    assert len(mock_llm.generate_multimodal_calls) == 1
