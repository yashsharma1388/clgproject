"""
Unit tests for PDF figure detection, image hash caching, and figure captioning.
"""

from pathlib import Path
import pymupdf

from mmassistant.ingestion.figures import (
    FigureCaptionCache,
    extract_and_caption_figures,
    page_has_visual_figures,
)
from tests.mock_provider import MockLLMProvider


def test_figure_caption_cache(tmp_path: Path):
    """Test get, set, and file persistence for FigureCaptionCache."""
    cache = FigureCaptionCache(cache_dir=tmp_path)
    assert cache.get("hash_123") is None

    cache.set("hash_123", "A bar chart of system telemetry.")
    assert cache.get("hash_123") == "A bar chart of system telemetry."

    # Verify reloading from disk
    cache2 = FigureCaptionCache(cache_dir=tmp_path)
    assert cache2.get("hash_123") == "A bar chart of system telemetry."


def test_page_has_visual_figures_detection(tmp_path: Path):
    """Verify figure detector detects pages with drawings or drawings count."""
    pdf_path = tmp_path / "diagram_test.pdf"
    doc = pymupdf.open()

    # Page 1: Pure text (no figures)
    page1 = doc.new_page()
    page1.insert_text((50, 50), "Plain narrative text without diagrams.")

    # Page 2: Has vector drawing paths
    page2 = doc.new_page()
    for i in range(10):
        page2.draw_rect(pymupdf.Rect(50, 50 + (i * 20), 100, 60 + (i * 20)))

    doc.save(str(pdf_path))
    doc.close()

    doc_read = pymupdf.open(str(pdf_path))
    assert page_has_visual_figures(doc_read[0]) is False
    assert page_has_visual_figures(doc_read[1]) is True
    doc_read.close()


def test_extract_and_caption_figures(tmp_path: Path):
    """Test generating figure_caption chunks from a diagram PDF."""
    pdf_path = tmp_path / "rover_schematic.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    for i in range(8):
        page.draw_rect(pymupdf.Rect(20, 20 + (i * 15), 150, 30 + (i * 15)))
    doc.save(str(pdf_path))
    doc.close()

    mock_llm = MockLLMProvider(
        response_text="Schematic diagram displaying the RAD750 processor and dual bus interconnects."
    )

    chunks = extract_and_caption_figures(
        pdf_path=pdf_path,
        doc_id="rover123",
        llm_provider=mock_llm,
        max_pages=2,
        cache_dir=tmp_path,
    )

    assert len(chunks) == 1
    assert chunks[0].chunk_type == "figure_caption"
    assert "RAD750 processor" in chunks[0].text
    assert chunks[0].page == 1
