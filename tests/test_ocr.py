"""
Unit tests for RapidOCR and multimodal transcription fallback.
"""

import io
from pathlib import Path
from PIL import Image, ImageDraw
import pytest

from mmassistant.ingestion.ocr import extract_text_from_image
from tests.mock_provider import MockLLMProvider


@pytest.fixture
def synthetic_ocr_image() -> bytes:
    """Generate an in-memory PNG with clear printed text."""
    img = Image.new("RGB", (300, 80), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 30), "PROJECT FALCON ROVER", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_rapid_ocr_extraction(synthetic_ocr_image: bytes):
    """Verify RapidOCR extracts text accurately and reports positive confidence."""
    text, conf = extract_text_from_image(synthetic_ocr_image)
    assert "PROJECT" in text.upper()
    assert "FALCON" in text.upper()
    assert conf > 0.5


def test_ocr_fallback_on_empty_image():
    """Verify Gemini Vision fallback is triggered when OCR detects no text."""
    # Blank white image has 0 OCR blocks
    blank_img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    blank_img.save(buf, format="PNG")

    mock_llm = MockLLMProvider(response_text="Transcribed by Vision: Emergency Beacon Active")

    text, conf = extract_text_from_image(buf.getvalue(), llm_fallback=mock_llm)
    assert "Emergency Beacon Active" in text
    assert conf == 0.90
    assert len(mock_llm.generate_multimodal_calls) == 1
