"""
Unit tests for VisionEngine and image loading.
"""

from pathlib import Path
import pytest
from PIL import Image

from mmassistant.vision import VisionEngine, load_image_bytes
from tests.mock_provider import MockLLMProvider


def test_load_image_bytes_png(tmp_path: Path):
    """Test loading and MIME validation for PNG image."""
    img_path = tmp_path / "chart.png"
    img = Image.new("RGB", (50, 50), color="blue")
    img.save(img_path, format="PNG")

    raw_bytes, mime = load_image_bytes(img_path)
    assert len(raw_bytes) > 0
    assert mime == "image/png"


def test_load_image_bytes_jpeg(tmp_path: Path):
    """Test loading and MIME validation for JPEG image."""
    img_path = tmp_path / "photo.jpg"
    img = Image.new("RGB", (50, 50), color="green")
    img.save(img_path, format="JPEG")

    raw_bytes, mime = load_image_bytes(img_path)
    assert len(raw_bytes) > 0
    assert mime == "image/jpeg"


def test_load_image_invalid(tmp_path: Path):
    """Verify invalid image format raises ValueError."""
    corrupt = tmp_path / "bad.png"
    corrupt.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid or corrupt image format"):
        load_image_bytes(corrupt)


def test_vision_engine_analyze(sample_image_bytes: bytes):
    """Test VisionEngine multimodal dispatch with MockLLMProvider."""
    mock_llm = MockLLMProvider(response_text="The image is a solid red square.")
    engine = VisionEngine(llm_provider=mock_llm)

    result = engine.analyze_image(sample_image_bytes, "What color is this?")
    assert "red square" in result
    assert len(mock_llm.generate_multimodal_calls) == 1
