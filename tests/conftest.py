"""
Pytest configuration and synthetic fixtures for mm-assistant.
Ensures tests run completely offline without touching external APIs.
"""

from __future__ import annotations

import io
from pathlib import Path
import pytest
from PIL import Image

from mmassistant.config import AppConfig
from tests.mock_provider import MockLLMProvider


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """Fixture providing a mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Generate a simple 100x100 red PNG in memory for testing."""
    img = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def tmp_config(tmp_path: Path) -> AppConfig:
    """Create a temporary AppConfig with ephemeral directories."""
    return AppConfig(
        google_api_key="mock_test_key",
        gemini_model_id="gemini-3.5-flash",
        chroma_persist_dir=tmp_path / "chroma",
        upload_dir=tmp_path / "uploads",
        cache_dir=tmp_path / "cache",
        top_k_chunks=3,
        similarity_threshold=0.3,
        enable_figure_extraction=False,
    )
