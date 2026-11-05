"""
Pytest configuration and synthetic fixtures for mm-assistant.
Ensures tests run completely offline without touching external APIs.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
import pytest
from PIL import Image

from mmassistant.config import AppConfig
from mmassistant.llm.base import LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    """Mock LLM Provider for unit testing."""

    def __init__(self, response_text: str = "This is a mocked response with citation [Doc: test.pdf, Page: 1].") -> None:
        self.response_text = response_text
        self.generate_text_calls: list[dict[str, Any]] = []
        self.generate_multimodal_calls: list[dict[str, Any]] = []

    def generate_text(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        self.generate_text_calls.append({
            "prompt": prompt,
            "system_instruction": system_instruction,
            "temperature": temperature,
        })
        return LLMResponse(
            content=self.response_text,
            model_name="mock-gemini",
            prompt_tokens=10,
            completion_tokens=15,
        )

    def generate_multimodal(
        self,
        prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
        system_instruction: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        self.generate_multimodal_calls.append({
            "prompt": prompt,
            "image_bytes": image_bytes,
            "mime_type": mime_type,
            "system_instruction": system_instruction,
            "temperature": temperature,
        })
        return LLMResponse(
            content="Mocked vision analysis: The image depicts a sample test chart.",
            model_name="mock-gemini-vision",
            prompt_tokens=25,
            completion_tokens=12,
        )

    def test_connection(self) -> tuple[bool, str]:
        return True, "Mock connection successful."


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
        gemini_model_id="gemini-2.0-flash",
        chroma_persist_dir=tmp_path / "chroma",
        upload_dir=tmp_path / "uploads",
        cache_dir=tmp_path / "cache",
        top_k_chunks=3,
        similarity_threshold=0.3,
        enable_figure_extraction=False,
    )
