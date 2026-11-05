"""
Reusable Mock LLM Provider for unit and integration testing.
"""

from __future__ import annotations

from typing import Any
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
