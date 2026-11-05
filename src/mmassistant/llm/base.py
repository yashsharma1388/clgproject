"""
Abstract Base Classes for LLM Providers.

Defines the contract for text and multimodal generation, decoupling application
logic from specific model vendors (e.g. Gemini, Ollama, OpenAI).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    """Standardized response container across all LLM backends."""

    content: str
    model_name: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    raw_response: Any = None


class LLMProvider(ABC):
    """Abstract interface for Large Language and Vision Model providers."""

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Generate text from a text prompt.

        Args:
            prompt: User or grounded RAG prompt.
            system_instruction: System prompt enforcing role and grounding rules.
            temperature: Sampling temperature (lower = more deterministic).

        Returns:
            LLMResponse containing generated content and metadata.
        """
        pass

    @abstractmethod
    def generate_multimodal(
        self,
        prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
        system_instruction: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Generate text given both an image and a text prompt.

        Args:
            prompt: Question or instruction regarding the image and context.
            image_bytes: Raw binary bytes of the image.
            mime_type: Image MIME type (e.g. 'image/png', 'image/jpeg').
            system_instruction: Optional system instruction.
            temperature: Sampling temperature.

        Returns:
            LLMResponse containing visual reasoning output.
        """
        pass

    @abstractmethod
    def test_connection(self) -> tuple[bool, str]:
        """Test API connectivity and model availability.

        Returns:
            Tuple of (success_boolean, status_message).
        """
        pass
