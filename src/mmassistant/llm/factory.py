"""
LLM Provider Factory.

Allows dynamically switching between local offline models (Ollama Phi-3 / TinyLlama)
and cloud models (Google Gemini).
"""

from __future__ import annotations

import logging
from typing import Literal

from mmassistant.config import AppConfig, get_config
from mmassistant.llm.base import LLMProvider

logger = logging.getLogger(__name__)


def get_llm_provider(
    provider_type: Literal["gemini", "ollama"] | str | None = None,
    model_id: str | None = None,
) -> LLMProvider:
    """Instantiate and return the configured LLMProvider.

    Args:
        provider_type: 'gemini' or 'ollama'. If omitted, reads from config.
        model_id: Optional model override.

    Returns:
        Configured LLMProvider instance.
    """
    cfg: AppConfig = get_config()
    chosen_type = (provider_type or cfg.llm_provider).strip().lower()

    if chosen_type == "ollama":
        from mmassistant.llm.ollama import OllamaProvider

        provider = OllamaProvider(model_id=model_id)
        if provider.is_available():
            logger.info("Using local Ollama provider with model '%s'", provider.model_id)
            return provider
        else:
            logger.warning(
                "Ollama selected but server at '%s' is not responding. "
                "Falling back to Gemini.",
                cfg.ollama_base_url,
            )

    # Default to Gemini
    from mmassistant.llm.gemini import GeminiProvider

    return GeminiProvider(model_id=model_id)
