"""
Unit tests for LLMProvider and GeminiProvider with mocked API calls.
"""

from typing import Any
from unittest.mock import MagicMock, patch
import pytest

from mmassistant.llm.base import LLMProvider, LLMResponse
from mmassistant.llm.gemini import GeminiProvider


def test_mock_llm_provider(mock_llm: Any, sample_image_bytes: bytes):
    """Test MockLLMProvider text and multimodal generation."""
    resp = mock_llm.generate_text("Hello, world!")
    assert isinstance(resp, LLMResponse)
    assert "mocked response" in resp.content
    assert resp.prompt_tokens == 10
    assert len(mock_llm.generate_text_calls) == 1

    img_resp = mock_llm.generate_multimodal("Describe this", sample_image_bytes)
    assert isinstance(img_resp, LLMResponse)
    assert "Mocked vision analysis" in img_resp.content
    assert len(mock_llm.generate_multimodal_calls) == 1


def test_gemini_provider_missing_key():
    """Verify GeminiProvider raises clear ValueError if API key is empty."""
    provider = GeminiProvider(api_key="")
    with pytest.raises(ValueError, match="Gemini API key is not configured"):
        provider.generate_text("Test query")


@patch("google.genai.Client")
def test_gemini_provider_generate_text_mocked(mock_client_class: MagicMock):
    """Verify GeminiProvider formats calls correctly to google-genai Client."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_resp = MagicMock()
    mock_resp.text = "Grounded answer from Gemini"
    mock_resp.usage_metadata.prompt_token_count = 15
    mock_resp.usage_metadata.candidates_token_count = 20
    mock_client.models.generate_content.return_value = mock_resp

    provider = GeminiProvider(api_key="valid_test_key", model_id="gemini-2.0-flash")
    provider._client = mock_client

    result = provider.generate_text("What is RAG?", system_instruction="Answer factually.")

    assert result.content == "Grounded answer from Gemini"
    assert result.model_name == "gemini-2.0-flash"
    assert result.prompt_tokens == 15
    assert result.completion_tokens == 20
    assert mock_client.models.generate_content.called


@patch("google.genai.Client")
def test_gemini_provider_retry_on_rate_limit(mock_client_class: MagicMock):
    """Verify exponential backoff retries when 429 Rate Limit is encountered."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    # First call raises 429 ResourceExhausted, second call succeeds
    mock_success = MagicMock()
    mock_success.text = "Recovered response after retry"
    mock_success.usage_metadata.prompt_token_count = 10
    mock_success.usage_metadata.candidates_token_count = 10

    mock_client.models.generate_content.side_effect = [
        Exception("429 ResourceExhausted: Quota exceeded"),
        mock_success,
    ]

    provider = GeminiProvider(
        api_key="valid_test_key",
        max_retries=2,
        base_backoff_seconds=0.01,  # Fast backoff for testing
    )
    provider._client = mock_client

    result = provider.generate_text("Test query")
    assert result.content == "Recovered response after retry"
    assert mock_client.models.generate_content.call_count == 2
