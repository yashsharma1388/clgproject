"""
Unit tests for the Ollama Local LLM provider.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest

from mmassistant.llm.base import LLMResponse
from mmassistant.llm.factory import get_llm_provider
from mmassistant.llm.ollama import OllamaProvider


def test_ollama_provider_init():
    """Test default initialization and configuration."""
    provider = OllamaProvider(base_url="http://localhost:11434", model_id="phi3:latest")
    assert provider.base_url == "http://localhost:11434"
    assert provider.model_id == "phi3:latest"


def test_ollama_generate_text_mocked():
    """Verify Ollama text generation parses JSON API response correctly."""
    provider = OllamaProvider()

    mock_resp_data = {
        "message": {"content": "Project Falcon is an autonomous exploration rover."},
        "prompt_eval_count": 25,
        "eval_count": 10,
    }

    mock_urlopen = MagicMock()
    mock_urlopen.__enter__.return_value.read.return_value = json.dumps(mock_resp_data).encode("utf-8")
    mock_urlopen.__enter__.return_value.status = 200

    with patch("urllib.request.urlopen", return_value=mock_urlopen):
        resp: LLMResponse = provider.generate_text("Tell me about Project Falcon")
        assert "Project Falcon" in resp.content
        assert resp.model_name == "ollama/phi3:latest"
        assert resp.prompt_tokens == 25
        assert resp.completion_tokens == 10


def test_ollama_is_available_mocked():
    """Test is_available returns True on successful API tags probe."""
    provider = OllamaProvider()
    mock_data = {"models": [{"name": "phi3:latest"}, {"name": "tinyllama:latest"}]}

    mock_urlopen = MagicMock()
    mock_urlopen.__enter__.return_value.read.return_value = json.dumps(mock_data).encode("utf-8")
    mock_urlopen.__enter__.return_value.status = 200

    with patch("urllib.request.urlopen", return_value=mock_urlopen):
        assert provider.is_available() is True
        assert "phi3:latest" in provider.list_installed_models()


def test_factory_returns_ollama_when_configured():
    """Verify factory dispatches to Ollama when requested."""
    with patch.object(OllamaProvider, "is_available", return_value=True):
        provider = get_llm_provider(provider_type="ollama")
        assert isinstance(provider, OllamaProvider)
