"""
Gemini LLM and Vision Provider.

Wraps Google's official `google-genai` SDK with retry logic, rate limit backoff,
multimodal handling, and graceful error messages.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

from mmassistant.config import AppConfig, get_config
from mmassistant.llm.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """LLM and Vision provider backed by Google's Gemini models via `google-genai`."""

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str | None = None,
        max_retries: int = 3,
        base_backoff_seconds: float = 2.0,
    ) -> None:
        """Initialize the Gemini Provider.

        Args:
            api_key: Optional Gemini API key. If omitted, loaded from AppConfig.
            model_id: Model name (e.g. 'gemini-2.0-flash'). Defaults to AppConfig.
            max_retries: Number of exponential backoff attempts on rate limits.
            base_backoff_seconds: Initial backoff delay in seconds.
        """
        cfg: AppConfig = get_config()
        self.api_key = (api_key or cfg.google_api_key).strip()
        self.model_id = (model_id or cfg.gemini_model_id).strip()
        self.max_retries = max_retries
        self.base_backoff_seconds = base_backoff_seconds

        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-initialize and return the official google.genai Client."""
        if not self.api_key:
            raise ValueError(
                "Gemini API key is not configured. Please set GOOGLE_API_KEY in "
                "your .env file or pass it to GeminiProvider."
            )

        if self._client is None:
            try:
                from google import genai

                self._client = genai.Client(api_key=self.api_key)
            except ImportError as err:
                raise ImportError(
                    "The `google-genai` package is required. Install it with "
                    "`pip install google-genai`."
                ) from err

        return self._client

    def _execute_with_retry(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute an API call with exponential backoff on rate-limit / transient errors."""
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                error_str = str(exc).lower()
                last_exception = exc

                is_rate_limit = "429" in error_str or "resource_exhausted" in error_str or "quota" in error_str
                is_transient = "503" in error_str or "unavailable" in error_str or "timeout" in error_str

                if (is_rate_limit or is_transient) and attempt < self.max_retries:
                    delay = (self.base_backoff_seconds * (2 ** attempt)) + random.uniform(0.1, 0.5)
                    logger.warning(
                        "Rate limit or transient error encountered (%s). Retrying in %.2fs (attempt %d/%d)...",
                        exc,
                        delay,
                        attempt + 1,
                        self.max_retries,
                    )
                    time.sleep(delay)
                    continue

                logger.error("Gemini API call failed: %s", exc)
                raise exc

        if last_exception:
            raise last_exception

    def generate_text(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Generate text using Gemini."""
        client = self._get_client()
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        )

        def _call() -> Any:
            return client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=config,
            )

        response = self._execute_with_retry(_call)

        text_content = response.text or ""
        tokens_in = getattr(response.usage_metadata, "prompt_token_count", None) if hasattr(response, "usage_metadata") else None
        tokens_out = getattr(response.usage_metadata, "candidates_token_count", None) if hasattr(response, "usage_metadata") else None

        return LLMResponse(
            content=text_content,
            model_name=self.model_id,
            prompt_tokens=tokens_in,
            completion_tokens=tokens_out,
            raw_response=response,
        )

    def generate_multimodal(
        self,
        prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
        system_instruction: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Generate analysis from an image and text prompt."""
        client = self._get_client()
        from google.genai import types

        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        )

        def _call() -> Any:
            return client.models.generate_content(
                model=self.model_id,
                contents=[image_part, prompt],
                config=config,
            )

        response = self._execute_with_retry(_call)

        text_content = response.text or ""
        tokens_in = getattr(response.usage_metadata, "prompt_token_count", None) if hasattr(response, "usage_metadata") else None
        tokens_out = getattr(response.usage_metadata, "candidates_token_count", None) if hasattr(response, "usage_metadata") else None

        return LLMResponse(
            content=text_content,
            model_name=self.model_id,
            prompt_tokens=tokens_in,
            completion_tokens=tokens_out,
            raw_response=response,
        )

    def test_connection(self) -> tuple[bool, str]:
        """Verify API key and model availability with a lightweight ping."""
        try:
            resp = self.generate_text(prompt="Reply with the single word: OK", temperature=0.0)
            if "ok" in resp.content.lower():
                return True, f"Successfully connected to Gemini model '{self.model_id}'."
            return True, f"Connected to Gemini, received response: '{resp.content.strip()}'."
        except Exception as exc:
            return False, f"Connection failed: {exc}"
