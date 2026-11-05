"""
Gemini LLM and Vision Provider.

Wraps Google's official `google-genai` SDK with retry logic, rate limit backoff,
multimodal handling, graceful error messages, and automatic model fallback.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

from mmassistant.config import AppConfig, get_config
from mmassistant.llm.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

# Known stable fallback models in priority order
FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-flash-latest"]


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
            model_id: Model name (e.g. 'gemini-3.5-flash'). Defaults to AppConfig.
            max_retries: Number of exponential backoff attempts on rate limits.
            base_backoff_seconds: Initial backoff delay in seconds.
        """
        cfg: AppConfig = get_config()
        self.api_key = (api_key if api_key is not None else cfg.google_api_key).strip()
        self.model_id = (model_id if model_id is not None else cfg.gemini_model_id).strip()
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
                is_transient = (
                    "503" in error_str
                    or "unavailable" in error_str
                    or "timeout" in error_str
                    or "disconnected" in error_str
                    or "protocol" in error_str
                    or "connection" in error_str
                    or "remoteprotocolerror" in error_str
                )

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

    def _call_with_fallback(self, execute_fn_builder: Any) -> tuple[Any, str]:
        """Attempt generation with active model, falling back if model is unavailable."""
        client = self._get_client()
        candidate_models = [self.model_id] + [m for m in FALLBACK_MODELS if m != self.model_id]

        last_err: Exception | None = None
        for model_name in candidate_models:
            call_fn, config = execute_fn_builder(client, model_name)
            try:
                response = self._execute_with_retry(call_fn)
                if model_name != self.model_id:
                    logger.info("Successfully recovered using fallback model '%s'.", model_name)
                return response, model_name
            except Exception as exc:
                err_str = str(exc).lower()
                is_unavailable = "503" in err_str or "404" in err_str or "unavailable" in err_str or "not_found" in err_str
                if is_unavailable and model_name != candidate_models[-1]:
                    logger.warning(
                        "Model '%s' unavailable (%s). Attempting fallback...",
                        model_name,
                        exc,
                    )
                    last_err = exc
                    continue
                raise exc

        if last_err:
            raise last_err
        raise RuntimeError("No models available for generation.")

    def generate_text(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Generate text using Gemini with automatic fallback on transient outages."""
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        )

        def _builder(client: Any, model_name: str) -> tuple[Any, Any]:
            def _call() -> Any:
                return client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
            return _call, config

        response, used_model = self._call_with_fallback(_builder)

        text_content = response.text or ""
        tokens_in = getattr(response.usage_metadata, "prompt_token_count", None) if hasattr(response, "usage_metadata") else None
        tokens_out = getattr(response.usage_metadata, "candidates_token_count", None) if hasattr(response, "usage_metadata") else None

        return LLMResponse(
            content=text_content,
            model_name=used_model,
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
        from google.genai import types

        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        )

        def _builder(client: Any, model_name: str) -> tuple[Any, Any]:
            def _call() -> Any:
                return client.models.generate_content(
                    model=model_name,
                    contents=[image_part, prompt],
                    config=config,
                )
            return _call, config

        response, used_model = self._call_with_fallback(_builder)

        text_content = response.text or ""
        tokens_in = getattr(response.usage_metadata, "prompt_token_count", None) if hasattr(response, "usage_metadata") else None
        tokens_out = getattr(response.usage_metadata, "candidates_token_count", None) if hasattr(response, "usage_metadata") else None

        return LLMResponse(
            content=text_content,
            model_name=used_model,
            prompt_tokens=tokens_in,
            completion_tokens=tokens_out,
            raw_response=response,
        )

    def test_connection(self) -> tuple[bool, str]:
        """Verify API key and model availability with a lightweight ping."""
        try:
            resp = self.generate_text(prompt="Reply with the single word: OK", temperature=0.0)
            if "ok" in resp.content.lower():
                return True, f"Successfully connected to Gemini model '{resp.model_name}'."
            return True, f"Connected to Gemini ({resp.model_name}), received response: '{resp.content.strip()}'."
        except Exception as exc:
            return False, f"Connection failed: {exc}"
