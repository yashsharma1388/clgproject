"""
Ollama Local LLM and Vision Provider.

Connects to a local Ollama daemon (default http://localhost:11434) to run
Small Language Models (e.g. Phi-3, TinyLlama, LLaVA) completely offline
with zero API cost, zero rate limits, and full data privacy.
"""

from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.request
from typing import Any

from mmassistant.config import AppConfig, get_config
from mmassistant.llm.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """LLM provider backed by a local Ollama instance."""

    def __init__(
        self,
        base_url: str | None = None,
        model_id: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        """Initialize the local Ollama provider.

        Args:
            base_url: Ollama HTTP endpoint (defaults to http://localhost:11434).
            model_id: Model tag (e.g. 'phi3:latest', 'tinyllama:latest').
            timeout_seconds: Request timeout in seconds.
        """
        cfg: AppConfig = get_config()
        self.base_url = (base_url or cfg.ollama_base_url).rstrip("/")
        self.model_id = (model_id or cfg.ollama_model_id).strip()
        self.timeout_seconds = timeout_seconds

    def test_connection(self) -> tuple[bool, str]:
        """Test API connectivity and model availability.

        Returns:
            Tuple of (success_boolean, status_message).
        """
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = [m.get("name", "") for m in data.get("models", [])]
                    return True, f"Ollama online with {len(models)} models ({', '.join(models[:3])})."
            return False, f"Unexpected Ollama status code {resp.status}."
        except Exception as exc:
            return False, f"Cannot reach Ollama at {self.base_url}: {exc}"

    def is_available(self) -> bool:
        """Check if local Ollama daemon is reachable and responding."""
        success, _ = self.test_connection()
        return success

    def list_installed_models(self) -> list[str]:
        """Return list of locally downloaded models in Ollama."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [m.get("name", "") for m in data.get("models", [])]
        except Exception as exc:
            logger.warning("Could not list Ollama models: %s", exc)
            return []

    def generate_text(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Generate text using local Ollama model via /api/chat."""
        messages: list[dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_id,
            "messages": messages,
            "options": {
                "temperature": temperature,
            },
            "stream": False,
        }

        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            content = data.get("message", {}).get("content", "").strip()
            prompt_eval_count = data.get("prompt_eval_count")
            eval_count = data.get("eval_count")

            return LLMResponse(
                content=content,
                model_name=f"ollama/{self.model_id}",
                prompt_tokens=prompt_eval_count,
                completion_tokens=eval_count,
                raw_response=data,
            )

        except urllib.error.URLError as url_err:
            error_msg = (
                f"Failed to connect to local Ollama at {self.base_url}. "
                "Ensure 'ollama serve' is running in the background."
            )
            logger.error("%s (%s)", error_msg, url_err)
            raise ConnectionError(error_msg) from url_err
        except Exception as exc:
            logger.error("Ollama generation failed: %s", exc)
            raise

    def generate_multimodal(
        self,
        prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
        system_instruction: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Generate multimodal output.

        If using a vision-capable Ollama model (e.g. llava, moondream), base64
        encodes the image. If using a text-only model (e.g. phi3), attempts
        Gemini fallback or provides clear diagnostic instruction.
        """
        installed = self.list_installed_models()
        vision_model = next((m for m in installed if any(v in m for v in ["llava", "moondream", "bakllava", "vision"])), None)

        target_model = vision_model or self.model_id
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        messages: list[dict[str, Any]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        messages.append({
            "role": "user",
            "content": prompt,
            "images": [encoded_image],
        })

        payload = {
            "model": target_model,
            "messages": messages,
            "options": {"temperature": temperature},
            "stream": False,
        }

        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            content = data.get("message", {}).get("content", "").strip()
            return LLMResponse(
                content=content,
                model_name=f"ollama/{target_model}",
                prompt_tokens=data.get("prompt_eval_count"),
                completion_tokens=data.get("eval_count"),
                raw_response=data,
            )
        except Exception as exc:
            # If target model does not support images, try Gemini fallback
            logger.warning(
                "Local model '%s' multimodal call failed (%s). Attempting Gemini fallback...",
                target_model,
                exc,
            )
            try:
                from mmassistant.llm.gemini import GeminiProvider
                gemini = GeminiProvider()
                return gemini.generate_multimodal(
                    prompt=prompt,
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                    system_instruction=system_instruction,
                    temperature=temperature,
                )
            except Exception as g_exc:
                logger.error("Multimodal fallback failed: %s", g_exc)
                raise exc
