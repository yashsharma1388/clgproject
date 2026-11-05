"""
Computer Vision and Visual Understanding Module.

Coordinates direct image question answering, chart/diagram interpretation,
and multimodal queries using Google Gemini Vision models.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any
from PIL import Image

from mmassistant.llm.base import LLMProvider, LLMResponse
from mmassistant.llm.gemini import GeminiProvider

logger = logging.getLogger(__name__)

SYSTEM_VISION_PROMPT = """You are an expert Computer Vision and Document Analysis AI.

Your role is to analyze user-provided images, charts, architectural diagrams, screenshots, and visual documents.

GROUNDING & VISUAL RULES:
1. TRUTHFULNESS: Base all statements strictly on visual elements present in the image. Do not invent details that are not visible.
2. CHARTS & GRAPHS: Identify titles, axes, legends, specific data points, and overall trends. Quote exact numbers where legible.
3. ARCHITECTURAL / SYSTEM DIAGRAMS: Identify components, modules, interfaces, and directional flow arrows accurately.
4. TEXT & LABELS: Transcribe any embedded text accurately. If text is blurry or partially obscured, state so explicitly.
5. If the image does not contain enough information to answer the question, state: "The provided image does not contain sufficient visual information to answer this question."
"""


def load_image_bytes(image_input: bytes | Path | str) -> tuple[bytes, str]:
    """Validate and read an image, returning raw bytes and standardized MIME type.

    Supported formats: PNG, JPEG, JPG, WEBP.
    """
    if isinstance(image_input, (Path, str)):
        path = Path(image_input).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {path}")
        raw_bytes = path.read_bytes()
    else:
        raw_bytes = image_input

    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        fmt = (pil_img.format or "PNG").upper()
    except Exception as exc:
        raise ValueError(f"Invalid or corrupt image format: {exc}") from exc

    mime_map = {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "WEBP": "image/webp",
    }
    mime_type = mime_map.get(fmt, "image/png")
    return raw_bytes, mime_type


class VisionEngine:
    """Handles visual reasoning and image question-answering."""

    def __init__(self, llm_provider: LLMProvider | None = None) -> None:
        """Initialize the vision engine with an LLM provider."""
        self.llm = llm_provider or GeminiProvider()

    def analyze_image(
        self,
        image_input: bytes | Path | str,
        query: str,
        system_instruction: str | None = None,
    ) -> str:
        """Analyze an image against a user question or instruction.

        Args:
            image_input: Image bytes or file path.
            query: Question or instruction regarding the image.
            system_instruction: Optional system instruction (defaults to SYSTEM_VISION_PROMPT).

        Returns:
            Grounded textual analysis of the image.
        """
        raw_bytes, mime_type = load_image_bytes(image_input)
        sys_prompt = system_instruction or SYSTEM_VISION_PROMPT

        response: LLMResponse = self.llm.generate_multimodal(
            prompt=query,
            image_bytes=raw_bytes,
            mime_type=mime_type,
            system_instruction=sys_prompt,
            temperature=0.2,
        )
        return response.content.strip()

    def describe_chart_or_diagram(self, image_input: bytes | Path | str) -> str:
        """Convenience method to extract detailed chart/diagram statistics and observations."""
        prompt = (
            "Analyze this diagram or chart in comprehensive detail. "
            "Report: 1) Title and type of diagram, 2) Key labels, categories, and values, "
            "3) Major trends or architectural flows, 4) Summary takeaway."
        )
        return self.analyze_image(image_input, prompt)
