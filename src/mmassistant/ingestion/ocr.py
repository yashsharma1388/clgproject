"""
Optical Character Recognition (OCR) module using RapidOCR.

Provides high-speed offline text extraction from scanned document pages and images
with automatic confidence evaluation and multimodal LLM fallback.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

from mmassistant.config import AppConfig, get_config
from mmassistant.llm.base import LLMProvider

logger = logging.getLogger(__name__)

# Lazy singleton for RapidOCR engine
_ocr_engine: Any = None


def get_ocr_engine() -> Any:
    """Lazy initialize the RapidOCR engine."""
    global _ocr_engine
    if _ocr_engine is None:
        logger.info("Initializing RapidOCR engine...")
        from rapidocr_onnxruntime import RapidOCR

        _ocr_engine = RapidOCR()
    return _ocr_engine


def extract_text_from_image(
    image_input: bytes | Path | str,
    confidence_threshold: float | None = None,
    llm_fallback: LLMProvider | None = None,
) -> tuple[str, float]:
    """Perform OCR on an image and return recognized text and average confidence.

    Args:
        image_input: Raw image bytes, Path object, or file path string.
        confidence_threshold: Minimum required average confidence (defaults to AppConfig).
        llm_fallback: Optional LLMProvider to invoke if OCR confidence is low.

    Returns:
        Tuple of (recognized_text, average_confidence).
    """
    cfg: AppConfig = get_config()
    min_conf = (
        confidence_threshold
        if confidence_threshold is not None
        else cfg.ocr_confidence_threshold
    )

    # Standardize input to bytes
    if isinstance(image_input, (Path, str)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        image_bytes = path.read_bytes()
    else:
        image_bytes = image_input

    engine = get_ocr_engine()
    raw_results, _ = engine(image_bytes)

    if not raw_results:
        logger.debug("RapidOCR detected no text blocks in image.")
        if llm_fallback and cfg.has_api_key:
            logger.info("Attempting Gemini Vision fallback for empty OCR result...")
            return _transcribe_with_vision(image_bytes, llm_fallback)
        return "", 0.0

    lines: list[str] = []
    confidences: list[float] = []

    for item in raw_results:
        # Each item format: [box_coordinates, text_string, confidence_float]
        if len(item) >= 3:
            text = str(item[1]).strip()
            conf = float(item[2])
            if text:
                lines.append(text)
                confidences.append(conf)

    full_text = "\n".join(lines)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    logger.debug(
        "RapidOCR recognized %d lines (avg confidence: %.2f)",
        len(lines),
        avg_conf,
    )

    # If OCR confidence is too low and an LLM fallback is configured, try Vision model
    if avg_conf < min_conf and llm_fallback and cfg.has_api_key:
        logger.warning(
            "RapidOCR confidence (%.2f) below threshold (%.2f). Invoking Gemini Vision fallback...",
            avg_conf,
            min_conf,
        )
        vision_text, vision_conf = _transcribe_with_vision(image_bytes, llm_fallback)
        if vision_text:
            return vision_text, vision_conf

    return full_text, avg_conf


def ocr_pdf_page(
    page: Any,
    dpi: int = 150,
    confidence_threshold: float | None = None,
    llm_fallback: LLMProvider | None = None,
) -> tuple[str, float]:
    """Rasterize a PyMuPDF PDF page and run OCR.

    Args:
        page: PyMuPDF page object (fitz.Page).
        dpi: Rendering resolution for the rasterized pixmap (150 is optimal for speed vs OCR accuracy).
        confidence_threshold: Minimum acceptable confidence.
        llm_fallback: Optional vision model fallback.

    Returns:
        Tuple of (recognized_text, average_confidence).
    """
    pixmap = page.get_pixmap(dpi=dpi)
    image_bytes = pixmap.tobytes("png")
    return extract_text_from_image(
        image_bytes,
        confidence_threshold=confidence_threshold,
        llm_fallback=llm_fallback,
    )


def _transcribe_with_vision(
    image_bytes: bytes,
    llm_provider: LLMProvider,
) -> tuple[str, float]:
    """Fallback transcription using Gemini Vision for heavily degraded or handwritten pages."""
    prompt = (
        "Transcribe all visible printed or handwritten text in this image verbatim. "
        "Preserve line breaks, headings, and numerical values. Do not add conversational commentary."
    )
    try:
        response = llm_provider.generate_multimodal(
            prompt=prompt,
            image_bytes=image_bytes,
            mime_type="image/png",
            temperature=0.0,
        )
        transcribed = response.content.strip()
        logger.info("Gemini Vision fallback successfully transcribed %d characters.", len(transcribed))
        return transcribed, 0.90  # Synthetic high confidence for model output
    except Exception as exc:
        logger.error("Gemini Vision OCR fallback failed: %s", exc)
        return "", 0.0
