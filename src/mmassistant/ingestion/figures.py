"""
PDF Figure Extraction and Semantic Captioning Module.

Detects diagrams, charts, and architectural illustrations in PDF pages,
renders them to images, generates semantic descriptions using Gemini Vision,
and indexes the captions as searchable chunks. Implements SHA-256 image caching
and per-document page caps to protect free-tier API quotas.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from mmassistant.config import AppConfig, get_config
from mmassistant.ingestion.chunking import Chunk
from mmassistant.llm.base import LLMProvider

logger = logging.getLogger(__name__)

FIGURE_DESCRIPTION_PROMPT = """Examine this document page and describe any diagrams, charts, flowcharts, graphs, or visual illustrations in concise, factual detail for search indexing.
Include:
- Diagram/Chart titles, axis labels, and units of measurement.
- High-level trends, key data values, or component relationships.
- Architectural layout or workflow sequence if applicable.
If the page contains only text and no diagrams, charts, or figures, reply with: NO_FIGURE."""


class FigureCaptionCache:
    """Persistent JSON cache mapping image SHA-256 hashes to generated captions."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        cfg: AppConfig = get_config()
        self.cache_file = Path(cache_dir or cfg.cache_dir) / "figures.json"
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self.cache_file.exists():
            try:
                self._cache = json.loads(self.cache_file.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Failed to load figure cache: %s. Re-initializing empty.", exc)
                self._cache = {}

    def _save(self) -> None:
        try:
            self.cache_file.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to save figure cache: %s", exc)

    def get(self, img_hash: str) -> str | None:
        return self._cache.get(img_hash)

    def set(self, img_hash: str, caption: str) -> None:
        self._cache[img_hash] = caption
        self._save()


def page_has_visual_figures(page: Any) -> bool:
    """Heuristic check to determine if a PDF page contains visual diagrams or images."""
    try:
        # Check embedded raster images
        images = page.get_images()
        if len(images) > 0:
            return True

        # Check vector drawing paths (e.g. flowcharts, architecture diagrams)
        drawings = page.get_drawings()
        if len(drawings) >= 5:
            return True
    except Exception as exc:
        logger.debug("Visual figure detection skipped: %s", exc)
    return False


def extract_and_caption_figures(
    pdf_path: Path,
    doc_id: str,
    llm_provider: LLMProvider | None = None,
    max_pages: int | None = None,
    start_chunk_idx: int = 1000,
    cache_dir: Path | None = None,
) -> list[Chunk]:
    """Inspect PDF pages for figures, generate vision captions, and return figure_caption chunks.

    Args:
        pdf_path: Path to the target PDF file.
        doc_id: Unique document identifier.
        llm_provider: Multimodal LLM provider (Gemini Vision).
        max_pages: Maximum number of figure pages to caption per document.
        start_chunk_idx: Starting chunk index to avoid collisions with text chunks.
        cache_dir: Optional custom cache directory for figure captions.

    Returns:
        List of Chunk objects with chunk_type="figure_caption".
    """
    cfg: AppConfig = get_config()
    cap_limit = max_pages if max_pages is not None else cfg.max_figure_pages

    if not cfg.has_api_key or llm_provider is None:
        logger.info("Figure captioning skipped: LLM provider or API key not available.")
        return []

    import pymupdf

    cache = FigureCaptionCache(cache_dir=cache_dir or cfg.cache_dir)
    figure_chunks: list[Chunk] = []
    processed_figure_pages = 0

    try:
        pdf_doc = pymupdf.open(str(pdf_path))
    except Exception as exc:
        logger.error("Could not open PDF for figure extraction: %s", exc)
        return []

    try:
        for page_idx in range(len(pdf_doc)):
            if processed_figure_pages >= cap_limit:
                logger.info("Reached maximum figure pages cap (%d). Halting figure extraction.", cap_limit)
                break

            page = pdf_doc[page_idx]
            page_num = page_idx + 1

            if not page_has_visual_figures(page):
                continue

            # Render page to PNG bytes
            pixmap = page.get_pixmap(dpi=150)
            img_bytes = pixmap.tobytes("png")
            img_hash = hashlib.sha256(img_bytes).hexdigest()

            # 1. Check cache
            cached_caption = cache.get(img_hash)
            if cached_caption:
                logger.debug("Figure on page %d found in cache (hash: %s).", page_num, img_hash[:12])
                caption_text = cached_caption
            else:
                # 2. Invoke Gemini Vision to caption the page diagram
                logger.info("Generating vision caption for figure on page %d of '%s'...", page_num, pdf_path.name)
                try:
                    response = llm_provider.generate_multimodal(
                        prompt=FIGURE_DESCRIPTION_PROMPT,
                        image_bytes=img_bytes,
                        mime_type="image/png",
                        temperature=0.2,
                    )
                    caption_text = response.content.strip()
                    if "no_figure" in caption_text.lower():
                        continue

                    cache.set(img_hash, caption_text)
                except Exception as exc:
                    logger.error("Figure captioning failed on page %d: %s", page_num, exc)
                    continue

            processed_figure_pages += 1
            chunk_id = f"{doc_id}_p{page_num}_fig_{processed_figure_pages}"
            figure_chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    filename=pdf_path.name,
                    page=page_num,
                    chunk_type="figure_caption",
                    chunk_index=start_chunk_idx + processed_figure_pages,
                    text=f"[VISUAL DIAGRAM & FIGURE ANALYSIS on Page {page_num}]:\n{caption_text}",
                    metadata={
                        "image_hash": img_hash,
                        "figure_index": processed_figure_pages,
                    },
                )
            )
    finally:
        pdf_doc.close()

    logger.info("Extracted and captioned %d figure chunks from '%s'", len(figure_chunks), pdf_path.name)
    return figure_chunks
