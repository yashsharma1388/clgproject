"""
Configuration management for Multimodal AI Assistant.

Loads environment variables from `.env` and provides strongly typed,
cross-platform configuration paths and hyperparameters.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Base project root (two levels up from this file)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load .env from project root if present
load_dotenv(dotenv_path=BASE_DIR / ".env")


@dataclass
class AppConfig:
    """Application configuration values and directory paths."""

    # Provider Selection ('ollama' or 'gemini')
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    )

    # Local Ollama Settings
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    )
    ollama_model_id: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL_ID", "phi3:latest").strip()
    )

    # Gemini API Settings
    google_api_key: str = field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY", "").strip()
    )
    gemini_model_id: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL_ID", "gemini-3.5-flash-lite").strip()
    )

    # Embedding & Vector Store Settings
    embedding_model_id: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL_ID", "BAAI/bge-small-en-v1.5"
        ).strip()
    )
    chroma_persist_dir: Path = field(
        default_factory=lambda: BASE_DIR
        / os.getenv("CHROMA_PERSIST_DIR", "storage/chroma")
    )
    upload_dir: Path = field(
        default_factory=lambda: BASE_DIR
        / os.getenv("UPLOAD_DIR", "storage/uploads")
    )
    cache_dir: Path = field(
        default_factory=lambda: BASE_DIR / os.getenv("CACHE_DIR", "storage/cache")
    )

    # Retrieval Hyperparameters
    top_k_chunks: int = field(
        default_factory=lambda: int(os.getenv("TOP_K_CHUNKS", "4"))
    )
    similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
    )

    # Ingestion & Vision Feature Toggles
    enable_figure_extraction: bool = field(
        default_factory=lambda: os.getenv(
            "ENABLE_FIGURE_EXTRACTION", "false"
        ).lower()
        in ("true", "1", "yes")
    )
    max_figure_pages: int = field(
        default_factory=lambda: int(os.getenv("MAX_FIGURE_PAGES", "5"))
    )
    ocr_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.60"))
    )

    def __post_init__(self) -> None:
        """Ensure all runtime directories exist on disk."""
        for path in (self.chroma_persist_dir, self.upload_dir, self.cache_dir):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def has_api_key(self) -> bool:
        """Return True if a non-empty Google API key is configured."""
        return bool(self.google_api_key and self.google_api_key != "your_gemini_api_key_here")


# Singleton instance for application-wide access
_config_instance: AppConfig | None = None


def get_config(reload: bool = False) -> AppConfig:
    """Retrieve or initialize the global application configuration."""
    global _config_instance
    if _config_instance is None or reload:
        if reload:
            load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)
        _config_instance = AppConfig()
    return _config_instance
