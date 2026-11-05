"""
Unit tests for configuration loading and directory creation.
"""

from pathlib import Path
from mmassistant.config import AppConfig, get_config


def test_app_config_initialization(tmp_path: Path):
    """Verify AppConfig populates default directories and ensures they exist."""
    chroma_dir = tmp_path / "chroma"
    upload_dir = tmp_path / "uploads"
    cache_dir = tmp_path / "cache"

    cfg = AppConfig(
        google_api_key="test_key_123",
        gemini_model_id="gemini-2.0-flash",
        chroma_persist_dir=chroma_dir,
        upload_dir=upload_dir,
        cache_dir=cache_dir,
    )

    assert cfg.google_api_key == "test_key_123"
    assert cfg.gemini_model_id == "gemini-2.0-flash"
    assert cfg.has_api_key is True
    assert chroma_dir.exists()
    assert upload_dir.exists()
    assert cache_dir.exists()


def test_app_config_missing_key():
    """Verify has_api_key returns False for empty or placeholder keys."""
    cfg_empty = AppConfig(google_api_key="")
    assert cfg_empty.has_api_key is False

    cfg_placeholder = AppConfig(google_api_key="your_gemini_api_key_here")
    assert cfg_placeholder.has_api_key is False


def test_get_config_singleton():
    """Verify get_config returns a valid singleton instance."""
    cfg1 = get_config()
    cfg2 = get_config()
    assert cfg1 is cfg2
    assert cfg1.top_k_chunks > 0
