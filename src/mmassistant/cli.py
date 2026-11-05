"""
Command Line Interface (CLI) for Multimodal AI Assistant.

Provides commands to test LLM connectivity, verify configuration, and in later
phases, ingest documents and execute RAG/vision queries without a web browser.
"""

from __future__ import annotations

import argparse
import logging
import sys

from mmassistant.config import get_config
from mmassistant.llm.gemini import GeminiProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mmassistant.cli")


def cmd_config(_args: argparse.Namespace) -> int:
    """Display active configuration paths and model settings."""
    cfg = get_config()
    print("=" * 60)
    print("  Multimodal AI Assistant - Active Configuration")
    print("=" * 60)
    print(f"Gemini Model ID:       {cfg.gemini_model_id}")
    print(f"API Key Configured:    {'Yes (masked)' if cfg.has_api_key else 'NO (Please set GOOGLE_API_KEY in .env)'}")
    print(f"Embedding Model ID:    {cfg.embedding_model_id}")
    print(f"Chroma Persist Dir:    {cfg.chroma_persist_dir}")
    print(f"Uploads Directory:     {cfg.upload_dir}")
    print(f"Cache Directory:       {cfg.cache_dir}")
    print(f"Top-K Chunks:          {cfg.top_k_chunks}")
    print(f"Similarity Threshold:  {cfg.similarity_threshold}")
    print(f"Figure Extraction:     {cfg.enable_figure_extraction}")
    print(f"Max Figure Pages:      {cfg.max_figure_pages}")
    print("=" * 60)
    return 0


def cmd_test_llm(_args: argparse.Namespace) -> int:
    """Test connectivity to Gemini LLM provider."""
    cfg = get_config()
    print("Testing connection to Gemini LLM Provider...")

    if not cfg.has_api_key:
        print("[ERROR] GOOGLE_API_KEY is not set.")
        print("Please create/update a `.env` file in the project root with:")
        print("  GOOGLE_API_KEY=your_actual_gemini_api_key")
        return 1

    try:
        provider = GeminiProvider()
        success, message = provider.test_connection()
        if success:
            print(f"[SUCCESS] {message}")
            return 0
        else:
            print(f"[FAILURE] {message}")
            return 1
    except Exception as exc:
        print(f"[ERROR] Exception during LLM test: {exc}")
        return 1


def main() -> None:
    """Entry point for CLI commands."""
    parser = argparse.ArgumentParser(
        prog="mmassistant",
        description="Multimodal AI Assistant for Intelligent Document & Visual Understanding",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: config
    subparsers.add_parser("config", help="Display active configuration and paths")

    # Command: test-llm
    subparsers.add_parser("test-llm", help="Verify connection to the configured Gemini model")

    # Subcommands to be fully populated in Phase 2 & 3
    ingest_p = subparsers.add_parser("ingest", help="Ingest and index a document (Phase 2)")
    ingest_p.add_argument("path", help="Path to document (PDF, DOCX, TXT, MD)")

    ask_p = subparsers.add_parser("ask", help="Ask a question grounded in indexed documents (Phase 2)")
    ask_p.add_argument("query", help="Natural language question")

    ask_img_p = subparsers.add_parser("ask-image", help="Ask a question about an image (Phase 3)")
    ask_img_p.add_argument("--image", required=True, help="Path to image file (PNG, JPG, WEBP)")
    ask_img_p.add_argument("query", help="Question about the image")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "config":
        sys.exit(cmd_config(args))
    elif args.command == "test-llm":
        sys.exit(cmd_test_llm(args))
    elif args.command in ("ingest", "ask", "ask-image"):
        print(f"Command '{args.command}' will be activated in Phase 2/3. Run 'mmassistant --help' for available commands.")
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
