"""
Command Line Interface (CLI) for Multimodal AI Assistant.

Provides commands to test LLM connectivity, verify configuration,
ingest documents, list/delete indexed files, and run RAG/vision queries.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from mmassistant.config import get_config
from mmassistant.llm.gemini import GeminiProvider
from mmassistant.pipeline import RAGPipeline

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


def cmd_ingest(args: argparse.Namespace) -> int:
    """Ingest and index a document into ChromaDB."""
    target_path = Path(args.path)
    if not target_path.exists():
        print(f"[ERROR] Target file does not exist: {target_path}")
        return 1

    print(f"Ingesting document: {target_path.name}...")
    pipeline = RAGPipeline()
    try:
        result = pipeline.ingest_document(target_path, force_reindex=args.force)
        if result["status"] == "skipped":
            print(f"[SKIPPED] Document '{result['filename']}' is already indexed (Doc ID: {result['doc_id']}).")
            print("Use --force to re-index.")
        elif result["status"] == "indexed":
            print(f"[SUCCESS] Successfully indexed '{result['filename']}'!")
            print(f"  Doc ID:      {result['doc_id']}")
            print(f"  Total Pages: {result['total_pages']}")
            print(f"  Chunks:      {result['chunks']}")
        else:
            print(f"[WARNING] Ingestion completed with status: {result['status']}")
        return 0
    except Exception as exc:
        logger.exception("Ingestion failed: %s", exc)
        print(f"[ERROR] Ingestion failed: {exc}")
        return 1


def cmd_list_docs(_args: argparse.Namespace) -> int:
    """List all documents currently indexed in ChromaDB."""
    pipeline = RAGPipeline()
    docs = pipeline.list_documents()
    print("=" * 65)
    print(f"  Indexed Documents in Vector Store (Total: {len(docs)})")
    print("=" * 65)
    if not docs:
        print("  No documents currently indexed. Run 'mmassistant ingest <path>' first.")
        print("=" * 65)
        return 0

    for doc in docs:
        print(f"Filename:    {doc['filename']}")
        print(f"Doc ID:      {doc['doc_id']}")
        print(f"Pages:       {doc['total_pages']} (Pages: {doc['pages']})")
        print(f"Chunks:      {doc['chunk_count']}")
        print("-" * 65)
    print("=" * 65)
    return 0


def cmd_delete_doc(args: argparse.Namespace) -> int:
    """Delete a document by its doc_id."""
    pipeline = RAGPipeline()
    count = pipeline.delete_document(args.doc_id)
    if count > 0:
        print(f"[SUCCESS] Deleted document '{args.doc_id}' ({count} chunks removed).")
    else:
        print(f"[INFO] Document '{args.doc_id}' was not found in the vector store.")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """Execute a grounded RAG query against indexed documents."""
    cfg = get_config()
    if not cfg.has_api_key:
        print("[ERROR] GOOGLE_API_KEY is not configured in .env.")
        return 1

    pipeline = RAGPipeline()
    print(f"Query: {args.query}\nSearching indexed documents...")

    doc_ids = [args.doc_id] if args.doc_id else None
    result = pipeline.ask(args.query, doc_ids=doc_ids)

    print("\n" + "=" * 60)
    print("  ANSWER")
    print("=" * 60)
    print(result.answer)
    print("=" * 60)

    if result.citations:
        print(f"Citations Found: {', '.join(result.citations)}")

    if args.show_sources and result.sources:
        print("\n" + "-" * 60)
        print(f"  RETRIEVED CONTEXT CHUNKS ({len(result.sources)} chunks)")
        print("-" * 60)
        for idx, src in enumerate(result.sources, 1):
            print(f"[{idx}] {src.citation} | Sim: {src.similarity:.3f} | Type: {src.chunk_type}")
            print(f"    {src.text[:180]}...")
            print("-" * 60)

    return 0


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

    # Command: ingest
    ingest_p = subparsers.add_parser("ingest", help="Ingest and index a document (PDF, DOCX, TXT, MD)")
    ingest_p.add_argument("path", help="Path to document file")
    ingest_p.add_argument("--force", action="store_true", help="Force re-indexing even if already indexed")

    # Command: list-docs
    subparsers.add_parser("list-docs", help="List all indexed documents and statistics")

    # Command: delete-doc
    del_p = subparsers.add_parser("delete-doc", help="Delete a document from vector store by Doc ID")
    del_p.add_argument("doc_id", help="Document ID to remove")

    # Command: ask
    ask_p = subparsers.add_parser("ask", help="Ask a question grounded in indexed documents")
    ask_p.add_argument("query", help="Natural language question")
    ask_p.add_argument("--doc-id", default=None, help="Optional Doc ID to restrict search to")
    ask_p.add_argument("--show-sources", action="store_true", default=True, help="Display retrieved source chunks")

    # Command: ask-image (Phase 3)
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
    elif args.command == "ingest":
        sys.exit(cmd_ingest(args))
    elif args.command == "list-docs":
        sys.exit(cmd_list_docs(args))
    elif args.command == "delete-doc":
        sys.exit(cmd_delete_doc(args))
    elif args.command == "ask":
        sys.exit(cmd_ask(args))
    elif args.command == "ask-image":
        print(f"Command '{args.command}' will be activated in Phase 3. Run 'mmassistant --help' for available commands.")
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
