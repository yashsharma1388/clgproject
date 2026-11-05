"""
Unit tests for InputRouter pipeline classification.
"""

from pathlib import Path
from mmassistant.router import InputRouter, InputType


def test_router_document_ingest():
    """Verify document file uploads are routed to DOCUMENT_INGEST."""
    route = InputRouter.route(file_path="manual.pdf", query=None)
    assert route == InputType.DOCUMENT_INGEST

    route_docx = InputRouter.route(file_path="report.docx", query=None)
    assert route_docx == InputType.DOCUMENT_INGEST


def test_router_text_query():
    """Verify text questions without images route to TEXT_DOC_QUERY."""
    route = InputRouter.route(query="What is the operating voltage?")
    assert route == InputType.TEXT_DOC_QUERY


def test_router_image_query():
    """Verify image uploads with questions route to IMAGE_QUERY."""
    route = InputRouter.route(
        query="What trend is shown in this graph?",
        image_bytes=b"fake_png_data",
        is_mixed_mode=False,
    )
    assert route == InputType.IMAGE_QUERY


def test_router_mixed_query():
    """Verify queries with both image and mixed mode enabled route to MIXED_QUERY."""
    route = InputRouter.route(
        query="Compare this diagram with the indexed architecture specs",
        image_bytes=b"fake_png_data",
        has_indexed_docs=True,
        is_mixed_mode=True,
    )
    assert route == InputType.MIXED_QUERY
