"""
Unit tests for document loaders (PDF, DOCX, TXT, MD).
"""

from pathlib import Path
import docx
import pymupdf
import pytest

from mmassistant.ingestion.loaders import (
    Document,
    compute_file_hash,
    load_docx,
    load_document,
    load_pdf,
    load_text,
)


def test_load_text_markdown(tmp_path: Path):
    """Test loading a markdown file."""
    md_file = tmp_path / "test.md"
    md_file.write_text("# Title\n\nThis is a sample markdown document.", encoding="utf-8")

    doc = load_text(md_file)
    assert isinstance(doc, Document)
    assert doc.filename == "test.md"
    assert doc.file_type == "md"
    assert doc.total_pages == 1
    assert "sample markdown document" in doc.pages[0].text
    assert len(doc.doc_id) == 16


def test_load_pdf_synthetic(tmp_path: Path):
    """Generate a synthetic 2-page PDF with PyMuPDF and verify extraction."""
    pdf_path = tmp_path / "sample.pdf"
    doc_fitz = pymupdf.open()

    # Page 1: Narrative text
    page1 = doc_fitz.new_page()
    page1.insert_text((50, 50), "Project Falcon is an autonomous exploration rover.")

    # Page 2: Text with simulated table or details
    page2 = doc_fitz.new_page()
    page2.insert_text((50, 50), "Subsystems include Dual MMRTG power generation delivering 240 Watts.")

    doc_fitz.save(str(pdf_path))
    doc_fitz.close()

    doc = load_pdf(pdf_path)
    assert doc.total_pages == 2
    assert doc.pages[0].page_number == 1
    assert "Project Falcon" in doc.pages[0].text
    assert doc.pages[1].page_number == 2
    assert "Dual MMRTG" in doc.pages[1].text


def test_load_docx_synthetic(tmp_path: Path):
    """Generate a synthetic Word DOCX file and verify extraction."""
    docx_path = tmp_path / "sample.docx"
    doc_docx = docx.Document()
    doc_docx.add_heading("Autonomous Systems", level=1)
    doc_docx.add_paragraph("High-radiation environments require radiation-hardened computing cores.")

    # Add a table
    table = doc_docx.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text = "Component"
    table.rows[0].cells[1].text = "Status"
    table.rows[1].cells[0].text = "RAD750 CPU"
    table.rows[1].cells[1].text = "Operational"

    doc_docx.save(str(docx_path))

    doc = load_docx(docx_path)
    assert doc.total_pages >= 1
    assert "Autonomous Systems" in doc.pages[0].text
    assert len(doc.pages[0].tables) >= 1
    assert "RAD750 CPU" in doc.pages[0].tables[0]


def test_universal_loader_dispatch(tmp_path: Path):
    """Verify load_document correctly dispatches based on extension."""
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("Plain text notes.", encoding="utf-8")

    doc = load_document(txt_file)
    assert doc.file_type == "txt"
    assert "Plain text notes." in doc.pages[0].text

    bad_file = tmp_path / "archive.zip"
    bad_file.write_text("fake binary", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_document(bad_file)


def test_compute_file_hash(tmp_path: Path):
    """Verify hash remains deterministic for identical contents."""
    f1 = tmp_path / "f1.txt"
    f2 = tmp_path / "f2.txt"
    f1.write_text("identical content", encoding="utf-8")
    f2.write_text("identical content", encoding="utf-8")

    assert compute_file_hash(f1) == compute_file_hash(f2)
