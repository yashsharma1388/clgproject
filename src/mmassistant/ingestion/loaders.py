"""
Document loaders for PDF, DOCX, TXT, and Markdown files.

Extracts text, structured tables, and page metadata, providing a unified
Document data structure with SHA-256 content deduplication.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DocumentPage:
    """Represents a single page or logical section within a document."""

    page_number: int  # 1-indexed
    text: str
    tables: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """Unified container for an ingested file across supported formats."""

    doc_id: str
    filename: str
    file_type: str
    file_path: Path
    doc_hash: str
    pages: list[DocumentPage]

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text.strip())


def compute_file_hash(path: Path) -> str:
    """Calculate the SHA-256 hash of a file for deduplication."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def _format_table_as_markdown(rows: list[list[Any]]) -> str:
    """Convert a 2D list of cells into a clean Markdown table string."""
    if not rows or not rows[0]:
        return ""

    cleaned_rows = [
        [str(cell or "").replace("\n", " ").strip() for cell in row]
        for row in rows
    ]

    header = cleaned_rows[0]
    separator = ["---"] * len(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]

    for row in cleaned_rows[1:]:
        # Pad or truncate row to match header length
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(row[: len(header)]) + " |")

    return "\n".join(lines)


def load_pdf(path: Path) -> Document:
    """Extract text and tables from a PDF using PyMuPDF."""
    import pymupdf

    doc_hash = compute_file_hash(path)
    doc_id = doc_hash[:16]
    pages: list[DocumentPage] = []

    try:
        pdf_doc = pymupdf.open(str(path))
    except Exception as exc:
        raise ValueError(f"Failed to open PDF document '{path.name}': {exc}") from exc

    try:
        for page_idx in range(len(pdf_doc)):
            page = pdf_doc[page_idx]
            page_num = page_idx + 1

            # Extract native text
            page_text = page.get_text("text").strip()

            # Extract tables if available
            extracted_tables: list[str] = []
            try:
                table_finder = page.find_tables()
                for table in table_finder:
                    table_data = table.extract()
                    md_table = _format_table_as_markdown(table_data)
                    if md_table:
                        extracted_tables.append(md_table)
            except Exception as tbl_err:
                logger.debug("Table extraction skipped on page %d: %s", page_num, tbl_err)

            pages.append(
                DocumentPage(
                    page_number=page_num,
                    text=page_text,
                    tables=extracted_tables,
                    metadata={"page_width": page.rect.width, "page_height": page.rect.height},
                )
            )
    finally:
        pdf_doc.close()

    logger.info("Loaded PDF '%s' with %d pages (hash: %s)", path.name, len(pages), doc_id)
    return Document(
        doc_id=doc_id,
        filename=path.name,
        file_type="pdf",
        file_path=path,
        doc_hash=doc_hash,
        pages=pages,
    )


def load_docx(path: Path) -> Document:
    """Extract paragraphs and tables from a Microsoft Word DOCX file."""
    import docx

    doc_hash = compute_file_hash(path)
    doc_id = doc_hash[:16]

    try:
        doc = docx.Document(str(path))
    except Exception as exc:
        raise ValueError(f"Failed to open DOCX document '{path.name}': {exc}") from exc

    # Group paragraphs and tables into logical sections/pages
    # Word documents don't have rigid pagination; we segment by explicit page breaks or paragraph batches
    pages: list[DocumentPage] = []
    current_page_text: list[str] = []
    current_tables: list[str] = []
    page_number = 1

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # Check for explicit page break in runs
        has_page_break = any(
            "w:br" in run._element.xml and 'type="page"' in run._element.xml
            for run in para.runs
        )

        current_page_text.append(text)

        # Segment if a page break was reached or section reached substantial length
        if has_page_break or len("\n".join(current_page_text)) > 2500:
            pages.append(
                DocumentPage(
                    page_number=page_number,
                    text="\n\n".join(current_page_text),
                    tables=list(current_tables),
                )
            )
            page_number += 1
            current_page_text = []
            current_tables = []

    # Extract tables in DOCX
    for table in doc.tables:
        rows: list[list[str]] = []
        for row in table.rows:
            rows.append([cell.text.strip() for cell in row.cells])
        md_table = _format_table_as_markdown(rows)
        if md_table:
            current_tables.append(md_table)

    if current_page_text or current_tables or not pages:
        pages.append(
            DocumentPage(
                page_number=page_number,
                text="\n\n".join(current_page_text),
                tables=current_tables,
            )
        )

    logger.info("Loaded DOCX '%s' with %d logical pages (hash: %s)", path.name, len(pages), doc_id)
    return Document(
        doc_id=doc_id,
        filename=path.name,
        file_type="docx",
        file_path=path,
        doc_hash=doc_hash,
        pages=pages,
    )


def load_text(path: Path) -> Document:
    """Extract content from plain text (.txt) and Markdown (.md) documents."""
    doc_hash = compute_file_hash(path)
    doc_id = doc_hash[:16]

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="latin-1", errors="replace")
    except Exception as exc:
        raise ValueError(f"Failed to read text file '{path.name}': {exc}") from exc

    # For markdown/text, segment by top-level headers (# Header) or treat as page 1
    # If the text is short (< 3000 chars), single page is natural; if long, split logically
    sections = content.split("\n# ")
    pages: list[DocumentPage] = []

    if len(sections) > 1 and len(content) > 3000:
        for idx, section in enumerate(sections):
            prefix = "" if idx == 0 else "# "
            section_text = (prefix + section).strip()
            if section_text:
                pages.append(DocumentPage(page_number=idx + 1, text=section_text))
    else:
        pages.append(DocumentPage(page_number=1, text=content.strip()))

    logger.info("Loaded text document '%s' with %d pages (hash: %s)", path.name, len(pages), doc_id)
    return Document(
        doc_id=doc_id,
        filename=path.name,
        file_type=path.suffix.lstrip(".").lower(),
        file_path=path,
        doc_hash=doc_hash,
        pages=pages,
    )


def load_document(file_path: str | Path) -> Document:
    """Universal document loader dispatching based on file extension.

    Supported formats: .pdf, .docx, .txt, .md
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Document file not found: {path}")

    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext == ".docx":
        return load_docx(path)
    elif ext in (".txt", ".md"):
        return load_text(path)
    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. Supported formats: PDF, DOCX, TXT, MD."
        )
