"""
Document parser for the SAM Vendor Contract Analyzer.

Responsibilities:
- Accept an uploaded PDF or DOCX file (as bytes, from Streamlit's file_uploader)
- Extract text page-by-page (PDF) or in reading order (DOCX)
- Fall back to OCR automatically for pages that have little/no extractable
  text (i.e. scanned/image-only pages)
- Return a structured result that preserves page numbers, so later on the
  LLM extraction step can cite "page N" for every field it pulls out

No data is written to permanent storage. Everything lives in a per-session
temp directory (see SessionWorkspace below) that the caller is responsible
for cleaning up at the end of the session.
"""

from __future__ import annotations

import io
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf  # PyMuPDF, imported as pymupdf (fitz is the deprecated alias)
import pytesseract
from docx import Document as DocxDocument
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# If a PDF page yields fewer than this many characters of extractable text,
# we treat it as a scanned/image page and OCR it instead.
MIN_CHARS_FOR_NATIVE_TEXT = 20

# DPI to render PDF pages at before handing them to Tesseract. Higher = more
# accurate OCR but slower. 300 is a good default for contract scans.
OCR_RENDER_DPI = 300


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PageResult:
    page_number: int  # 1-indexed
    text: str
    source: str  # "native" or "ocr"


@dataclass
class ParsedDocument:
    filename: str
    file_type: str  # "pdf" or "docx"
    pages: list[PageResult] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Concatenated text with page markers, ready to feed to an LLM."""
        parts = []
        for p in self.pages:
            parts.append(f"\n\n--- Page {p.page_number} ---\n{p.text}")
        return "".join(parts).strip()

    @property
    def ocr_page_count(self) -> int:
        return sum(1 for p in self.pages if p.source == "ocr")

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)


# ---------------------------------------------------------------------------
# Session workspace (temp-only storage, no persistence)
# ---------------------------------------------------------------------------

class SessionWorkspace:
    """
    A throwaway temp directory scoped to one user session.

    Usage:
        ws = SessionWorkspace()
        path = ws.save_upload(uploaded_file.name, uploaded_file.getvalue())
        ... do work ...
        ws.cleanup()   # call this when the session/download is done

    Nothing here touches a database or any location outside the OS temp
    directory, and cleanup() removes it entirely.
    """

    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.dir = Path(tempfile.mkdtemp(prefix=f"sam_analyzer_{self.session_id}_"))

    def save_upload(self, filename: str, file_bytes: bytes) -> Path:
        safe_name = Path(filename).name  # strip any path components
        dest = self.dir / safe_name
        dest.write_bytes(file_bytes)
        return dest

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# ---------------------------------------------------------------------------
# PDF parsing (native text + OCR fallback)
# ---------------------------------------------------------------------------

def _ocr_page(page: "pymupdf.Page") -> str:
    """Render a PDF page to an image and OCR it with Tesseract."""
    zoom = OCR_RENDER_DPI / 72  # PDF base is 72 DPI
    matrix = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)


def parse_pdf(file_path: Path) -> ParsedDocument:
    doc = ParsedDocument(filename=file_path.name, file_type="pdf")

    with pymupdf.open(file_path) as pdf:
        for i, page in enumerate(pdf, start=1):
            native_text = page.get_text().strip()

            if len(native_text) >= MIN_CHARS_FOR_NATIVE_TEXT:
                doc.pages.append(PageResult(page_number=i, text=native_text, source="native"))
            else:
                # Likely a scanned page (or a blank one) -- try OCR
                ocr_text = _ocr_page(page).strip()
                doc.pages.append(PageResult(page_number=i, text=ocr_text, source="ocr"))

    return doc


# ---------------------------------------------------------------------------
# DOCX parsing
# ---------------------------------------------------------------------------

def parse_docx(file_path: Path) -> ParsedDocument:
    """
    DOCX has no native concept of "pages" (pagination is a rendering detail),
    so we treat the whole document as a single logical page, but we do walk
    paragraphs AND tables in document order so nothing gets silently dropped.
    """
    docx_doc = DocxDocument(file_path)

    chunks: list[str] = []
    for element in docx_doc.element.body:
        tag = element.tag.split("}")[-1]
        if tag == "p":
            para_text = "".join(node.text or "" for node in element.iter() if node.tag.endswith("}t"))
            if para_text.strip():
                chunks.append(para_text)
        elif tag == "tbl":
            chunks.append(_table_to_text(element))

    full_text = "\n".join(chunks).strip()

    doc = ParsedDocument(filename=file_path.name, file_type="docx")
    doc.pages.append(PageResult(page_number=1, text=full_text, source="native"))
    return doc


def _table_to_text(tbl_element) -> str:
    """Flatten a docx table XML element into pipe-separated rows."""
    rows_text = []
    for row in tbl_element.iter():
        if row.tag.endswith("}tr"):
            cells = [
                "".join(node.text or "" for node in cell.iter() if node.tag.endswith("}t"))
                for cell in row if cell.tag.endswith("}tc")
            ]
            rows_text.append(" | ".join(c.strip() for c in cells))
    return "\n".join(rows_text)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_document(file_path: Path) -> ParsedDocument:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(file_path)
    elif suffix in (".docx",):
        return parse_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Only .pdf and .docx are supported.")
