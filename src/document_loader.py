"""
Step 1: Raw Text Extraction
Extract text from PDF documents using Docling and split into chunks.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from docling.document_converter import DocumentConverter


@dataclass
class TextChunk:
    """A chunk of text with provenance metadata."""
    text: str
    doc_id: str
    page: int
    chunk_id: int
    start_char: int = 0
    end_char: int = 0


def extract_text_from_pdf(pdf_path: str) -> list[tuple[int, str]]:
    """Extract text from a PDF via Docling, returning (page_number, text) pairs."""
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    full_text = result.document.export_to_markdown()
    # Docling returns a single markdown string; treat as page 1
    pages: list[tuple[int, str]] = []
    if full_text.strip():
        pages.append((1, full_text))
    return pages


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[tuple[int, int]]:
    """Split text into overlapping character-level windows.
    Returns list of (start, end) offsets.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        spans.append((start, end))
        if end >= len(text):
            break
        start += chunk_size - chunk_overlap
    return spans


def load_documents(
    input_dir: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[TextChunk]:
    """Load all PDFs from *input_dir* and return a flat list of TextChunks."""
    all_chunks: list[TextChunk] = []
    chunk_counter = 0

    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(".pdf"):
            continue
        fpath = os.path.join(input_dir, fname)
        doc_id = os.path.splitext(fname)[0]

        pages = extract_text_from_pdf(fpath)
        for page_num, page_text in pages:
            spans = chunk_text(page_text, chunk_size, chunk_overlap)
            for start, end in spans:
                all_chunks.append(TextChunk(
                    text=page_text[start:end],
                    doc_id=doc_id,
                    page=page_num,
                    chunk_id=chunk_counter,
                    start_char=start,
                    end_char=end,
                ))
                chunk_counter += 1

    return all_chunks
