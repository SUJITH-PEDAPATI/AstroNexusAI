from __future__ import annotations

import re
from typing import Optional


_DOI_RE = re.compile(
    r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)",
    re.IGNORECASE,
)

_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")
_ARXIV_RE = re.compile(
    r"arXiv[:\s]*([\d]{4}\.\d{4,5}(?:v\d+)?)",
    re.IGNORECASE,
)
_AUTHOR_LINE_RE = re.compile(
    r"^([A-Z][a-z]+(?:\s+[A-Z]\.?)+(?:[,\s]+(?:and\s+)?[A-Z][a-z]+(?:\s+[A-Z]\.?)+)*)",
    re.MULTILINE,
)

def extract_doi(text: str) -> Optional[str]:
    match = _DOI_RE.search(text)
    return match.group(1) if match else None

def extract_arxiv_id(text: str) -> Optional[str]:
    match = _ARXIV_RE.search(text)
    return match.group(1) if match else None


def extract_year(text: str) -> Optional[int]:
    """Return the most plausible publication year."""
    matches = _YEAR_RE.findall(text[:3000])   # look in first ~3000 chars
    if not matches:
        return None
    return int(matches[0])

def extract_title_heuristic(text: str, pdf_title: Optional[str] = None) -> Optional[str]:
    """
    Best-effort title extraction.

    Priority:
    1. PDF metadata title (if non-generic)
    2. First non-empty, non-URL line of the document that looks like a title
       (longer than 20 chars, shorter than 200, no trailing colon)
    """
    if pdf_title and len(pdf_title) > 15 and pdf_title.lower() not in ("untitled", "microsoft word"):
        return pdf_title.strip()

    # Walk first 50 lines looking for a title-like line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines[:50]:
        if (
            20 < len(line) < 200
            and not line.endswith(":")
            and not line.startswith("http")
            and not _DOI_RE.search(line)
            and not re.search(r'permission|attribution|copyright|rights reserved', line, re.IGNORECASE)
            and sum(1 for c in line if c.isupper()) / max(len(line), 1) < 0.6  
        ):
            return line
    return None


def extract_abstract(text: str) -> Optional[str]:
    """Extract the abstract section if present."""
    match = re.search(
        r"(?:abstract|summary)[.\s\-—:]+(.+?)(?:\n\n|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        abstract = match.group(1).strip()
        # Trim to reasonable length
        return abstract[:2000] if len(abstract) > 2000 else abstract
    return None


def extract_metadata(
    text: str,
    loader_metadata: dict,
) -> dict:
    """
    Combine loader-level metadata with text-extracted fields.

    Returns a flat dict ready to hydrate PaperMetadata.
    """
    pdf_title = loader_metadata.get("pdf_title") or loader_metadata.get("md_title") or loader_metadata.get("docx_title")

    return {
        "source_file":  loader_metadata.get("source_file"),
        "file_type":    loader_metadata.get("file_type"),
        "page_count":   loader_metadata.get("page_count"),
        "title":        extract_title_heuristic(text, pdf_title),
        "authors":      [],    # advanced author extraction = Week 3 (GraphAgent)
        "doi":          extract_doi(text),
        "arxiv_id":     extract_arxiv_id(text),
        "year":         extract_year(text),
        "abstract":     extract_abstract(text),
    }
