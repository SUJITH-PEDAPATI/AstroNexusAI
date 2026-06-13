from __future__ import annotations

import logging
from pathlib import Path

from backend.loaders.factory_loader import get_loader
from backend.ingestion.text_cleaner import clean_text, clean_pages
from backend.ingestion.metadata_extractor import extract_metadata
from backend.ingestion.models import PaperMetadata, RawDocument

logger = logging.getLogger(__name__)


def ingest_paper(file_path: str | Path) -> RawDocument:
    """
    Full ingestion pipeline for a single research paper.

    Steps:
        1. Route to the correct loader via factory
        2. Clean raw text (strip noise, fix ligatures, normalize whitespace)
        3. Extract metadata (title, DOI, year, abstract)
        4. Return a RawDocument ready for the chunking stage

    Usage:
        doc = ingest_paper("papers/attention_is_all_you_need.pdf")
        print(doc.metadata.title)
        print(doc.metadata.doi)
    """
    path = Path(file_path)
    logger.info(f"[Ingestion] Starting: {path.name}")

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    loader = get_loader(path)
    raw = loader.load()

    raw_text: str = raw["full_text"]
    raw_pages: list[str] = raw.get("pages", [])
    loader_meta: dict = raw.get("metadata", {})

    logger.info(f"[Ingestion] Loaded {len(raw_text):,} chars, {len(raw_pages)} pages")

    # ── Step 2: Clean ─────────────────────────────────────────────────────────
    clean_full_text = clean_text(raw_text)
    clean_page_list = clean_pages(raw_pages)

    logger.info(f"[Ingestion] Cleaned text: {len(clean_full_text):,} chars")

    # ── Step 3: Extract metadata ──────────────────────────────────────────────
    extracted = extract_metadata(clean_full_text, loader_meta)

    paper_metadata = PaperMetadata(
        title=extracted.get("title"),
        authors=extracted.get("authors", []),
        doi=extracted.get("doi"),
        year=extracted.get("year"),
        abstract=extracted.get("abstract"),
        source_file=extracted.get("source_file"),
        file_type=extracted.get("file_type"),
        page_count=extracted.get("page_count"),
    )

    logger.info(
        f"[Ingestion] Metadata — title: '{paper_metadata.title}' | "
        f"DOI: {paper_metadata.doi} | year: {paper_metadata.year}"
    )

    # ── Step 4: Return RawDocument ────────────────────────────────────────────
    return RawDocument(
        full_text=clean_full_text,
        metadata=paper_metadata,
        pages=clean_page_list,
    )


def ingest_batch(file_paths: list[str | Path]) -> list[RawDocument]:
    """
    Ingest multiple papers, skipping failures with a warning.

    Usage:
        docs = ingest_batch(["paper1.pdf", "paper2.pdf"])
    """
    results: list[RawDocument] = []
    for fp in file_paths:
        try:
            results.append(ingest_paper(fp))
        except Exception as e:
            logger.warning(f"[Ingestion] Skipped {fp}: {e}")
    return results
