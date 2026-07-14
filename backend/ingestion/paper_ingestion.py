from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from backend.loaders.factory_loader import get_loader
from backend.ingestion.text_cleaner import clean_text, clean_pages
from backend.ingestion.metadata_extractor import extract_metadata
from backend.ingestion.models import PaperMetadata, RawDocument

logger = logging.getLogger(__name__)


def ingest_paper(
    file_path:        str | Path,
    override_paper_id: str | None = None,
) -> RawDocument:
    """
    Full ingestion pipeline for a single research paper.

    Steps:
        1. Route to the correct loader via factory
        2. Clean raw text
        3. Extract metadata
        4. Return a RawDocument ready for chunking

    Args:
        file_path:         Path to the document
        override_paper_id: If provided, use this as paper_id instead of
                           the auto-generated one. Used by the evaluation
                           pipeline to preserve QASPER paper IDs.

    paper_id is now DETERMINISTIC:
        - When override_paper_id is given  → use it directly (QASPER eval)
        - Otherwise → MD5 hash of filename (same file = same ID always)
          This prevents duplicate upserts and fixes evaluation ID mismatch.
    """
    path = Path(file_path)
    logger.info(f"[Ingestion] Starting: {path.name}")

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    loader = get_loader(path)
    raw    = loader.load()

    raw_text:   str       = raw["full_text"]
    raw_pages:  list[str] = raw.get("pages", [])
    loader_meta: dict     = raw.get("metadata", {})

    logger.info(f"[Ingestion] Loaded {len(raw_text):,} chars, {len(raw_pages)} pages")

    # ── Step 2: Clean ─────────────────────────────────────────────────────────
    clean_full_text = clean_text(raw_text)
    clean_page_list = clean_pages(raw_pages)

    logger.info(f"[Ingestion] Cleaned text: {len(clean_full_text):,} chars")

    # ── Step 3: Extract metadata ──────────────────────────────────────────────
    extracted = extract_metadata(clean_full_text, loader_meta)

    paper_metadata = PaperMetadata(
        title=       extracted.get("title"),
        authors=     extracted.get("authors", []),
        doi=         extracted.get("doi"),
        year=        extracted.get("year"),
        abstract=    extracted.get("abstract"),
        source_file= extracted.get("source_file"),
        file_type=   extracted.get("file_type"),
        page_count=  extracted.get("page_count"),
    )

    logger.info(
        f"[Ingestion] Metadata — title: '{paper_metadata.title}' | "
        f"DOI: {paper_metadata.doi} | year: {paper_metadata.year}"
    )

    # ── Step 4: Assign deterministic paper_id ────────────────────────────────
    if override_paper_id:
        # Evaluation mode — preserve the benchmark's paper ID exactly
        paper_id = override_paper_id
        logger.info(f"[Ingestion] Using override paper_id: {paper_id}")
    else:
        # Production mode — deterministic hash of filename
        # Same file always gets same paper_id → dedup works correctly
        paper_id = hashlib.md5(path.name.encode("utf-8")).hexdigest()
        logger.info(f"[Ingestion] Generated paper_id (MD5): {paper_id[:12]}...")

    # ── Step 5: Return RawDocument ────────────────────────────────────────────
    return RawDocument(
        paper_id=  paper_id,
        full_text= clean_full_text,
        metadata=  paper_metadata,
        pages=     clean_page_list,
    )


def ingest_batch(
    file_paths:         list[str | Path],
    override_paper_ids: list[str] | None = None,
) -> list[RawDocument]:
    """
    Ingest multiple papers, skipping failures with a warning.

    Args:
        file_paths:          List of file paths
        override_paper_ids:  Optional list of paper IDs (same order as file_paths)
    """
    results: list[RawDocument] = []

    for i, fp in enumerate(file_paths):
        override_id = (
            override_paper_ids[i]
            if override_paper_ids and i < len(override_paper_ids)
            else None
        )
        try:
            results.append(ingest_paper(fp, override_paper_id=override_id))
        except Exception as e:
            logger.warning(f"[Ingestion] Skipped {fp}: {e}")

    return results