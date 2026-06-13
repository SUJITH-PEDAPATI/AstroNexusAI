# backend/ingestion/__init__.py
from .paper_ingestion import ingest_paper, ingest_batch
from .models import PaperMetadata, RawDocument, IngestedChunk

__all__ = ["ingest_paper", "ingest_batch", "PaperMetadata", "RawDocument", "IngestedChunk"]
