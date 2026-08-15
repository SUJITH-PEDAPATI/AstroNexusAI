from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    """Metadata extracted from a research paper."""

    paper_id: str = Field(default_factory=lambda: str(uuid4()))
    title: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    doi: Optional[str] = None
    year: Optional[int] = None
    abstract: Optional[str] = None
    source_file: Optional[str] = None          # original filename
    file_type: Optional[str] = None            # pdf, docx, md, txt
    page_count: Optional[int] = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class RawDocument(BaseModel):
    """
    Output from any loader — raw text + metadata before chunking.

    paper_id is the SINGLE stable identifier that flows through the
    entire pipeline:  loader → chunker → Qdrant payload → Neo4j Paper node.

    It is taken from metadata.paper_id (not generated separately) so that
    Qdrant and Neo4j always refer to the same UUID.
    """

    full_text: str
    metadata: PaperMetadata
    pages: list[str] = Field(default_factory=list)   # per-page text for PDFs

    @property
    def paper_id(self) -> str:
        """Single source of truth: always the metadata paper_id."""
        return self.metadata.paper_id


class IngestedChunk(BaseModel):
    """A single chunk ready for embedding and Qdrant upsert."""

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    paper_id: str
    chunk_index: int
    text: str
    section: Optional[str] = None              # e.g. "Abstract", "Introduction"
    page_num: Optional[int] = None
    metadata: PaperMetadata