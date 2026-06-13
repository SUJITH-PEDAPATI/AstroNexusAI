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
    source_file: Optional[str] = None          
    file_type: Optional[str] = None            
    page_count: Optional[int] = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class RawDocument(BaseModel):
    """Output from any loader — raw text + metadata before chunking."""

    paper_id: str = Field(default_factory=lambda: str(uuid4()))
    full_text: str
    metadata: PaperMetadata
    pages: list[str] = Field(default_factory=list)   


class IngestedChunk(BaseModel):
    """A single chunk ready for embedding and Qdrant upsert."""

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    paper_id: str
    chunk_index: int
    text: str
    section: Optional[str] = None              
    page_num: Optional[int] = None
    metadata: PaperMetadata
