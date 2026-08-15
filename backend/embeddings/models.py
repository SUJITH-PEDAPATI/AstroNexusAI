from __future__ import annotations

from backend.ingestion.models import IngestedChunk
from pydantic import BaseModel, Field


class EmbeddedChunk(BaseModel):
    """
    An IngestedChunk enriched with its BAAI dense vector.
    This is the final object upserted into Qdrant.
    """

    chunk: IngestedChunk
    vector: list[float]              # 1024-dim dense vector from Qwen3-Embedding
    vector_dim: int = 384
    embedding_model: str = "BAAI/bge-m3" # IImplemented the embedding model name as a string field to keep track of which embedding model was used for this chunk. This can be useful for future reference or if multiple embedding models are used in the system.   

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def paper_id(self) -> str:
        return self.chunk.paper_id

    def qdrant_payload(self) -> dict:
        """
        Flat payload dict stored alongside the vector in Qdrant.
        Enables metadata filtering during retrieval.
        """
        return {
            "chunk_id":    self.chunk.chunk_id,
            "paper_id":    self.chunk.paper_id,
            "chunk_index": self.chunk.chunk_index,
            "text":        self.chunk.text,
            "section":     self.chunk.section,
            "page_num":    self.chunk.page_num,
            # Paper-level metadata
            "title":       self.chunk.metadata.title,
            "authors":     self.chunk.metadata.authors,
            "doi":         self.chunk.metadata.doi,
            "year":        self.chunk.metadata.year,
            "source_file": self.chunk.metadata.source_file,
        }