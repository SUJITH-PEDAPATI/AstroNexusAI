from __future__ import annotations
 
import logging
from uuid import uuid4
 
from langchain_text_splitters import RecursiveCharacterTextSplitter
 
from backend.ingestion.models import RawDocument, IngestedChunk
from backend.ingestion.chunking.section_detector import split_by_sections, detect_section
 
logger = logging.getLogger(__name__)
 
#Chunk Size of 512 and Chuck Overlap of size 64 are mainted in order to extract better perfomance while chunking
CHUNK_SIZE    = 1024   
CHUNK_OVERLAP = 128    
 
# Separators tried in order — prefer natural paragraph/sentence breaks
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
SKIP_SECTIONS = {
    "References",
    "Acknowledgements",
    "Acknowledgments",   
    "Preamble",
}


def chunk_document(
    doc: RawDocument,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[IngestedChunk]:
    """
    Split a RawDocument into a list of IngestedChunks.
 
    Strategy:
        1. Split the cleaned full_text into named sections (Abstract, Introduction, …)
        2. Apply RecursiveCharacterTextSplitter inside each section
        3. Attach section name + page estimate + paper metadata to every chunk
 
    Returns:
        list[IngestedChunk] — ready for the embedding stage
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
        length_function=len,
    )
 
    sections = split_by_sections(doc.full_text)
    logger.info(f"[Chunker] {len(sections)} sections detected for paper '{doc.metadata.title}'")
 
    chunks: list[IngestedChunk] = []
    chunk_index = 0
 
    for section in sections:
        section_name = section["section"]
        section_text = section["text"]

        if section_name in SKIP_SECTIONS:
            logger.info(f"[Chunker] Skipping section: {section_name}")
            continue
        raw_chunks: list[str] = splitter.split_text(section_text)
 
        for raw_chunk in raw_chunks:
            text = raw_chunk.strip()
            if not text:
                continue
 
            page_num = _estimate_page(doc, text)
 
            chunk = IngestedChunk(
                chunk_id=str(uuid4()),
                paper_id=doc.paper_id,
                chunk_index=chunk_index,
                text=text,
                section=section_name,
                page_num=page_num,
                metadata=doc.metadata,
            )
            chunks.append(chunk)
            chunk_index += 1
 
    logger.info(f"[Chunker] Produced {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks
 
def _estimate_page(doc: RawDocument, chunk_text: str) -> int | None:
    """
    Estimate which page a chunk came from by comparing it against per-page text.
    Returns 1-based page number, or None if pages are unavailable.
    """
    if not doc.pages:
        return None
 
    best_page = 1
    best_score = 0
 
    fingerprint = chunk_text[:80].strip()
 
    for i, page_text in enumerate(doc.pages, start=1):
        if fingerprint in page_text:
            return i
        # Fallback: count shared words
        score = sum(1 for word in fingerprint.split() if word in page_text)
        if score > best_score:
            best_score = score
            best_page = i
 
    return best_page