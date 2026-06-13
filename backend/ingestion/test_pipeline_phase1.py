"""
Full RAG Pipeline Test — Steps 1 through 5.

Run from project root:
    python -m backend.ingestion.test_pipeline <path_to_file>

Requires:
    HF_API_TOKEN=hf_xxxxxxxxxxxx  in your .env
    Qdrant running on localhost:6333

Output saved to:
    output/ingestion_result.json
    output/chunks_result.json
    output/embedded_result.json
    output/embeddings_full.txt
    output/retrieval_results.txt
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ── Edit these to match your ingested paper topic ──────────────────────────────
TEST_QUERIES = [
    "What is the main contribution of this paper?",
    "What datasets were used for evaluation?",
    "What is the proposed model architecture?",
    "What were the experimental results?",
    "What are the limitations of this approach?",
]


def write_embeddings_txt(embedded, out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ASTRONEXUS AI — EMBEDDING REPORT\n")
        f.write(f"Model   : {embedded[0].embedding_model}\n")
        f.write(f"Dim     : {embedded[0].vector_dim}\n")
        f.write(f"Chunks  : {len(embedded)}\n")
        f.write("=" * 60 + "\n\n")
        for ec in embedded:
            f.write("=" * 60 + "\n")
            f.write(f"CHUNK {ec.chunk.chunk_index}\n")
            f.write("=" * 60 + "\n")
            f.write(f"chunk_id  : {ec.chunk_id}\n")
            f.write(f"paper_id  : {ec.paper_id}\n")
            f.write(f"section   : {ec.chunk.section}\n")
            f.write(f"page      : {ec.chunk.page_num}\n")
            f.write(f"text_len  : {len(ec.chunk.text)} chars\n")
            f.write(f"model     : {ec.embedding_model}\n")
            f.write(f"dim       : {ec.vector_dim}\n")
            f.write(f"\ntext:\n")
            words = ec.chunk.text.split()
            line, lines = [], []
            for word in words:
                if sum(len(w) + 1 for w in line) + len(word) > 80:
                    lines.append(" ".join(line))
                    line = [word]
                else:
                    line.append(word)
            if line:
                lines.append(" ".join(line))
            for l in lines:
                f.write(f"  {l}\n")
            f.write(f"\nvector ({ec.vector_dim} dims):\n")
            for i, val in enumerate(ec.vector):
                f.write(f"  [{i:>4}]  {val:+.8f}\n")
            norm = sum(x * x for x in ec.vector) ** 0.5
            f.write(f"\nnorm      : {norm:.8f}  {'✓ normalized' if abs(norm - 1.0) < 1e-3 else '⚠ check normalization'}\n")
            f.write("-" * 60 + "\n\n")
        f.write("END OF REPORT\n")


def write_retrieval_txt(query_results: list[tuple], rag_context: str, out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ASTRONEXUS AI — RETRIEVAL TEST RESULTS\n")
        f.write("=" * 60 + "\n\n")

        for i, (query, results) in enumerate(query_results, start=1):
            f.write(f"QUERY {i}: {query}\n")
            f.write("-" * 60 + "\n")
            if not results:
                f.write("  ⚠ No results above score threshold.\n")
            else:
                for j, chunk in enumerate(results, start=1):
                    f.write(f"  Result {j}:\n")
                    f.write(f"    score   : {chunk.score:.6f}\n")
                    f.write(f"    paper   : {chunk.title or 'Unknown'} ({chunk.year or 'n/a'})\n")
                    f.write(f"    section : {chunk.section}\n")
                    f.write(f"    page    : {chunk.page_num}\n")
                    f.write(f"    doi     : {chunk.doi or 'n/a'}\n")
                    f.write(f"    text    :\n")
                    words = chunk.text.split()
                    line, lines = [], []
                    for word in words:
                        if sum(len(w) + 1 for w in line) + len(word) > 72:
                            lines.append("      " + " ".join(line))
                            line = [word]
                        else:
                            line.append(word)
                    if line:
                        lines.append("      " + " ".join(line))
                    f.write("\n".join(lines) + "\n\n")
            f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("RAG CONTEXT FORMAT PREVIEW (Query 1)\n")
        f.write("=" * 60 + "\n")
        f.write(rag_context + "\n")
        f.write("\nEND OF RETRIEVAL RESULTS\n")


def run(file_path: str) -> None:
    from backend.ingestion.paper_ingestion import ingest_paper
    from backend.ingestion.chunking import chunk_document
    from backend.embeddings import embed_chunks
    from backend.rag import upsert_chunks, get_collection_info
    from backend.rag.retriever import retrieve, retrieve_for_rag

    path = Path(file_path)
    if not path.exists():
        print(f"[Error] File not found: {path}")
        sys.exit(1)

    # ── Step 1: Ingestion ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Step 1] Ingesting: {path.name}")
    print(f"{'='*60}")
    doc = ingest_paper(path)
    print(f"  title    : {doc.metadata.title}")
    print(f"  doi      : {doc.metadata.doi}")
    print(f"  year     : {doc.metadata.year}")
    print(f"  pages    : {doc.metadata.page_count}")
    print(f"  text_len : {len(doc.full_text):,} chars")

    # ── Step 2: Chunking ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Step 2] Chunking")
    print(f"{'='*60}")
    chunks = chunk_document(doc)
    print(f"  total chunks : {len(chunks)}")
    for section, count in Counter(c.section for c in chunks).most_common():
        print(f"    {section:<25} {count} chunks")

    # ── Step 3: Embedding ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Step 3] Embedding — BAAI/bge-m3 via HF API")
    print(f"{'='*60}")
    embedded = embed_chunks(chunks)
    print(f"  vectors produced : {len(embedded)}")
    print(f"  vector dim       : {embedded[0].vector_dim}")
    norm = sum(x * x for x in embedded[0].vector) ** 0.5
    print(f"  norm check       : {norm:.6f}  {'✓' if abs(norm - 1.0) < 1e-3 else '⚠'}")

    # ── Step 4: Qdrant Upsert ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Step 4] Qdrant Upsert")
    print(f"{'='*60}")
    upsert_chunks(embedded)
    info = get_collection_info()
    print(f"  collection   : {info.get('collection')}")
    print(f"  total_points : {info.get('total_points')}")
    print(f"  vector_size  : {info.get('vector_size')}")
    print(f"  distance     : {info.get('distance')}")
    print(f"  status       : {info.get('status')}")

    # ── Step 5: Retrieval Test ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Step 5] Retrieval Test")
    print(f"{'='*60}")

    query_results = []
    for i, query in enumerate(TEST_QUERIES, start=1):
        print(f"\n  [Query {i}] {query}")
        results = retrieve(query, top_k=3)
        query_results.append((query, results))
        if not results:
            print(f"  ⚠  No results above threshold")
        else:
            for j, chunk in enumerate(results, start=1):
                print(f"    [{j}] score={chunk.score:.4f} | {chunk.section} | p.{chunk.page_num}")
                print(f"         {chunk.text[:100]}...")

    # RAG context preview
    print(f"\n  --- RAG Context Preview (Query 1) ---")
    rag_context = retrieve_for_rag(TEST_QUERIES[0], top_k=3)
    print(rag_context)

    # ── Save all outputs ───────────────────────────────────────────────────────
    out_dir1 = Path("output/test_folder/ingestion_output")
    out_dir1.mkdir(exist_ok=True)

    out_dir2 = Path("output/test_folder/chunking_output")
    out_dir2.mkdir(exist_ok=True)

    out_dir3 = Path("output/test_folder/embedding_output")
    out_dir3.mkdir(exist_ok=True)
    with open(out_dir1 / "ingestion_result.json", "w", encoding="utf-8") as f:
        json.dump(doc.model_dump(), f, indent=2, default=str)

    with open(out_dir2 / "chunks_result.json", "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in chunks], f, indent=2, default=str)

    with open(out_dir3 / "embedded_result.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    **ec.chunk.model_dump(),
                    "vector_preview": ec.vector[:8],
                    "vector_dim": ec.vector_dim,
                    "embedding_model": ec.embedding_model,
                }
                for ec in embedded
            ],
            f, indent=2, default=str
        )
    out_dir4 = Path("output/txt_files")
    out_dir4.mkdir(exist_ok=True)
    write_embeddings_txt(embedded, out_dir4 / "embeddings_full.txt")
    write_retrieval_txt(query_results, rag_context, out_dir4 / "retrieval_results.txt")

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"   Step 1 → output/ingestion_result.json")
    print(f"   Step 2 → output/chunks_result.json")
    print(f"   Step 3 → output/embedded_result.json")
    print(f"   Step 3 → output/embeddings_full.txt")
    print(f"   Step 4 → Qdrant '{info.get('collection')}' — {info.get('total_points')} points")
    print(f"   Step 5 → output/retrieval_results.txt")
    print(f"\n  Week 2 RAG pipeline complete.")
    print(f"  Ready for Week 3 — Knowledge Graph.\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m backend.ingestion.test_pipeline <path_to_file>")
        sys.exit(1)
    run(sys.argv[1])