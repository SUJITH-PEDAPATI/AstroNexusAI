"""
AstroNexus AI — Full Pipeline Test
Covers: Ingestion → Chunking → Embedding → Qdrant → Retrieval → Knowledge Graph → Evaluation

Run from project root:
    python -m backend.ingestion.test_pipeline <path_to_file>

Requires:
    HF_API_TOKEN=hf_xxxxxxxxxxxx  in your .env
    Qdrant  running on localhost:6333
    Neo4j   running on localhost:7687

Outputs:
    output/ingestion_result.json
    output/chunks_result.json
    output/embedded_result.json
    output/embeddings_full.txt
    output/retrieval_results.txt
    output/graph_result.json
    output/evaluation/evaluation_report.csv
    output/evaluation/evaluation_summary.csv
    output/evaluation/evaluation_summary.txt
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

# ── Config ─────────────────────────────────────────────────────────────────────
TOP_K             = 5
SCORE_THRESHOLD   = 0.40
USE_RAGAS         = False    # set True if you have OpenAI key
RUN_EVALUATION    = True     # set False to skip QASPER evaluation

TEST_QUERIES = [
    "What is the main contribution of this paper?",
    "What datasets were used for evaluation?",
    "What is the proposed model architecture?",
    "What were the experimental results?",
    "What are the limitations of this approach?",
]

OUT_DIR      = Path("output")
EVAL_OUT_DIR = OUT_DIR / "evaluation"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def section_header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def section_footer(label: str, path: str | None = None) -> None:
    if path:
        print(f"\n  ✓ {label} → {path}")
    else:
        print(f"\n  ✓ {label}")


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
            f.write(
                f"\nnorm : {norm:.8f}  "
                f"{'✓ normalized' if abs(norm - 1.0) < 1e-3 else '⚠ check normalization'}\n"
            )
            f.write("-" * 60 + "\n\n")
        f.write("END OF REPORT\n")


def write_retrieval_txt(
    query_results: list[tuple],
    rag_context:   str,
    out_path:      Path,
) -> None:
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


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(file_path: str) -> None:

    # ── Imports ────────────────────────────────────────────────────────────────
    from backend.ingestion.paper_ingestion      import ingest_paper
    from backend.ingestion.chunking             import chunk_document
    from backend.embeddings                     import embed_chunks
    from backend.rag                            import upsert_chunks, get_collection_info
    from backend.rag.retriever                  import retrieve, retrieve_for_rag
    from backend.graph                          import build_graph_from_document, get_graph_stats

    path = Path(file_path)
    if not path.exists():
        print(f"\n  [Error] File not found: {path}")
        sys.exit(1)

    OUT_DIR.mkdir(exist_ok=True)
    EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — INGESTION
    # ══════════════════════════════════════════════════════════════════════════
    section_header("STEP 1 — INGESTION")

    doc = ingest_paper(path)

    print(f"  file     : {path.name}")
    print(f"  title    : {doc.metadata.title}")
    print(f"  doi      : {doc.metadata.doi}")
    print(f"  year     : {doc.metadata.year}")
    print(f"  pages    : {doc.metadata.page_count}")
    print(f"  text_len : {len(doc.full_text):,} chars")

    with open(OUT_DIR / "ingestion_result.json", "w", encoding="utf-8") as f:
        json.dump(doc.model_dump(), f, indent=2, default=str)

    section_footer("ingestion_result.json", "output/ingestion_result.json")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — CHUNKING
    # ══════════════════════════════════════════════════════════════════════════
    section_header("STEP 2 — CHUNKING")

    chunks = chunk_document(doc)

    print(f"  total chunks : {len(chunks)}")
    print(f"  sections     :")
    for section, count in Counter(c.section for c in chunks).most_common():
        print(f"    {section:<25} {count} chunks")

    with open(OUT_DIR / "chunks_result.json", "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in chunks], f, indent=2, default=str)

    section_footer("chunks_result.json", "output/chunks_result.json")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — EMBEDDING
    # ══════════════════════════════════════════════════════════════════════════
    section_header("STEP 3 — EMBEDDING  (BAAI/bge-m3 via HF API)")

    embedded = embed_chunks(chunks)

    norm = sum(x * x for x in embedded[0].vector) ** 0.5
    print(f"  vectors produced : {len(embedded)}")
    print(f"  vector dim       : {embedded[0].vector_dim}")
    print(f"  model            : {embedded[0].embedding_model}")
    print(f"  norm check       : {norm:.6f}  {'✓' if abs(norm - 1.0) < 1e-3 else '⚠'}")

    with open(OUT_DIR / "embedded_result.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    **ec.chunk.model_dump(),
                    "vector_preview":  ec.vector[:8],
                    "vector_dim":      ec.vector_dim,
                    "embedding_model": ec.embedding_model,
                }
                for ec in embedded
            ],
            f, indent=2, default=str,
        )

    write_embeddings_txt(embedded, OUT_DIR / "embeddings_full.txt")

    section_footer("embedded_result.json + embeddings_full.txt",
                   "output/embedded_result.json")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — QDRANT UPSERT
    # ══════════════════════════════════════════════════════════════════════════
    section_header("STEP 4 — QDRANT UPSERT")

    upsert_chunks(embedded)
    info = get_collection_info()

    print(f"  collection   : {info.get('collection')}")
    print(f"  total_points : {info.get('total_points')}")
    print(f"  vector_size  : {info.get('vector_size')}")
    print(f"  distance     : {info.get('distance')}")
    print(f"  status       : {info.get('status')}")

    section_footer(f"Qdrant '{info.get('collection')}' — {info.get('total_points')} points")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — RETRIEVAL TEST
    # ══════════════════════════════════════════════════════════════════════════
    section_header("STEP 5 — RETRIEVAL TEST")

    query_results = []
    for i, query in enumerate(TEST_QUERIES, start=1):
        print(f"\n  [Query {i}] {query}")
        results = retrieve(query, top_k=TOP_K, score_threshold=SCORE_THRESHOLD)
        query_results.append((query, results))
        if not results:
            print(f"    ⚠ No results above threshold {SCORE_THRESHOLD}")
        else:
            for j, chunk in enumerate(results, start=1):
                print(
                    f"    [{j}] score={chunk.score:.4f} | "
                    f"{chunk.section} | p.{chunk.page_num}"
                )
                print(f"         {chunk.text[:100]}...")

    print(f"\n  --- RAG Context Preview (Query 1) ---")
    rag_context = retrieve_for_rag(TEST_QUERIES[0], top_k=TOP_K)
    print(rag_context)

    write_retrieval_txt(query_results, rag_context, OUT_DIR / "retrieval_results.txt")
    section_footer("retrieval_results.txt", "output/retrieval_results.txt")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6 — KNOWLEDGE GRAPH
    # ══════════════════════════════════════════════════════════════════════════
    section_header("STEP 6 — KNOWLEDGE GRAPH  (Neo4j)")

    extraction = build_graph_from_document(doc)
    stats      = get_graph_stats()

    print(f"  authors   : {[a.name for a in extraction.authors]}")
    print(f"  models    : {[m.name for m in extraction.models]}")
    print(f"  datasets  : {[d.name for d in extraction.datasets]}")
    print(f"  tasks     : {[t.name for t in extraction.tasks]}")
    print(f"  relations : {len(extraction.relations)}")
    print(f"\n  Neo4j stats:")
    print(f"    nodes     : {stats.get('nodes', {})}")
    print(f"    relations : {stats.get('relations', {})}")
    print(f"\n  Visualize → http://localhost:7474")
    print(f"  Cypher    → MATCH (n) RETURN n")

    with open(OUT_DIR / "graph_result.json", "w", encoding="utf-8") as f:
        json.dump(extraction.model_dump(), f, indent=2, default=str)

    section_footer("graph_result.json", "output/graph_result.json")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 7 — EVALUATION  (QASPER benchmark)
    # ══════════════════════════════════════════════════════════════════════════
    if RUN_EVALUATION:
        section_header("STEP 7 — EVALUATION  (QASPER benchmark)")

        from backend.evaluation.dataset_loader     import load_qasper
        from backend.evaluation.retrieval_evaluator import (
            retrieve_for_question, evaluate_retrieval,
        )
        from backend.evaluation.generation_evaluator import evaluate_generation
        from backend.evaluation.answer_generator    import generate_answer
        from backend.evaluation.report_generator    import (
            aggregate_metrics, save_csv, save_summary_csv, save_txt_report,
        )

        # Load a small subset for quick evaluation
        print(f"\n  Loading QASPER (10 papers, 30 questions)...")
        qa_pairs, paper_texts = load_qasper(
            split="validation",
            max_papers=10,
        )
        qa_pairs = qa_pairs[:30]
        print(f"  Loaded {len(qa_pairs)} QA pairs from {len(paper_texts)} papers")

        # Retrieval evaluation
        print(f"\n  Running retrieval evaluation...")
        retrieval_metrics_list = []
        for qa in qa_pairs:
            try:
                result  = retrieve_for_question(qa, top_k=TOP_K)
                metrics = evaluate_retrieval(result)
                retrieval_metrics_list.append(metrics)
            except Exception as e:
                logging.warning(f"  Retrieval skipped {qa.question_id}: {e}")

        # Generation evaluation
        print(f"\n  Running generation evaluation...")
        generation_metrics_list = []
        for qa in qa_pairs:
            try:
                gen_result  = generate_answer(qa, top_k=TOP_K)
                gen_metrics = evaluate_generation(gen_result, use_ragas=USE_RAGAS)
                generation_metrics_list.append(gen_metrics)
            except Exception as e:
                logging.warning(f"  Generation skipped {qa.question_id}: {e}")

        # Aggregate + save
        qa_type_map = {qa.question_id: qa.answer_type for qa in qa_pairs}
        report = aggregate_metrics(
            retrieval_metrics_list,
            generation_metrics_list,
            qa_type_map,
        )

        save_csv(retrieval_metrics_list, generation_metrics_list, EVAL_OUT_DIR)
        save_summary_csv(report, EVAL_OUT_DIR)
        save_txt_report(report, EVAL_OUT_DIR)

        print(f"\n  RETRIEVAL RESULTS")
        print(f"    Hit Rate    : {report.avg_hit_rate:.4f}")
        print(f"    MRR         : {report.avg_mrr:.4f}")
        print(f"    MAP         : {report.avg_map:.4f}")
        print(f"    nDCG        : {report.avg_ndcg:.4f}")
        print(f"    Precision@3 : {report.avg_precision_at_3:.4f}")
        print(f"    Recall@5    : {report.avg_recall_at_5:.4f}")
        print(f"\n  GENERATION RESULTS")
        print(f"    Exact Match : {report.avg_exact_match:.4f}")
        print(f"    F1 Score    : {report.avg_f1:.4f}")
        print(f"    ROUGE-1     : {report.avg_rouge_1:.4f}")
        print(f"    ROUGE-2     : {report.avg_rouge_2:.4f}")
        print(f"    ROUGE-L     : {report.avg_rouge_l:.4f}")
        print(f"    BLEU        : {report.avg_bleu:.4f}")

        section_footer(
            "evaluation_report.csv + evaluation_summary.csv + evaluation_summary.txt",
            "output/evaluation/",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    section_header("PIPELINE COMPLETE")

    print(f"  Step 1 ✓  output/ingestion_result.json")
    print(f"  Step 2 ✓  output/chunks_result.json")
    print(f"  Step 3 ✓  output/embedded_result.json")
    print(f"  Step 3 ✓  output/embeddings_full.txt")
    print(f"  Step 4 ✓  Qdrant '{info.get('collection')}' — {info.get('total_points')} points")
    print(f"  Step 5 ✓  output/retrieval_results.txt")
    print(f"  Step 6 ✓  output/graph_result.json")
    if RUN_EVALUATION:
        print(f"  Step 7 ✓  output/evaluation/evaluation_report.csv")
        print(f"  Step 7 ✓  output/evaluation/evaluation_summary.csv")
        print(f"  Step 7 ✓  output/evaluation/evaluation_summary.txt")

    print(f"\n  Weeks covered : 1 (infra) · 2 (RAG) · 3 (graph) · eval framework")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python -m backend.ingestion.test_pipeline <path_to_file>")
        print("Example: python -m backend.ingestion.test_pipeline data/papers/paper.pdf\n")
        sys.exit(1)
    run(sys.argv[1])