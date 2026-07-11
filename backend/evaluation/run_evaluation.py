"""
AstroNexus AI — Complete RAG Evaluation Pipeline.

Run from project root:
    python -m backend.evaluation.run_evaluation

What it does:
    1. Downloads QASPER dataset (20 papers by default)
    2. Ingests papers through your existing pipeline → Qdrant
    3. Runs all questions through retriever
    4. Generates answers via LLM
    5. Computes all retrieval + generation metrics
    6. Saves CSV + TXT report to output/evaluation/

Requires:
    HF_API_TOKEN in .env
    Qdrant running on localhost:6333
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUT_DIR = Path("output/evaluation")


def run(
    max_papers:  int  = 20,      # number of QASPER papers to evaluate
    max_questions: int = 50,     # cap questions for quick runs
    top_k:       int  = 5,       # chunks to retrieve per question
    use_ragas:   bool = False,   # set True if you have OpenAI key for Ragas
    ingest_papers: bool = True,  # set False if papers already in Qdrant
) -> None:

    from backend.evaluation.dataset_loader import load_qasper
    from backend.evaluation.retrieval_evaluator import (
        retrieve_for_question, evaluate_retrieval
    )
    from backend.evaluation.generation_evaluator import evaluate_generation
    from backend.evaluation.answer_generator import generate_answer
    from backend.evaluation.report_generator import (
        aggregate_metrics, save_csv, save_summary_csv, save_txt_report
    )

    # ── Step 1: Load QASPER ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("[Step 1] Loading QASPER dataset")
    print(f"{'='*60}")

    qa_pairs, paper_texts = load_qasper(
        split="validation",
        max_papers=max_papers,
    )

    # Cap questions for quick evaluation
    qa_pairs = qa_pairs[:max_questions]
    print(f"  Questions loaded : {len(qa_pairs)}")
    print(f"  Papers loaded    : {len(paper_texts)}")

    # ── Step 2: Ingest papers into Qdrant ─────────────────────────────────────
    if ingest_papers:
        print(f"\n{'='*60}")
        print("[Step 2] Ingesting papers into Qdrant")
        print(f"{'='*60}")

        from backend.ingestion.paper_ingestion import ingest_paper
        from backend.ingestion.chunking import chunk_document
        from backend.embeddings import embed_chunks
        from backend.rag import upsert_chunks
        from backend.ingestion.models import RawDocument, PaperMetadata
        import tempfile, os

        for paper_id, full_text in paper_texts.items():
            # Write paper text to temp file and ingest
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(full_text)
                tmp_path = tmp.name

            try:
                doc = ingest_paper(tmp_path)
                # Override paper_id to match QASPER's ID
                doc.paper_id = paper_id
                chunks   = chunk_document(doc)
                embedded = embed_chunks(chunks)
                upserted = upsert_chunks(embedded)
                print(f"  Ingested paper {paper_id[:12]}... — {upserted} chunks")
            except Exception as e:
                logger.warning(f"  Skipped {paper_id}: {e}")
            finally:
                os.unlink(tmp_path)
    else:
        print(f"\n[Step 2] Skipping ingestion — using existing Qdrant data")

    # ── Step 3: Retrieval evaluation ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Step 3] Retrieval evaluation — {len(qa_pairs)} questions")
    print(f"{'='*60}")

    retrieval_results  = []
    retrieval_metrics_list = []

    for i, qa in enumerate(qa_pairs, start=1):
        try:
            result  = retrieve_for_question(qa, top_k=top_k)
            metrics = evaluate_retrieval(result)
            retrieval_results.append(result)
            retrieval_metrics_list.append(metrics)

            if i % 10 == 0 or i == len(qa_pairs):
                hit_rate = sum(float(m.hit_rate) for m in retrieval_metrics_list) / i
                mrr      = sum(m.mrr             for m in retrieval_metrics_list) / i
                print(f"  [{i}/{len(qa_pairs)}] Hit Rate={hit_rate:.3f} | MRR={mrr:.3f}")

        except Exception as e:
            logger.warning(f"  Retrieval failed for {qa.question_id}: {e}")

    # ── Step 4: Answer generation + evaluation ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Step 4] Generation evaluation — {len(qa_pairs)} questions")
    print(f"{'='*60}")

    generation_metrics_list = []

    for i, qa in enumerate(qa_pairs, start=1):
        try:
            gen_result = generate_answer(qa, top_k=top_k)
            gen_metrics = evaluate_generation(gen_result, use_ragas=use_ragas)
            generation_metrics_list.append(gen_metrics)

            if i % 10 == 0 or i == len(qa_pairs):
                avg_f1 = sum(m.f1_score for m in generation_metrics_list) / i
                avg_rouge = sum(m.rouge_l for m in generation_metrics_list) / i
                print(f"  [{i}/{len(qa_pairs)}] F1={avg_f1:.3f} | ROUGE-L={avg_rouge:.3f}")

        except Exception as e:
            logger.warning(f"  Generation failed for {qa.question_id}: {e}")

    # ── Step 5: Aggregate + save reports ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("[Step 5] Generating reports")
    print(f"{'='*60}")

    qa_type_map = {qa.question_id: qa.answer_type for qa in qa_pairs}

    report = aggregate_metrics(
        retrieval_metrics_list,
        generation_metrics_list,
        qa_type_map,
    )

    save_csv(retrieval_metrics_list, generation_metrics_list, OUT_DIR)
    save_summary_csv(report, OUT_DIR)
    save_txt_report(report, OUT_DIR)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Questions evaluated : {report.total_questions}")
    print(f"  Answer types        : {report.answer_types}")
    print()
    print("  RETRIEVAL")
    print(f"    Hit Rate    : {report.avg_hit_rate:.4f}")
    print(f"    MRR         : {report.avg_mrr:.4f}")
    print(f"    MAP         : {report.avg_map:.4f}")
    print(f"    nDCG        : {report.avg_ndcg:.4f}")
    print(f"    Precision@3 : {report.avg_precision_at_3:.4f}")
    print(f"    Recall@5    : {report.avg_recall_at_5:.4f}")
    print()
    print("  GENERATION")
    print(f"    Exact Match : {report.avg_exact_match:.4f}")
    print(f"    F1          : {report.avg_f1:.4f}")
    print(f"    ROUGE-L     : {report.avg_rouge_l:.4f}")
    print(f"    BLEU        : {report.avg_bleu:.4f}")
    if use_ragas:
        print(f"    Faithfulness: {report.avg_faithfulness:.4f}")
        print(f"    Relevancy   : {report.avg_answer_relevancy:.4f}")
    print()
    print(f"  ✓ output/evaluation/evaluation_report.csv")
    print(f"  ✓ output/evaluation/evaluation_summary.csv")
    print(f"  ✓ output/evaluation/evaluation_summary.txt\n")


if __name__ == "__main__":
    run(
        max_papers=   20,
        max_questions=50,
        top_k=        5,
        use_ragas=    False,   # set True if you have OpenAI API key
        ingest_papers=True,
    )