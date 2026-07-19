"""
AstroNexus AI — Evaluation Pipeline (Step 1 Fixed Version)

Critical fixes applied:
    1. QASPER papers are ingested with their original paper_id preserved
    2. Qdrant dedup check uses the QASPER paper_id (not a random UUID)
    3. Retrieval filters by the correct paper_id during evaluation
    4. is_relevant() uses direct containment + lower overlap threshold

Run from project root:
    python -m backend.evaluation.run_evaluation

Requires:
    HF_API_TOKEN in .env
    Qdrant running on localhost:6333
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=  logging.INFO,
    format= "%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUT_DIR = Path("output/evaluation")


def ingest_qasper_paper(
    paper_id:  str,
    full_text: str,
) -> int:
    """
    Ingest a single QASPER paper into Qdrant, preserving the QASPER paper_id.

    This is the core fix: we pass override_paper_id=paper_id so the chunks
    stored in Qdrant have the same paper_id that QASPER uses in its QA pairs.
    When the evaluator calls retrieve(filter_paper_id=qa.paper_id), it will
    find these chunks.

    Returns number of chunks upserted (0 if already indexed).
    """
    from backend.ingestion.paper_ingestion import ingest_paper
    from backend.ingestion.chunking        import chunk_document
    from backend.embeddings                import embed_chunks
    from backend.rag                       import upsert_chunks

    # Write paper text to a temp file so our existing loaders can handle it
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(full_text)
        tmp_path = tmp.name

    try:
        # override_paper_id preserves QASPER's paper_id in every chunk
        doc = ingest_paper(tmp_path, override_paper_id=paper_id)
        chunks   = chunk_document(doc)
        embedded = embed_chunks(chunks)
        upserted = upsert_chunks(embedded)   # dedup check uses paper_id
        return upserted
    except Exception as e:
        logger.warning(f"  [Ingest] Failed for {paper_id}: {e}")
        return 0
    finally:
        os.unlink(tmp_path)


def verify_ingestion(paper_id: str, expected_chunks: int) -> bool:
    """
    Verify that a paper's chunks are actually in Qdrant with the correct paper_id.
    This is the diagnostic step that confirms Fix 1 is working.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client = QdrantClient(host="localhost", port=6333)

    try:
        results, _ = client.scroll(
            collection_name="papers",
            scroll_filter=Filter(
                must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        found = len(results) > 0
        if found:
            logger.info(f"  [Verify] ✓ paper_id '{paper_id[:20]}...' found in Qdrant")
        else:
            logger.warning(f"  [Verify] ✗ paper_id '{paper_id[:20]}...' NOT found in Qdrant")
        return found
    except Exception as e:
        logger.warning(f"  [Verify] Error checking paper_id: {e}")
        return False


def run(
    max_papers:    int  = 10,
    max_questions: int  = 30,
    top_k:         int  = 5,
    use_ragas:     bool = False,
    re_ingest:     bool = True,   # set False if papers already indexed
) -> None:

    from backend.evaluation.dataset_loader      import load_qasper
    from backend.evaluation.retrieval_evaluator import (
        retrieve_for_question, evaluate_retrieval,
    )
    from backend.evaluation.generation_evaluator import evaluate_generation
    from backend.evaluation.answer_generator     import generate_answer
    from backend.evaluation.report_generator     import (
        aggregate_metrics, save_csv, save_summary_csv, save_txt_report,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1: Load QASPER
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"[Step 1] Loading QASPER ({max_papers} papers, {max_questions} questions)")
    print(f"{'='*60}")

    qa_pairs, paper_texts = load_qasper(
        split=      "validation",
        max_papers= max_papers,
    )
    qa_pairs = qa_pairs[:max_questions]

    print(f"  Papers loaded    : {len(paper_texts)}")
    print(f"  QA pairs loaded  : {len(qa_pairs)}")

    # Show answer type distribution
    from collections import Counter
    type_dist = Counter(qa.answer_type for qa in qa_pairs)
    print(f"  Answer types     : {dict(type_dist)}")

    # Show sample paper IDs — these must match what goes into Qdrant
    sample_ids = list(paper_texts.keys())[:3]
    print(f"  Sample paper IDs : {sample_ids}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2: Ingest QASPER papers into Qdrant with correct paper_ids
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"[Step 2] Ingesting QASPER papers into Qdrant")
    print(f"{'='*60}")
    print(f"  re_ingest = {re_ingest}")
    print()

    if re_ingest:
        total_chunks = 0
        for paper_id, full_text in paper_texts.items():
            chunks_added = ingest_qasper_paper(paper_id, full_text)
            total_chunks += chunks_added
            print(f"  {paper_id[:30]:<30} → {chunks_added} chunks upserted")

        print(f"\n  Total chunks indexed : {total_chunks}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3: Verify ingestion — paper IDs must be in Qdrant
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"[Step 3] Verifying paper_id alignment")
    print(f"{'='*60}")

    verified = 0
    for paper_id in list(paper_texts.keys())[:5]:   # check first 5
        if verify_ingestion(paper_id, expected_chunks=0):
            verified += 1

    print(f"\n  Verified {verified}/5 papers found in Qdrant")

    if verified == 0:
        print("\n  ✗ CRITICAL: No papers found in Qdrant with QASPER paper IDs.")
        print("    This means the paper_id mismatch is still present.")
        print("    Check that override_paper_id is being set correctly.")
        sys.exit(1)

    print(f"\n  ✓ Paper ID alignment confirmed — evaluation will work correctly")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4: Retrieval evaluation
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"[Step 4] Retrieval evaluation — {len(qa_pairs)} questions")
    print(f"{'='*60}")

    retrieval_metrics_list = []
    retrieval_zeros        = 0

    for i, qa in enumerate(qa_pairs, start=1):
        try:
            result  = retrieve_for_question(qa, top_k=top_k)
            metrics = evaluate_retrieval(result)
            retrieval_metrics_list.append(metrics)

            if not metrics.hit_rate:
                retrieval_zeros += 1

            # Progress every 10 questions
            if i % 10 == 0 or i == len(qa_pairs):
                hit  = sum(float(m.hit_rate) for m in retrieval_metrics_list) / i
                mrr  = sum(m.mrr             for m in retrieval_metrics_list) / i
                print(f"  [{i:>3}/{len(qa_pairs)}]  Hit Rate={hit:.3f}  MRR={mrr:.3f}")

        except Exception as e:
            logger.warning(f"  Retrieval failed for {qa.question_id}: {e}")

    # Diagnostic: if still all zeros, print why
    if retrieval_zeros == len(retrieval_metrics_list):
        print(f"\n  ⚠ WARNING: All retrieval metrics are still zero.")
        print(f"  Possible causes:")
        print(f"    1. Qdrant collection was not refreshed — reset and re-ingest")
        print(f"    2. QASPER evidence sentences don't overlap with chunks")
        print(f"    3. Wrong Qdrant collection being queried")
        print(f"\n  Run this to inspect a sample question:")
        if qa_pairs:
            print(f"    paper_id = '{qa_pairs[0].paper_id}'")
            print(f"    question = '{qa_pairs[0].question}'")
            print(f"    evidence = {qa_pairs[0].evidence[:1]}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5: Generation evaluation
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"[Step 5] Generation evaluation — {len(qa_pairs)} questions")
    print(f"{'='*60}")

    generation_metrics_list = []

    for i, qa in enumerate(qa_pairs, start=1):
        try:
            gen_result  = generate_answer(qa, top_k=top_k)
            gen_metrics = evaluate_generation(gen_result, use_ragas=use_ragas)
            generation_metrics_list.append(gen_metrics)

            if i % 10 == 0 or i == len(qa_pairs):
                f1    = sum(m.f1_score for m in generation_metrics_list) / i
                rouge = sum(m.rouge_l  for m in generation_metrics_list) / i
                print(f"  [{i:>3}/{len(qa_pairs)}]  F1={f1:.3f}  ROUGE-L={rouge:.3f}")

        except Exception as e:
            logger.warning(f"  Generation failed for {qa.question_id}: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6: Aggregate and save
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"[Step 6] Saving reports")
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

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Questions evaluated : {report.total_questions}")
    print(f"  Answer types        : {report.answer_types}")
    print()
    print("  RETRIEVAL")
    print(f"    Hit Rate    : {report.avg_hit_rate:.4f}")
    print(f"    MRR         : {report.avg_mrr:.4f}")
    print(f"    MAP         : {report.avg_map:.4f}")
    print(f"    nDCG        : {report.avg_ndcg:.4f}")
    print(f"    Precision@1 : {report.avg_precision_at_1:.4f}")
    print(f"    Precision@3 : {report.avg_precision_at_3:.4f}")
    print(f"    Precision@5 : {report.avg_precision_at_5:.4f}")
    print(f"    Recall@1    : {report.avg_recall_at_1:.4f}")
    print(f"    Recall@3    : {report.avg_recall_at_3:.4f}")
    print(f"    Recall@5    : {report.avg_recall_at_5:.4f}")
    print()
    print("  GENERATION")
    print(f"    Exact Match : {report.avg_exact_match:.4f}")
    print(f"    F1 Score    : {report.avg_f1:.4f}")
    print(f"    ROUGE-1     : {report.avg_rouge_1:.4f}")
    print(f"    ROUGE-2     : {report.avg_rouge_2:.4f}")
    print(f"    ROUGE-L     : {report.avg_rouge_l:.4f}")
    print(f"    BLEU        : {report.avg_bleu:.4f}")
    print()
    print(f"  ✓ output/evaluation/evaluation_report.csv")
    print(f"  ✓ output/evaluation/evaluation_summary.csv")
    print(f"  ✓ output/evaluation/evaluation_summary.txt")
    print()


if __name__ == "__main__":
    run(
        max_papers=    10,
        max_questions= 30,
        top_k=         5,
        use_ragas=     False,
        re_ingest=     True,
    )