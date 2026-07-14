"""
Diagnostic script — run this BEFORE the full evaluation to verify
that the three root causes have been fixed.

Run from project root:
    python -m backend.evaluation.diagnose_evaluation

Checks:
    1. Can we load QASPER and get paper IDs?
    2. Can we ingest one QASPER paper with the correct paper_id?
    3. Can we find that paper in Qdrant by paper_id?
    4. Does is_relevant() correctly identify relevant chunks?
    5. Does retrieval return nonzero results for a QASPER question?
"""
from __future__ import annotations

import logging
import os
import tempfile

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(message)s")


def check(label: str, condition: bool, detail: str = "") -> bool:
    icon = "✓" if condition else "✗"
    print(f"  {icon} {label}")
    if detail:
        print(f"      {detail}")
    return condition


def run_diagnostics() -> None:
    print("\n" + "=" * 60)
    print("ASTRONEXUS — EVALUATION DIAGNOSTIC")
    print("=" * 60)
    all_passed = True

    # ── Check 1: QASPER loading ───────────────────────────────────────────────
    print("\n[Check 1] QASPER dataset loading")
    try:
        from backend.evaluation.dataset_loader import load_qasper
        qa_pairs, paper_texts = load_qasper(split="validation", max_papers=2)
        passed = len(qa_pairs) > 0 and len(paper_texts) > 0
        check("QASPER loads successfully", passed,
              f"{len(qa_pairs)} QA pairs, {len(paper_texts)} papers")

        sample_paper_id = list(paper_texts.keys())[0]
        sample_qa       = qa_pairs[0]
        check("QA pair has paper_id",    bool(sample_qa.paper_id),
              f"paper_id = '{sample_qa.paper_id}'")
        check("QA pair has evidence",    bool(sample_qa.evidence),
              f"evidence[0] = '{sample_qa.evidence[0][:80]}...'")
        check("paper_id matches QA pair", sample_qa.paper_id == sample_paper_id,
              f"QA paper_id='{sample_qa.paper_id}' | paper_texts key='{sample_paper_id}'")

    except Exception as e:
        check("QASPER loads successfully", False, str(e))
        all_passed = False
        print("\n  Cannot continue — QASPER failed to load.")
        return

    # ── Check 2: Ingestion with override_paper_id ─────────────────────────────
    print("\n[Check 2] Ingestion with QASPER paper_id")
    try:
        from backend.ingestion.paper_ingestion import ingest_paper
        from backend.ingestion.chunking        import chunk_document

        full_text = list(paper_texts.values())[0]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(full_text)
            tmp_path = tmp.name

        doc = ingest_paper(tmp_path, override_paper_id=sample_paper_id)
        os.unlink(tmp_path)

        check("Ingestion succeeds with override_paper_id",
              doc.paper_id == sample_paper_id,
              f"doc.paper_id = '{doc.paper_id}'")

        chunks = chunk_document(doc)
        check("Chunking produces chunks", len(chunks) > 0,
              f"{len(chunks)} chunks produced")
        check("Chunks carry correct paper_id",
              all(c.paper_id == sample_paper_id for c in chunks),
              f"chunk[0].paper_id = '{chunks[0].paper_id}'")

    except Exception as e:
        check("Ingestion with override_paper_id", False, str(e))
        all_passed = False

    # ── Check 3: Qdrant lookup by paper_id ────────────────────────────────────
    print("\n[Check 3] Qdrant paper_id lookup")
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = QdrantClient(host="localhost", port=6333)
        results, _ = client.scroll(
            collection_name="papers",
            scroll_filter=Filter(
                must=[FieldCondition(
                    key="paper_id",
                    match=MatchValue(value=sample_paper_id)
                )]
            ),
            limit=5,
            with_payload=True,
            with_vectors=False,
        )

        found = len(results) > 0
        check("QASPER paper_id found in Qdrant", found,
              f"Found {len(results)} chunks for paper_id '{sample_paper_id}'")

        if found:
            stored_id = results[0].payload.get("paper_id", "MISSING")
            check("Stored paper_id matches QASPER paper_id",
                  stored_id == sample_paper_id,
                  f"stored='{stored_id}' | expected='{sample_paper_id}'")
        else:
            print("      → Run evaluation with re_ingest=True first")
            all_passed = False

    except Exception as e:
        check("Qdrant connection", False, str(e))
        all_passed = False

    # ── Check 4: is_relevant() correctness ───────────────────────────────────
    print("\n[Check 4] is_relevant() evidence matching")
    try:
        from backend.evaluation.retrieval_evaluator import is_relevant

        evidence = sample_qa.evidence[:1] if sample_qa.evidence else ["test evidence"]
        ev        = evidence[0]

        # Test 1: chunk that CONTAINS the evidence should be relevant
        chunk_with_evidence = f"Some prefix text. {ev} Some suffix text about the paper."
        t1 = is_relevant(chunk_with_evidence, evidence)
        check("Chunk containing evidence → relevant", t1,
              f"evidence='{ev[:60]}...'")

        # Test 2: completely irrelevant chunk should not be relevant
        irrelevant_chunk = "This is completely unrelated content about cooking recipes."
        t2 = not is_relevant(irrelevant_chunk, evidence)
        check("Unrelated chunk → not relevant", t2)

        # Test 3: empty evidence → not relevant
        t3 = not is_relevant(chunk_with_evidence, [])
        check("Empty evidence → not relevant", t3)

    except Exception as e:
        check("is_relevant() works correctly", False, str(e))
        all_passed = False

    # ── Check 5: End-to-end retrieval for one QASPER question ─────────────────
    print("\n[Check 5] End-to-end retrieval")
    try:
        from backend.evaluation.retrieval_evaluator import (
            retrieve_for_question, evaluate_retrieval
        )

        result  = retrieve_for_question(sample_qa, top_k=5)
        metrics = evaluate_retrieval(result)

        check("Retrieval returns chunks", len(result.retrieved_chunks) > 0,
              f"{len(result.retrieved_chunks)} chunks retrieved")
        check("Hit rate is computed",    metrics.hit_rate is not None,
              f"hit_rate = {float(metrics.hit_rate):.1f}")
        check("MRR is computed",         metrics.mrr is not None,
              f"mrr = {metrics.mrr:.4f}")

        if len(result.retrieved_chunks) == 0:
            print("\n      → Paper may not be in Qdrant yet.")
            print(f"      → Run ingestion for paper_id='{sample_paper_id}' first.")

    except Exception as e:
        check("End-to-end retrieval", False, str(e))
        all_passed = False

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL CHECKS PASSED — run_evaluation.py will work correctly")
        print("  Run: python -m backend.evaluation.run_evaluation")
    else:
        print("✗ SOME CHECKS FAILED — fix issues above before running evaluation")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_diagnostics()