"""
AstroNexus AI — Corpus-Level BM25 Retrieval Evaluation

Evaluates BM25 retrieval at the CORPUS level.
Builds ONE global BM25 index over all chunks in Qdrant,
then retrieves without any paper_id filter.

Run from project root:
    python -m backend.evaluation.run_corpus_bm25_evaluation

Requires:
    pip install rank-bm25 numpy
    Qdrant running with ALL QASPER papers already indexed.

Outputs (all prefixed corpus_bm25_):
    output/evaluation/corpus_bm25_retrieval_report.csv
    output/evaluation/corpus_bm25_paper_report.csv
    output/evaluation/corpus_bm25_summary.csv
    output/evaluation/corpus_bm25_summary.txt
    output/evaluation/corpus_comparison.csv   (Dense vs BM25 at corpus level)
"""
from __future__ import annotations

import csv
import logging
import re
from collections import Counter
from pathlib import Path
from statistics import mean

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


# ── Global BM25 index ─────────────────────────────────────────────────────────

def build_global_bm25_index(collection: str = "papers") -> object:
    """
    Build one BM25 index over ALL chunks in Qdrant.

    This is the corpus-level equivalent of the per-paper BM25 index
    used in the single-paper evaluation. Scrolling all chunks once
    and building a single index is more efficient than building many
    per-paper indexes and merging results.

    Returns:
        dict with keys: bm25, chunks, payloads
    """
    try:
        from rank_bm25 import BM25Okapi
        import numpy as np
    except ImportError as e:
        raise ImportError("Install: pip install rank-bm25 numpy") from e

    from qdrant_client import QdrantClient

    client = QdrantClient(host="localhost", port=6333)

    all_chunks:   list[str]  = []
    all_payloads: list[dict] = []
    offset = None

    logger.info("[BM25-Corpus] Scrolling all chunks from Qdrant...")

    while True:
        results, next_offset = client.scroll(
            collection_name= collection,
            limit=           1000,
            offset=          offset,
            with_payload=    True,
            with_vectors=    False,
        )

        for point in results:
            payload = point.payload or {}
            text    = payload.get("text", "").strip()
            if text:
                all_chunks.append(text)
                all_payloads.append(payload)

        if next_offset is None:
            break
        offset = next_offset

    logger.info(f"[BM25-Corpus] Building global index over {len(all_chunks)} chunks")

    tokenized = [_tokenize(c) for c in all_chunks]
    bm25      = BM25Okapi(tokenized)

    return {"bm25": bm25, "chunks": all_chunks, "payloads": all_payloads}


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 — same as bm25_retriever.py."""
    text   = text.lower()
    text   = re.sub(r"[^\w\s\-]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]


def bm25_corpus_retrieve(
    index: dict,
    query: str,
    top_k: int = 5,
) -> object:   # RetrievalResult
    """
    Query the global BM25 index and return top-k results.

    Returns RetrievalResult with retrieved_paper_ids populated
    for paper-level metric computation.
    """
    import numpy as np
    from backend.evaluation.models import RetrievalResult
    from backend.rag.retriever import RetrievedChunk

    bm25     = index["bm25"]
    chunks   = index["chunks"]
    payloads = index["payloads"]

    query_tokens = _tokenize(query)
    if not query_tokens or not chunks:
        return RetrievalResult(
            question_id="", question=query,
            retrieved_chunks=[], retrieved_scores=[],
            evidence_chunks=[], retrieved_paper_ids=[],
        )

    scores      = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]

    max_score = float(scores[top_indices[0]]) if len(top_indices) > 0 else 1.0
    if max_score == 0.0:
        max_score = 1.0

    retrieved_chunks:     list[str]   = []
    retrieved_scores:     list[float] = []
    retrieved_paper_ids:  list[str]   = []

    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0.0:
            continue
        p = payloads[idx]
        retrieved_chunks.append(p.get("text", ""))
        retrieved_scores.append(round(score / max_score, 6))
        retrieved_paper_ids.append(p.get("paper_id", ""))

    return RetrievalResult(
        question_id=         "",    # filled by caller
        question=            query,
        retrieved_chunks=    retrieved_chunks,
        retrieved_scores=    retrieved_scores,
        evidence_chunks=     [],    # filled by caller
        retrieved_paper_ids= retrieved_paper_ids,
    )


def save_corpus_comparison(
    dense_csv:  Path,
    bm25_report: object,
    paper_bm25:  object,
    out_path:    Path,
) -> None:
    """
    Write Dense vs BM25 corpus-level comparison table.
    Loads dense corpus summary from previously saved CSV.
    """
    # Load dense corpus summary
    dense_data = {}
    if dense_csv.exists():
        with open(dense_csv, encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) == 2:
                    dense_data[row[0]] = row[1]
    else:
        logger.warning(f"[Comparison] Dense corpus summary not found at {dense_csv}")
        logger.warning("  Run run_corpus_evaluation.py first")

    def d(key): return dense_data.get(key, "n/a")

    rows = [
        # Paper-level
        ("Top-1 Paper Accuracy", d("top1_paper_accuracy"),     f"{paper_bm25.avg_top1_paper_accuracy:.4f}"),
        ("Paper Hit Rate",       d("paper_hit_rate"),          f"{paper_bm25.avg_paper_hit_rate:.4f}"),
        ("Paper MRR",            d("paper_mrr"),               f"{paper_bm25.avg_paper_mrr:.4f}"),
        ("Paper Recall@1",       d("paper_recall@1"),          f"{paper_bm25.avg_paper_recall_at_1:.4f}"),
        ("Paper Recall@3",       d("paper_recall@3"),          f"{paper_bm25.avg_paper_recall_at_3:.4f}"),
        ("Paper Recall@5",       d("paper_recall@5"),          f"{paper_bm25.avg_paper_recall_at_5:.4f}"),
        # Chunk-level
        ("Chunk Hit Rate",       d("hit_rate"),                f"{bm25_report.avg_hit_rate:.4f}"),
        ("Chunk MRR",            d("mrr"),                     f"{bm25_report.avg_mrr:.4f}"),
        ("MAP",                  d("map"),                     f"{bm25_report.avg_map:.4f}"),
        ("nDCG",                 d("ndcg"),                    f"{bm25_report.avg_ndcg:.4f}"),
        ("Precision@1",          d("precision@1"),             f"{bm25_report.avg_precision_at_1:.4f}"),
        ("Precision@3",          d("precision@3"),             f"{bm25_report.avg_precision_at_3:.4f}"),
        ("Precision@5",          d("precision@5"),             f"{bm25_report.avg_precision_at_5:.4f}"),
        ("Recall@5",             d("recall@5"),                f"{bm25_report.avg_recall_at_5:.4f}"),
        # Generation
        ("Exact Match",          d("exact_match"),             f"{bm25_report.avg_exact_match:.4f}"),
        ("F1",                   d("f1"),                      f"{bm25_report.avg_f1:.4f}"),
        ("ROUGE-L",              d("rouge_l"),                 f"{bm25_report.avg_rouge_l:.4f}"),
        ("BLEU",                 d("bleu"),                    f"{bm25_report.avg_bleu:.4f}"),
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Dense (BGE-M3) Corpus", "BM25 (Okapi) Corpus"])
        writer.writerows(rows)

    logger.info(f"[Comparison] Corpus comparison → {out_path}")

    # Print to terminal
    print(f"\n{'='*65}")
    print("CORPUS-LEVEL: DENSE vs BM25")
    print(f"{'='*65}")
    print(f"  {'Metric':<22}  {'Dense':>12}  {'BM25':>12}")
    print(f"  {'-'*22}  {'-'*12}  {'-'*12}")
    for metric, dense_val, bm25_val in rows:
        print(f"  {metric:<22}  {dense_val:>12}  {bm25_val:>12}")
    print(f"{'='*65}")


def run(
    max_papers:    int  = 10,
    max_questions: int  = 30,
    top_k:         int  = 5,
    use_ragas:     bool = False,
) -> None:

    from backend.evaluation.dataset_loader       import load_qasper
    from backend.evaluation.retrieval_evaluator  import evaluate_retrieval
    from backend.evaluation.generation_evaluator  import evaluate_generation
    from backend.evaluation.report_generator      import aggregate_metrics, save_csv
    from backend.evaluation.corpus_evaluator      import (
        evaluate_paper_retrieval, aggregate_paper_metrics, save_paper_metrics_csv
    )
    from backend.evaluation.models               import GenerationResult
    from backend.evaluation.answer_generator     import _generate, _QA_PROMPT
    from backend.evaluation.run_corpus_evaluation import (
        save_corpus_summary_csv, save_corpus_summary_txt
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load QASPER ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Step 1] Loading QASPER  ({max_papers} papers, {max_questions} questions)")
    print(f"{'='*60}")

    qa_pairs, paper_texts = load_qasper(split="validation", max_papers=max_papers)
    qa_pairs = qa_pairs[:max_questions]
    print(f"  QA pairs : {len(qa_pairs)}")

    # ── Step 2: Build global BM25 index ───────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Step 2] Building global BM25 index (all papers in Qdrant)")
    print(f"{'='*60}")

    global_index = build_global_bm25_index()
    print(f"  Index size : {len(global_index['chunks'])} chunks")

    # ── Step 3: Corpus retrieval ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Step 3] Corpus-level BM25 retrieval")
    print(f"{'='*60}")

    chunk_metrics_list = []
    paper_metrics_list = []

    for i, qa in enumerate(qa_pairs, start=1):
        try:
            result = bm25_corpus_retrieve(global_index, qa.question, top_k=top_k)
            # Fill in fields that the retrieval function left empty
            result.question_id   = qa.question_id
            result.evidence_chunks = qa.evidence

            chunk_metrics = evaluate_retrieval(result)
            paper_metrics = evaluate_paper_retrieval(
                question_id=         qa.question_id,
                correct_paper_id=    qa.paper_id,
                retrieved_paper_ids= result.retrieved_paper_ids,
            )
            chunk_metrics_list.append(chunk_metrics)
            paper_metrics_list.append(paper_metrics)

            if i % 10 == 0 or i == len(qa_pairs):
                chunk_hit = mean(float(m.hit_rate)       for m in chunk_metrics_list)
                paper_hit = mean(float(m.paper_hit_rate) for m in paper_metrics_list)
                print(
                    f"  [{i:>3}/{len(qa_pairs)}]  "
                    f"Chunk Hit={chunk_hit:.3f}  "
                    f"Paper Hit={paper_hit:.3f}"
                )
        except Exception as e:
            logger.warning(f"  BM25 retrieval failed {qa.question_id}: {e}")

    # ── Step 4: Generation ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Step 4] Generation evaluation")
    print(f"{'='*60}")

    gen_metrics_list = []
    for i, qa in enumerate(qa_pairs, start=1):
        try:
            result   = bm25_corpus_retrieve(global_index, qa.question, top_k=top_k)
            contexts = result.retrieved_chunks

            if not contexts:
                gen_result = GenerationResult(
                    question_id="", question=qa.question,
                    generated_answer="[NO_CONTEXT_RETRIEVED]",
                    ground_truth=qa.ground_truth, context_used=[],
                )
            else:
                context_str = "\n\n---\n\n".join(
                    f"[Chunk {j+1}]\n{c}" for j, c in enumerate(contexts)
                )
                generated = _generate(
                    _QA_PROMPT.format(context=context_str, question=qa.question)
                )
                gen_result = GenerationResult(
                    question_id=     qa.question_id,
                    question=        qa.question,
                    generated_answer=generated,
                    ground_truth=    qa.ground_truth,
                    context_used=    contexts,
                )

            gen_metrics = evaluate_generation(gen_result, use_ragas=use_ragas)
            gen_metrics_list.append(gen_metrics)

            if i % 10 == 0 or i == len(qa_pairs):
                f1 = mean(m.f1_score for m in gen_metrics_list)
                print(f"  [{i:>3}/{len(qa_pairs)}]  F1={f1:.3f}")

        except Exception as e:
            logger.warning(f"  Generation failed {qa.question_id}: {e}")

    # ── Step 5: Aggregate and save ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Step 5] Saving BM25 corpus reports")
    print(f"{'='*60}")

    qa_type_map  = {qa.question_id: qa.answer_type for qa in qa_pairs}
    chunk_report = aggregate_metrics(chunk_metrics_list, gen_metrics_list, qa_type_map)
    paper_report = aggregate_paper_metrics(paper_metrics_list)

    save_csv(chunk_metrics_list, gen_metrics_list, OUT_DIR)
    (OUT_DIR / "evaluation_report.csv").rename(OUT_DIR / "corpus_bm25_retrieval_report.csv")

    save_paper_metrics_csv(paper_metrics_list, OUT_DIR / "corpus_bm25_paper_report.csv")
    save_corpus_summary_csv(chunk_report, paper_report, OUT_DIR / "corpus_bm25_summary.csv")
    save_corpus_summary_txt(
        chunk_report, paper_report, "BM25 (Okapi)", OUT_DIR / "corpus_bm25_summary.txt"
    )

    # ── Step 6: Comparison table ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Step 6] Corpus-level comparison table")
    print(f"{'='*60}")

    save_corpus_comparison(
        dense_csv=   OUT_DIR / "corpus_dense_summary.csv",
        bm25_report= chunk_report,
        paper_bm25=  paper_report,
        out_path=    OUT_DIR / "corpus_comparison.csv",
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("BM25 CORPUS EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"  PAPER-LEVEL")
    print(f"    Paper Hit Rate  : {paper_report.avg_paper_hit_rate:.4f}")
    print(f"    Paper MRR       : {paper_report.avg_paper_mrr:.4f}")
    print(f"    Top-1 Accuracy  : {paper_report.avg_top1_paper_accuracy:.4f}")
    print()
    print(f"  CHUNK-LEVEL")
    print(f"    Hit Rate        : {chunk_report.avg_hit_rate:.4f}")
    print(f"    MRR             : {chunk_report.avg_mrr:.4f}")
    print(f"    Precision@3     : {chunk_report.avg_precision_at_3:.4f}")
    print(f"    Recall@5        : {chunk_report.avg_recall_at_5:.4f}")
    print()
    print(f"  GENERATION")
    print(f"    F1              : {chunk_report.avg_f1:.4f}")
    print(f"    ROUGE-L         : {chunk_report.avg_rouge_l:.4f}")
    print()
    print("  OUTPUT FILES")
    print(f"    ✓ output/evaluation/corpus_bm25_retrieval_report.csv")
    print(f"    ✓ output/evaluation/corpus_bm25_paper_report.csv")
    print(f"    ✓ output/evaluation/corpus_bm25_summary.csv")
    print(f"    ✓ output/evaluation/corpus_bm25_summary.txt")
    print(f"    ✓ output/evaluation/corpus_comparison.csv")
    print()


if __name__ == "__main__":
    run(max_papers=10, max_questions=30, top_k=5)