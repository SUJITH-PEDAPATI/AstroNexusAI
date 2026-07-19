"""
AstroNexus AI — Corpus-Level Dense Retrieval Evaluation

Evaluates dense retrieval (BGE-M3 + Qdrant) at the CORPUS level.
The retriever searches ALL papers simultaneously — no paper_id filter.

This is the realistic evaluation mode: the system must identify
the correct paper AND the correct chunk from the full indexed corpus.

Produces two layers of metrics:

    Layer 1 — Paper Retrieval
        Did the system find the correct paper in its top-k results?
        Top-1 Paper Accuracy, Paper Hit Rate, Paper MRR, Paper Recall@k

    Layer 2 — Chunk Retrieval
        Did the system find the relevant chunk(s)?
        Hit Rate, MRR, MAP, nDCG, Precision@k, Recall@k

    Layer 3 — Generation (unchanged)
        F1, ROUGE-L, BLEU, Exact Match

Run from project root:
    python -m backend.evaluation.run_corpus_evaluation

Requires:
    Qdrant running with ALL QASPER papers already indexed.
    Run run_evaluation.py first to index the papers.

Outputs (all prefixed corpus_dense_):
    output/evaluation/corpus_dense_retrieval_report.csv
    output/evaluation/corpus_dense_paper_report.csv
    output/evaluation/corpus_dense_summary.csv
    output/evaluation/corpus_dense_summary.txt
"""
from __future__ import annotations

import csv
import logging
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


def retrieve_corpus_dense(
    qa:    object,   # QAPair
    top_k: int = 5,
) -> object:          # RetrievalResult
    """
    Dense retrieval across the full corpus (no paper_id filter).

    Wraps the existing retrieve() function with filter_paper_id=None.
    Also populates retrieved_paper_ids from chunk payloads for
    paper-level metric computation.
    """
    from backend.rag.retriever import retrieve
    from backend.evaluation.models import RetrievalResult

    results = retrieve(
        query=           qa.question,
        top_k=           top_k,
        score_threshold= 0.0,
        filter_paper_id= None,   # ← corpus-level: no paper filter
    )

    return RetrievalResult(
        question_id=         qa.question_id,
        question=            qa.question,
        retrieved_chunks=    [r.text     for r in results],
        retrieved_scores=    [r.score    for r in results],
        evidence_chunks=     qa.evidence,
        retrieved_paper_ids= [r.paper_id for r in results],
    )


def save_corpus_summary_txt(
    chunk_report:  object,   # EvaluationReport
    paper_report:  object,   # CorpusPaperReport
    retriever:     str,
    out_path:      Path,
) -> None:
    """Save a human-readable corpus evaluation summary."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"ASTRONEXUS AI — CORPUS-LEVEL EVALUATION ({retriever})\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Questions : {chunk_report.total_questions}\n\n")

        f.write("─" * 60 + "\n")
        f.write("PAPER-LEVEL RETRIEVAL\n")
        f.write("(Did the system find the correct paper from the corpus?)\n")
        f.write("─" * 60 + "\n")
        f.write(f"  Top-1 Paper Accuracy : {paper_report.avg_top1_paper_accuracy:.4f}\n")
        f.write(f"  Paper Hit Rate       : {paper_report.avg_paper_hit_rate:.4f}\n")
        f.write(f"  Paper MRR            : {paper_report.avg_paper_mrr:.4f}\n")
        f.write(f"  Paper Recall@1       : {paper_report.avg_paper_recall_at_1:.4f}\n")
        f.write(f"  Paper Recall@3       : {paper_report.avg_paper_recall_at_3:.4f}\n")
        f.write(f"  Paper Recall@5       : {paper_report.avg_paper_recall_at_5:.4f}\n\n")

        f.write("─" * 60 + "\n")
        f.write("CHUNK-LEVEL RETRIEVAL\n")
        f.write("(Did the system find the relevant chunk from the corpus?)\n")
        f.write("─" * 60 + "\n")
        f.write(f"  Hit Rate    : {chunk_report.avg_hit_rate:.4f}\n")
        f.write(f"  MRR         : {chunk_report.avg_mrr:.4f}\n")
        f.write(f"  MAP         : {chunk_report.avg_map:.4f}\n")
        f.write(f"  nDCG        : {chunk_report.avg_ndcg:.4f}\n")
        f.write(f"  Precision@1 : {chunk_report.avg_precision_at_1:.4f}\n")
        f.write(f"  Precision@3 : {chunk_report.avg_precision_at_3:.4f}\n")
        f.write(f"  Precision@5 : {chunk_report.avg_precision_at_5:.4f}\n")
        f.write(f"  Recall@1    : {chunk_report.avg_recall_at_1:.4f}\n")
        f.write(f"  Recall@3    : {chunk_report.avg_recall_at_3:.4f}\n")
        f.write(f"  Recall@5    : {chunk_report.avg_recall_at_5:.4f}\n\n")

        f.write("─" * 60 + "\n")
        f.write("GENERATION\n")
        f.write("─" * 60 + "\n")
        f.write(f"  Exact Match : {chunk_report.avg_exact_match:.4f}\n")
        f.write(f"  F1          : {chunk_report.avg_f1:.4f}\n")
        f.write(f"  ROUGE-1     : {chunk_report.avg_rouge_1:.4f}\n")
        f.write(f"  ROUGE-2     : {chunk_report.avg_rouge_2:.4f}\n")
        f.write(f"  ROUGE-L     : {chunk_report.avg_rouge_l:.4f}\n")
        f.write(f"  BLEU        : {chunk_report.avg_bleu:.4f}\n\n")

        f.write("─" * 60 + "\n")
        f.write("INTERPRETATION\n")
        f.write("─" * 60 + "\n")
        f.write("  Paper Hit Rate > 0.70 → system reliably finds correct paper\n")
        f.write("  Paper MRR     > 0.50 → correct paper usually in top-2 results\n")
        f.write("  Chunk Hit Rate > 0.60 → most questions have relevant chunk retrieved\n")
        f.write("  Chunk Hit Rate (corpus) < Chunk Hit Rate (single-paper) → expected\n")
        f.write("    Gap reflects real-world retrieval difficulty\n")
        f.write("=" * 60 + "\n")

    logger.info(f"[CorpusEval] Summary → {out_path}")


def save_corpus_summary_csv(
    chunk_report: object,
    paper_report: object,
    out_path:     Path,
) -> None:
    """Save aggregated corpus summary to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows([
            ["total_questions",          chunk_report.total_questions],
            # Paper-level
            ["top1_paper_accuracy",      f"{paper_report.avg_top1_paper_accuracy:.4f}"],
            ["paper_hit_rate",           f"{paper_report.avg_paper_hit_rate:.4f}"],
            ["paper_mrr",                f"{paper_report.avg_paper_mrr:.4f}"],
            ["paper_recall@1",           f"{paper_report.avg_paper_recall_at_1:.4f}"],
            ["paper_recall@3",           f"{paper_report.avg_paper_recall_at_3:.4f}"],
            ["paper_recall@5",           f"{paper_report.avg_paper_recall_at_5:.4f}"],
            # Chunk-level
            ["hit_rate",                 f"{chunk_report.avg_hit_rate:.4f}"],
            ["mrr",                      f"{chunk_report.avg_mrr:.4f}"],
            ["map",                      f"{chunk_report.avg_map:.4f}"],
            ["ndcg",                     f"{chunk_report.avg_ndcg:.4f}"],
            ["precision@1",              f"{chunk_report.avg_precision_at_1:.4f}"],
            ["precision@3",              f"{chunk_report.avg_precision_at_3:.4f}"],
            ["precision@5",              f"{chunk_report.avg_precision_at_5:.4f}"],
            ["recall@1",                 f"{chunk_report.avg_recall_at_1:.4f}"],
            ["recall@3",                 f"{chunk_report.avg_recall_at_3:.4f}"],
            ["recall@5",                 f"{chunk_report.avg_recall_at_5:.4f}"],
            # Generation
            ["exact_match",              f"{chunk_report.avg_exact_match:.4f}"],
            ["f1",                       f"{chunk_report.avg_f1:.4f}"],
            ["rouge_1",                  f"{chunk_report.avg_rouge_1:.4f}"],
            ["rouge_2",                  f"{chunk_report.avg_rouge_2:.4f}"],
            ["rouge_l",                  f"{chunk_report.avg_rouge_l:.4f}"],
            ["bleu",                     f"{chunk_report.avg_bleu:.4f}"],
        ])
    logger.info(f"[CorpusEval] Summary CSV → {out_path}")


def run(
    max_papers:    int  = 10,
    max_questions: int  = 30,
    top_k:         int  = 5,
    use_ragas:     bool = False,
) -> object:
    """
    Run corpus-level dense retrieval evaluation.

    Returns the chunk_report and paper_report objects so they can be
    consumed by the comparison table generator.
    """
    from backend.evaluation.dataset_loader      import load_qasper
    from backend.evaluation.retrieval_evaluator import evaluate_retrieval
    from backend.evaluation.generation_evaluator import evaluate_generation
    from backend.evaluation.answer_generator    import generate_answer
    from backend.evaluation.report_generator    import (
        aggregate_metrics, save_csv,
    )
    from backend.evaluation.corpus_evaluator    import (
        evaluate_paper_retrieval,
        aggregate_paper_metrics,
        save_paper_metrics_csv,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load QASPER ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Step 1] Loading QASPER  ({max_papers} papers, {max_questions} questions)")
    print(f"{'='*60}")

    qa_pairs, paper_texts = load_qasper(split="validation", max_papers=max_papers)
    qa_pairs = qa_pairs[:max_questions]

    print(f"  Papers   : {len(paper_texts)}")
    print(f"  QA pairs : {len(qa_pairs)}")
    print(f"  Types    : {dict(Counter(qa.answer_type for qa in qa_pairs))}")

    # ── Step 2: Corpus-level retrieval ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Step 2] Corpus-level dense retrieval  (no paper_id filter)")
    print(f"{'='*60}")

    chunk_metrics_list = []
    paper_metrics_list = []

    for i, qa in enumerate(qa_pairs, start=1):
        try:
            result = retrieve_corpus_dense(qa, top_k=top_k)

            # Chunk-level metrics
            chunk_metrics = evaluate_retrieval(result)
            chunk_metrics_list.append(chunk_metrics)

            # Paper-level metrics
            paper_metrics = evaluate_paper_retrieval(
                question_id=         qa.question_id,
                correct_paper_id=    qa.paper_id,
                retrieved_paper_ids= result.retrieved_paper_ids,
            )
            paper_metrics_list.append(paper_metrics)

            if i % 10 == 0 or i == len(qa_pairs):
                chunk_hit = mean(float(m.hit_rate) for m in chunk_metrics_list)
                paper_hit = mean(float(m.paper_hit_rate) for m in paper_metrics_list)
                mrr       = mean(m.mrr for m in chunk_metrics_list)
                print(
                    f"  [{i:>3}/{len(qa_pairs)}]  "
                    f"Chunk Hit={chunk_hit:.3f}  "
                    f"Paper Hit={paper_hit:.3f}  "
                    f"MRR={mrr:.3f}"
                )

        except Exception as e:
            logger.warning(f"  Retrieval failed {qa.question_id}: {e}")

    # ── Step 3: Generation ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Step 3] Generation evaluation")
    print(f"{'='*60}")

    gen_metrics_list = []
    for i, qa in enumerate(qa_pairs, start=1):
        try:
            # generate_answer uses its own retrieval internally
            # We override it to use corpus-level context
            result   = retrieve_corpus_dense(qa, top_k=top_k)
            contexts = result.retrieved_chunks

            if not contexts:
                from backend.evaluation.models import GenerationResult
                gen_result = GenerationResult(
                    question_id=     qa.question_id,
                    question=        qa.question,
                    generated_answer="[NO_CONTEXT_RETRIEVED]",
                    ground_truth=    qa.ground_truth,
                    context_used=    [],
                )
            else:
                from backend.evaluation.answer_generator import _generate, _QA_PROMPT
                from backend.evaluation.models import GenerationResult
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
                f1    = mean(m.f1_score for m in gen_metrics_list)
                rouge = mean(m.rouge_l  for m in gen_metrics_list)
                print(f"  [{i:>3}/{len(qa_pairs)}]  F1={f1:.3f}  ROUGE-L={rouge:.3f}")

        except Exception as e:
            logger.warning(f"  Generation failed {qa.question_id}: {e}")

    # ── Step 4: Aggregate and save ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Step 4] Saving corpus-level dense reports")
    print(f"{'='*60}")

    qa_type_map   = {qa.question_id: qa.answer_type for qa in qa_pairs}
    chunk_report  = aggregate_metrics(chunk_metrics_list, gen_metrics_list, qa_type_map)
    paper_report  = aggregate_paper_metrics(paper_metrics_list)

    # Save per-question reports
    save_csv(chunk_metrics_list, gen_metrics_list, OUT_DIR)
    (OUT_DIR / "evaluation_report.csv").rename(
        OUT_DIR / "corpus_dense_retrieval_report.csv"
    )

    save_paper_metrics_csv(paper_metrics_list, OUT_DIR / "corpus_dense_paper_report.csv")
    save_corpus_summary_csv(chunk_report, paper_report, OUT_DIR / "corpus_dense_summary.csv")
    save_corpus_summary_txt(chunk_report, paper_report, "Dense (BGE-M3)",
                            OUT_DIR / "corpus_dense_summary.txt")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("CORPUS-LEVEL DENSE EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Questions : {chunk_report.total_questions}")
    print()
    print("  PAPER-LEVEL")
    print(f"    Top-1 Paper Accuracy : {paper_report.avg_top1_paper_accuracy:.4f}")
    print(f"    Paper Hit Rate       : {paper_report.avg_paper_hit_rate:.4f}")
    print(f"    Paper MRR            : {paper_report.avg_paper_mrr:.4f}")
    print(f"    Paper Recall@5       : {paper_report.avg_paper_recall_at_5:.4f}")
    print()
    print("  CHUNK-LEVEL")
    print(f"    Hit Rate    : {chunk_report.avg_hit_rate:.4f}")
    print(f"    MRR         : {chunk_report.avg_mrr:.4f}")
    print(f"    nDCG        : {chunk_report.avg_ndcg:.4f}")
    print(f"    Precision@3 : {chunk_report.avg_precision_at_3:.4f}")
    print(f"    Recall@5    : {chunk_report.avg_recall_at_5:.4f}")
    print()
    print("  GENERATION")
    print(f"    F1      : {chunk_report.avg_f1:.4f}")
    print(f"    ROUGE-L : {chunk_report.avg_rouge_l:.4f}")
    print(f"    BLEU    : {chunk_report.avg_bleu:.4f}")
    print()
    print("  OUTPUT FILES")
    print(f"    ✓ output/evaluation/corpus_dense_retrieval_report.csv")
    print(f"    ✓ output/evaluation/corpus_dense_paper_report.csv")
    print(f"    ✓ output/evaluation/corpus_dense_summary.csv")
    print(f"    ✓ output/evaluation/corpus_dense_summary.txt")
    print()

    return chunk_report, paper_report


if __name__ == "__main__":
    run(max_papers=10, max_questions=30, top_k=5)