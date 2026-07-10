"""
QASPER Dataset Loader for AstroNexus RAG Evaluation.

Downloads QASPER from HuggingFace and converts it into
QAPair objects compatible with the evaluation pipeline.

Install: pip install datasets
"""
from __future__ import annotations

import logging
from typing import Optional

from backend.evaluation.models import QAPair

logger = logging.getLogger(__name__)


def load_qasper(
    split:       str = "validation",   # "train" | "validation" | "test"
    max_papers:  int = 20,             # limit for quick testing
    answer_types: list[str] | None = None,  # filter by type if needed
) -> tuple[list[QAPair], dict[str, str]]:
    """
    Load QASPER dataset and return QA pairs + paper full texts.

    Args:
        split:        Dataset split to load
        max_papers:   Max number of papers to load (None = all)
        answer_types: Filter to specific answer types
                      ["extractive", "abstractive", "yes_no", "unanswerable"]

    Returns:
        qa_pairs:    list[QAPair]
        paper_texts: dict[paper_id → full_text]
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Install datasets: pip install datasets") from e

    logger.info(f"[DatasetLoader] Loading QASPER ({split} split)...")
    dataset = load_dataset("allenai/qasper", split=split, trust_remote_code=True)

    qa_pairs:    list[QAPair] = []
    paper_texts: dict[str, str] = {}
    papers_processed = 0

    for item in dataset:
        if max_papers and papers_processed >= max_papers:
            break

        paper_id = item["id"]

        # ── Extract full paper text from QASPER structure ──────────────────────
        full_text = _extract_paper_text(item)
        if not full_text.strip():
            continue

        paper_texts[paper_id] = full_text
        papers_processed += 1

        # ── Extract QA pairs ──────────────────────────────────────────────────
        qas = item.get("qas", {})
        questions  = qas.get("question", [])
        answers    = qas.get("answers", [])
        question_ids = qas.get("question_id", [])

        for q_idx, (question, answer_list, q_id) in enumerate(
            zip(questions, answers, question_ids)
        ):
            parsed = _parse_answers(answer_list)

            if not parsed["ground_truth"]:
                continue

            # Filter by answer type if requested
            if answer_types and parsed["answer_type"] not in answer_types:
                continue

            qa_pairs.append(QAPair(
                question_id=  q_id or f"{paper_id}_q{q_idx}",
                paper_id=     paper_id,
                question=     question.strip(),
                ground_truth= parsed["ground_truth"],
                evidence=     parsed["evidence"],
                answer_type=  parsed["answer_type"],
            ))

    logger.info(
        f"[DatasetLoader] Loaded {len(qa_pairs)} QA pairs "
        f"from {papers_processed} papers"
    )

    # Log answer type distribution
    from collections import Counter
    type_dist = Counter(qa.answer_type for qa in qa_pairs)
    logger.info(f"[DatasetLoader] Answer types: {dict(type_dist)}")

    return qa_pairs, paper_texts


def _extract_paper_text(item: dict) -> str:
    """
    Reconstruct full paper text from QASPER's section structure.

    QASPER stores papers as: title + abstract + sections (heading + paragraphs)
    """
    parts = []

    # Title
    title = item.get("title", "").strip()
    if title:
        parts.append(title)

    # Abstract
    abstract = item.get("abstract", "").strip()
    if abstract:
        parts.append(f"Abstract\n{abstract}")

    # Full text sections
    full_text = item.get("full_text", {})
    headings   = full_text.get("section_name", [])
    paragraphs = full_text.get("paragraphs", [])

    for heading, para_list in zip(headings, paragraphs):
        if heading:
            parts.append(f"\n{heading}")
        if isinstance(para_list, list):
            parts.extend(p for p in para_list if p.strip())
        elif isinstance(para_list, str) and para_list.strip():
            parts.append(para_list)

    return "\n\n".join(parts)


def _parse_answers(answer_list: list | dict) -> dict:
    """
    Parse QASPER answer structure into ground truth + evidence.

    QASPER answers can be: extractive, abstractive, yes/no, or unanswerable.
    Multiple annotators may provide different answers — we collect all.
    """
    ground_truth: list[str] = []
    evidence:     list[str] = []
    answer_type               = "unanswerable"

    # Handle both list and dict formats
    if isinstance(answer_list, dict):
        answer_list = [answer_list]

    for answer_item in (answer_list or []):
        ann_list = answer_item.get("answer", [])
        if isinstance(ann_list, dict):
            ann_list = [ann_list]

        for ann in (ann_list or []):
            # Extractive answers
            extractive = ann.get("extractive_spans", [])
            if extractive:
                ground_truth.extend(extractive)
                answer_type = "extractive"

            # Abstractive answers
            free_form = ann.get("free_form_answer", "").strip()
            if free_form and free_form.lower() != "unanswerable":
                ground_truth.append(free_form)
                answer_type = "abstractive"

            # Yes/No answers
            yes_no = ann.get("yes_no", None)
            if yes_no is not None:
                ground_truth.append("yes" if yes_no else "no")
                answer_type = "yes_no"

            # Evidence sentences
            evid = ann.get("evidence", [])
            if isinstance(evid, list):
                evidence.extend(e for e in evid if e.strip())

    return {
        "ground_truth": list(set(ground_truth)),  # deduplicate
        "evidence":     list(set(evidence)),
        "answer_type":  answer_type,
    }