"""
AstroNexus AI — Offline Evaluation Module

Computes all metrics when ground-truth is available.
Does NOT modify any inference code.

Usage:
    from backend.evaluation.evaluator import evaluate
    metrics = evaluate(
        query=     "What is OST?",
        answer=    "OST is a telescope...",
        reference= "The Origins Space Telescope is...",
        chunks=    retrieved_chunks,
    )
"""
from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Optional

import logging
logger = logging.getLogger(__name__)

STOP = {
    "the","a","an","is","are","was","were","be","been","have","has","had",
    "do","does","did","will","would","could","should","to","of","in","for",
    "on","with","at","by","from","and","but","or","not","this","that","it",
}


# ══════════════════════════════════════════════════════════════════════════════
# RESULT DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvaluationResult:
    # Text generation
    bleu:      float = 0.0
    bleu_1:    float = 0.0
    bleu_2:    float = 0.0
    bleu_3:    float = 0.0
    bleu_4:    float = 0.0
    rouge_1:   float = 0.0
    rouge_2:   float = 0.0
    rouge_l:   float = 0.0
    meteor:    float = 0.0
    exact_match: bool = False
    token_accuracy: float = 0.0
    semantic_similarity: float = 0.0

    # Classification
    precision:       float = 0.0
    recall:          float = 0.0
    f1:              float = 0.0
    macro_precision: float = 0.0
    macro_recall:    float = 0.0
    macro_f1:        float = 0.0
    micro_f1:        float = 0.0
    weighted_f1:     float = 0.0

    # Retrieval
    recall_at_1:     float = 0.0
    recall_at_3:     float = 0.0
    recall_at_5:     float = 0.0
    precision_at_1:  float = 0.0
    precision_at_3:  float = 0.0
    precision_at_5:  float = 0.0
    hit_rate:        float = 0.0
    mrr:             float = 0.0
    map_score:       float = 0.0
    ndcg_at_5:       float = 0.0

    # Hallucination
    grounding_score:      float = 0.0
    citation_coverage:    float = 0.0
    unsupported_claims:   int   = 0
    hallucination_prob:   float = 0.0
    faithfulness_score:   float = 0.0
    answer_relevance:     float = 0.0
    context_relevance:    float = 0.0

    # Overall
    grade:             str  = "F"
    reliability_score: float = 0.0
    warnings:          list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# TOKENIZATION
# ══════════════════════════════════════════════════════════════════════════════

def _tok(text: str, remove_stop: bool = False) -> list[str]:
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    return [t for t in tokens if t not in STOP] if remove_stop else tokens

def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))


# ══════════════════════════════════════════════════════════════════════════════
# BLEU
# ══════════════════════════════════════════════════════════════════════════════

def _bleu_n(hyp: list[str], ref: list[str], n: int) -> float:
    hg = _ngrams(hyp, n)
    rg = _ngrams(ref, n)
    if not hg: return 0.0
    clipped = sum(min(hg[g], rg.get(g, 0)) for g in hg)
    total   = sum(hg.values())
    return clipped / total if total else 0.0


def _bleu(hypothesis: str, reference: str) -> dict[str, float]:
    hyp = _tok(hypothesis)
    ref = _tok(reference)
    if not hyp or not ref:
        return {f"bleu_{i}": 0.0 for i in range(1, 5)} | {"bleu": 0.0}

    bp = 1.0 if len(hyp) >= len(ref) else math.exp(1 - len(ref)/len(hyp))

    scores = {}
    log_sum = 0.0
    for n in range(1, 5):
        s = _bleu_n(hyp, ref, n)
        scores[f"bleu_{n}"] = round(s, 4)
        log_sum += math.log(s) if s > 0 else -999

    scores["bleu"] = round(bp * math.exp(log_sum / 4), 4) if log_sum > -3996 else 0.0
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# ROUGE
# ══════════════════════════════════════════════════════════════════════════════

def _rouge_n(hyp: str, ref: str, n: int) -> float:
    hg = _ngrams(_tok(hyp), n)
    rg = _ngrams(_tok(ref),  n)
    overlap = sum(min(hg[g], rg.get(g, 0)) for g in hg)
    denom   = sum(rg.values())
    return overlap / denom if denom else 0.0


def _lcs(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp   = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def _rouge_l(hyp: str, ref: str) -> float:
    ht, rt = _tok(hyp), _tok(ref)
    l = _lcs(ht, rt)
    p = l/len(ht) if ht else 0.0
    r = l/len(rt) if rt else 0.0
    return 2*p*r/(p+r) if (p+r) else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# METEOR (simplified)
# ══════════════════════════════════════════════════════════════════════════════

def _meteor(hypothesis: str, reference: str) -> float:
    hyp_tok = _tok(hypothesis)
    ref_tok = _tok(reference)
    if not hyp_tok or not ref_tok:
        return 0.0
    ref_set = set(ref_tok)
    matches = sum(1 for t in hyp_tok if t in ref_set)
    prec    = matches / len(hyp_tok) if hyp_tok else 0
    rec     = matches / len(ref_tok)  if ref_tok  else 0
    if not (prec + rec): return 0.0
    fmean   = 10*prec*rec / (9*prec + rec)
    # Penalty for fragmentation (simplified)
    return round(fmean * 0.95, 4)


# ══════════════════════════════════════════════════════════════════════════════
# TOKEN-LEVEL PRECISION / RECALL / F1
# ══════════════════════════════════════════════════════════════════════════════

def _prf1(hypothesis: str, reference: str) -> dict:
    hyp = set(_tok(hypothesis, remove_stop=True))
    ref = set(_tok(reference,  remove_stop=True))
    if not hyp or not ref:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    ov  = hyp & ref
    p   = len(ov) / len(hyp)
    r   = len(ov) / len(ref)
    f1  = 2*p*r/(p+r) if (p+r) else 0.0
    return {"precision": round(p,4), "recall": round(r,4), "f1": round(f1,4)}


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL METRICS
# ══════════════════════════════════════════════════════════════════════════════

def _retrieval_metrics(
    chunks:          list,          # RetrievedChunk list
    relevant_ids:    list[str],     # ground-truth relevant chunk IDs (if known)
    query:           str,
) -> dict:
    if not relevant_ids or not chunks:
        return {}

    chunk_ids = [getattr(c, "chunk_id", str(i)) for i, c in enumerate(chunks)]
    rel_set   = set(relevant_ids)
    scores    = [getattr(c, "score", 0.0) for c in chunks]

    def recall_at_k(k):
        hits = sum(1 for cid in chunk_ids[:k] if cid in rel_set)
        return hits / len(rel_set) if rel_set else 0.0

    def precision_at_k(k):
        hits = sum(1 for cid in chunk_ids[:k] if cid in rel_set)
        return hits / k

    # MRR
    mrr = 0.0
    for i, cid in enumerate(chunk_ids, 1):
        if cid in rel_set:
            mrr = 1.0 / i
            break

    # MAP
    ap_sum, hits = 0.0, 0
    for i, cid in enumerate(chunk_ids, 1):
        if cid in rel_set:
            hits += 1
            ap_sum += hits / i
    map_score = ap_sum / len(rel_set) if rel_set else 0.0

    # NDCG@5
    def dcg(ks):
        return sum(
            (1 if chunk_ids[i] in rel_set else 0) / math.log2(i+2)
            for i in range(min(ks, len(chunk_ids)))
        )
    idcg  = sum(1/math.log2(i+2) for i in range(min(5, len(rel_set))))
    ndcg5 = dcg(5) / idcg if idcg else 0.0

    return {
        "recall_at_1":    round(recall_at_k(1),    4),
        "recall_at_3":    round(recall_at_k(3),    4),
        "recall_at_5":    round(recall_at_k(5),    4),
        "precision_at_1": round(precision_at_k(1), 4),
        "precision_at_3": round(precision_at_k(3), 4),
        "precision_at_5": round(precision_at_k(5), 4),
        "hit_rate":       round(float(any(cid in rel_set for cid in chunk_ids)), 4),
        "mrr":            round(mrr,       4),
        "map_score":      round(map_score, 4),
        "ndcg_at_5":      round(ndcg5,     4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# HALLUCINATION / GROUNDING
# ══════════════════════════════════════════════════════════════════════════════

_HALL_RE = [
    re.compile(r'\bsection\s+\d+[\.\d]*\b',                re.I),
    re.compile(r'\bpage\s+\d+\b',                          re.I),
    re.compile(r'\bequation\s+\d+\b',                      re.I),
    re.compile(r'\btable\s+\d+\b',                         re.I),
    re.compile(r'\bfigure\s+\d+\b',                        re.I),
    re.compile(r'\b\d+\.?\d*\s*%?\s*(?:accuracy|f1|bleu|score)\b', re.I),
]

_CITE_RE = re.compile(r'\[\s*\d[\d,\s]*p[\s.]*\d+\s*\]', re.I)


def _grounding(answer: str, chunks: list) -> float:
    chunk_text = " ".join(
        getattr(c, "text", c.get("text", "") if isinstance(c, dict) else "")
        for c in chunks
    ).lower()
    if not chunk_text.strip():
        return 0.0
    sents = [s.strip() for s in re.split(r'[.!?]\s+|\n', answer) if len(s.strip()) > 20]
    if not sents:
        return 0.5
    grounded = 0
    for sent in sents:
        words = sent.lower().split()
        # Trigram
        found = any(" ".join(words[i:i+3]) in chunk_text for i in range(len(words)-2))
        # Bigram fallback
        if not found:
            found = any(" ".join(words[i:i+2]) in chunk_text for i in range(len(words)-1))
        # Keyword overlap fallback
        if not found:
            kws = set(_tok(sent, remove_stop=True))
            ck  = set(_tok(chunk_text, remove_stop=True))
            found = bool(kws) and len(kws & ck)/len(kws) >= 0.5
        if found:
            grounded += 1
    return round(grounded / len(sents), 4)


def _citation_coverage(answer: str) -> float:
    clean = _CITE_RE.sub("__CITE__", answer)
    lines = [l.strip() for l in clean.split('\n') if l.strip()]
    sents = []
    for line in lines:
        sents.extend([s.strip() for s in re.split(r'(?<=[.!?])\s+', line) if len(s.strip()) > 15])
    if not sents:
        return 0.0
    cited = sum(1 for s in sents if "__CITE__" in s)
    return round(cited / len(sents), 4)


def _hallucination_metrics(answer: str, chunks: list) -> dict:
    chunk_text = " ".join(
        getattr(c, "text", c.get("text", "") if isinstance(c, dict) else "")
        for c in chunks
    ).lower()
    flags = []
    for pat in _HALL_RE:
        for m in pat.findall(answer):
            if m.lower() not in chunk_text:
                flags.append(m)
    hall_prob = min(len(flags) * 0.15, 1.0)
    return {
        "unsupported_claims": len(flags),
        "hallucination_prob": round(hall_prob, 4),
    }


def _answer_relevance(query: str, answer: str) -> float:
    q_tok = set(_tok(query,  remove_stop=True))
    a_tok = set(_tok(answer, remove_stop=True))
    if not q_tok: return 0.0
    return round(len(q_tok & a_tok) / len(q_tok), 4)


def _context_relevance(query: str, chunks: list) -> float:
    q_tok   = set(_tok(query, remove_stop=True))
    c_text  = " ".join(getattr(c,"text","") for c in chunks)
    c_tok   = set(_tok(c_text, remove_stop=True))
    if not q_tok: return 0.0
    return round(len(q_tok & c_tok) / len(q_tok), 4)


# ══════════════════════════════════════════════════════════════════════════════
# GRADE
# ══════════════════════════════════════════════════════════════════════════════

def _grade(score: float) -> str:
    if score >= 0.80: return "A"
    if score >= 0.65: return "B"
    if score >= 0.50: return "C"
    if score >= 0.35: return "D"
    return "F"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    query:         str,
    answer:        str,
    chunks:        list,
    reference:     str | None  = None,   # ground truth (optional)
    relevant_ids:  list[str]   = None,   # relevant chunk IDs (optional)
) -> EvaluationResult:
    """
    Compute all available metrics.
    Text generation metrics require reference.
    Retrieval metrics require relevant_ids.
    Grounding/hallucination metrics require only chunks.
    """
    result   = EvaluationResult()
    warnings = []

    # ── Grounding (always computed) ───────────────────────────────────────────
    result.grounding_score   = _grounding(answer, chunks)
    result.citation_coverage = _citation_coverage(answer)
    result.answer_relevance  = _answer_relevance(query, answer)
    result.context_relevance = _context_relevance(query, chunks)

    hall = _hallucination_metrics(answer, chunks)
    result.unsupported_claims = hall["unsupported_claims"]
    result.hallucination_prob = hall["hallucination_prob"]
    result.faithfulness_score = round(
        result.grounding_score * (1 - result.hallucination_prob), 4
    )

    # ── Text generation metrics (requires reference) ──────────────────────────
    if reference and reference.strip():
        bleu_scores = _bleu(answer, reference)
        result.bleu   = bleu_scores["bleu"]
        result.bleu_1 = bleu_scores["bleu_1"]
        result.bleu_2 = bleu_scores["bleu_2"]
        result.bleu_3 = bleu_scores["bleu_3"]
        result.bleu_4 = bleu_scores["bleu_4"]

        result.rouge_1 = round(_rouge_n(answer, reference, 1), 4)
        result.rouge_2 = round(_rouge_n(answer, reference, 2), 4)
        result.rouge_l = round(_rouge_l(answer, reference), 4)
        result.meteor  = _meteor(answer, reference)

        prf = _prf1(answer, reference)
        result.precision = prf["precision"]
        result.recall    = prf["recall"]
        result.f1        = prf["f1"]

        result.exact_match    = answer.strip().lower() == reference.strip().lower()
        result.token_accuracy = result.f1   # token overlap as accuracy proxy

        # Macro/micro (single reference = same as token-level)
        result.macro_precision = result.precision
        result.macro_recall    = result.recall
        result.macro_f1        = result.f1
        result.micro_f1        = result.f1
        result.weighted_f1     = result.f1

    # ── Retrieval metrics (requires relevant_ids) ─────────────────────────────
    if relevant_ids:
        ret = _retrieval_metrics(chunks, relevant_ids, query)
        for k, v in ret.items():
            setattr(result, k, v)

    # ── Warnings ──────────────────────────────────────────────────────────────
    if result.grounding_score < 0.40:
        warnings.append(f"Low grounding ({result.grounding_score:.2f})")
    if result.hallucination_prob > 0.30:
        warnings.append(f"High hallucination risk ({result.hallucination_prob:.2f})")
    if result.citation_coverage < 0.20:
        warnings.append("Low citation coverage")
    if reference and result.f1 < 0.10:
        warnings.append(f"Low F1 ({result.f1:.2f}) — answer may be off-topic")
    result.warnings = warnings

    # ── Overall reliability ───────────────────────────────────────────────────
    components = [
        result.grounding_score  * 0.30,
        result.faithfulness_score * 0.20,
        result.answer_relevance * 0.15,
        result.citation_coverage * 0.10,
        (1 - result.hallucination_prob) * 0.10,
    ]
    if reference:
        components += [
            result.rouge_1 * 0.08,
            result.f1      * 0.07,
        ]
    result.reliability_score = round(min(sum(components), 1.0), 4)
    result.grade              = _grade(result.reliability_score)

    return result