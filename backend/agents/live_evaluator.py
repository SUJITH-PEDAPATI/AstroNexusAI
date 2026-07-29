"""
AstroNexus AI — Live Evaluator v3.0

Fixes:
    1. Citation regex now handles [1, p.1], [3, p.5], [Section 3, p.2]
       without splitting on dots inside brackets
    2. Grounding uses sliding window + partial match fallback
    3. All metrics computed consistently
"""
from __future__ import annotations

import math
import re
import logging
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

STOP = {
    "the","a","an","is","are","was","were","be","been","have","has","had",
    "do","does","did","will","would","could","should","to","of","in","for",
    "on","with","at","by","from","and","but","or","not","this","that","it",
    "its","as","if","then","than","so","yet","both","either","neither",
    "just","about","into","through","during","before","after","between",
}

# ── Citation pattern — handles all formats used in the system ─────────────────
# Matches: [1, p.1]  [3, p.5]  [Section 3, p.2]  [Methods p.4]
_CITE_RE = re.compile(
    r'\[\s*(?:Section\s+)?[\w\s,]+p[\s.]*\d+\s*\]',
    re.IGNORECASE,
)

# ── Sentence splitter — does NOT split on dots inside brackets ────────────────
_SENT_SPLIT_RE = re.compile(
    r'(?<!\[\d)(?<!\d,\s*p)(?<!\w\.\w)(?<=[.!?])\s+'
)


@dataclass
class LiveEvalResult:
    top_score:        float = 0.0
    mean_score:       float = 0.0
    score_variance:   float = 0.0
    coverage:         float = 0.0
    chunks_used:      int   = 0
    bleu:             float = 0.0
    rouge1:           float = 0.0
    rouge2:           float = 0.0
    rougeL:           float = 0.0
    precision:        float = 0.0
    recall:           float = 0.0
    f1:               float = 0.0
    grounding:        float = 0.0
    hallucination_flags: int = 0
    citation_coverage: float = 0.0
    reliability_score: float = 0.0
    grade:            str   = "F"
    warnings:         list  = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "retrieval": {
                "top_score":      round(self.top_score, 4),
                "mean_score":     round(self.mean_score, 4),
                "score_variance": round(self.score_variance, 4),
                "coverage":       round(self.coverage, 4),
                "chunks_used":    self.chunks_used,
            },
            "answer_quality": {
                "bleu":      round(self.bleu, 4),
                "rouge1":    round(self.rouge1, 4),
                "rouge2":    round(self.rouge2, 4),
                "rougeL":    round(self.rougeL, 4),
                "precision": round(self.precision, 4),
                "recall":    round(self.recall, 4),
                "f1":        round(self.f1, 4),
            },
            "faithfulness": {
                "grounding":           round(self.grounding, 4),
                "hallucination_flags": self.hallucination_flags,
                "citation_coverage":   round(self.citation_coverage, 4),
            },
            "overall": {
                "reliability_score": round(self.reliability_score, 4),
                "grade":             self.grade,
                "warnings":          self.warnings,
            },
        }


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

def _bleu(hyp: str, refs: list[str], max_n: int = 4) -> float:
    hyp_t = _tok(hyp)
    if not hyp_t: return 0.0
    ref_ts = [_tok(r) for r in refs]
    ref_len = min(len(r) for r in ref_ts) if ref_ts else 0
    bp = 1.0 if len(hyp_t) >= ref_len else math.exp(1 - ref_len/len(hyp_t))
    log_avg = 0.0
    for n in range(1, max_n+1):
        hg = _ngrams(hyp_t, n)
        if not hg: return 0.0
        clipped = sum(
            min(cnt, max(_ngrams(r, n).get(g, 0) for r in ref_ts))
            for g, cnt in hg.items()
        )
        total = sum(hg.values())
        if total == 0 or clipped == 0: return 0.0
        log_avg += math.log(clipped/total)
    return bp * math.exp(log_avg/max_n)


# ══════════════════════════════════════════════════════════════════════════════
# ROUGE
# ══════════════════════════════════════════════════════════════════════════════

def _rouge_n(hyp: str, ref: str, n: int) -> float:
    hg, rg = _ngrams(_tok(hyp), n), _ngrams(_tok(ref), n)
    overlap = sum(min(hg[g], rg[g]) for g in hg if g in rg)
    denom   = sum(rg.values())
    return overlap/denom if denom else 0.0


def _lcs(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp   = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j],dp[i][j-1])
    return dp[m][n]


def _rouge_l(hyp: str, ref: str) -> float:
    ht, rt = _tok(hyp), _tok(ref)
    l = _lcs(ht, rt)
    p = l/len(ht) if ht else 0.0
    r = l/len(rt) if rt else 0.0
    return 2*p*r/(p+r) if (p+r) else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# GROUNDING — Fixed: multi-level fallback, not just exact trigram match
# ══════════════════════════════════════════════════════════════════════════════

def _grounding_score(answer: str, chunks: list[dict]) -> float:
    """
    Multi-level grounding check per sentence:
        Level 1: 3-gram exact match       (strict)
        Level 2: 2-gram exact match       (moderate)
        Level 3: keyword overlap >= 0.5   (permissive fallback)

    A sentence is grounded if ANY level matches.
    This handles paraphrasing without inflating scores.
    """
    chunk_text = " ".join(
        c.get("text", c.get("payload", {}).get("text", "")).lower()
        for c in chunks
    )
    if not chunk_text.strip():
        return 0.0

    sentences = [
        s.strip() for s in re.split(r'[.!?]\s+|\n', answer)
        if len(s.strip()) > 20
    ]
    if not sentences:
        return 0.5

    grounded = 0
    for sent in sentences:
        words = sent.lower().split()
        found = False

        # Level 1: trigram
        for i in range(len(words)-2):
            if " ".join(words[i:i+3]) in chunk_text:
                found = True; break

        # Level 2: bigram
        if not found:
            for i in range(len(words)-1):
                if " ".join(words[i:i+2]) in chunk_text:
                    found = True; break

        # Level 3: keyword overlap
        if not found:
            kws     = set(_tok(sent, remove_stop=True))
            ck_text = set(_tok(chunk_text, remove_stop=True))
            if kws and len(kws & ck_text)/len(kws) >= 0.5:
                found = True

        if found:
            grounded += 1

    return grounded/len(sentences)


# ══════════════════════════════════════════════════════════════════════════════
# CITATION COVERAGE — Fixed: preserve dots inside brackets before splitting
# ══════════════════════════════════════════════════════════════════════════════

def _citation_coverage(answer: str) -> float:
    """
    Correct approach:
        1. Find all citations in the full answer
        2. Find all sentences
        3. Count sentences that have a citation on the same line
    Does NOT split on periods inside brackets.
    """
    # Strip citations temporarily to find sentences
    clean   = _CITE_RE.sub("__CITE__", answer)
    lines   = [l.strip() for l in clean.split('\n') if l.strip()]

    # Sentence-level check: does this line contain a citation marker?
    all_sentences = []
    for line in lines:
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', line)
                 if len(s.strip()) > 15]
        all_sentences.extend(sents)

    if not all_sentences:
        return 0.0

    cited = sum(1 for s in all_sentences if "__CITE__" in s)
    return cited/len(all_sentences)


# ══════════════════════════════════════════════════════════════════════════════
# HALLUCINATION DETECTION
# ══════════════════════════════════════════════════════════════════════════════

_HALL_PATTERNS = [
    ("section_ref",  re.compile(r'\bsection\s+\d+[\.\d]*\b',       re.I)),
    ("page_ref",     re.compile(r'\bpage\s+\d+\b',                 re.I)),
    ("equation_ref", re.compile(r'\bequation\s+\d+\b',             re.I)),
    ("table_ref",    re.compile(r'\btable\s+\d+\b',                re.I)),
    ("figure_ref",   re.compile(r'\bfigure\s+\d+\b',               re.I)),
    ("score_claim",  re.compile(r'\b\d+\.?\d*\s*%?\s*(?:accuracy|bleu|rouge|f1|score|mrr|ndcg)\b', re.I)),
]

def _hallucination_flags(answer: str, chunks: list[dict]) -> list[str]:
    chunk_text = " ".join(
        c.get("text", c.get("payload", {}).get("text", "")).lower()
        for c in chunks
    )
    flags = []
    for pname, pattern in _HALL_PATTERNS:
        for match in pattern.findall(answer):
            if match.lower() not in chunk_text:
                flags.append(f"{pname}: '{match}'")
    return flags


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL METRICS
# ══════════════════════════════════════════════════════════════════════════════

def _retrieval_metrics(query: str, chunks: list[dict]) -> dict:
    if not chunks:
        return {"top":0.0,"mean":0.0,"var":0.0,"coverage":0.0}
    scores = [c.get("score",0.0) for c in chunks]
    top    = max(scores)
    mean   = sum(scores)/len(scores)
    var    = sum((s-mean)**2 for s in scores)/len(scores)
    qtoks  = set(_tok(query, remove_stop=True))
    ctxt   = " ".join(c.get("text","") for c in chunks).lower()
    cov    = sum(1 for w in qtoks if w in ctxt)/len(qtoks) if qtoks else 0.0
    return {"top":top,"mean":mean,"var":var,"coverage":cov}


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
# MAIN EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_paper_answer(
    query:   str,
    answer:  str,
    chunks:  list[dict],
) -> LiveEvalResult:
    result   = LiveEvalResult()
    warnings = []

    if not answer.strip() or not chunks:
        result.warnings = ["Empty answer or no chunks"]
        return result

    chunk_texts = [c.get("text", c.get("payload", {}).get("text", "")) for c in chunks]

    # Retrieval
    ret = _retrieval_metrics(query, chunks)
    result.top_score      = ret["top"]
    result.mean_score     = ret["mean"]
    result.score_variance = ret["var"]
    result.coverage       = ret["coverage"]
    result.chunks_used    = len(chunks)

    # Answer quality
    best_ref     = max(chunk_texts, key=lambda r: _rouge_n(answer, r, 1))
    result.bleu  = _bleu(answer, chunk_texts)
    result.rouge1= _rouge_n(answer, best_ref, 1)
    result.rouge2= _rouge_n(answer, best_ref, 2)
    result.rougeL= _rouge_l(answer, best_ref)

    hyp_w = set(_tok(answer, remove_stop=True))
    ref_w = set()
    for t in chunk_texts: ref_w.update(_tok(t, remove_stop=True))
    ov    = hyp_w & ref_w
    result.precision = len(ov)/len(hyp_w) if hyp_w else 0.0
    result.recall    = len(ov)/len(ref_w)  if ref_w  else 0.0
    result.f1 = (
        2*result.precision*result.recall/(result.precision+result.recall)
        if (result.precision+result.recall) else 0.0
    )

    # Faithfulness
    result.grounding          = _grounding_score(answer, chunks)
    hall                      = _hallucination_flags(answer, chunks)
    result.hallucination_flags= len(hall)
    result.citation_coverage  = _citation_coverage(answer)

    # Warnings
    if result.top_score < 0.35:
        warnings.append(f"Low retrieval score ({result.top_score:.2f})")
    if result.grounding < 0.50:
        warnings.append(f"Low grounding ({result.grounding:.2f})")
    if result.citation_coverage < 0.30:
        warnings.append("Low citation coverage")
    for h in hall[:2]:
        warnings.append(f"Possible hallucination: {h}")

    # Reliability
    result.reliability_score = min(1.0, (
        result.top_score       * 0.20 +
        result.coverage        * 0.10 +
        result.rouge1          * 0.15 +
        result.f1              * 0.15 +
        result.grounding       * 0.25 +
        result.citation_coverage * 0.10 +
        (1 - min(result.hallucination_flags * 0.2, 1.0)) * 0.05
    ))
    result.grade    = _grade(result.reliability_score)
    result.warnings = warnings

    logger.info(
        f"[Eval] BLEU={result.bleu:.3f} R1={result.rouge1:.3f} "
        f"F1={result.f1:.3f} gnd={result.grounding:.3f} "
        f"cite={result.citation_coverage:.3f} grade={result.grade}"
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS
# ══════════════════════════════════════════════════════════════════════════════

def _run_tests():
    print("Running evaluator unit tests...")

    # Test 1: Citation coverage with bracket-dot format
    ans1   = "OST operates from 5-600 µm [1, p.1]. It has a 5.9m mirror [3, p.5]."
    chunks1= [{"text":"OST operates from 5 to 600 µm infrared","score":0.8}]
    cov1   = _citation_coverage(ans1)
    assert cov1 > 0, f"FAIL T1: citation_coverage={cov1}, expected >0"
    print(f"  T1 citation_coverage: {cov1:.3f} ✓")

    # Test 2: Grounding — paraphrase should still ground
    ans2   = "The telescope provides unprecedented infrared sensitivity"
    chunks2= [{"text":"OST will open new discovery space in infrared astronomy","score":0.7}]
    gnd2   = _grounding_score(ans2, chunks2)
    assert gnd2 > 0, f"FAIL T2: grounding={gnd2}"
    print(f"  T2 grounding (paraphrase): {gnd2:.3f} ✓")

    # Test 3: Empty answer
    r3 = evaluate_paper_answer("test", "", [{"text":"some text","score":0.5}])
    assert r3.grade == "F", f"FAIL T3: expected F got {r3.grade}"
    print(f"  T3 empty answer: grade={r3.grade} ✓")

    # Test 4: Full evaluation
    ans4    = "The Origins Space Telescope is designed for infrared observations [1, p.1]."
    chunks4 = [{"text":"Origins Space Telescope designed for infrared","score":0.82}]
    r4      = evaluate_paper_answer("What is OST?", ans4, chunks4)
    assert r4.citation_coverage > 0, f"FAIL T4: citation_coverage={r4.citation_coverage}"
    print(f"  T4 full eval: grade={r4.grade} cite={r4.citation_coverage:.3f} ✓")

    print("All tests passed.\n")


if __name__ == "__main__":
    _run_tests()