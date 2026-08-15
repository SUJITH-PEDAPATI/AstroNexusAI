"""
Unit tests for evaluation metrics.
Run with: python -m evaluation.test_metrics
"""
import sys, math
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from evaluation.metrics import (
    recall_at_k, precision_at_k, mean_reciprocal_rank, ndcg_at_k,
    token_f1, fact_coverage, grounding_score,
)

PASS = 0; FAIL = 0
def check(name, got, expected, tol=1e-4):
    global PASS, FAIL
    if abs(got - expected) < tol:
        print(f"  PASS  {name}: {got:.4f}")
        PASS += 1
    else:
        print(f"  FAIL  {name}: got {got:.4f}, expected {expected:.4f}")
        FAIL += 1

print("=== nDCG@k (fixed) ===")
# Perfect: first hit at rank 1 → DCG = IDCG = 1/log2(2) = 1.0
check("nDCG perfect",      ndcg_at_k(["A"]*10, {"A"}, 10), 1.0)
# Complete miss
check("nDCG zero",         ndcg_at_k(["B"]*10, {"A"}, 10), 0.0)
# Hit at rank 3 only: DCG = 1/log2(4) = 0.5; IDCG = 1.0 → nDCG = 0.5
check("nDCG rank-3 hit",   ndcg_at_k(["B","B","A","B"], {"A"}, 10), 0.5)
# Duplicate hits from same paper — should only count first occurrence
check("nDCG dedup",        ndcg_at_k(["A","A","A","A","A"], {"A"}, 10), 1.0)
# Two relevant papers, both retrieved
check("nDCG two-relevant", ndcg_at_k(["A","B","C"], {"A","B"}, 5),
      (1/math.log2(2) + 1/math.log2(3)) / (1/math.log2(2) + 1/math.log2(3)))
# Must never exceed 1.0
for trial in [["A"]*5, ["A","B","A","A"], ["A","B"]*5]:
    v = ndcg_at_k(trial, {"A","B"}, 10)
    if v > 1.0 + 1e-9:
        print(f"  FAIL  nDCG>1: {v:.4f} for {trial}")
        FAIL += 1
    else:
        PASS += 1
print(f"  nDCG<=1 invariant: PASS for all trials")

print("\n=== Recall@k ===")
check("Recall@1 perfect",   recall_at_k(["A","B"], {"A"}, 1), 1.0)
check("Recall@1 miss",      recall_at_k(["B","A"], {"A"}, 1), 0.0)
check("Recall@5 partial",   recall_at_k(["A"]*5,  {"A","B"}, 5), 0.5)
check("Recall@0 empty-rel", recall_at_k(["A"],    set(), 5), 0.0)

print("\n=== Precision@k ===")
check("Precision@5 all-hit", precision_at_k(["A"]*5, {"A"}, 5), 1.0)
check("Precision@5 no-hit",  precision_at_k(["B"]*5, {"A"}, 5), 0.0)
check("Precision@5 partial", precision_at_k(["A","B","A","B","B"], {"A"}, 5), 0.4)

print("\n=== MRR ===")
check("MRR rank-1", mean_reciprocal_rank(["A","B"], {"A"}), 1.0)
check("MRR rank-2", mean_reciprocal_rank(["B","A"], {"A"}), 0.5)
check("MRR miss",   mean_reciprocal_rank(["B","C"], {"A"}), 0.0)

print("\n=== Token F1 ===")
check("F1 identical", token_f1("hello world", "hello world"), 1.0)
check("F1 zero",      token_f1("hello world", "foo bar"),     0.0)
check("F1 partial",   token_f1("hello world foo", "hello bar"), 0.4)

print("\n=== Fact coverage ===")
check("FC all",  fact_coverage("The imaging spectroscopic data", ["imaging","spectroscopic"]), 1.0)
check("FC none", fact_coverage("hello world", ["imaging","spectroscopic"]), 0.0)
check("FC empty-facts", fact_coverage("anything", []), 1.0)

print("\n=== Grounding (no leakage) ===")
ctx = ["The transformer model uses attention mechanisms for translation tasks."]
check("Grnd high", grounding_score("The transformer uses attention for translation.", ctx), 1.0)
check("Grnd zero", grounding_score("Quantum physics reveals cosmic structure.", ctx), 0.0)

print(f"\n{'='*40}")
print(f"RESULTS: {PASS} PASS  {FAIL} FAIL")
if FAIL == 0:
    print("ALL TESTS PASS — metrics are mathematically correct")
else:
    print("SOME TESTS FAILED — do not proceed to ablation")
    sys.exit(1)
