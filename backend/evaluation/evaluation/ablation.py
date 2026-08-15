"""
AstroNexusAI — Retrieval-Only Ablation & Parameter Sensitivity Study
=====================================================================
All experiments in this module are retrieval-only.  No LLM, no Gemini,
no answer generation of any kind.  If --no-generation is NOT set, the
runner still refuses to call Gemini — generation must be explicitly
re-enabled via --enable-generation for future use when quota is restored.

Usage:
    # Baseline retrieval (35 questions)
    python -m evaluation.ablation --phase baseline --no-generation

    # Top-k sweep
    python -m evaluation.ablation --phase topk --no-generation

    # Score-threshold sweep
    python -m evaluation.ablation --phase threshold --no-generation

    # Dense vs KG vs Hybrid ablation
    python -m evaluation.ablation --phase hybrid --no-generation

    # All retrieval phases
    python -m evaluation.ablation --phase retrieval-all --no-generation

    # Smoke test (5 questions)
    python -m evaluation.ablation --phase baseline --no-generation --limit 5

Architecture note
-----------------
"KG retrieval" in AstroNexusAI is not a separate index — it is the Neo4j
graph supplementing context AFTER Qdrant retrieval in knowledge_fusion.fuse().
The dense Qdrant retrieval is always the primary retrieval mechanism.

Therefore the three retrieval modes tested are:
  A. Dense-only (Qdrant, paper filter, no KG context call)
  B. Dense + KG  (Qdrant, paper filter, Neo4j graph_context appended)
  C. Corpus-wide (Qdrant, no paper filter — simulates cross-paper retrieval)

Trust weighting is NOT implemented as a retrieval-side parameter in the
current codebase (it is applied at generation/prompt-building time).
It is correctly reported as NOT IMPLEMENTED for retrieval-side metrics.

Chunk-size ablation requires re-indexing Qdrant. This file does NOT do that.
Chunk-size results are reported as NOT TESTED (requires re-indexing).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import logging

# Windows terminals default to cp1252 which cannot encode Unicode chars like → ═ ─.
# Reconfigure stdout/stderr to UTF-8 so print() never crashes on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from datetime import datetime
from pathlib import Path
from typing import Optional

_PROJECT      = Path(__file__).resolve().parent.parent   # backend/evaluation/
_PROJECT_ROOT = _PROJECT.parent.parent                    # project root (contains backend/)
for _p in [str(_PROJECT), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Metrics import works whether invoked as:
#   python -m evaluation.ablation          (from backend/)
#   python -m backend.evaluation.evaluation.ablation  (from project root)
try:
    from evaluation.metrics import (
        recall_at_k, precision_at_k, mean_reciprocal_rank, ndcg_at_k,
    )
except ModuleNotFoundError:
    from backend.evaluation.evaluation.metrics import (  # type: ignore[no-redef]
        recall_at_k, precision_at_k, mean_reciprocal_rank, ndcg_at_k,
    )

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("ablation")

# ── LLM BLOCK ────────────────────────────────────────────────────────────────
# When _LLM_BLOCKED=True any attempt to call Gemini raises immediately.
_LLM_BLOCKED: bool = True   # default: always blocked until explicitly released

def _block_llm_imports() -> None:
    """
    Patch sys.modules so that any import of Gemini-related modules raises.
    Called unconditionally at startup. Only skipped if --enable-generation
    is passed (future use when quota is restored).
    """
    class _BlockedModule:
        def __getattr__(self, name):
            raise RuntimeError(
                f"\n{'='*60}\n"
                f"  BLOCKED: Attempt to import Gemini/LLM module.\n"
                f"  This evaluation runs in --no-generation mode.\n"
                f"  No Gemini API calls are permitted.\n"
                f"{'='*60}"
            )
    for name in ("google.genai", "google.generativeai", "genai"):
        sys.modules[name] = _BlockedModule()   # type: ignore

_block_llm_imports()

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_BASE = _PROJECT / "evaluation" / "results"
DATASET      = _PROJECT / "evaluation" / "dataset.jsonl"

for d in [
    RESULTS_BASE / "retrieval_ablation",
    RESULTS_BASE / "top_k",
    RESULTS_BASE / "threshold",
    RESULTS_BASE / "latency",
    RESULTS_BASE / "tables",
]:
    d.mkdir(parents=True, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────

def load_qs(limit: Optional[int] = None) -> list[dict]:
    qs = []
    with open(DATASET) as f:
        for line in f:
            if line.strip():
                qs.append(json.loads(line))
    return qs[:limit] if limit else qs


def get_paper_id_map() -> dict[str, str]:
    """Scan Qdrant payloads and return {label: paper_id}."""
    from backend.rag.vector_store import _get_client, COLLECTION_NAME
    client  = _get_client()
    results, _ = client.scroll(
        collection_name=COLLECTION_NAME, limit=2000,
        with_payload=True, with_vectors=False,
    )
    papers: dict[str, dict] = {}
    for r in results:
        p   = r.payload or {}
        pid = p.get("paper_id", "")
        if pid and pid not in papers:
            papers[pid] = {"title": p.get("title", ""), "chunks": 0}
        if pid:
            papers[pid]["chunks"] += 1

    mapping: dict[str, str] = {}
    for pid, info in papers.items():
        t = info["title"].lower()
        if "aion" in t:
            mapping["AION-1"] = pid
        elif "astrom3" in t or "astro-m3" in t:
            mapping["AstroM3"] = pid
        elif "knowledge" in t or "interdisciplinary" in t:
            mapping["KnowledgeGraph"] = pid

    print(f"\nPaper ID map: {json.dumps(mapping, indent=2)}")
    if not mapping:
        raise RuntimeError("No papers found in Qdrant. Ingest papers first.")
    return mapping


# ── Hit-rate metric ────────────────────────────────────────────────────────────

def hit_at_k(retrieved_ids: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(p in relevant for p in retrieved_ids[:k]) else 0.0


# ── Core retrieval for one question ──────────────────────────────────────────

def _eval_one_retrieval(
    q: dict,
    paper_id: Optional[str],
    top_k: int,
    score_threshold: float,
    relevant: set[str],
    with_kg: bool = False,
) -> tuple[dict, dict]:
    """
    Run one retrieval query and return (metric_dict, latency_dict).
    with_kg=True fetches Neo4j graph context (does NOT call any LLM).
    """
    from backend.rag.retriever import retrieve
    from backend.embeddings.embedder import embed_query

    # Embedding
    t0    = time.perf_counter()
    _     = embed_query(q["question"])
    t_emb = (time.perf_counter() - t0) * 1000

    # Qdrant retrieval
    t1     = time.perf_counter()
    chunks = retrieve(
        query=           q["question"],
        top_k=           top_k,
        score_threshold= score_threshold,
        filter_paper_id= paper_id,
    )
    t_qdrant = (time.perf_counter() - t1) * 1000

    # Optional Neo4j graph context (retrieval-only, no LLM)
    t_kg = 0.0
    if with_kg and paper_id:
        try:
            t2   = time.perf_counter()
            from backend.graph.neo4j_client import _get_driver
            driver = _get_driver()
            with driver.session() as s:
                s.run(
                    "MATCH (p:Paper {paper_id:$pid}) "
                    "OPTIONAL MATCH (p)-[:TAGGED]->(k:Keyword) "
                    "RETURN k.name AS kw LIMIT 20",
                    pid=paper_id,
                ).data()
            t_kg = (time.perf_counter() - t2) * 1000
        except Exception as e:
            logger.debug(f"Neo4j skipped: {e}")

    total_ms = t_emb + t_qdrant + t_kg

    retrieved_ids = [c.paper_id for c in chunks]
    scores        = [c.score    for c in chunks]

    metrics = {
        "recall@1":     recall_at_k(retrieved_ids, relevant, 1),
        "recall@3":     recall_at_k(retrieved_ids, relevant, 3),
        "recall@5":     recall_at_k(retrieved_ids, relevant, 5),
        "recall@10":    recall_at_k(retrieved_ids, relevant, 10),
        "precision@1":  precision_at_k(retrieved_ids, relevant, 1),
        "precision@3":  precision_at_k(retrieved_ids, relevant, 3),
        "precision@5":  precision_at_k(retrieved_ids, relevant, 5),
        "mrr":          mean_reciprocal_rank(retrieved_ids, relevant),
        "ndcg@10":      ndcg_at_k(retrieved_ids, relevant, 10),
        "hit@1":        hit_at_k(retrieved_ids, relevant, 1),
        "hit@5":        hit_at_k(retrieved_ids, relevant, 5),
        "n_retrieved":  len(chunks),
        "top_score":    scores[0] if scores else 0.0,
        "mean_score":   round(statistics.mean(scores), 4) if scores else 0.0,
    }
    latency = {
        "embed_ms":  round(t_emb, 2),
        "qdrant_ms": round(t_qdrant, 2),
        "kg_ms":     round(t_kg, 2),
        "total_ms":  round(total_ms, 2),
    }
    return metrics, latency


# ── Aggregate over all questions ──────────────────────────────────────────────

def run_config(
    label:          str,
    questions:      list[dict],
    paper_id_map:   dict[str, str],
    top_k:          int,
    score_threshold: float,
    paper_filter:   str = "per_paper",   # "per_paper" | "none"
    with_kg:        bool = False,
) -> dict:
    """
    Evaluate one retrieval configuration over the full question set.

    paper_filter="per_paper": filter_paper_id=<paper's id> (standard mode)
    paper_filter="none":      filter_paper_id=None (corpus-wide search)
    """
    all_metrics:  list[dict] = []
    all_latency:  list[dict] = []
    failures:     list[dict] = []

    print(f"\n  [{label}] top_k={top_k} threshold={score_threshold} "
          f"filter={paper_filter} kg={with_kg}")

    for q in questions:
        label_q = q["paper_label"]

        if paper_filter == "none":
            paper_id = None
            relevant = set(paper_id_map.values())
        elif label_q == "cross":
            paper_id = None
            relevant = set(paper_id_map.values())
        else:
            paper_id = paper_id_map.get(label_q)
            relevant = {paper_id} if paper_id else set()

        try:
            m, lat = _eval_one_retrieval(
                q, paper_id, top_k, score_threshold, relevant, with_kg=with_kg,
            )
            all_metrics.append(m)
            all_latency.append(lat)
        except Exception as e:
            logger.warning(f"  FAIL [{q['id']}]: {e}")
            failures.append({"id": q["id"], "error": str(e)})

    n = len(all_metrics)
    if n == 0:
        return {"config": label, "n": 0, "error": "all queries failed"}

    def _avg(key):
        return round(sum(m[key] for m in all_metrics) / n, 4)

    def _lat_stat(key):
        vals = [l[key] for l in all_latency]
        return {
            "mean":   round(statistics.mean(vals), 1),
            "median": round(statistics.median(vals), 1),
            "p95":    round(sorted(vals)[max(0, int(0.95 * len(vals)) - 1)], 1),
        }

    result = {
        "config":       label,
        "top_k":        top_k,
        "threshold":    score_threshold,
        "paper_filter": paper_filter,
        "with_kg":      with_kg,
        "n":            n,
        "failures":     len(failures),
        # Retrieval metrics
        "recall@1":     _avg("recall@1"),
        "recall@3":     _avg("recall@3"),
        "recall@5":     _avg("recall@5"),
        "recall@10":    _avg("recall@10"),
        "precision@1":  _avg("precision@1"),
        "precision@3":  _avg("precision@3"),
        "precision@5":  _avg("precision@5"),
        "mrr":          _avg("mrr"),
        "ndcg@10":      _avg("ndcg@10"),
        "hit@1":        _avg("hit@1"),
        "hit@5":        _avg("hit@5"),
        "top_score_mean": _avg("top_score"),
        # Latency (retrieval only, no generation)
        "latency_embed_ms":  _lat_stat("embed_ms"),
        "latency_qdrant_ms": _lat_stat("qdrant_ms"),
        "latency_kg_ms":     _lat_stat("kg_ms"),
        "latency_total_ms":  _lat_stat("total_ms"),
        # Short fields for table display
        "lat_mean_ms":   _lat_stat("total_ms")["mean"],
        "lat_median_ms": _lat_stat("total_ms")["median"],
        "lat_p95_ms":    _lat_stat("total_ms")["p95"],
        # Generation metrics (not run in this session)
        "token_f1":      "N/A — generation disabled",
        "fact_coverage": "N/A — generation disabled",
        "grounding":     "N/A — generation disabled",
    }

    r5   = result["recall@5"]
    mrr  = result["mrr"]
    ndcg = result["ndcg@10"]
    lat  = result["lat_mean_ms"]
    print(f"    → Recall@5={r5}  MRR={mrr}  nDCG@10={ndcg}  Latency={lat}ms")
    return result


# ── Phases ────────────────────────────────────────────────────────────────────

def phase_baseline(qs, pid_map) -> list[dict]:
    print("\n" + "="*56)
    print("PHASE 1 — BASELINE (top_k=5, threshold=0.55, no-gen)")
    print("="*56)
    return [run_config("baseline", qs, pid_map, top_k=5, score_threshold=0.55)]


def phase_topk(qs, pid_map) -> list[dict]:
    print("\n" + "="*56)
    print("PHASE 2 — TOP-K SENSITIVITY (threshold=0.0)")
    print("="*56)
    results = []
    for k in [1, 3, 5, 8, 10, 15]:
        r = run_config(f"topk_{k}", qs, pid_map, top_k=k, score_threshold=0.0)
        results.append(r)
    return results


def phase_threshold(qs, pid_map) -> list[dict]:
    print("\n" + "="*56)
    print("PHASE 3 — SCORE THRESHOLD SENSITIVITY (top_k=5)")
    print("="*56)
    results = []
    for thr in [0.30, 0.40, 0.50, 0.55, 0.60, 0.70]:
        r = run_config(f"thr_{thr}", qs, pid_map, top_k=5, score_threshold=thr)
        results.append(r)
    return results


def phase_hybrid(qs, pid_map) -> list[dict]:
    """
    Dense-only vs Dense+KG vs Corpus-wide retrieval.

    Implementation reality (from code inspection):
    - "KG-only" retrieval does not exist as a standalone path in AstroNexusAI.
      Neo4j provides supplementary graph context AFTER Qdrant retrieval.
    - "Dense+KG" = Qdrant retrieval + Neo4j graph context fetch (no LLM).
    - "Corpus-wide" = Qdrant with no per-paper filter (all 3 papers searched).
    - Trust weighting: NOT IMPLEMENTED at retrieval level — it is prompt-
      construction logic inside knowledge_fusion._build_prompt, not a
      retrieval ranking parameter. Reported as NOT IMPLEMENTED.
    """
    print("\n" + "="*56)
    print("PHASE 4 — RETRIEVAL MODE ABLATION")
    print("  Dense-only | Dense+KG | Corpus-wide")
    print("  Trust weighting: NOT IMPLEMENTED at retrieval level")
    print("="*56)
    results = []

    # A. Dense-only (no Neo4j call)
    results.append(run_config(
        "dense_only", qs, pid_map,
        top_k=5, score_threshold=0.0, with_kg=False,
    ))

    # B. Dense + KG context (Neo4j fetch, no LLM)
    results.append(run_config(
        "dense_plus_kg", qs, pid_map,
        top_k=5, score_threshold=0.0, with_kg=True,
    ))

    # C. Corpus-wide (no paper filter — multi-document retrieval)
    results.append(run_config(
        "corpus_wide", qs, pid_map,
        top_k=5, score_threshold=0.0, paper_filter="none",
    ))

    return results


# ── Output helpers ────────────────────────────────────────────────────────────

RET_COLS = [
    "recall@1","recall@3","recall@5","recall@10",
    "precision@5","mrr","ndcg@10","hit@1","hit@5",
    "lat_mean_ms","lat_median_ms","lat_p95_ms",
]

def print_table(results: list[dict], cols: list[str] = RET_COLS) -> None:
    w = 14
    header = f"{'config':<22} " + " ".join(f"{c[:w]:<{w}}" for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        row = f"{r.get('config',''):<22} " + " ".join(
            f"{str(r.get(c,'—'))[:w]:<{w}}" for c in cols
        )
        print(row)


def write_md(results: list[dict], cols: list[str], path: Path) -> None:
    hdrs = ["Config"] + [c.replace("_"," ").replace("@","@") for c in cols]
    lines = ["| " + " | ".join(hdrs) + " |",
             "| " + " | ".join(["---"]*len(hdrs)) + " |"]
    for r in results:
        cells = [str(r.get("config",""))] + [str(r.get(c,"—")) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")
    print(f"  Table → {path}")


def save_json(data, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AstroNexusAI retrieval ablation")
    parser.add_argument("--phase", default="retrieval-all",
        choices=["baseline","topk","threshold","hybrid","retrieval-all"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-generation", action="store_true",
        help="Retrieval-only mode (Gemini blocked). REQUIRED for this study.")
    parser.add_argument("--enable-generation", action="store_true",
        help="Future flag: re-enable LLM generation when quota is restored.")
    args = parser.parse_args()

    if not args.no_generation and not args.enable_generation:
        print("WARNING: Neither --no-generation nor --enable-generation specified.")
        print("Defaulting to --no-generation (retrieval-only) to protect Gemini quota.")
        args.no_generation = True

    # ── Hard LLM block banner ─────────────────────────────────────────────────
    if args.no_generation:
        print("\n" + "="*56)
        print("  RETRIEVAL-ONLY EVALUATION")
        print("  Gemini/LLM generation DISABLED")
        print("  No Gemini API calls will be made")
        print("="*56)

    # Run unit tests first (fast, no Qdrant needed)
    print("\nVerifying metric implementations...")
    import math as _m
    from evaluation.metrics import ndcg_at_k as _ndcg
    assert _ndcg(["A"]*10, {"A"}, 10) == 1.0,   "nDCG perfect != 1.0"
    assert _ndcg(["B"]*10, {"A"}, 10) == 0.0,   "nDCG miss != 0.0"
    assert _ndcg(["A"]*10, {"A"}, 10) <= 1.0,   "nDCG > 1"
    print("  nDCG unit tests: PASS")

    qs      = load_qs(args.limit)
    pid_map = get_paper_id_map()
    print(f"\nLoaded {len(qs)} questions | papers: {list(pid_map.keys())}")

    RD = RESULTS_BASE

    if args.phase in ("baseline", "retrieval-all"):
        r = phase_baseline(qs, pid_map)
        print_table(r)
        save_json(r, RD / "retrieval_ablation" / "baseline.json")
        write_md(r, RET_COLS, RD / "tables" / "baseline.md")

    if args.phase in ("topk", "retrieval-all"):
        r = phase_topk(qs, pid_map)
        print_table(r)
        save_json(r, RD / "top_k" / "results.json")
        write_md(r, RET_COLS, RD / "tables" / "topk_table.md")

    if args.phase in ("threshold", "retrieval-all"):
        r = phase_threshold(qs, pid_map)
        print_table(r)
        save_json(r, RD / "threshold" / "results.json")
        write_md(r, RET_COLS, RD / "tables" / "threshold_table.md")

    if args.phase in ("hybrid", "retrieval-all"):
        r = phase_hybrid(qs, pid_map)
        print_table(r)
        for row in r:
            fn = row["config"].replace(" ","_") + ".json"
            save_json(row, RD / "retrieval_ablation" / fn)
        write_md(r, RET_COLS, RD / "tables" / "hybrid_ablation.md")

        # Trust-weighting note
        note = {
            "trust_weighting":  "NOT IMPLEMENTED at retrieval level",
            "explanation": (
                "Trust weighting in AstroNexusAI is applied inside "
                "knowledge_fusion._build_prompt() during context assembly "
                "for LLM generation. It does not affect Qdrant ranking or "
                "Neo4j graph retrieval. No retrieval-side trust-weighting "
                "experiment is possible without generation."
            ),
        }
        save_json(note, RD / "retrieval_ablation" / "trust_weighting_note.json")

        # Chunk-size note
        chunk_note = {
            "chunk_size_ablation":  "NOT TESTED — requires Qdrant re-indexing",
            "explanation": (
                "Changing chunk size requires re-running the full ingestion "
                "pipeline (text → chunks → embeddings → Qdrant upsert). "
                "This cannot be done without overwriting or creating a new "
                "Qdrant collection. Chunk-size ablation is deferred to "
                "a future experiment session."
            ),
            "current_config": {"chunk_size": 512, "chunk_overlap": 64},
        }
        save_json(chunk_note, RD / "retrieval_ablation" / "chunk_size_note.json")

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "timestamp":         datetime.now().isoformat(),
        "n_questions":       len(qs),
        "gemini_used":       False,
        "llm_used":          False,
        "generation_metrics": "N/A — generation disabled",
        "note":              (
            "Gemini API quota was exhausted. All experiments are retrieval-only. "
            "Generation metrics (Token F1, Fact Coverage, Grounding, Correctness, "
            "Hallucination) must be recalculated when Gemini quota is restored."
        ),
    }
    save_json(summary, RD / "retrieval_ablation" / "summary.json")
    print(f"\nAll results saved to: {RD}")
    print("\nGemini generation was NOT used in this experiment because "
          "the Gemini API quota was exhausted.")
    print("\nRETRIEVAL ABLATION STATUS: READY")


if __name__ == "__main__":
    main()
