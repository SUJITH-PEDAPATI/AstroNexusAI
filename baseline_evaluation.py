"""
AstroNexus AI — Baseline Comparison Evaluation
================================================
Measures Recall@k, Grounding, Accuracy (Token F1), and Latency
for all four baselines on the SAME query set:

    1. LLM-only        (no retrieval, no graph)
    2. Dense RAG       (Qdrant retrieval + LLM, no graph, no web)
    3. KG-only QA      (Neo4j graph agent only)
    4. AstroNexus AI   (full orchestrator pipeline)

Usage:
    1. Edit QUERIES below with your test questions + ground-truth answers.
    2. Ingest papers into Qdrant + Neo4j first (POST /ingest).
    3. Run:   python baseline_evaluation.py
    4. Results saved to:  output/baseline_results.json
                          output/baseline_results.csv
                          output/baseline_latex_table.tex

All metrics are computed — nothing is fabricated.
"""
from __future__ import annotations

import json
import os
import sys
import time
import statistics
from pathlib import Path
from dataclasses import dataclass, field, asdict

# ── Add project root to path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Query set ──────────────────────────────────────────────────────────────────
# EDIT THIS: add your actual test queries with ground-truth answers.
# ground_truth_answer: the expected answer text (for Token F1 / accuracy).
# ground_truth_section: the expected paper section (for Recall@k relevance).
# paper_id: the paper_id in Qdrant (set to None for general queries).

QUERIES = [
    {
        "id": "R01",
        "question": "What is the main contribution of this paper?",
        "ground_truth_answer": "The Transformer model architecture based entirely on attention mechanisms.",
        "ground_truth_section": "Abstract",
        "paper_id": None,  # set to your actual paper_id after ingestion
    },
    {
        "id": "R02",
        "question": "What dataset was used for training?",
        "ground_truth_answer": "WMT 2014 English-German and English-French datasets.",
        "ground_truth_section": "Training",
        "paper_id": None,
    },
    {
        "id": "R03",
        "question": "What baseline models were compared?",
        "ground_truth_answer": "ConvS2S, ByteNet, and various LSTM-based models.",
        "ground_truth_section": "Results",
        "paper_id": None,
    },
    {
        "id": "R04",
        "question": "What evaluation metrics were reported?",
        "ground_truth_answer": "BLEU score on English-German and English-French translation.",
        "ground_truth_section": "Results",
        "paper_id": None,
    },
    {
        "id": "R05",
        "question": "What loss function was used?",
        "ground_truth_answer": "Cross-entropy loss with label smoothing.",
        "ground_truth_section": "Methods",
        "paper_id": None,
    },
    # ADD MORE QUERIES HERE — aim for 30+
]

# ── Retrieval top-k (same for all systems) ─────────────────────────────────────
TOP_K = 5

# ── Output directory ───────────────────────────────────────────────────────────
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)


# ── Metric computation ────────────────────────────────────────────────────────

def compute_token_f1(generated: str, reference: str) -> float:
    """Token-level F1 between generated answer and ground truth."""
    gen_tokens = set(generated.lower().split())
    ref_tokens = set(reference.lower().split())
    if not gen_tokens or not ref_tokens:
        return 0.0
    common = gen_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(gen_tokens)
    recall    = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_recall_at_k(
    retrieved_sections: list[str],
    ground_truth_section: str,
) -> float:
    """
    Recall@k: 1.0 if any of the top-k retrieved chunks comes from the
    correct section, else 0.0.  (Binary hit — standard for single-answer QA.)
    """
    if not ground_truth_section:
        return 0.0
    gt = ground_truth_section.lower().strip()
    for sec in retrieved_sections:
        if gt in sec.lower():
            return 1.0
    return 0.0


def compute_grounding(answer: str, context_chunks: list[str]) -> float:
    """
    Grounding score: fraction of answer sentences whose key terms appear
    in the retrieved context.  Matches the answer_evaluator.py approach.
    """
    if not answer.strip() or not context_chunks:
        return 0.0

    import re
    context_text = " ".join(context_chunks).lower()
    sentences = [s.strip() for s in re.split(r'[.!?]+', answer) if len(s.strip()) > 10]
    if not sentences:
        return 0.0

    grounded = 0
    for sent in sentences:
        words = [w for w in sent.lower().split() if len(w) > 3]
        if not words:
            continue
        overlap = sum(1 for w in words if w in context_text) / len(words)
        if overlap >= 0.3:
            grounded += 1

    return grounded / len(sentences)


# ── System runners ────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    system:     str
    query_id:   str
    answer:     str            = ""
    chunks:     list[str]      = field(default_factory=list)
    sections:   list[str]      = field(default_factory=list)
    scores:     list[float]    = field(default_factory=list)
    latency_s:  float          = 0.0
    error:      str            = ""


def run_llm_only(query: dict) -> RunResult:
    """Baseline 1: LLM answers without any retrieval or graph."""
    t0 = time.perf_counter()
    try:
        # Use Ollama directly — no retrieval, no graph
        import urllib.request
        import json as _json

        base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.environ.get("OLLAMA_MODEL", "qwen3:4b")

        payload = _json.dumps({
            "model": model,
            "prompt": f"Answer this scientific question concisely:\n\n{query['question']}",
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 300},
        }).encode()

        req = urllib.request.Request(
            f"{base}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            answer = _json.loads(resp.read()).get("response", "").strip()

        return RunResult(
            system="LLM-only",
            query_id=query["id"],
            answer=answer,
            latency_s=time.perf_counter() - t0,
        )
    except Exception as e:
        return RunResult(
            system="LLM-only",
            query_id=query["id"],
            latency_s=time.perf_counter() - t0,
            error=str(e),
        )


def run_dense_rag(query: dict) -> RunResult:
    """Baseline 2: Dense Qdrant retrieval + LLM generation (no graph, no web)."""
    t0 = time.perf_counter()
    try:
        from backend.rag.retriever import retrieve
        from backend.embeddings.embedder import embed_query

        results = retrieve(
            query=query["question"],
            top_k=TOP_K,
            score_threshold=0.0,  # accept all for fair comparison
            filter_paper_id=query.get("paper_id"),
        )

        chunks   = [r.text for r in results]
        sections = [r.section or "" for r in results]
        scores   = [r.score for r in results]

        # Generate answer from retrieved context only
        context = "\n\n".join(
            f"[{i+1}] ({sections[i]}) {chunks[i][:500]}"
            for i in range(len(chunks))
        )

        import urllib.request
        import json as _json
        base  = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.environ.get("OLLAMA_MODEL", "qwen3:4b")

        prompt = (
            f"Based ONLY on the following retrieved passages, answer the question.\n\n"
            f"PASSAGES:\n{context}\n\nQUESTION: {query['question']}\n\nANSWER:"
        )
        payload = _json.dumps({
            "model": model, "prompt": prompt,
            "stream": False, "options": {"temperature": 0.1, "num_predict": 300},
        }).encode()
        req = urllib.request.Request(
            f"{base}/api/generate", data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            answer = _json.loads(resp.read()).get("response", "").strip()

        return RunResult(
            system="Dense RAG",
            query_id=query["id"],
            answer=answer,
            chunks=chunks,
            sections=sections,
            scores=scores,
            latency_s=time.perf_counter() - t0,
        )
    except Exception as e:
        return RunResult(
            system="Dense RAG",
            query_id=query["id"],
            latency_s=time.perf_counter() - t0,
            error=str(e),
        )


def run_kg_only(query: dict) -> RunResult:
    """Baseline 3: Knowledge graph retrieval only (Neo4j Cypher, no vector search)."""
    t0 = time.perf_counter()
    try:
        from backend.graph.neo4j_client import _get_driver

        # Extract keywords from query
        words = [w.lower() for w in query["question"].split() if len(w) > 3]

        driver = _get_driver()
        graph_text = []
        with driver.session() as session:
            for kw in words[:5]:
                rows = session.run(
                    "MATCH (n) WHERE toLower(n.name) CONTAINS $kw "
                    "OPTIONAL MATCH (n)-[r]-(m) "
                    "RETURN n.name AS source, type(r) AS rel, m.name AS target "
                    "LIMIT 10",
                    kw=kw,
                ).data()
                for row in rows:
                    if row.get("rel") and row.get("target"):
                        graph_text.append(
                            f"{row['source']} --[{row['rel']}]--> {row['target']}"
                        )

        context = "\n".join(graph_text) if graph_text else "No graph data found."

        import urllib.request
        import json as _json
        base  = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.environ.get("OLLAMA_MODEL", "qwen3:4b")

        prompt = (
            f"Based ONLY on these knowledge graph relationships, answer the question.\n\n"
            f"GRAPH:\n{context}\n\nQUESTION: {query['question']}\n\nANSWER:"
        )
        payload = _json.dumps({
            "model": model, "prompt": prompt,
            "stream": False, "options": {"temperature": 0.1, "num_predict": 300},
        }).encode()
        req = urllib.request.Request(
            f"{base}/api/generate", data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            answer = _json.loads(resp.read()).get("response", "").strip()

        return RunResult(
            system="KG-only QA",
            query_id=query["id"],
            answer=answer,
            chunks=graph_text,
            latency_s=time.perf_counter() - t0,
        )
    except Exception as e:
        return RunResult(
            system="KG-only QA",
            query_id=query["id"],
            latency_s=time.perf_counter() - t0,
            error=str(e),
        )


def run_astronexus(query: dict) -> RunResult:
    """Full AstroNexus AI system via orchestrator.run()."""
    t0 = time.perf_counter()
    try:
        from backend.agents.orchestrator import run

        result = run(
            query=query["question"],
            paper_loaded=bool(query.get("paper_id")),
            paper_id=query.get("paper_id"),
        )

        answer = result.get("final_answer", "")
        meta   = result.get("metadata") or {}
        evl    = meta.get("evaluation") or {}

        # Extract retrieved chunks from metadata if available
        rag_ctx = result.get("rag_context", "")
        chunks  = [rag_ctx] if rag_ctx else []

        return RunResult(
            system="AstroNexus AI",
            query_id=query["id"],
            answer=answer,
            chunks=chunks,
            latency_s=time.perf_counter() - t0,
        )
    except Exception as e:
        return RunResult(
            system="AstroNexus AI",
            query_id=query["id"],
            latency_s=time.perf_counter() - t0,
            error=str(e),
        )


# ── Main evaluation loop ─────────────────────────────────────────────────────

def main():
    if len(QUERIES) < 3:
        print("WARNING: Only {} queries defined. Add more for reliable results.".format(len(QUERIES)))

    systems = [
        ("LLM-only",      run_llm_only),
        ("Dense RAG",     run_dense_rag),
        ("KG-only QA",    run_kg_only),
        ("AstroNexus AI", run_astronexus),
    ]

    all_results: list[dict] = []

    for sys_name, runner in systems:
        print(f"\n{'='*50}")
        print(f"  Evaluating: {sys_name}")
        print(f"{'='*50}")

        recall_scores    = []
        grounding_scores = []
        f1_scores        = []
        latencies        = []

        for i, q in enumerate(QUERIES, 1):
            print(f"  [{i}/{len(QUERIES)}] {q['id']}: {q['question'][:50]}...")

            result = runner(q)

            if result.error:
                print(f"    ERROR: {result.error}")
                continue

            # Recall@k
            r_at_k = compute_recall_at_k(
                result.sections,
                q.get("ground_truth_section", ""),
            )
            recall_scores.append(r_at_k)

            # Grounding
            grounding = compute_grounding(result.answer, result.chunks)
            grounding_scores.append(grounding)

            # Token F1 (accuracy proxy)
            f1 = compute_token_f1(result.answer, q.get("ground_truth_answer", ""))
            f1_scores.append(f1)

            # Latency
            latencies.append(result.latency_s)

            print(f"    Recall@{TOP_K}={r_at_k:.2f}  Grounding={grounding:.3f}  "
                  f"F1={f1:.3f}  Latency={result.latency_s:.2f}s")

            all_results.append({
                "system":     sys_name,
                "query_id":   q["id"],
                "recall_at_k": r_at_k,
                "grounding":  grounding,
                "token_f1":   f1,
                "latency_s":  result.latency_s,
                "answer_len": len(result.answer),
                "chunks_used": len(result.chunks),
                "error":      result.error,
            })

        # Aggregate
        n = len(recall_scores)
        if n == 0:
            print(f"\n  {sys_name}: ALL QUERIES FAILED — no metrics to report")
            continue

        avg_recall    = statistics.mean(recall_scores)
        avg_grounding = statistics.mean(grounding_scores)
        avg_f1        = statistics.mean(f1_scores)
        avg_latency   = statistics.mean(latencies)

        print(f"\n  {sys_name} AGGREGATE ({n} queries):")
        print(f"    Recall@{TOP_K}  = {avg_recall:.4f}")
        print(f"    Grounding = {avg_grounding:.4f}")
        print(f"    Token F1  = {avg_f1:.4f}")
        print(f"    Latency   = {avg_latency:.2f}s")

    # ── Save results ──────────────────────────────────────────────────────────
    with open(OUT_DIR / "baseline_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'baseline_results.json'}")

    # CSV
    import csv
    with open(OUT_DIR / "baseline_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Saved: {OUT_DIR / 'baseline_results.csv'}")

    # LaTeX table
    _generate_latex_table(all_results)


def _generate_latex_table(results: list[dict]):
    """Generate the IEEE LaTeX table from aggregated results."""
    from collections import defaultdict

    by_system = defaultdict(list)
    for r in results:
        if not r["error"]:
            by_system[r["system"]].append(r)

    lines = [
        r"\begin{table}[t]",
        r"\caption{Baseline Comparison (Higher is Better Unless Noted)}",
        r"\label{tab:results}",
        r"\centering",
        r"\renewcommand{\arraystretch}{1.25}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"\textbf{System} & \textbf{Recall@" + str(TOP_K) + r"} & \textbf{Grounding} & \textbf{Token F1} & \textbf{Lat.\,(s)$\downarrow$} \\",
        r"\midrule",
    ]

    order = ["LLM-only", "Dense RAG", "KG-only QA", "AstroNexus AI"]

    for sys_name in order:
        rows = by_system.get(sys_name, [])
        if not rows:
            recall = grounding = f1 = latency = "N/A"
        else:
            recall    = f"{statistics.mean(r['recall_at_k'] for r in rows):.2f}"
            grounding = f"{statistics.mean(r['grounding'] for r in rows):.2f}"
            f1        = f"{statistics.mean(r['token_f1'] for r in rows):.2f}"
            latency   = f"{statistics.mean(r['latency_s'] for r in rows):.1f}"

        if sys_name == "AstroNexus AI":
            lines.append(
                rf"\textbf{{{sys_name}}} & \textbf{{{recall}}} & "
                rf"\textbf{{{grounding}}} & \textbf{{{f1}}} & "
                rf"\textbf{{{latency}}} \\"
            )
        else:
            lines.append(
                rf"{sys_name} & {recall} & {grounding} & {f1} & {latency} \\"
            )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    tex_path = OUT_DIR / "baseline_latex_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {tex_path}")
    print("\nLaTeX table content:")
    print("\n".join(lines))


if __name__ == "__main__":
    main()