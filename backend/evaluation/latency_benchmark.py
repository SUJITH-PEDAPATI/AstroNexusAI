"""
AstroNexus AI — End-to-End Latency Benchmark
==============================================
Measures ACTUAL latency of all implemented pipeline configurations by
calling each service layer directly with time.perf_counter() instrumentation.

Configurations tested:
    1. Local Ollama  — qwen3:4b via /api/generate
    2. Gemini API    — gemini-3.1-flash-lite  (if GEMINI_API_KEY is set)
    3. RAG           — Qdrant retrieval → Gemini generation
    4. Knowledge Graph — Neo4j query → Gemini generation
    5. Full Agentic  — POST /chat (FastAPI → Router → LangGraph → agents)

Usage (from project root):
    python -m backend.evaluation.latency_benchmark

    # Quick smoke test (5 iterations, skip slow configs)
    python -m backend.evaluation.latency_benchmark --n 5 --quick

    # Custom server URL
    python -m backend.evaluation.latency_benchmark --api-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Path bootstrap ──────────────────────────────────────────────────────────────
_BACKEND = Path(__file__).resolve().parent.parent
_ROOT    = _BACKEND.parent
for _p in [str(_BACKEND), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_ROOT / ".env", override=False)
    load_dotenv(dotenv_path=_ROOT / "backend" / ".env", override=False)
except ImportError:
    pass

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ── Output directory ────────────────────────────────────────────────────────────
_RESULTS_DIR = _BACKEND / "evaluation" / "results" / "latency"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def _ok(s):   return f"[OK]  {s}"
def _warn(s): return f"[WARN] {s}"
def _err(s):  return f"[ERR] {s}"
def _hdr(s):  return f"\n{'-'*60}\n  {s}\n{'-'*60}"


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICS HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _stats(samples: list[float]) -> dict:
    """Compute full latency statistics from a list of millisecond values."""
    if not samples:
        return {}
    n = len(samples)
    s = sorted(samples)
    mean_v   = statistics.mean(s)
    median_v = statistics.median(s)
    std_v    = statistics.stdev(s) if n > 1 else 0.0
    p95_idx  = min(int(math.ceil(0.95 * n)) - 1, n - 1)
    p95_v    = s[p95_idx]
    return {
        "n":       n,
        "min_ms":  round(s[0],     1),
        "max_ms":  round(s[-1],    1),
        "mean_ms": round(mean_v,   1),
        "med_ms":  round(median_v, 1),
        "p95_ms":  round(p95_v,    1),
        "std_ms":  round(std_v,    1),
        "samples": [round(x, 1) for x in samples],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT PROBE
# ═══════════════════════════════════════════════════════════════════════════════

def probe_environment() -> dict:
    env: dict = {}
    env["machine"] = {
        "os":      platform.system() + " " + platform.version()[:40],
        "python":  platform.python_version(),
        "cpu":     platform.processor() or "unknown",
    }
    try:
        import torch
        env["pytorch"] = {
            "version":    torch.__version__,
            "cuda_avail": torch.cuda.is_available(),
            "gpu":        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only",
        }
    except ImportError:
        env["pytorch"] = {"version": "not installed"}

    try:
        import psutil
        env["machine"]["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        env["machine"]["ram_gb"] = "psutil not installed"

    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as r:
            data   = json.loads(r.read())
            models = [m["name"] for m in data.get("models", [])]
        env["ollama"] = {"status": "running", "models": models}
    except Exception as e:
        env["ollama"] = {"status": f"offline: {e}"}

    try:
        import requests
        r = requests.get("http://localhost:6333/collections", timeout=3)
        data = r.json()
        colls = [c["name"] for c in data.get("result", {}).get("collections", [])]
        env["qdrant"] = {"status": "running", "collections": colls, "host": "localhost:6333"}
    except Exception as e:
        env["qdrant"] = {"status": f"offline: {e}"}

    try:
        from neo4j import GraphDatabase
        d = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "astronexus123"))
        d.verify_connectivity()
        d.close()
        env["neo4j"] = {"status": "running", "uri": "bolt://localhost:7687"}
    except Exception as e:
        env["neo4j"] = {"status": f"offline: {e}"}

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    env["gemini"] = {
        "key_set": bool(gemini_key),
        "model":   os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
    }

    pkg_versions = {}
    for pkg, attr in [
        ("langchain_core", "__version__"),
        ("langgraph",      "__version__"),
        ("qdrant_client",  "__version__"),
        ("neo4j",          "__version__"),
    ]:
        try:
            mod = __import__(pkg)
            pkg_versions[pkg] = getattr(mod, attr, "installed")
        except ImportError:
            pkg_versions[pkg] = "not installed"
    env["packages"] = pkg_versions
    return env


# ═══════════════════════════════════════════════════════════════════════════════
# TEST QUERIES
# ═══════════════════════════════════════════════════════════════════════════════

_QUERIES = [
    "What is the architecture of the AION model and how does it handle multimodal inputs?",
    "Explain the training methodology used in AstroM3 for astronomical classification.",
    "What datasets were used to train the models described in the papers?",
    "How does the knowledge graph represent relationships between astronomical entities?",
    "What are the key evaluation metrics reported in the astronomy research papers?",
    "Describe the attention mechanism used in the transformer-based astronomy model.",
    "What is the role of contrastive learning in the AstroM3 architecture?",
    "How does the model handle class imbalance in the astronomical dataset?",
    "What pre-training strategies are described for the astronomy foundation model?",
    "Summarise the main findings of the knowledge graph paper.",
    "What are the limitations mentioned by the authors of the AION paper?",
    "How does the model performance compare across different astronomical surveys?",
]

_GRAPH_QUERIES = [
    "Who authored the AION paper?",
    "Which papers are related to transformer models?",
    "What datasets are tagged in the knowledge graph?",
    "List algorithms related to classification in the papers.",
    "Which authors have published on astronomical surveys?",
    "What institutions are associated with the research papers?",
    "Which models are connected to the AstroM3 dataset?",
    "What entities are linked to contrastive learning?",
    "Who wrote about knowledge graphs in astronomy?",
    "List the keywords tagged to the uploaded papers.",
]


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG 1 — LOCAL OLLAMA
# ═══════════════════════════════════════════════════════════════════════════════

def bench_ollama(queries: list[str], n: int, model: str = "qwen3:4b") -> dict:
    print(_hdr(f"CONFIG 1 -- Local Ollama ({model})"))
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    samples:    list[float] = []
    cold_start: Optional[float] = None
    errors = 0

    for i in range(n):
        q = queries[i % len(queries)]
        payload = json.dumps({
            "model":  model,
            "prompt": q,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 300},
        }).encode()

        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(
                f"{base_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                json.loads(resp.read())
            ms = (time.perf_counter() - t0) * 1000
            if cold_start is None:
                cold_start = ms
                print(f"  [{i+1:>2}/{n}] Cold start: {ms:>8.0f} ms")
            else:
                samples.append(ms)
                print(f"  [{i+1:>2}/{n}] Warm:       {ms:>8.0f} ms")
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            errors += 1
            print(_err(f"  [{i+1:>2}/{n}] FAILED ({ms:.0f} ms): {e}"))

    result = {
        "config": "Local Ollama", "model": model,
        "n_total": n, "n_errors": errors,
        "cold_start_ms": round(cold_start, 1) if cold_start else None,
        "warm": _stats(samples),
    }
    _print_stats("Ollama warm", result["warm"])
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG 2 — GEMINI API
# ═══════════════════════════════════════════════════════════════════════════════

def bench_gemini(queries: list[str], n: int) -> dict:
    print(_hdr("CONFIG 2 -- Gemini API"))
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

    if not gemini_key:
        print(_warn("GEMINI_API_KEY not set -- skipping"))
        return {"config": "Gemini", "status": "SKIPPED -- GEMINI_API_KEY not set"}

    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=gemini_key)
    except ImportError as e:
        return {"config": "Gemini", "status": f"SKIPPED -- google-genai not installed: {e}"}

    samples:    list[float] = []
    cold_start: Optional[float] = None
    errors = 0
    system = "You are a concise astronomical AI assistant. Answer in 2-3 sentences."

    for i in range(n):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        try:
            resp = client.models.generate_content(
                model=    model_name,
                contents= f"{system}\n\nQuestion: {q}\n\nAnswer:",
                config=   types.GenerateContentConfig(temperature=0.1, max_output_tokens=300),
            )
            _ = resp.text
            ms = (time.perf_counter() - t0) * 1000
            if cold_start is None:
                cold_start = ms
                print(f"  [{i+1:>2}/{n}] Cold start: {ms:>8.0f} ms")
            else:
                samples.append(ms)
                print(f"  [{i+1:>2}/{n}] Warm:       {ms:>8.0f} ms")
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            errors += 1
            print(_err(f"  [{i+1:>2}/{n}] FAILED ({ms:.0f} ms): {e}"))

    result = {
        "config": "Gemini API", "model": model_name,
        "n_total": n, "n_errors": errors,
        "cold_start_ms": round(cold_start, 1) if cold_start else None,
        "warm": _stats(samples),
    }
    _print_stats("Gemini warm", result["warm"])
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG 3 — RAG (Qdrant + Gemini)
# ═══════════════════════════════════════════════════════════════════════════════

def bench_rag(queries: list[str], n: int) -> dict:
    print(_hdr("CONFIG 3 -- RAG (Qdrant retrieval + Gemini generation)"))
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

    try:
        from backend.rag.retriever import retrieve
    except Exception as e:
        return {"config": "RAG", "status": f"SKIPPED -- retriever import failed: {e}"}

    gemini_client = None
    _gtypes = None
    if gemini_key:
        try:
            from google import genai
            from google.genai import types
            gemini_client = genai.Client(api_key=gemini_key)
            _gtypes = types
        except ImportError:
            pass

    total_samples:  list[float] = []
    ret_samples:    list[float] = []
    gen_samples:    list[float] = []
    cold_start:     Optional[float] = None
    errors = 0

    for i in range(n):
        q = queries[i % len(queries)]

        t_ret = time.perf_counter()
        try:
            chunks = retrieve(q, top_k=5, score_threshold=0.0)
            ret_ms = (time.perf_counter() - t_ret) * 1000
            ctx    = [c.text for c in chunks]
        except Exception as e:
            errors += 1
            print(_err(f"  [{i+1:>2}/{n}] Retrieval FAILED: {e}"))
            continue

        gen_ms = 0.0
        if gemini_client and ctx:
            ctx_str = "\n\n".join(f"[Chunk {j+1}]\n{t[:600]}" for j, t in enumerate(ctx))
            prompt  = f"CONTEXT:\n{ctx_str}\n\nQUESTION: {q}\n\nANSWER:"
            t_gen   = time.perf_counter()
            try:
                resp = gemini_client.models.generate_content(
                    model=    model_name,
                    contents= prompt,
                    config=   _gtypes.GenerateContentConfig(temperature=0.1, max_output_tokens=400),
                )
                _ = resp.text
                gen_ms = (time.perf_counter() - t_gen) * 1000
            except Exception as e:
                gen_ms = (time.perf_counter() - t_gen) * 1000
                errors += 1
                print(_err(f"  [{i+1:>2}/{n}] Generation FAILED ({gen_ms:.0f} ms): {e}"))

        total_ms = ret_ms + gen_ms
        if cold_start is None:
            cold_start = total_ms
            print(f"  [{i+1:>2}/{n}] Cold:  {total_ms:>8.0f} ms  (ret={ret_ms:.0f}  gen={gen_ms:.0f}  chunks={len(chunks)})")
        else:
            total_samples.append(total_ms)
            ret_samples.append(ret_ms)
            gen_samples.append(gen_ms)
            print(f"  [{i+1:>2}/{n}] Warm:  {total_ms:>8.0f} ms  (ret={ret_ms:.0f}  gen={gen_ms:.0f}  chunks={len(chunks)})")

    result = {
        "config": "RAG (Qdrant + Gemini)", "model": model_name,
        "n_total": n, "n_errors": errors,
        "cold_start_ms": round(cold_start, 1) if cold_start else None,
        "warm": _stats(total_samples),
        "breakdown": {
            "qdrant_retrieval":  _stats(ret_samples),
            "gemini_generation": _stats(gen_samples),
        },
    }
    _print_stats("RAG E2E warm", result["warm"])
    print(f"    Qdrant mean:  {result['breakdown']['qdrant_retrieval'].get('mean_ms','N/A')} ms")
    print(f"    Gemini mean:  {result['breakdown']['gemini_generation'].get('mean_ms','N/A')} ms")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG 4 — KNOWLEDGE GRAPH (Neo4j + Gemini)
# ═══════════════════════════════════════════════════════════════════════════════

def bench_knowledge_graph(queries: list[str], n: int) -> dict:
    print(_hdr("CONFIG 4 -- Knowledge Graph (Neo4j + Gemini)"))
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "astronexus123"))
        driver.verify_connectivity()
    except Exception as e:
        return {"config": "Knowledge Graph", "status": f"SKIPPED -- Neo4j offline: {e}"}

    gemini_client = None
    _gtypes = None
    if gemini_key:
        try:
            from google import genai
            from google.genai import types
            gemini_client = genai.Client(api_key=gemini_key)
            _gtypes = types
        except ImportError:
            pass

    total_samples: list[float] = []
    neo4j_samples: list[float] = []
    gen_samples:   list[float] = []
    cold_start:    Optional[float] = None
    errors = 0
    stop = {"who","what","when","where","how","the","is","are","was","did",
            "does","a","an","of","in","to","for","and","or","by","at","on",
            "explain","define","describe","list","which"}

    for i in range(n):
        q    = queries[i % len(queries)]
        kws  = [w.strip("?.,!\"'").lower() for w in q.split()
                if len(w.strip("?.,!\"'")) > 3 and w.strip("?.,!\"'").lower() not in stop][:5]

        t_neo = time.perf_counter()
        rows: list[dict] = []
        try:
            with driver.session() as session:
                for kw in kws:
                    rows.extend(session.run(
                        "MATCH (n) WHERE (n:Paper OR n:Model OR n:Dataset OR n:Author "
                        "OR n:Entity OR n:Keyword) "
                        "AND toLower(coalesce(n.name, n.title, '')) CONTAINS toLower($kw) "
                        "OPTIONAL MATCH (n)-[r]->(m) "
                        "RETURN labels(n)[0] AS type, coalesce(n.name, n.title,'') AS name, "
                        "type(r) AS relation, coalesce(m.name, m.title,'') AS related LIMIT 6",
                        kw=kw,
                    ).data())
            neo_ms = (time.perf_counter() - t_neo) * 1000
        except Exception as e:
            errors += 1
            print(_err(f"  [{i+1:>2}/{n}] Neo4j FAILED: {e}"))
            continue

        lines = []
        seen  = set()
        for row in rows[:15]:
            key = f"{row.get('name')}:{row.get('related')}"
            if key in seen:
                continue
            seen.add(key)
            if row.get("relation") and row.get("related"):
                lines.append(f"  {row.get('type')}: {row.get('name')} -> [{row.get('relation')}] -> {row.get('related')}")
            else:
                lines.append(f"  {row.get('type')}: {row.get('name')}")
        graph_ctx = "\n".join(lines) if lines else "No graph data found."

        gen_ms = 0.0
        if gemini_client:
            prompt = f"KNOWLEDGE GRAPH:\n{graph_ctx}\n\nQUESTION: {q}\n\nANSWER:"
            t_gen  = time.perf_counter()
            try:
                resp = gemini_client.models.generate_content(
                    model=    model_name,
                    contents= prompt,
                    config=   _gtypes.GenerateContentConfig(temperature=0.1, max_output_tokens=300),
                )
                _ = resp.text
                gen_ms = (time.perf_counter() - t_gen) * 1000
            except Exception as e:
                gen_ms = (time.perf_counter() - t_gen) * 1000
                errors += 1
                print(_err(f"  [{i+1:>2}/{n}] Generation FAILED ({gen_ms:.0f} ms): {e}"))

        total_ms = neo_ms + gen_ms
        if cold_start is None:
            cold_start = total_ms
            print(f"  [{i+1:>2}/{n}] Cold:  {total_ms:>8.0f} ms  (neo4j={neo_ms:.0f}  gen={gen_ms:.0f}  rows={len(rows)})")
        else:
            total_samples.append(total_ms)
            neo4j_samples.append(neo_ms)
            gen_samples.append(gen_ms)
            print(f"  [{i+1:>2}/{n}] Warm:  {total_ms:>8.0f} ms  (neo4j={neo_ms:.0f}  gen={gen_ms:.0f}  rows={len(rows)})")

    driver.close()
    result = {
        "config": "Knowledge Graph (Neo4j + Gemini)", "model": model_name,
        "n_total": n, "n_errors": errors,
        "cold_start_ms": round(cold_start, 1) if cold_start else None,
        "warm": _stats(total_samples),
        "breakdown": {
            "neo4j_query":       _stats(neo4j_samples),
            "gemini_generation": _stats(gen_samples),
        },
    }
    _print_stats("KG E2E warm", result["warm"])
    print(f"    Neo4j mean:  {result['breakdown']['neo4j_query'].get('mean_ms','N/A')} ms")
    print(f"    Gemini mean: {result['breakdown']['gemini_generation'].get('mean_ms','N/A')} ms")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG 5 — FULL AGENTIC PIPELINE (POST /chat)
# ═══════════════════════════════════════════════════════════════════════════════

def bench_full_agentic(queries: list[str], n: int, api_url: str) -> dict:
    print(_hdr(f"CONFIG 5 -- Full Agentic Pipeline (POST {api_url}/chat)"))
    try:
        import requests as _req
    except ImportError:
        return {"config": "Full Agentic", "status": "SKIPPED -- requests not installed"}

    try:
        _req.get(f"{api_url}/docs", timeout=5)
    except Exception as e:
        return {"config": "Full Agentic", "status": f"SKIPPED -- server not reachable: {e}"}

    total_samples:  list[float] = []
    cold_start:     Optional[float] = None
    errors = 0
    route_counts: dict[str, int] = {}

    for i in range(n):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        try:
            resp = _req.post(
                f"{api_url}/chat",
                json={"query": q, "paper_id": None},
                timeout=180,
            )
            resp.raise_for_status()
            body  = resp.json()
            ms    = (time.perf_counter() - t0) * 1000
            qtype = body.get("query_type", "unknown")
            route_counts[qtype] = route_counts.get(qtype, 0) + 1

            if cold_start is None:
                cold_start = ms
                print(f"  [{i+1:>2}/{n}] Cold start: {ms:>8.0f} ms  route={qtype}")
            else:
                total_samples.append(ms)
                print(f"  [{i+1:>2}/{n}] Warm:       {ms:>8.0f} ms  route={qtype}")
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            errors += 1
            print(_err(f"  [{i+1:>2}/{n}] FAILED ({ms:.0f} ms): {e}"))

    result = {
        "config": "Full Agentic Pipeline",
        "endpoint": f"{api_url}/chat",
        "n_total": n, "n_errors": errors,
        "cold_start_ms": round(cold_start, 1) if cold_start else None,
        "route_counts": route_counts,
        "warm": _stats(total_samples),
    }
    _print_stats("Full Agentic warm", result["warm"])
    print(f"    Route distribution: {route_counts}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MICRO BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════

def bench_query_router(queries: list[str], n: int) -> dict:
    print(_hdr("MICRO -- Query Router (pure regex, no I/O)"))
    try:
        from backend.services.query_router import classify
    except Exception as e:
        return {"component": "QueryRouter", "status": f"SKIPPED: {e}"}
    samples: list[float] = []
    for i in range(n):
        t0 = time.perf_counter()
        classify(queries[i % len(queries)])
        samples.append((time.perf_counter() - t0) * 1000)
    result = {"component": "QueryRouter", **_stats(samples)}
    print(f"  Mean: {result.get('mean_ms')} ms  p95: {result.get('p95_ms')} ms")
    return result


def bench_qdrant_only(queries: list[str], n: int) -> dict:
    print(_hdr("MICRO -- Qdrant Retrieval (BGE-M3 embed + ANN search)"))
    try:
        from backend.rag.retriever import retrieve
    except Exception as e:
        return {"component": "Qdrant", "status": f"SKIPPED: {e}"}
    samples:    list[float] = []
    cold_start: Optional[float] = None
    for i in range(n):
        t0 = time.perf_counter()
        try:
            chunks = retrieve(queries[i % len(queries)], top_k=5, score_threshold=0.0)
            ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            print(_err(f"  [{i+1:>2}/{n}] FAILED: {e}"))
            continue
        if cold_start is None:
            cold_start = ms
            print(f"  [{i+1:>2}/{n}] Cold (model load): {ms:>8.0f} ms  chunks={len(chunks)}")
        else:
            samples.append(ms)
            print(f"  [{i+1:>2}/{n}] Warm:              {ms:>8.0f} ms  chunks={len(chunks)}")
    result = {
        "component": "Qdrant (BGE-M3 embed + ANN)",
        "cold_start_ms": round(cold_start, 1) if cold_start else None,
        **_stats(samples),
    }
    _print_stats("Qdrant warm", result)
    return result


def bench_neo4j_only(queries: list[str], n: int) -> dict:
    print(_hdr("MICRO -- Neo4j Raw Cypher Query"))
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "astronexus123"))
        driver.verify_connectivity()
    except Exception as e:
        return {"component": "Neo4j", "status": f"SKIPPED: {e}"}
    samples: list[float] = []
    for i in range(n):
        q   = queries[i % len(queries)]
        kws = [w.lower() for w in q.split() if len(w) > 4][:3]
        t0  = time.perf_counter()
        try:
            with driver.session() as session:
                for kw in kws:
                    session.run(
                        "MATCH (n) WHERE toLower(coalesce(n.name,n.title,'')) "
                        "CONTAINS toLower($kw) RETURN n LIMIT 5",
                        kw=kw,
                    ).data()
            ms = (time.perf_counter() - t0) * 1000
            samples.append(ms)
            print(f"  [{i+1:>2}/{n}] {ms:>8.0f} ms")
        except Exception as e:
            print(_err(f"  [{i+1:>2}/{n}] FAILED: {e}"))
    driver.close()
    result = {"component": "Neo4j Cypher", **_stats(samples)}
    _print_stats("Neo4j Cypher", result)
    return result


def bench_fastapi_overhead(api_url: str, n: int) -> dict:
    print(_hdr(f"MICRO -- FastAPI HTTP overhead ({api_url})"))
    try:
        import requests as _req
    except ImportError:
        return {"component": "FastAPI overhead", "status": "SKIPPED"}
    endpoint = "/docs"
    for ep in ["/health", "/docs", "/"]:
        try:
            _req.get(f"{api_url}{ep}", timeout=5)
            endpoint = ep
            break
        except Exception:
            continue
    samples: list[float] = []
    for i in range(n):
        t0 = time.perf_counter()
        try:
            _req.get(f"{api_url}{endpoint}", timeout=10)
            ms = (time.perf_counter() - t0) * 1000
            samples.append(ms)
            print(f"  [{i+1:>2}/{n}] {ms:.1f} ms")
        except Exception as e:
            print(_err(f"  [{i+1:>2}/{n}] FAILED: {e}"))
    result = {"component": f"FastAPI HTTP ({endpoint})", **_stats(samples)}
    print(f"  Mean round-trip: {result.get('mean_ms')} ms")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PRINT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _print_stats(label: str, s: dict) -> None:
    n = s.get("n", 0) if s else 0
    if not n:
        print(f"  {label}: no successful samples")
        return
    print(
        f"  {label} (n={n}): "
        f"mean={s.get('mean_ms')} ms  "
        f"median={s.get('med_ms')} ms  "
        f"p95={s.get('p95_ms')} ms  "
        f"min={s.get('min_ms')} ms  "
        f"max={s.get('max_ms')} ms  "
        f"std={s.get('std_ms')} ms"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary_table(results: dict) -> None:
    print("\n\n" + "="*82)
    print("  FINAL LATENCY COMPARISON TABLE  (warm requests only, all values in ms)")
    print("="*82)

    configs = [
        ("Local Ollama (qwen3:4b)", results.get("ollama",          {})),
        ("Gemini API",              results.get("gemini",          {})),
        ("RAG (Qdrant+Gemini)",     results.get("rag",             {})),
        ("Knowledge Graph",         results.get("knowledge_graph", {})),
        ("Full Agentic Pipeline",   results.get("full_agentic",    {})),
    ]
    header = f"  {'Configuration':<24} {'Mean':>8} {'Median':>8} {'P95':>8} {'Min':>8} {'Max':>8} {'StdDev':>8} {'N':>5}"
    print(header)
    print("  " + "-"*78)

    for label, r in configs:
        if not r:
            print(f"  {label:<24}  {'not run':>8}")
            continue
        status = r.get("status", "")
        if status.startswith("SKIPPED"):
            print(f"  {label:<24}  {status[:56]}")
            continue
        s = r.get("warm", {})
        if not s:
            print(f"  {label:<24}  no warm data")
            continue
        print(
            f"  {label:<24}  "
            f"{str(s.get('mean_ms','-')):>8}  "
            f"{str(s.get('med_ms','-')):>8}  "
            f"{str(s.get('p95_ms','-')):>8}  "
            f"{str(s.get('min_ms','-')):>8}  "
            f"{str(s.get('max_ms','-')):>8}  "
            f"{str(s.get('std_ms','-')):>8}  "
            f"{str(s.get('n','-')):>5}"
        )

    print("\n  COLD-START LATENCY (first request, ms)")
    print("  " + "-"*50)
    for label, r in configs:
        cs = r.get("cold_start_ms")
        if cs is not None:
            print(f"  {label:<28}  {cs:>10.1f} ms")

    print("\n  COMPONENT BREAKDOWN (mean ms, warm)")
    print("  " + "-"*60)
    micro = results.get("micro", {})
    rows  = [
        ("FastAPI HTTP overhead",    micro.get("fastapi_overhead", {}).get("mean_ms")),
        ("Query Router (regex)",     micro.get("query_router",     {}).get("mean_ms")),
        ("Qdrant (embed+ANN warm)",  micro.get("qdrant_only",      {}).get("mean_ms")),
        ("Qdrant cold start",        micro.get("qdrant_only",      {}).get("cold_start_ms")),
        ("Neo4j Cypher",             micro.get("neo4j_only",       {}).get("mean_ms")),
        ("Ollama LLM (qwen3:4b)",    results.get("ollama", {}).get("warm", {}).get("mean_ms")),
        ("Gemini LLM (API)",         results.get("gemini", {}).get("warm", {}).get("mean_ms")),
    ]
    rag_bd = results.get("rag", {}).get("breakdown", {})
    if rag_bd:
        rows += [
            ("RAG Qdrant retrieval",      rag_bd.get("qdrant_retrieval", {}).get("mean_ms")),
            ("RAG Gemini generation",     rag_bd.get("gemini_generation",{}).get("mean_ms")),
        ]
    kg_bd = results.get("knowledge_graph", {}).get("breakdown", {})
    if kg_bd:
        rows += [
            ("KG Neo4j query",            kg_bd.get("neo4j_query",       {}).get("mean_ms")),
            ("KG Gemini generation",      kg_bd.get("gemini_generation", {}).get("mean_ms")),
        ]
    for label, val in rows:
        v = f"{val} ms" if val is not None else "N/A"
        print(f"  {label:<32}  {v:>12}")

    print("="*82)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="AstroNexus AI -- End-to-End Latency Benchmark"
    )
    parser.add_argument("--n",            type=int, default=12)
    parser.add_argument("--api-url",      type=str, default="http://localhost:8000")
    parser.add_argument("--quick",        action="store_true",
                        help="5 iterations, skip Ollama (slow)")
    parser.add_argument("--skip-ollama",  action="store_true")
    parser.add_argument("--skip-gemini",  action="store_true")
    parser.add_argument("--skip-rag",     action="store_true")
    parser.add_argument("--skip-kg",      action="store_true")
    parser.add_argument("--skip-agentic", action="store_true")
    args = parser.parse_args()

    n = 5 if args.quick else args.n
    print(f"\n{'='*60}")
    print(f"  AstroNexus AI -- Latency Benchmark")
    print(f"  n={n} iterations per config  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    print("\nProbing environment...")
    env = probe_environment()
    print(f"  CPU     : {env['machine'].get('cpu','?')[:60]}")
    print(f"  RAM     : {env['machine'].get('ram_gb','?')} GB")
    print(f"  Python  : {env['machine']['python']}")
    pt = env.get("pytorch", {})
    print(f"  PyTorch : {pt.get('version','?')}  CUDA={pt.get('cuda_avail',False)}  GPU={pt.get('gpu','?')}")
    print(f"  Ollama  : {env['ollama']['status']}  models={env['ollama'].get('models',[])}")
    print(f"  Qdrant  : {env['qdrant']['status']}  colls={env['qdrant'].get('collections',[])}")
    print(f"  Neo4j   : {env['neo4j']['status']}")
    print(f"  Gemini  : key={'SET' if env['gemini']['key_set'] else 'NOT SET'}  model={env['gemini']['model']}")

    results: dict = {"env": env, "timestamp": datetime.now().isoformat(), "n": n}

    # ── Micro benchmarks ───────────────────────────────────────────────────────
    micro: dict = {}
    micro["fastapi_overhead"] = bench_fastapi_overhead(args.api_url, min(n, 20))
    micro["query_router"]     = bench_query_router(_QUERIES, min(n * 4, 50))
    micro["qdrant_only"]      = bench_qdrant_only(_QUERIES, n)
    micro["neo4j_only"]       = bench_neo4j_only(_GRAPH_QUERIES, n)
    results["micro"] = micro

    # ── Configuration benchmarks ───────────────────────────────────────────────
    if not args.skip_ollama and not args.quick:
        results["ollama"]          = bench_ollama(_QUERIES, n)
    else:
        results["ollama"]          = {"config": "Local Ollama", "status": "SKIPPED -- quick mode or --skip-ollama"}

    if not args.skip_gemini:
        results["gemini"]          = bench_gemini(_QUERIES, n)
    else:
        results["gemini"]          = {"config": "Gemini", "status": "SKIPPED -- --skip-gemini"}

    if not args.skip_rag:
        results["rag"]             = bench_rag(_QUERIES, n)
    else:
        results["rag"]             = {"config": "RAG", "status": "SKIPPED -- --skip-rag"}

    if not args.skip_kg:
        results["knowledge_graph"] = bench_knowledge_graph(_GRAPH_QUERIES, n)
    else:
        results["knowledge_graph"] = {"config": "Knowledge Graph", "status": "SKIPPED -- --skip-kg"}

    if not args.skip_agentic:
        results["full_agentic"]    = bench_full_agentic(_QUERIES, n, args.api_url)
    else:
        results["full_agentic"]    = {"config": "Full Agentic", "status": "SKIPPED -- --skip-agentic"}

    # ── Summary ────────────────────────────────────────────────────────────────
    print_summary_table(results)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = _RESULTS_DIR / f"latency_{ts}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved -> {out}")


if __name__ == "__main__":
    main()
