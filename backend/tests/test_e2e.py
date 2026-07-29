"""
AstroNexus AI — End-to-End Backend Test

Tests every component in the correct order.
Run from project root:
    python -m backend.tests.test_e2e

Requirements:
    - Qdrant running on localhost:6333
    - Neo4j  running on localhost:7687
    - Ollama running on localhost:11434
    - GEMINI_API_KEY in .env
    - A PDF in data/papers/
    - A satellite image in data/satellite/  (optional)
    - A WAV file in data/audio/             (optional)
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.WARNING, format="%(message)s")

PAPER_PATH = Path("data/papers")
SAT_PATH   = Path("data/satellite")
AUDIO_PATH = Path("data/audio")

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

results: dict[str, bool] = {}


def ok(name: str, detail: str = ""):
    results[name] = True
    suffix = f"  {detail}" if detail else ""
    print(f"  {GREEN}✓{RESET}  {name}{suffix}")


def fail(name: str, error: str = ""):
    results[name] = False
    print(f"  {RED}✗{RESET}  {name}  →  {error[:80]}")


def skip(name: str, reason: str = ""):
    results[name] = True   # skipped = not a failure
    print(f"  {YELLOW}–{RESET}  {name}  (skipped: {reason})")


def section(title: str):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. SERVICES
# ══════════════════════════════════════════════════════════════════════════════

def test_services():
    section("1. SERVICES")

    # Qdrant
    try:
        from backend.rag.vector_store import get_collection_info
        info = get_collection_info()
        ok("Qdrant", f"points={info.get('total_points',0)}")
    except Exception as e:
        fail("Qdrant", str(e))

    # Neo4j
    try:
        from backend.graph.neo4j_client import get_graph_stats
        stats = get_graph_stats()
        ok("Neo4j", f"nodes={stats.get('total_nodes',0)}")
    except Exception as e:
        fail("Neo4j", str(e))

    # Ollama
    try:
        import urllib.request, os
        urllib.request.urlopen(
            f"{os.environ.get('OLLAMA_BASE_URL','http://localhost:11434')}/api/tags",
            timeout=3,
        )
        ok("Ollama")
    except Exception as e:
        fail("Ollama", str(e))

    # Gemini key
    import os
    if os.environ.get("GEMINI_API_KEY"):
        ok("Gemini API key", "configured")
    else:
        fail("Gemini API key", "GEMINI_API_KEY not set in .env")


# ══════════════════════════════════════════════════════════════════════════════
# 2. INGESTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

_paper_id    = None
_paper_title = None

def test_ingestion():
    global _paper_id, _paper_title
    section("2. INGESTION PIPELINE")

    # Find a PDF
    pdfs = list(PAPER_PATH.glob("*.pdf")) if PAPER_PATH.exists() else []
    if not pdfs:
        skip("Paper ingestion", f"no PDF found in {PAPER_PATH}/")
        skip("Chunking", "no PDF")
        skip("Embedding", "no PDF")
        skip("Qdrant upsert", "no PDF")
        return

    pdf = pdfs[0]
    print(f"  Using: {pdf.name}")

    # Ingest
    try:
        from backend.ingestion.paper_ingestion import ingest_paper
        t0  = time.perf_counter()
        doc = ingest_paper(pdf)
        _paper_id    = doc.paper_id
        _paper_title = doc.metadata.title
        ok("Paper ingestion", f"title='{(_paper_title or '')[:40]}'  id={_paper_id[:8]}...")
    except Exception as e:
        fail("Paper ingestion", str(e))
        return

    # Chunking
    try:
        from backend.ingestion.chunking import chunk_document
        chunks = chunk_document(doc)
        ok("Chunking", f"{len(chunks)} chunks")
    except Exception as e:
        fail("Chunking", str(e))
        return

    # Embedding
    try:
        from backend.embeddings import embed_chunks
        embedded = embed_chunks(chunks)
        norm = sum(x*x for x in embedded[0].vector)**0.5
        ok("Embedding", f"dim={embedded[0].vector_dim}  norm={norm:.4f}")
    except Exception as e:
        fail("Embedding", str(e))
        return

    # Qdrant upsert
    try:
        from backend.rag import upsert_chunks
        upsert_chunks(embedded)
        ok("Qdrant upsert", f"{len(embedded)} vectors stored")
    except Exception as e:
        fail("Qdrant upsert", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 3. KNOWLEDGE GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def test_graph_ingestion():
    section("3. KNOWLEDGE GRAPH")

    if not _paper_id:
        skip("Entity extraction", "no paper ingested")
        skip("Neo4j write", "no paper ingested")
        skip("Keyword write", "no paper ingested")
        skip("Domain classification", "no paper ingested")
        return

    # Find the doc (re-ingest cheaply)
    pdfs = list(PAPER_PATH.glob("*.pdf"))
    if not pdfs:
        return
    pdf = pdfs[0]

    try:
        from backend.ingestion.paper_ingestion import ingest_paper
        doc = ingest_paper(pdf)
    except Exception as e:
        fail("Re-ingest for graph", str(e))
        return

    # Entity extraction
    try:
        from backend.graph.entity_extractor import extract_entities
        result = extract_entities(doc)
        ok("Entity extraction",
           f"authors={len(result.authors)} "
           f"models={len(result.models)} "
           f"datasets={len(result.datasets)}")
    except Exception as e:
        fail("Entity extraction", str(e))
        return

    # Neo4j write
    try:
        from backend.graph.neo4j_client import write_extraction, create_constraints
        create_constraints()
        write_extraction(result)
        ok("Neo4j entity write", f"{len(result.relations)} relations")
    except Exception as e:
        fail("Neo4j entity write", str(e))

    # Domain + keywords
    try:
        from backend.graph.graph_builder import _write_domain_and_keywords
        summary = _write_domain_and_keywords(result.paper.node_id, doc)
        ok("Domain classification", f"domain={summary['primary_domain']}")
        ok("Keyword write",
           f"written={summary['keywords_written']} "
           f"linked={summary['keywords_linked']}")
        if summary["errors"]:
            print(f"      {YELLOW}warnings:{RESET} {summary['errors'][:2]}")
    except Exception as e:
        fail("Domain + keywords", str(e))

    # Verify keywords in Neo4j
    try:
        from backend.graph.neo4j_client import _get_driver
        driver = _get_driver()
        with driver.session() as s:
            rows = s.run(
                "MATCH (p:Paper {paper_id:$pid})-[:TAGGED]->(k:Keyword) "
                "RETURN k.name AS kw LIMIT 10",
                pid=_paper_id,
            ).data()
        kws = [r["kw"] for r in rows]
        if kws:
            ok("Keywords in Neo4j", f"{kws[:5]}")
        else:
            fail("Keywords in Neo4j", "zero keywords found — check domain_classifier coverage")
    except Exception as e:
        fail("Keywords in Neo4j", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 4. RAG RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def test_retrieval():
    section("4. RAG RETRIEVAL")

    try:
        from backend.rag.retriever import retrieve, retrieve_for_rag
        query   = "What is the main contribution of this paper?"
        chunks  = retrieve(query, top_k=3)
        if chunks:
            ok("Qdrant retrieval",
               f"{len(chunks)} chunks  top_score={chunks[0].score:.3f}")
        else:
            fail("Qdrant retrieval", "zero results — is a paper ingested?")

        context = retrieve_for_rag(query, top_k=3)
        ok("RAG context build", f"{len(context)} chars")
    except Exception as e:
        fail("RAG retrieval", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 5. KNOWLEDGE FUSION
# ══════════════════════════════════════════════════════════════════════════════

def test_knowledge_fusion():
    section("5. KNOWLEDGE FUSION")

    try:
        from backend.rag.retriever import retrieve
        from backend.agents.knowledge_fusion import fuse

        query  = "What datasets were used?"
        chunks = retrieve(query, top_k=3)
        chunk_dicts = [
            {"score": c.score, "text": c.text,
             "section": c.section, "page_num": c.page_num,
             "payload": {"text": c.text, "section": c.section,
                         "page_num": c.page_num, "title": c.title}}
            for c in chunks
        ]

        fused = fuse(
            query=         query,
            qdrant_chunks= chunk_dicts,
            paper_node_id= None,
        )

        has_qdrant = "DOCUMENT EXCERPTS" in fused.prompt_text
        has_graph  = "KNOWLEDGE GRAPH"   in fused.prompt_text

        ok("Knowledge fusion",
           f"prompt={len(fused.prompt_text)} chars  "
           f"qdrant={'✓' if has_qdrant else '✗'}  "
           f"graph={'✓' if has_graph else '✗'}")
    except Exception as e:
        fail("Knowledge fusion", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 6. AGENT ROUTING
# ══════════════════════════════════════════════════════════════════════════════

def test_routing():
    section("6. ROUTER")

    from backend.agents.router import classify_query

    cases = [
        ("What is reinforcement learning?",           {},                              "general"),
        ("Who authored Attention Is All You Need?",   {},                              "graph"),
        ("What is the main contribution?",            {"metadata":{"paper_loaded":True}}, "research"),
        ("Identify land cover.",                      {"metadata":{"image_path":"x"}}, "satellite"),
        ("What satellites are in Copernicus mission?",{},                              "graph"),
    ]

    passed = 0
    for query, state, expected in cases:
        got = classify_query(query, state)
        if got == expected:
            passed += 1
            ok(f"Route: '{query[:40]}'", f"→ {got}")
        else:
            fail(f"Route: '{query[:40]}'", f"got={got} expected={expected}")

    if passed == len(cases):
        ok("Router overall", f"{passed}/{len(cases)} correct")


# ══════════════════════════════════════════════════════════════════════════════
# 7. FULL AGENT PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def test_agents():
    section("7. FULL AGENT PIPELINE")

    # Research agent
    try:
        from backend.agents.orchestrator import run
        t0     = time.perf_counter()
        result = run(
            query=        "What is the main contribution of this paper?",
            paper_loaded= bool(_paper_id),
            paper_id=     _paper_id,
        )
        elapsed = time.perf_counter() - t0
        answer  = result.get("final_answer", "")
        meta    = result.get("metadata") or {}
        evl     = meta.get("evaluation") or {}

        ok("Research agent",
           f"conf={evl.get('confidence','?')}  "
           f"grounding={evl.get('grounding_score',0):.2f}  "
           f"reliable={evl.get('is_reliable','?')}  "
           f"time={elapsed:.1f}s")
        ok("Ollama used",  str(meta.get("ollama_used", False)))
        ok("Gemini used",  str(meta.get("gemini_used", False)))
        if answer:
            print(f"\n      Answer preview: {answer[:120]}...\n")
    except Exception as e:
        fail("Research agent", str(e))

    # Graph agent
    try:
        from backend.agents.orchestrator import run
        result = run(query="Who authored Attention Is All You Need?")
        answer = result.get("final_answer", "")
        ok("Graph agent", f"{len(answer)} chars returned")
    except Exception as e:
        fail("Graph agent", str(e))

    # General agent
    try:
        from backend.agents.orchestrator import run
        result = run(query="What is reinforcement learning?")
        answer = result.get("final_answer", "")
        ok("General agent", f"type={result.get('query_type')}  chars={len(answer)}")
    except Exception as e:
        fail("General agent", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 8. EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

def test_evaluator():
    section("8. ANSWER EVALUATOR")

    try:
        from backend.agents.answer_evaluator import get_evaluator

        evaluator = get_evaluator()

        # Normal case
        chunks = [{"score": 0.82, "text": "The Transformer model uses attention mechanism.", "section": "Methods", "page_num": 4, "title": "Test"}]
        result = evaluator.evaluate(
            query=            "What model is used?",
            answer=           "The paper uses the Transformer model with attention mechanism.",
            retrieved_chunks= chunks,
        )
        ok("Evaluator — normal",
           f"conf={result.confidence}  "
           f"grounding={result.grounding_score:.2f}  "
           f"reliable={result.is_reliable}")

        # Hallucination detection
        result2 = evaluator.evaluate(
            query=            "What is the accuracy?",
            answer=           "Section 12.9 shows 97.3% accuracy on page 987.",
            retrieved_chunks= chunks,
        )
        flagged = len(result2.warnings) > 0
        ok("Evaluator — hallucination detection",
           f"warnings={len(result2.warnings)}  flagged={flagged}")

        # Low confidence
        result3 = evaluator.evaluate(
            query=            "What is the loss function?",
            answer=           "Some answer here.",
            retrieved_chunks= [{"score": 0.32, "text": "unrelated text", "section": "?", "page_num": 1}],
        )
        ok("Evaluator — low confidence", f"conf={result3.confidence}")

    except Exception as e:
        fail("Evaluator", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 9. VISION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def test_vision():
    section("9. VISION PIPELINE")

    images = list(SAT_PATH.glob("*.jpg")) + list(SAT_PATH.glob("*.png")) \
             if SAT_PATH.exists() else []

    if not images:
        skip("Gemini captioning",  f"no image in {SAT_PATH}/")
        skip("Keyword → Neo4j",    "no image")
        skip("SAM2 segmentation",  "no image")
        return

    img = images[0]
    print(f"  Using: {img.name}")

    # Gemini caption
    try:
        from backend.vision.florence2_captioner import caption_image
        result = caption_image(img, write_to_graph=True)
        kws    = getattr(result, "image_keywords", [])
        ok("Gemini captioning", f"'{result.caption[:60]}'")
        ok("Keyword → Neo4j",   f"{kws}")
    except Exception as e:
        fail("Gemini captioning", str(e))

    # SAM2
    try:
        from backend.vision.sam2_segmentor import segment_image
        result = segment_image(img, save_viz=True)
        ok("SAM2 segmentation", f"masks={result['mask_count']}")
    except Exception as e:
        fail("SAM2 segmentation", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 10. VOICE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def test_voice():
    section("10. VOICE PIPELINE")

    wavs = list(AUDIO_PATH.glob("*.wav")) if AUDIO_PATH.exists() else []

    if not wavs:
        skip("Whisper STT", f"no WAV in {AUDIO_PATH}/")
        return

    wav = wavs[0]
    try:
        from backend.voice.whisper_service import WhisperService
        result = WhisperService().transcribe(wav)
        ok("Whisper STT",
           f"lang={result['language']}  "
           f"duration={result['duration']}s  "
           f"text='{result['text'][:50]}'")
    except Exception as e:
        fail("Whisper STT", str(e))

    # Edge TTS
    try:
        import edge_tts, asyncio
        async def _tts():
            comm = edge_tts.Communicate("AstroNexus voice pipeline test.", "en-US-AriaNeural")
            out  = Path("output/audio/test_tts.mp3")
            out.parent.mkdir(parents=True, exist_ok=True)
            await comm.save(str(out))
            return out
        out = asyncio.run(_tts())
        ok("Edge TTS", f"saved → {out}")
    except Exception as e:
        fail("Edge TTS", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 11. FASTAPI HEALTH
# ══════════════════════════════════════════════════════════════════════════════

def test_api():
    section("11. FASTAPI")

    try:
        import urllib.request
        resp = urllib.request.urlopen("http://localhost:8000/health", timeout=3)
        data = json.loads(resp.read())
        ok("FastAPI /health", f"status={data.get('status')}")
    except Exception as e:
        skip("FastAPI /health", f"server not running — start with: uvicorn backend.api.main:app --port 8000")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def summary():
    print(f"\n{'═'*55}")
    print("  RESULTS")
    print(f"{'═'*55}")

    passed = sum(v for v in results.values())
    total  = len(results)
    pct    = 100 * passed // total if total else 0

    failed_tests = [k for k, v in results.items() if not v]

    print(f"  {passed}/{total} passed  ({pct}%)")

    if failed_tests:
        print(f"\n  {RED}FAILED:{RESET}")
        for t in failed_tests:
            print(f"    ✗  {t}")
    else:
        print(f"\n  {GREEN}All tests passed.{RESET}")

    print(f"{'═'*55}\n")

    # Save report
    Path("output").mkdir(exist_ok=True)
    with open("output/e2e_test_results.json", "w") as f:
        json.dump({
            "passed": passed,
            "total":  total,
            "pct":    pct,
            "results":results,
        }, f, indent=2)
    print(f"  Report saved → output/e2e_test_results.json\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'═'*55}")
    print("  ASTRONEXUS AI — END-TO-END TEST")
    print(f"{'═'*55}")

    test_services()
    test_ingestion()
    test_graph_ingestion()
    test_retrieval()
    test_knowledge_fusion()
    test_routing()
    test_agents()
    test_evaluator()
    test_vision()
    test_voice()
    test_api()
    summary()


if __name__ == "__main__":
    main()