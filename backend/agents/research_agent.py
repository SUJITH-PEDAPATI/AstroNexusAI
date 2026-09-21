"""
AstroNexus AI — Research Agent v5.0

Added: Chain-of-Thought reasoning before answering.
The model now thinks step by step before generating the final answer.

Pipeline per query:
    1. Retrieve chunks (Qdrant + BM25)
    2. Fuse with graph context + history
    3. THINK  — reason about what the evidence means
    4. ANSWER — generate answer from reasoning + evidence
    5. REFINE — Gemini improves structure and citations
    6. EVALUATE — score the answer
"""
from __future__ import annotations

import json, logging, os, re, time, urllib.request
from backend.agents.state    import AgentState
from backend.agents.identity import REFINE_SYSTEM, GENERAL_SYSTEM, ASTRONOMY_SYSTEM

logger = logging.getLogger(__name__)

OLLAMA_BASE   = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL  = os.environ.get("OLLAMA_MODEL",    "qwen3:4b")
GEMINI_KEY    = os.environ.get("GEMINI_API_KEY",  "")
ABSTAIN_SCORE = 0.20

# ── System prompt with reasoning ──────────────────────────────────────────────

_THINK_SYSTEM = """\
You are AstroNexus AI, a scientific research assistant built at NIT Kurukshetra.

You have been given:
  - Document excerpts from an uploaded research paper
  - Knowledge graph facts (authors, models, datasets, domain)
  - Conversation history

Before answering, you MUST reason step by step:

STEP 1 — UNDERSTAND THE QUESTION
  What exactly is the user asking?
  What type of answer is needed? (fact / explanation / comparison / summary)

STEP 2 — ANALYSE THE EVIDENCE
  What do the document excerpts say about this topic?
  What do the graph facts add?
  Are there any gaps in the evidence?

STEP 3 — REASON
  Connect the evidence to the question.
  Identify key relationships, causes, comparisons.
  Think about what the authors intended.

STEP 4 — FORMULATE ANSWER
  Write a comprehensive answer based ONLY on the evidence.
  Every fact must have a citation [chunk_number, p.page].
  If evidence is missing → say "AstroNexus AI could not find this in the paper."

NEVER skip the reasoning steps.
NEVER answer from training data — only from the provided context."""

_THINK_PROMPT = """\
{fused_context}

QUESTION: {question}

Now reason step by step before answering.

<thinking>
STEP 1 — UNDERSTAND THE QUESTION:
[What is being asked? What kind of answer is needed?]

STEP 2 — ANALYSE THE EVIDENCE:
[What do the chunks say? What does the graph add? Any gaps?]

STEP 3 — REASON:
[Connect evidence to the question. Key relationships?]

STEP 4 — FORMULATE:
[Plan the answer structure]
</thinking>

<answer>
[Your comprehensive, cited answer here. Minimum 200 words if evidence exists.
Cite every fact: [chunk_number, p.page]
If not found: "AstroNexus AI could not find this in the uploaded paper."]
</answer>"""

_REFINE_PROMPT = """\
CONTEXT (document + graph):
{fused_context}

REASONING AND DRAFT:
{draft}

QUESTION: {question}

Extract the <answer> section and improve it:
- Keep all citations [N, p.page]
- Add any missing citations
- Improve clarity and structure
- Use graph facts (authors, models) where relevant
- Minimum 200 words

FINAL ANSWER:"""

_LIVE_RE = re.compile(
    r'\b(weather|forecast|right now|today|air quality|real.?time)\b',
    re.IGNORECASE,
)


def _extract_answer(raw: str) -> str:
    """Extract the <answer> block from CoT output."""
    match = re.search(r'<answer>(.*?)</answer>', raw, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # If no tags, return everything after "STEP 4" or the last 60% of output
    if "STEP 4" in raw.upper() or "<thinking>" in raw.lower():
        parts = re.split(r'STEP\s+4|</thinking>|<answer>', raw, flags=re.IGNORECASE)
        return parts[-1].strip() if parts else raw.strip()
    return raw.strip()


def _ollama(system: str, prompt: str) -> str:
    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 2000},
    }).encode()
    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/generate",
            data=payload, headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read()).get("response", "").strip()
    except Exception as e:
        logger.error(f"[ResearchAgent] Ollama: {e}")
        return ""


def _gemini(system: str, prompt: str) -> str:
    if not GEMINI_KEY: return ""
    try:
        from google import genai
        from google.genai import types
        client   = genai.Client(api_key=GEMINI_KEY)
        response = client.models.generate_content(
            model=    "gemini-2.5-flash",
            contents= f"{system}\n\n{prompt}",
            config=   types.GenerateContentConfig(
                temperature=0.1, max_output_tokens=1500
            ),
        )
        return response.text.strip()
    except Exception as e:
        logger.warning(f"[ResearchAgent] Gemini: {e}")
        return ""


def research_agent_node(state: AgentState) -> AgentState:
    t0         = time.perf_counter()
    query      = state.get("query", "")
    metadata   = state.get("metadata") or {}
    paper_id   = metadata.get("paper_id")
    history    = state.get("conversation_history") or []
    turn_count = state.get("turn_count", 1)
    paper_loaded = (
        metadata.get("paper_loaded", False)
        or state.get("paper_loaded", False)
    )

    logger.info(f"[ResearchAgent] T{turn_count} paper={paper_loaded} '{query[:50]}'")

    # ── Mode (determine BEFORE retrieval so we skip it for general queries) ────
    if _LIVE_RE.search(query):
        mode = "live_data"
    elif paper_loaded:
        mode = "paper_qa"
    else:
        mode = "general"

    logger.info(f"[ResearchAgent] mode={mode}")

    # ── Retrieve (skip for general/conversational queries) ────────────────────
    # Retrieval loads the embedding model into GPU memory on first call (~60s).
    # General questions ("hello", "who are you", etc.) don't need RAG context
    # and should go straight to the LLM. Only retrieve when there is a paper
    # in context (paper_qa) or when a live-data lookup is needed.
    chunk_dicts = []
    top_score   = 0.0
    if mode in ("paper_qa", "live_data"):
        try:
            from backend.rag.retriever import retrieve
            chunks = retrieve(query, top_k=5, paper_id=paper_id)
            chunk_dicts = [
                {
                    "score":   c.score, "text": c.text,
                    "section": c.section, "page_num": c.page_num,
                    "title":   c.title,
                    "payload": {"text":c.text,"section":c.section,"page_num":c.page_num},
                }
                for c in chunks
            ]
            top_score = chunks[0].score if chunks else 0.0
            logger.info(f"[ResearchAgent] {len(chunk_dicts)} chunks top={top_score:.4f}")
        except Exception as e:
            logger.error(f"[ResearchAgent] Retrieval: {e}")
    else:
        logger.info("[ResearchAgent] Skipping retrieval (general mode — no paper loaded)")
    final_answer = ""
    eval_dict    = {}

    # ══════════════════════════════════════════════════════════════════════════
    # PAPER QA — with Chain-of-Thought reasoning
    # ══════════════════════════════════════════════════════════════════════════
    if mode == "paper_qa":
        if not chunk_dicts or top_score < ABSTAIN_SCORE:
            final_answer = (
                f"**AstroNexus AI** could not find sufficient evidence in the "
                f"uploaded paper to answer this question "
                f"(retrieval score: {top_score:.3f}). "
                f"Please try rephrasing your question."
            )
        else:
            # Fuse: Qdrant + Graph + History
            fused_prompt = ""
            try:
                from backend.agents.knowledge_fusion import fuse
                fused        = fuse(
                    query=                query,
                    qdrant_chunks=        chunk_dicts,
                    paper_node_id=        paper_id,
                    conversation_history= history,
                )
                fused_prompt = fused.prompt_text
                g            = fused.graph_entities
                logger.info(
                    f"[ResearchAgent] Graph authors={len(g.get('authors',[]))} "
                    f"kws={len(g.get('keywords',[]))} domain='{g.get('domain','')}'"
                )
            except Exception as e:
                logger.warning(f"[ResearchAgent] Fusion: {e}")
                fused_prompt = "\n\n".join(
                    f"[Chunk {i+1} | {c['section']} p.{c['page_num']}]\n{c['text']}"
                    for i, c in enumerate(chunk_dicts[:5])
                )

            # ── THINK + ANSWER (Ollama CoT) ───────────────────────────────────
            logger.info("[ResearchAgent] Thinking...")
            cot_output = _ollama(
                _THINK_SYSTEM,
                _THINK_PROMPT.format(fused_context=fused_prompt, question=query),
            )

            # Extract answer from reasoning
            draft = _extract_answer(cot_output)
            logger.info(f"[ResearchAgent] CoT output={len(cot_output)} answer={len(draft)}")

            # ── REFINE (Gemini) ───────────────────────────────────────────────
            final_answer = _gemini(
                REFINE_SYSTEM,
                _REFINE_PROMPT.format(
                    fused_context=fused_prompt,
                    draft=cot_output,   # pass full CoT so Gemini sees the reasoning
                    question=query,
                ),
            ) or draft or "AstroNexus AI was unable to generate an answer."

            # ── Evaluate ──────────────────────────────────────────────────────
            try:
                from backend.agents.live_evaluator import evaluate_paper_answer
                ev        = evaluate_paper_answer(query, final_answer, chunk_dicts)
                eval_dict = ev.to_dict()
                logger.info(
                    f"[ResearchAgent] grade={ev.grade} BLEU={ev.bleu:.3f} "
                    f"F1={ev.f1:.3f} gnd={ev.grounding:.3f} cite={ev.citation_coverage:.3f}"
                )
                try:
                    from backend.agents.eval_logger import save_eval
                    save_eval(
                        query=query, answer=final_answer, mode=mode,
                        eval_dict=eval_dict, paper_id=paper_id,
                        top_score=top_score, latency_s=time.perf_counter()-t0,
                    )
                except Exception: pass
            except Exception as e:
                logger.warning(f"[ResearchAgent] Eval: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # LIVE DATA
    # ══════════════════════════════════════════════════════════════════════════
    elif mode == "live_data":
        try:
            from backend.agents.api_router import select_apis
            from backend.agents.api_fusion import call_apis, format_for_prompt
            api_data = format_for_prompt(call_apis(select_apis(query)))
            if api_data:
                final_answer = _gemini(
                    ASTRONOMY_SYSTEM,
                    f"Real-time data:\n{api_data}\n\nQuestion: {query}\n\nAnswer:"
                )
        except Exception as e:
            logger.warning(f"[ResearchAgent] Live: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # GENERAL
    # ══════════════════════════════════════════════════════════════════════════
    else:
        # Even for general questions, use light CoT
        cot_prompt = f"""\
Think briefly about this question, then answer clearly.

Question: {query}

<thinking>
What is being asked? What do I know about this?
</thinking>

<answer>
[Clear, helpful answer here]
</answer>"""
        raw          = _ollama(GENERAL_SYSTEM, cot_prompt)
        draft        = _extract_answer(raw) or raw
        final_answer = _gemini(GENERAL_SYSTEM,
            f"Improve this answer:\n{draft}\n\nQuestion: {query}") or draft

    if not final_answer:
        final_answer = "AstroNexus AI was unable to generate an answer. Check Ollama is running."

    updated_history = list(history) + [{
        "turn": turn_count, "query": query,
        "answer": final_answer[:500], "mode": mode,
    }]

    elapsed = time.perf_counter() - t0
    logger.info(f"[ResearchAgent] done {elapsed:.1f}s")

    return {
        **state,
        "final_answer":         final_answer,
        "conversation_history": updated_history,
        "metadata": {
            **metadata,
            "mode":       mode,
            "top_score":  round(top_score, 4),
            "evaluation": eval_dict,
            "latency_s":  round(elapsed, 2),
        },
    }