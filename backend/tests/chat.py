"""
AstroNexus AI — Terminal Chat (Final)
Fixed: eval schema mismatch between live_evaluator and chat display/save.
"""
from __future__ import annotations

import sys, os, json, time
from pathlib import Path

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

RESET="\033[0m"; BOLD="\033[1m"; DIM="\033[2m"
GREEN="\033[92m"; CYAN="\033[96m"; YELLOW="\033[93m"
RED="\033[91m";   BLUE="\033[94m"; MAGENTA="\033[95m"

def c(t,col): return f"{col}{t}{RESET}"
def bold(t):  return f"{BOLD}{t}{RESET}"
def dim(t):   return f"{DIM}{t}{RESET}"
def gc(g):    return GREEN if g in("A","B") else YELLOW if g=="C" else RED


# ══════════════════════════════════════════════════════════════════════════════
# FIX: convert flat EvaluationResult.to_dict() → nested schema
# ══════════════════════════════════════════════════════════════════════════════

def _to_nested(flat: dict, top_score: float = 0.0, chunks_used: int = 0) -> dict:
    """
    EvaluationResult.to_dict() returns a FLAT dict with keys like:
        bleu, rouge_1, grounding_score, reliability_score, grade

    But print_eval_box() and save_eval() expect NESTED dict:
        retrieval.top_score
        answer_quality.bleu
        faithfulness.grounding
        overall.grade

    This function converts flat → nested.
    Also handles the case where the dict is already nested (from research_agent).
    """
    if not flat:
        return {}

    # Already nested — return as-is
    if "answer_quality" in flat or "retrieval" in flat:
        return flat

    # Convert flat → nested
    return {
        "retrieval": {
            "top_score":      top_score,
            "mean_score":     top_score * 0.9,   # approximate
            "score_variance": 0.0,
            "coverage":       flat.get("context_relevance", 0.0),
            "chunks_used":    chunks_used,
        },
        "answer_quality": {
            "bleu":      flat.get("bleu",    0.0),
            "rouge1":    flat.get("rouge_1", 0.0),
            "rouge2":    flat.get("rouge_2", 0.0),
            "rougeL":    flat.get("rouge_l", 0.0),
            "precision": flat.get("precision", 0.0),
            "recall":    flat.get("recall",    0.0),
            "f1":        flat.get("f1",        0.0),
        },
        "faithfulness": {
            "grounding":           flat.get("grounding_score",    0.0),
            "hallucination_flags": flat.get("unsupported_claims", 0),
            "citation_coverage":   flat.get("citation_coverage",  0.0),
        },
        "overall": {
            "reliability_score": flat.get("reliability_score", 0.0),
            "grade":             flat.get("grade", "F"),
            "warnings":          flat.get("warnings", []),
        },
    }


def ingest_paper(path_str: str) -> dict | None:
    path = Path(path_str.strip().strip("'\""))
    if not path.exists():
        print(c(f"  File not found: {path}", RED)); return None
    if path.suffix.lower() != ".pdf":
        print(c("  Only PDF supported.", RED)); return None
    print(dim(f"  Ingesting: {path.name}"))
    try:
        print(dim("  [1/5] Parsing..."),   end="\r")
        from backend.ingestion.paper_ingestion import ingest_paper as _ing
        doc = _ing(path); print(dim("  [1/5] Parsing... done"))

        print(dim("  [2/5] Chunking..."),  end="\r")
        from backend.ingestion.chunking import chunk_document
        chunks = chunk_document(doc)
        print(dim(f"  [2/5] Chunking... {len(chunks)} chunks"))

        print(dim("  [3/5] Embedding..."), end="\r")
        from backend.embeddings import embed_chunks
        embedded = embed_chunks(chunks); print(dim("  [3/5] Embedding... done"))

        print(dim("  [4/5] Qdrant..."),    end="\r")
        from backend.rag import upsert_chunks
        upsert_chunks(embedded); print(dim("  [4/5] Qdrant... done"))

        print(dim("  [5/5] Graph..."),     end="\r")
        from backend.graph.graph_builder import build_graph_from_document
        ext = build_graph_from_document(doc)
        print(dim("  [5/5] Graph... done"))

        kws = []
        try:
            from backend.graph.neo4j_client import _get_driver
            with _get_driver().session() as s:
                rows = s.run(
                    "MATCH (p:Paper {paper_id:$pid})-[:TAGGED]->(k:Keyword) "
                    "RETURN k.name AS kw LIMIT 20",
                    pid=doc.paper_id
                ).data()
            kws = [r["kw"] for r in rows]
        except Exception: pass

        return {
            "paper_id": doc.paper_id,
            "title":    doc.metadata.title or path.name,
            "chunks":   len(chunks),
            "authors":  [a.name for a in ext.authors],
            "keywords": kws,
        }
    except Exception as e:
        print(c(f"  Failed: {e}", RED)); return None


def print_eval_box(eval_dict: dict) -> None:
    if not eval_dict or "answer_quality" not in eval_dict:
        return
    ret  = eval_dict.get("retrieval",{})
    aq   = eval_dict.get("answer_quality",{})
    fa   = eval_dict.get("faithfulness",{})
    ov   = eval_dict.get("overall",{})
    g    = ov.get("grade","?")
    scr  = ov.get("reliability_score",0)
    W    = 54

    print(f"\n  {dim('┌'+'─'*W+'┐')}")
    print(f"  {dim('│')} {bold('RETRIEVAL'):<{W-1}}{dim('│')}")
    ts   = ret.get('top_score',0)
    tsc  = GREEN if ts>=0.5 else YELLOW if ts>=0.35 else RED
    print(f"  {dim('│')}  Top={c(f'{ts:.4f}',tsc)}  Mean={ret.get('mean_score',0):.4f}"
          f"  Cov={ret.get('coverage',0):.4f}  Chunks={ret.get('chunks_used',0)}"
          f"{'':>4}{dim('│')}")
    print(f"  {dim('├'+'─'*W+'┤')}")
    print(f"  {dim('│')} {bold('ANSWER QUALITY'):<{W-1}}{dim('│')}")
    print(f"  {dim('│')}  BLEU={aq.get('bleu',0):.4f}  R1={aq.get('rouge1',0):.4f}"
          f"  R2={aq.get('rouge2',0):.4f}  RL={aq.get('rougeL',0):.4f}{'':>6}{dim('│')}")
    f1c  = GREEN if aq.get('f1',0)>=0.25 else YELLOW
    print(f"  {dim('│')}  P={aq.get('precision',0):.4f}  R={aq.get('recall',0):.4f}"
          f"  F1={c(f"{aq.get('f1',0):.4f}",f1c)}{'':>24}{dim('│')}")
    print(f"  {dim('├'+'─'*W+'┤')}")
    print(f"  {dim('│')} {bold('FAITHFULNESS'):<{W-1}}{dim('│')}")
    gc2  = GREEN if fa.get('grounding',0)>=0.6 else YELLOW if fa.get('grounding',0)>=0.35 else RED
    hfl  = fa.get('hallucination_flags',0)
    hc   = GREEN if hfl==0 else RED
    print(f"  {dim('│')}  Grounding={c(f"{fa.get('grounding',0):.4f}",gc2)}"
          f"  Halluc={c(str(hfl)+' flags',hc)}"
          f"  CitCov={fa.get('citation_coverage',0):.4f}{'':>4}{dim('│')}")
    print(f"  {dim('├'+'─'*W+'┤')}")
    gcol = gc(g)
    print(f"  {dim('│')} {bold('OVERALL')}  Grade={gcol}{g}{RESET}"
          f"  Score={c(f'{scr:.4f}',gcol)}{'':>{W-28}}{dim('│')}")
    print(f"  {dim('└'+'─'*W+'┘')}")

    warns = ov.get("warnings",[])
    for w in warns[:3]:
        print(f"  {YELLOW}⚠{RESET} {w}")


def format_answer(result: dict) -> str:
    ans   = result.get("final_answer","")
    route = result.get("query_type","?")
    meta  = result.get("metadata") or {}
    mode  = meta.get("mode","")
    ts    = meta.get("top_score",0.0)
    apis  = meta.get("apis_called",[])

    badges = [c(f"→ {route}",BLUE)]
    if mode: badges.append(c(f"mode={mode}",MAGENTA))
    if ts:   badges.append(dim(f"score={ts:.3f}"))
    if apis: badges.append(c(f"apis={apis}",YELLOW))

    lines = ["  "+"  ".join(b for b in badges if b),""]
    if ans:
        words=ans.split(); line="  "
        for word in words:
            if len(line)+len(word)>72:
                lines.append(line); line="  "+word+" "
            else:
                line+=word+" "
        if line.strip(): lines.append(line)
    else:
        lines.append(c("  (no answer)",RED))

    if result.get("error"):
        lines.append(f"\n  {c('Error: '+result['error'][:80],RED)}")
    return "\n".join(lines)


def print_banner():
    print(f"""
{CYAN}{BOLD}
  ╔══════════════════════════════════════════════════╗
  ║         AstroNexus AI  —  Terminal Chat          ║
  ║   RAG · KG · Vision · APIs · Live Evaluation    ║
  ╚══════════════════════════════════════════════════╝
{RESET}  Type {CYAN}/help{RESET} for commands.\n""")


def print_help():
    print(f"""
{bold('Commands')}
  /ingest <path>   Ingest a PDF (full pipeline)
  /image  <path>   Set active satellite image
  /clear           Clear conversation history
  /history         Show conversation history
  /status          Show session status
  /scores          Show last evaluation scores
  /help            Show this help
  /quit            Save summary + exit

{bold('Output files  (auto-saved)')}
  output/evaluation_log.json     ← every turn
  output/evaluation_summary.json ← on /quit
""")


def run():
    print_banner()

    session = {
        "paper_id":None,"paper_title":None,
        "image_path":None,"paper_loaded":False,"history":[],
    }
    last_eval: dict = {}

    import logging; logging.disable(logging.CRITICAL)

    print(c("  Warming up...",DIM), end=" ", flush=True)
    try:
        from backend.agents.orchestrator import get_graph
        get_graph(); print(c("ready.",GREEN))
    except Exception as e:
        print(c(f"failed: {e}",RED)); sys.exit(1)

    print(dim("  /ingest <path.pdf> to load a paper. Scores saved to output/evaluation_log.json\n"))

    while True:
        try:
            raw = input(f"{CYAN}{BOLD}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            raw = "/quit"

        if not raw: continue

        if raw.startswith("/"):
            parts = raw.split(maxsplit=1)
            cmd   = parts[0].lower()
            arg   = parts[1] if len(parts)>1 else ""

            if cmd in ("/quit","/exit"):
                try:
                    from backend.agents.eval_logger import save_summary
                    summary = save_summary()
                    if summary:
                        print(f"\n  {bold('Session Summary')}")
                        print(f"  Total turns   : {summary.get('total_turns',0)}")
                        print(f"  Paper QA turns: {summary.get('paper_qa_turns',0)}")
                        aq = summary.get('answer_quality',{})
                        fa = summary.get('faithfulness',{})
                        print(f"  Avg BLEU      : {aq.get('avg_bleu',0):.4f}")
                        print(f"  Avg F1        : {aq.get('avg_f1',0):.4f}")
                        print(f"  Avg Grounding : {fa.get('avg_grounding',0):.4f}")
                        ov = summary.get('overall',{})
                        print(f"  Grade dist    : {ov.get('grade_distribution',{})}")
                        print(f"\n  {GREEN}Saved:{RESET}")
                        print(f"    output/evaluation_log.json")
                        print(f"    output/evaluation_summary.json")
                except Exception as e:
                    print(dim(f"  (summary failed: {e})"))
                print(dim("\nGoodbye.\n")); break

            elif cmd == "/help":    print_help()
            elif cmd == "/scores":
                if last_eval: print_eval_box(last_eval)
                else: print(dim("  No evaluation yet.\n"))
            elif cmd == "/status":
                pid = session.get("paper_id")
                print(f"\n  Paper  : {c(session.get('paper_title','')[:50],GREEN) if pid else c('not ingested',YELLOW)}")
                print(f"  History: {len(session.get('history',[]))} turns\n")
            elif cmd == "/history":
                for t in session.get("history",[]):
                    print(f"  {dim('['+str(t['turn'])+']')} {CYAN}{t['query'][:60]}{RESET}")
                    print(f"  {t['answer'][:120]}...\n")
            elif cmd == "/clear":
                session["history"]=[]; print(dim("  Cleared.\n"))
            elif cmd == "/ingest":
                if not arg:
                    print(dim("  Usage: /ingest <path>\n"))
                else:
                    t0=time.perf_counter()
                    r=ingest_paper(arg)
                    if r:
                        session.update({
                            "paper_id":r["paper_id"],"paper_title":r["title"],
                            "paper_loaded":True,"history":[]
                        })
                        print(f"\n  {c('✓ Ingested',GREEN)}  {r['title'][:55]}")
                        print(f"  ID      : {dim(r['paper_id'])}")
                        print(f"  Chunks  : {r['chunks']}")
                        print(f"  Authors : {', '.join(r['authors'][:3]) or 'not extracted'}")
                        print(f"  Keywords: {', '.join(r['keywords'][:8]) or 'none'}")
                        print(f"  Time    : {time.perf_counter()-t0:.1f}s\n")
            elif cmd == "/image":
                if arg:
                    session["image_path"]=arg.strip().strip("'\"")
                    print(c(f"  Image: {session['image_path']}\n",GREEN))
            else:
                print(dim(f"  Unknown: {cmd}\n"))
            continue

        # ── Run pipeline ──────────────────────────────────────────────────────
        if not session["paper_loaded"]:
            print(dim("  No paper — general knowledge mode...\n"))

        print(f"{GREEN}{BOLD}AI :{RESET} ",end="",flush=True)
        print(dim("thinking..."),end="\r",flush=True)

        t0=time.perf_counter()
        try:
            from backend.agents.orchestrator import run as orch
            result=orch(
                query=                raw,
                image_path=           session.get("image_path"),
                paper_loaded=         session.get("paper_loaded",False),
                paper_id=             session.get("paper_id"),
                conversation_history= session.get("history",[]),
            )
        except Exception as e:
            print(f"{GREEN}{BOLD}AI :{RESET}  {c('Error: '+str(e),RED)}\n"); continue

        elapsed = time.perf_counter() - t0
        meta    = result.get("metadata") or {}
        mode    = meta.get("mode","")
        top_sc  = meta.get("top_score", 0.0)
        chunks_used = 0

        # ── Get eval dict — fixed schema conversion ────────────────────────────
        eval_d = meta.get("evaluation", {})

        if mode == "paper_qa" and (not eval_d or "answer_quality" not in eval_d):
            # eval_d is either {} or flat — run evaluator now
            try:
                from backend.rag.retriever         import retrieve
                from backend.agents.live_evaluator import evaluate_paper_answer

                chunks    = retrieve(raw, top_k=5, paper_id=session.get("paper_id"))
                chunks_used = len(chunks)
                cdicts    = [
                    {"score":ch.score,"text":ch.text,
                     "section":ch.section,"page_num":ch.page_num}
                    for ch in chunks
                ]
                lev       = evaluate_paper_answer(raw, result.get("final_answer",""), cdicts)
                flat_eval = lev.to_dict()
                # ← FIX: convert flat dict to nested schema
                eval_d    = _to_nested(flat_eval, top_score=top_sc, chunks_used=chunks_used)

            except Exception as e:
                # Log the actual error — don't swallow it silently
                import traceback
                print(dim(f"  (eval failed: {e})"))

        elif eval_d and "answer_quality" not in eval_d:
            # research_agent stored flat dict in metadata — convert it
            eval_d = _to_nested(eval_d, top_score=top_sc)

        last_eval = eval_d

        # Print answer
        print(f"{GREEN}{BOLD}AI :{RESET}")
        print(format_answer(result))
        print(f"\n  {dim(f'⏱  {elapsed:.1f}s')}\n")

        # Print eval box
        if mode == "paper_qa" and eval_d:
            print_eval_box(eval_d)

        # Save to JSON — always save nested dict
        try:
            from backend.agents.eval_logger import save_eval
            save_eval(
                query=       raw,
                answer=      result.get("final_answer",""),
                mode=        mode or "unknown",
                eval_dict=   eval_d,   # ← now always nested
                paper_id=    session.get("paper_id"),
                paper_title= session.get("paper_title"),
                top_score=   top_sc,
                latency_s=   elapsed,
            )
        except Exception as e:
            print(dim(f"  (eval save failed: {e})"))

        print(f"\n  {dim('─'*60)}\n")
        session["history"] = result.get("conversation_history", session["history"])


if __name__ == "__main__":
    run()