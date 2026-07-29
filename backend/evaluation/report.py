"""
AstroNexus AI — Evaluation Report Generator

Reads logs/ JSONL files and produces a summary report.

Run: python -m backend.evaluation.report
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

LOG_DIR = Path("logs")

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists(): return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try: records.append(json.loads(line))
                except Exception: pass
    return records


def generate_report() -> dict:
    runtime  = _read_jsonl(LOG_DIR / "runtime_logs.jsonl")
    evals    = _read_jsonl(LOG_DIR / "evaluation_metrics.jsonl")

    if not runtime:
        print("No logs found in logs/")
        return {}

    def avg(records, *keys):
        vals = []
        for r in records:
            node = r
            for k in keys:
                node = node.get(k, {}) if isinstance(node, dict) else None
                if node is None: break
            if isinstance(node, (int, float)):
                vals.append(node)
        return round(statistics.mean(vals), 4) if vals else 0.0

    report = {
        "total_queries":     len(runtime),
        "sessions":          len(set(r.get("session_id","") for r in runtime)),
        "routes": {},
        "retrieval": {
            "avg_top_score":  avg(runtime, "retrieval", "max_score"),
            "avg_chunks":     avg(runtime, "retrieval", "chunks_sent_to_llm"),
            "avg_latency_ms": avg(runtime, "retrieval", "total_retrieval_latency_ms"),
        },
        "generation": {
            "avg_response_len": avg(runtime, "generation", "response_length"),
            "avg_latency_ms":   avg(runtime, "generation", "total_latency_ms"),
            "avg_tokens":       avg(runtime, "generation", "completion_tokens"),
        },
    }

    # Route distribution
    routes = [r.get("routing", {}).get("selected_route", "unknown") for r in runtime]
    for route in set(routes):
        report["routes"][route] = routes.count(route)

    # Evaluation summary (if available)
    if evals:
        report["evaluation"] = {
            "total_evaluated": len(evals),
            "avg_grounding":   avg(evals, "evaluation", "grounding_score"),
            "avg_f1":          avg(evals, "evaluation", "f1"),
            "avg_rouge1":      avg(evals, "evaluation", "rouge_1"),
            "avg_bleu":        avg(evals, "evaluation", "bleu"),
            "avg_faithfulness":avg(evals, "evaluation", "faithfulness_score"),
            "avg_citation_cov":avg(evals, "evaluation", "citation_coverage"),
            "grade_distribution": {},
        }
        grades = [r.get("evaluation", {}).get("grade", "F") for r in evals]
        for g in "ABCDF":
            report["evaluation"]["grade_distribution"][g] = grades.count(g)

    return report


def run():
    print("\n" + "="*55)
    print("  ASTRONEXUS AI — EVALUATION REPORT")
    print("="*55)

    report = generate_report()
    if not report:
        return

    print(f"\n  Total queries : {report['total_queries']}")
    print(f"  Sessions      : {report['sessions']}")
    print(f"\n  Route distribution:")
    for route, count in report["routes"].items():
        print(f"    {route:<15} {count}")

    print(f"\n  Retrieval:")
    for k, v in report["retrieval"].items():
        print(f"    {k:<25} {v}")

    print(f"\n  Generation:")
    for k, v in report["generation"].items():
        print(f"    {k:<25} {v}")

    if "evaluation" in report:
        print(f"\n  Evaluation ({report['evaluation']['total_evaluated']} queries):")
        for k, v in report["evaluation"].items():
            if k not in ("total_evaluated", "grade_distribution"):
                print(f"    {k:<25} {v}")
        print(f"\n  Grade distribution:")
        for g, count in report["evaluation"]["grade_distribution"].items():
            print(f"    {g}  {'█'*count} {count}")

    # Save report
    import json
    LOG_DIR.mkdir(exist_ok=True)
    with open(LOG_DIR / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved → logs/report.json\n")
    print("="*55 + "\n")


if __name__ == "__main__":
    run()