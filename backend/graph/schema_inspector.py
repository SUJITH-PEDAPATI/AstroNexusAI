"""
AstroNexus AI — Neo4j Schema Inspector

Inspects the live Neo4j graph: node counts, relationship types,
sample papers, and a per-paper context dump.

Run:
    python -m backend.graph.schema_inspector
    python -m backend.graph.schema_inspector --paper <paper_id>
    python -m backend.graph.schema_inspector --entity <name>
"""
from __future__ import annotations

import sys
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Colours ───────────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"

def c(text, colour): return f"{colour}{text}{RESET}"
def bold(text):      return f"{BOLD}{text}{RESET}"
def dim(text):       return f"{DIM}{text}{RESET}"

SEP = dim("  " + "─" * 58)


def print_banner():
    print(f"""
{CYAN}{BOLD}  Neo4j Schema Inspector — AstroNexus AI{RESET}
  URI: {os.environ.get('NEO4J_URI', 'bolt://localhost:7687')}
""")


def inspect_schema():
    from backend.graph.neo4j_client import get_graph_stats

    print(bold("  [1] Node Counts"))
    stats = get_graph_stats()

    nodes = stats.get("nodes", {})
    rels  = stats.get("relations", {})

    if not nodes:
        print(c("      No nodes found — is Neo4j populated? Run /ingest first.", YELLOW))
    else:
        for label, count in sorted(nodes.items(), key=lambda x: -x[1]):
            bar = GREEN if count > 0 else DIM
            print(f"      {bar}{label:<20}{RESET}  {count:>6}")

    print(f"\n      {dim('Total nodes     :')} {stats.get('total_nodes', 0)}")
    print(f"      {dim('Total relations :')} {stats.get('total_relations', 0)}")

    print(f"\n{SEP}\n")
    print(bold("  [2] Relationship Types"))
    if not rels:
        print(c("      No relationships found.", YELLOW))
    else:
        for rtype, count in sorted(rels.items(), key=lambda x: -x[1]):
            print(f"      {CYAN}{rtype:<30}{RESET}  {count:>6}")


def inspect_papers(limit: int = 10):
    from backend.graph.neo4j_client import _run_read

    print(f"\n{SEP}\n")
    print(bold(f"  [3] Sample Papers (up to {limit})"))
    rows = _run_read(
        "MATCH (p:Paper) "
        "OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author) "
        "RETURN p.title AS title, p.paper_id AS pid, p.node_id AS nid, "
        "       p.year AS year, collect(DISTINCT a.name)[..3] AS authors "
        "LIMIT $lim",
        lim=limit,
    )
    if not rows:
        print(c("      No Paper nodes found.", YELLOW))
        return

    for i, r in enumerate(rows, 1):
        title   = (r.get("title") or "untitled")[:55]
        pid     = r.get("pid") or r.get("nid") or "—"
        year    = r.get("year") or "?"
        authors = ", ".join(r.get("authors") or []) or "—"
        print(f"      {dim(str(i)+'.')} {bold(title)}")
        print(f"           id={dim(pid)}  year={year}")
        print(f"           authors: {authors}\n")


def inspect_paper_context(paper_id: str):
    from backend.graph.neo4j_client import get_paper_graph_context

    print(f"\n{SEP}\n")
    print(bold(f"  [4] Paper Context: {dim(paper_id)}"))
    ctx = get_paper_graph_context(paper_id)
    if not ctx:
        print(c(f"      No data found for paper_id={paper_id!r}", RED))
        print("      Make sure the id matches p.node_id in Neo4j.")
        return

    fields = [
        ("Title",        ctx.get("title", "")),
        ("Year",         ctx.get("year", "")),
        ("DOI",          ctx.get("doi", "")),
        ("Venue",        ctx.get("venue", "")),
        ("Domain",       ctx.get("domain", "")),
        ("Authors",      ", ".join(ctx.get("authors", []))),
        ("Institutions", ", ".join(ctx.get("institutions", []))),
        ("Models",       ", ".join(ctx.get("models", []))),
        ("Algorithms",   ", ".join(ctx.get("algorithms", []))),
        ("Datasets",     ", ".join(ctx.get("datasets", []))),
        ("Tasks",        ", ".join(ctx.get("tasks", []))),
        ("Metrics",      ", ".join(ctx.get("metrics", []))),
        ("Keywords",     ", ".join(ctx.get("keywords", []))),
        ("Satellites",   ", ".join(ctx.get("satellites", []))),
    ]
    for label, value in fields:
        val_str = str(value)[:80] if value else dim("—")
        print(f"      {CYAN}{label:<14}{RESET}  {val_str}")


def inspect_entity(name: str):
    from backend.graph.neo4j_client import get_entity_neighbours

    print(f"\n{SEP}\n")
    print(bold(f"  [5] Entity Neighbours: {dim(name)}"))
    rows = get_entity_neighbours(name, depth=2, limit=25)
    if not rows:
        print(c(f"      No neighbours found for {name!r}", YELLOW))
        return

    for r in rows:
        rels   = " → ".join(r.get("relations", []))
        target = r.get("target") or "?"
        ttype  = r.get("target_type") or "?"
        print(f"      {GREEN}{r['source']}{RESET}  --[{CYAN}{rels}{RESET}]-->  "
              f"{YELLOW}{target}{RESET}  {dim('('+ttype+')')}")


def main():
    args = sys.argv[1:]
    paper_id = None
    entity   = None

    i = 0
    while i < len(args):
        if args[i] == "--paper" and i + 1 < len(args):
            paper_id = args[i + 1]; i += 2
        elif args[i] == "--entity" and i + 1 < len(args):
            entity = args[i + 1]; i += 2
        else:
            i += 1

    print_banner()

    try:
        inspect_schema()
        inspect_papers()

        if paper_id:
            inspect_paper_context(paper_id)

        if entity:
            inspect_entity(entity)

        print(f"\n{SEP}\n")
        print(c("  Done.", GREEN))

    except Exception as e:
        print(c(f"\n  ERROR: {e}", RED))
        print(dim("  Make sure Neo4j is running and NEO4J_URI/USER/PASSWORD are set."))
        sys.exit(1)


if __name__ == "__main__":
    main()
