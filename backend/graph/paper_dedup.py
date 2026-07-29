"""
Fix 5: Duplicate papers — production dedup strategy

Run to find and fix duplicates:
    python -m backend.graph.paper_dedup
"""
from __future__ import annotations
import sys, logging
logging.basicConfig(level=logging.WARNING)


def find_duplicates() -> list[dict]:
    from backend.graph.neo4j_client import _get_driver
    driver = _get_driver()
    with driver.session() as s:
        rows = s.run(
            """
            MATCH (p:Paper)
            WITH p.title AS title, collect(p) AS papers, count(*) AS cnt
            WHERE cnt > 1
            RETURN title, cnt,
                   [p in papers | {id: id(p), paper_id: p.paper_id, node_id: p.node_id}] AS nodes
            ORDER BY cnt DESC LIMIT 20
            """
        ).data()
    return rows


def deduplicate() -> int:
    """Keep the node with paper_id set, delete others."""
    from backend.graph.neo4j_client import _get_driver
    driver  = _get_driver()
    deleted = 0

    dupes = find_duplicates()
    if not dupes:
        print("  No duplicates found.")
        return 0

    with driver.session() as s:
        for row in dupes:
            nodes    = row["nodes"]
            # Keep node with paper_id, or first node if none have it
            keeper   = next((n for n in nodes if n.get("paper_id")), nodes[0])
            to_delete= [n for n in nodes if n["id"] != keeper["id"]]

            for node in to_delete:
                s.run(
                    "MATCH (p) WHERE id(p)=$iid DETACH DELETE p",
                    iid=node["id"],
                )
                deleted += 1
                print(f"  Deleted duplicate: {row['title'][:50]}")

    return deleted


def add_uniqueness_constraint():
    """Add paper_id uniqueness constraint to prevent future duplicates."""
    from backend.graph.neo4j_client import _get_driver
    driver = _get_driver()
    with driver.session() as s:
        s.run(
            "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS "
            "FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE"
        )
    print("  ✓ Uniqueness constraint added on Paper.paper_id")


def run():
    print("\n" + "="*50)
    print("ASTRONEXUS — PAPER DEDUPLICATION")
    print("="*50)

    dupes = find_duplicates()
    if not dupes:
        print("\n  No duplicate papers found.")
    else:
        print(f"\n  Found {len(dupes)} duplicate paper titles:")
        for d in dupes:
            print(f"    {d['title'][:50]} ({d['cnt']} copies)")

        deleted = deduplicate()
        print(f"\n  Deleted {deleted} duplicate nodes.")

    add_uniqueness_constraint()
    print("\n✓ Done\n")


if __name__ == "__main__":
    run()