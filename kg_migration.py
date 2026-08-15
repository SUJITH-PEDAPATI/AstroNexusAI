"""
AstroNexusAI — Knowledge Graph Migration
==========================================
Run ONCE after applying the neo4j_client.py fixes.

What it does:
  1. Creates new uniqueness constraints (paper_id, keyword name, entity name)
  2. Backfills paper_id on existing Paper nodes that only have node_id
  3. Adds HAS_ENTITY and HAS_KEYWORD edges where only AUTHORED_BY/TAGGED exist
  4. Creates Entity nodes for Authors/Models/Datasets already in the graph

Safe to run multiple times — all writes use MERGE.

Usage:
    cd <project_root>
    python kg_migration.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.graph.neo4j_client import _get_driver

def run_migration():
    driver = _get_driver()

    print("=== AstroNexusAI Knowledge Graph Migration ===\n")

    with driver.session() as s:
        # 1. New constraints
        print("Step 1: Creating new constraints...")
        for cq in [
            "CREATE CONSTRAINT paper_paper_id_unique IF NOT EXISTS FOR (p:Paper)      REQUIRE p.paper_id IS UNIQUE",
            "CREATE CONSTRAINT keyword_name_unique   IF NOT EXISTS FOR (k:Keyword)    REQUIRE k.name     IS UNIQUE",
            "CREATE CONSTRAINT entity_name_unique    IF NOT EXISTS FOR (e:Entity)     IS UNIQUE",
            "CREATE CONSTRAINT query_id_unique       IF NOT EXISTS FOR (q:Query)      REQUIRE q.query_id IS UNIQUE",
            "CREATE CONSTRAINT voice_id_unique       IF NOT EXISTS FOR (v:VoiceInput) REQUIRE v.voice_id IS UNIQUE",
        ]:
            try: s.run(cq)
            except Exception as e: print(f"  Warning: {e}")
        print("  Constraints done.\n")

        # 2. Backfill paper_id on Paper nodes that lack it
        print("Step 2: Backfilling paper_id on Paper nodes...")
        r = s.run("""
            MATCH (p:Paper)
            WHERE p.paper_id IS NULL AND p.node_id IS NOT NULL
            SET p.paper_id = p.node_id
            RETURN count(p) AS fixed
        """)
        print(f"  Fixed: {r.single()['fixed']} nodes.\n")

        # 3. Migrate AUTHORED_BY → HAS_ENTITY
        print("Step 3: Migrating AUTHORED_BY to HAS_ENTITY...")
        r = s.run("""
            MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author)
            MERGE (e:Entity {name: a.name})
            SET e.type = 'Author'
            WITH p, e
            MERGE (p)-[:HAS_ENTITY]->(e)
            RETURN count(e) AS migrated
        """)
        print(f"  Migrated: {r.single()['migrated']} Author entities.\n")

        # 4. Migrate Model/Dataset/Task → HAS_ENTITY
        for label, etype in [("Model","Model"),("Dataset","Dataset"),("Task","Task")]:
            r = s.run(f"""
                MATCH (p:Paper)-[]->(n:{label})
                MERGE (e:Entity {{name: n.name}})
                SET e.type = $etype
                WITH p, e
                MERGE (p)-[:HAS_ENTITY]->(e)
                RETURN count(e) AS migrated
            """, etype=etype)
            print(f"  Migrated: {r.single()['migrated']} {label} entities.")

        # 5. Migrate TAGGED → HAS_KEYWORD
        print("\nStep 5: Migrating TAGGED to HAS_KEYWORD...")
        r = s.run("""
            MATCH (p:Paper)-[:TAGGED]->(k:Keyword)
            MERGE (p)-[:HAS_KEYWORD]->(k)
            RETURN count(k) AS migrated
        """)
        print(f"  Migrated: {r.single()['migrated']} keyword links.\n")

        # Summary
        print("Step 6: Migration summary...")
        for label in ["Paper","Author","Model","Dataset","Task","Keyword","Entity","Query","VoiceInput","VoiceSession"]:
            r = s.run(f"MATCH (n:{label}) RETURN count(n) AS c")
            print(f"  {label:<18}: {r.single()['c']}")

        print("\n=== Migration complete ===")

if __name__ == "__main__":
    run_migration()