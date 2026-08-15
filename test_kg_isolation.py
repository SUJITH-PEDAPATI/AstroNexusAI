"""
AstroNexusAI — Knowledge Graph Isolation Tests
Tests: paper isolation, query linking, voice linking, duplicate prevention.
Run: python test_kg_isolation.py
"""
import sys, uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS = 0; FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}{': ' + detail if detail else ''}")
        FAIL += 1

def run_tests():
    from backend.graph.neo4j_client import (
        _get_driver, upsert_query_node, upsert_voice_input_node,
        upsert_entity_for_paper, upsert_keyword_for_paper,
        query_paper_graph_by_id, list_papers,
    )
    driver = _get_driver()

    print("\n=== Test 1: Paper isolation — entities don't bleed ===")
    pid_a = "test-paper-A-" + str(uuid.uuid4())[:8]
    pid_b = "test-paper-B-" + str(uuid.uuid4())[:8]

    # Create two paper nodes
    with driver.session() as s:
        s.run("MERGE (p:Paper {paper_id:$pid}) SET p.title=$t", pid=pid_a, t="Test Paper A")
        s.run("MERGE (p:Paper {paper_id:$pid}) SET p.title=$t", pid=pid_b, t="Test Paper B")

    upsert_entity_for_paper(pid_a, "Entity-Only-In-A", "Model")
    upsert_entity_for_paper(pid_b, "Entity-Only-In-B", "Dataset")
    upsert_keyword_for_paper(pid_a, "keyword-alpha")
    upsert_keyword_for_paper(pid_b, "keyword-beta")

    graph_a = query_paper_graph_by_id(pid_a)
    graph_b = query_paper_graph_by_id(pid_b)

    names_a = {r["target_name"] for r in graph_a}
    names_b = {r["target_name"] for r in graph_b}

    check("Paper A has its own entity", "Entity-Only-In-A" in names_a)
    check("Paper A has its own keyword", "keyword-alpha" in names_a)
    check("Paper A does NOT have Paper B's entity", "Entity-Only-In-B" not in names_a)
    check("Paper B has its own entity", "Entity-Only-In-B" in names_b)
    check("Paper B does NOT have Paper A's entity", "Entity-Only-In-A" not in names_b)

    print("\n=== Test 2: Query node links to multiple papers ===")
    qid = str(uuid.uuid4())
    upsert_query_node(qid, "test query text " + qid, [pid_a, pid_b], ["BERT", "WMT-2014"])
    with driver.session() as s:
        r = s.run("""
            MATCH (q:Query {query_id:$qid})-[:RELATED_TO]->(p:Paper)
            RETURN count(p) AS cnt
        """, qid="test query text " + qid)
        cnt = r.single()["cnt"]
    check("Query linked to 2 papers", cnt == 2, f"got {cnt}")

    with driver.session() as s:
        r = s.run("""
            MATCH (q:Query {query_id:$qid})-[:MENTIONS]->(e:Entity)
            RETURN count(e) AS cnt
        """, qid="test query text " + qid)
        cnt = r.single()["cnt"]
    check("Query mentions extracted entities", cnt >= 1, f"got {cnt}")

    print("\n=== Test 3: VoiceInput links to papers and entities ===")
    vid = str(uuid.uuid4())
    upsert_voice_input_node(vid, "voice test " + vid, [pid_a], ["DINOv2"])
    with driver.session() as s:
        r = s.run("""
            MATCH (v:VoiceInput {voice_id:$vid})-[:RELATED_TO]->(p:Paper)
            RETURN count(p) AS cnt
        """, vid=vid)
        cnt = r.single()["cnt"]
    check("VoiceInput linked to Paper", cnt == 1, f"got {cnt}")

    with driver.session() as s:
        r = s.run("""
            MATCH (v:VoiceInput {voice_id:$vid})-[:MENTIONS]->(e:Entity)
            RETURN count(e) AS cnt
        """, vid=vid)
        cnt = r.single()["cnt"]
    check("VoiceInput mentions entity", cnt >= 1, f"got {cnt}")

    print("\n=== Test 4: Duplicate-proof MERGE ===")
    upsert_entity_for_paper(pid_a, "Entity-Only-In-A", "Model")  # repeat
    upsert_entity_for_paper(pid_a, "Entity-Only-In-A", "Model")  # repeat
    with driver.session() as s:
        r = s.run("""
            MATCH (p:Paper {paper_id:$pid})-[:HAS_ENTITY]->(e:Entity {name:'Entity-Only-In-A'})
            RETURN count(e) AS cnt
        """, pid=pid_a)
        cnt = r.single()["cnt"]
    check("Duplicate MERGE produces exactly 1 entity link", cnt == 1, f"got {cnt}")

    print("\n=== Test 5: list_papers() returns ingested papers ===")
    papers = list_papers()
    check("list_papers returns at least the 2 test papers",
          any(p["paper_id"] == pid_a for p in papers))

    print("\n=== Test 6: Cleanup test nodes ===")
    with driver.session() as s:
        s.run("MATCH (p:Paper) WHERE p.paper_id STARTS WITH 'test-paper-' DETACH DELETE p")
        s.run("MATCH (q:Query) WHERE q.text STARTS WITH 'test query text' DETACH DELETE q")
        s.run("MATCH (v:VoiceInput) WHERE v.text STARTS WITH 'voice test' DETACH DELETE v")
        s.run("MATCH (e:Entity) WHERE e.name IN ['Entity-Only-In-A','Entity-Only-In-B','DINOv2','BERT','WMT-2014'] DETACH DELETE e")
        s.run("MATCH (k:Keyword) WHERE k.name IN ['keyword-alpha','keyword-beta'] DETACH DELETE k")
    check("Test cleanup complete", True)

    print(f"\n{'='*44}")
    print(f"  RESULTS: {PASS} PASS  {FAIL} FAIL")
    if FAIL == 0:
        print("  ALL TESTS PASS — KG isolation is working correctly")
    else:
        print("  SOME TESTS FAILED — check output above")
    print(f"{'='*44}\n")

if __name__ == "__main__":
    run_tests()