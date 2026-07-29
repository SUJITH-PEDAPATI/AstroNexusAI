from dotenv import load_dotenv; load_dotenv()

# Step 1: Check Neo4j
print('[1] Neo4j...')
try:
    from backend.graph.neo4j_client import _get_driver, get_graph_stats
    stats = get_graph_stats()
    nodes = stats.get('total_nodes', 0)
    rels  = stats.get('total_relations', 0)
    print(f'    OK - nodes={nodes} relations={rels}')
except Exception as e:
    print(f'    FAIL - {e}')

# Step 2: Check paper exists in Neo4j
print('[2] Paper in Neo4j...')
try:
    from backend.graph.neo4j_client import _get_driver
    with _get_driver().session() as s:
        rows = s.run('MATCH (p:Paper) RETURN p.title AS t, p.paper_id AS pid LIMIT 5').data()
    if rows:
        for r in rows:
            print(f'    {r["t"][:50]} | {r["pid"]}')
    else:
        print('    NO PAPERS in Neo4j - re-ingest required')
except Exception as e:
    print(f'    FAIL - {e}')

# Step 3: Check paper in Qdrant
print('[3] Paper in Qdrant...')
try:
    from backend.rag.retriever import retrieve
    chunks = retrieve('scientific goals', top_k=3)
    if chunks:
        print(f'    OK - {len(chunks)} chunks, top score={chunks[0].score:.3f}')
    else:
        print('    NO CHUNKS in Qdrant - re-ingest required')
except Exception as e:
    print(f'    FAIL - {e}')

# Step 4: Check graph context retrieval
print('[4] Graph context...')
try:
    from backend.agents.knowledge_fusion import _get_graph_context
    ctx = _get_graph_context('scientific goals', None)
    print(f'    authors  = {ctx.get("authors")}')
    print(f'    keywords = {ctx.get("keywords")}')
    print(f'    facts    = {len(ctx.get("related_facts", []))}')
except Exception as e:
    print(f'    FAIL - {e}')

# Step 5: Full orchestrator run
print('[5] Full pipeline...')
try:
    from backend.agents.orchestrator import run
    result = run(
        query='What are the scientific goals?',
        paper_loaded=True,
        paper_id='d542e31bb239eca29da245eea786d9e5',
    )
    meta = result.get('metadata') or {}
    print(f'    route        = {result.get("query_type")}')
    print(f'    mode         = {meta.get("mode")}')
    print(f'    top_score    = {meta.get("top_score")}')
    print(f'    paper_loaded = {meta.get("paper_loaded")}')
except Exception as e:
    print(f'    FAIL - {e}')
