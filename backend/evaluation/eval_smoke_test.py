"""
AstroNexusAI -- Evaluation Smoke Test (Task 6)
==============================================
Runs single-query retrieval tests at multiple top_k / threshold values.
Run from backend/ directory:
    python evaluation/eval_smoke_test.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Make backend importable — add both project root and backend/ itself
_BACKEND     = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BACKEND.parent
for _p in [str(_BACKEND), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(_BACKEND.parent / ".env")
except ImportError:
    pass

print("\n" + "="*60)
print("SMOKE TEST -- AstroNexusAI Retrieval Pipeline")
print("="*60)

# -- Qdrant collection info ---------------------------------------------------
try:
    from qdrant_client import QdrantClient
    client = QdrantClient("localhost", port=6333)
    info   = client.get_collection("papers")
    qdrant_dim    = info.config.params.vectors.size
    qdrant_dist   = info.config.params.vectors.distance.name
    qdrant_points = info.points_count
    print(f"\nQdrant 'papers' collection:")
    print(f"  vector_size : {qdrant_dim}")
    print(f"  distance    : {qdrant_dist}")
    print(f"  total points: {qdrant_points}")
except Exception as e:
    print(f"[ERROR] Qdrant connection failed: {e}")
    sys.exit(1)

# -- Embedding test -----------------------------------------------------------
print("\n-- Embedding Test --")
from backend.embeddings.embedder import embed_query, MODEL_NAME, VECTOR_DIM, USE_LOCAL_EMBEDDING

backend_name = "local sentence-transformers" if USE_LOCAL_EMBEDDING else "HF Inference API"
print(f"  Embedding model  : {MODEL_NAME}")
print(f"  Expected dim     : {VECTOR_DIM}")
print(f"  Backend          : {backend_name}")

t0     = time.perf_counter()
vec    = embed_query("What is the orbital period of AION-1 satellite?")
t_emb  = (time.perf_counter() - t0) * 1000

actual_dim = len(vec)
print(f"  Actual dim       : {actual_dim}")
print(f"  Embed latency    : {t_emb:.1f} ms")

if actual_dim != qdrant_dim:
    print(f"[CRITICAL] Dimension mismatch: embedding={actual_dim}, Qdrant={qdrant_dim}")
    print("  Re-index Qdrant or switch embedding model.")
    sys.exit(1)
else:
    print(f"  Dimensions match ({actual_dim}-dim) OK")

# -- Retrieval tests ----------------------------------------------------------
from backend.rag.retriever import retrieve

TEST_QUERY = "What is the orbital period of AION-1 satellite?"

configs = [
    dict(top_k=1,  score_threshold=0.0,  label="top_k=1,  thr=0.0 "),
    dict(top_k=3,  score_threshold=0.0,  label="top_k=3,  thr=0.0 "),
    dict(top_k=5,  score_threshold=0.0,  label="top_k=5,  thr=0.0 "),
    dict(top_k=5,  score_threshold=0.30, label="top_k=5,  thr=0.30"),
    dict(top_k=5,  score_threshold=0.55, label="top_k=5,  thr=0.55"),
]

print(f"\n-- Retrieval Tests --")
print(f"   Query: '{TEST_QUERY[:55]}...'")

all_ok = True
for cfg in configs:
    t1     = time.perf_counter()
    chunks = retrieve(
        query=           TEST_QUERY,
        top_k=           cfg["top_k"],
        score_threshold= cfg["score_threshold"],
    )
    t_ret = (time.perf_counter() - t1) * 1000

    scores    = [round(c.score, 4) for c in chunks]
    paper_ids = [c.paper_id for c in chunks]

    print(f"\n  [{cfg['label']}]")
    print(f"    n_retrieved : {len(chunks)}")
    print(f"    scores      : {scores}")
    print(f"    paper_ids   : {paper_ids}")
    print(f"    latency     : {t_ret:.1f} ms")

    if len(chunks) == 0 and cfg["score_threshold"] == 0.0:
        print(f"    [WARN] Zero results with threshold=0.0 -- check paper_id filter and Qdrant")
        all_ok = False

if all_ok:
    print("\nSmoke test PASSED -- pipeline is working correctly.")
else:
    print("\nSmoke test completed with warnings -- review results above.")

print("="*60)
print("\nRun the full ablation from backend/:")
print("  python -m evaluation.ablation --phase retrieval-all --no-generation")
