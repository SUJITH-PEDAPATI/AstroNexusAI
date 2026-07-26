import json
from pathlib import Path
from backend.agents.orchestrator import run

# Paste your benchmark rows as a list
TESTS = [
    # ==========================
    # GENERAL KNOWLEDGE
    # ==========================
    {"id": "G01", "category": "General", "question": "What is reinforcement learning?", "expected_agent": "general"},
    {"id": "G02", "category": "General", "question": "When was ISRO founded?", "expected_agent": "general"},
    {"id": "G03", "category": "General", "question": "What is a transformer in electronics?", "expected_agent": "general"},
    {"id": "G04", "category": "General", "question": "Explain the difference between supervised and unsupervised learning.", "expected_agent": "general"},
    {"id": "G05", "category": "General", "question": "What is a black hole?", "expected_agent": "general"},
    {"id": "G06", "category": "General", "question": "Who is the current director of NASA?", "expected_agent": "general"},
    {"id": "G07", "category": "General", "question": "What programming language is Python named after?", "expected_agent": "general"},
    {"id": "G08", "category": "General", "question": "What is the difference between RAM and ROM?", "expected_agent": "general"},
    {"id": "G09", "category": "General", "question": "How does GPS work?", "expected_agent": "general"},
    {"id": "G10", "category": "General", "question": "What is quantum computing?", "expected_agent": "general"},
    {"id": "G11", "category": "General", "question": "What is the speed of light?", "expected_agent": "general"},
    {"id": "G12", "category": "General", "question": "Explain gradient descent.", "expected_agent": "general"},
    {"id": "G13", "category": "General", "question": "What is the difference between precision and recall?", "expected_agent": "general"},
    {"id": "G14", "category": "General", "question": "What does CNN stand for in deep learning?", "expected_agent": "general"},
    {"id": "G15", "category": "General", "question": "What is overfitting?", "expected_agent": "general"},

    # ==========================
    # RAG
    # ==========================
    {"id": "R01", "category": "RAG", "question": "What is the main contribution of this paper?", "expected_agent": "research"},
    {"id": "R02", "category": "RAG", "question": "What dataset was used for training?", "expected_agent": "research"},
    {"id": "R03", "category": "RAG", "question": "What baseline models were compared?", "expected_agent": "research"},
    {"id": "R04", "category": "RAG", "question": "What hyperparameters were chosen?", "expected_agent": "research"},
    {"id": "R05", "category": "RAG", "question": "What loss function was used?", "expected_agent": "research"},
    {"id": "R06", "category": "RAG", "question": "What limitations were mentioned by the authors?", "expected_agent": "research"},
    {"id": "R07", "category": "RAG", "question": "What is Algorithm 2 in this paper?", "expected_agent": "research"},
    {"id": "R08", "category": "RAG", "question": "What evaluation metrics were reported?", "expected_agent": "research"},
    {"id": "R09", "category": "RAG", "question": "Who funded this research?", "expected_agent": "research"},
    {"id": "R10", "category": "RAG", "question": "What future work do the authors suggest?", "expected_agent": "research"},
    {"id": "R11", "category": "RAG", "question": "What is the computational complexity of the model?", "expected_agent": "research"},
    {"id": "R12", "category": "RAG", "question": "What BLEU score was reported?", "expected_agent": "research"},
    {"id": "R13", "category": "RAG", "question": "How many parameters does the model have?", "expected_agent": "research"},
    {"id": "R14", "category": "RAG", "question": "What preprocessing steps were applied?", "expected_agent": "research"},
    {"id": "R15", "category": "RAG", "question": "What ablation studies were performed?", "expected_agent": "research"},

    # ==========================
    # FOLLOW-UP
    # ==========================
    {"id": "F01", "category": "Follow-up", "question": "Explain that in simple words.", "expected_agent": "research"},
    {"id": "F02", "category": "Follow-up", "question": "Who proposed this approach?", "expected_agent": "research"},
    {"id": "F03", "category": "Follow-up", "question": "Is that dataset publicly available?", "expected_agent": "research"},
    {"id": "F04", "category": "Follow-up", "question": "Why did they choose that loss function?", "expected_agent": "research"},
    {"id": "F05", "category": "Follow-up", "question": "Which metric showed the biggest improvement?", "expected_agent": "research"},
    {"id": "F06", "category": "Follow-up", "question": "How could those limitations be addressed?", "expected_agent": "research"},
    {"id": "F07", "category": "Follow-up", "question": "Has any paper already addressed one of those limitations?", "expected_agent": "graph"},
    {"id": "F08", "category": "Follow-up", "question": "Compare this paper with the previous one.", "expected_agent": "research"},
    {"id": "F09", "category": "Follow-up", "question": "Summarize the paper for a research blog.", "expected_agent": "research"},
    {"id": "F10", "category": "Follow-up", "question": "I meant the second paper I uploaded.", "expected_agent": "research"},

    # ==========================
    # GRAPH
    # ==========================
    {"id": "KG01", "category": "Graph", "question": "Who authored Attention Is All You Need?", "expected_agent": "graph"},
    {"id": "KG02", "category": "Graph", "question": "What papers use the Transformer model?", "expected_agent": "graph"},
    {"id": "KG03", "category": "Graph", "question": "Which satellites belong to the Copernicus mission?", "expected_agent": "graph"},
    {"id": "KG04", "category": "Graph", "question": "What sensor does Sentinel-1 carry?", "expected_agent": "graph"},
    {"id": "KG05", "category": "Graph", "question": "Which papers are about flood detection?", "expected_agent": "graph"},
    {"id": "KG06", "category": "Graph", "question": "Which datasets are related to semantic segmentation?", "expected_agent": "graph"},
    {"id": "KG07", "category": "Graph", "question": "Which algorithms are tagged under Remote Sensing?", "expected_agent": "graph"},
    {"id": "KG08", "category": "Graph", "question": "What missions does NASA operate?", "expected_agent": "graph"},
    {"id": "KG09", "category": "Graph", "question": "Which papers were presented at NeurIPS?", "expected_agent": "graph"},
    {"id": "KG10", "category": "Graph", "question": "What metrics are used for segmentation tasks?", "expected_agent": "graph"}
]

results = []
for test in TESTS:
    output   = run(test["question"])
    routed   = output.get("query_type", "unknown")
    passed   = routed == test["expected_agent"]
    results.append({
        "id":       test["id"],
        "passed":   passed,
        "expected": test["expected_agent"],
        "got":      routed,
        "answer":   output.get("final_answer", "")[:120],
        "error":    output.get("error"),
    })
    print(f"{'✓' if passed else '✗'} [{test['id']}] routed={routed}")

# Save report
Path("output").mkdir(exist_ok=True)
with open("output/benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

passed = sum(r["passed"] for r in results)
print(f"\n{passed}/{len(results)} routing tests passed")