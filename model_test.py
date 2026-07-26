# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# from langchain_core.messages import HumanMessage

# llm = HuggingFaceEndpoint(
#     # repo_id="google/gemma-3-4b-it",
#     repo_id= "Qwen/Qwen2.5-7B-Instruct",
#     # repo_id= "meta-llama/Llama-3.1-70B-Instruct",
#     max_new_tokens=256,
#     temperature=0.1,
# )

# chat = ChatHuggingFace(llm=llm)

# response = chat.invoke([
#     HumanMessage(content="What is the use of Artificial Intelligence in today's world?")
# ])

# print(response.content)

from backend.agents.orchestrator import run

# Text query (paper already ingested)
result = run(
    query=        "What is the main contribution of this paper?",
    paper_loaded= True,
    paper_id=     "your_paper_id",   # from ingestion output
)
print(result["final_answer"])
print(result["metadata"].get("evaluation", "No evaluation data"))

# With satellite image
result = run(
    query=      "What land cover is visible?",
    image_path= "data/satellite/image.jpg",
)

# With voice note
result = run(
    query=      "",
    audio_path= "data/audio/question.wav",
    paper_loaded= True,
)