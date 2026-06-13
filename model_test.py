from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)

client.delete_collection("papers")

print("Collection deleted successfully.")