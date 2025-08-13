from typing import List, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def search(self, query: str, collection: str, top_k: int = 3) -> List[Tuple[str, str]]:
        query_vec = self.model.encode([query])[0].tolist()
        hits = self.client.search(collection, query_vector=query_vec, limit=top_k)
        return [(h.payload.get("source"), h.payload.get("text")) for h in hits]
