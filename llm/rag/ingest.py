import argparse, os, glob
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

def ingest(collection: str, docs_dir: str, host: str = "localhost", port: int = 6333):
    client = QdrantClient(host=host, port=port)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    client.recreate_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    docs = []
    for path in glob.glob(os.path.join(docs_dir, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            docs.append((os.path.basename(path), f.read().strip()))

    texts = [t for _, t in docs]
    vecs = model.encode(texts).tolist()

    points = [
        PointStruct(id=i, vector=vecs[i], payload={"source": docs[i][0], "text": texts[i]})
        for i in range(len(texts))
    ]
    client.upsert(collection_name=collection, points=points)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--docs", required=True)
    args = parser.parse_args()
    ingest(collection=args.collection, docs_dir=args.docs)
