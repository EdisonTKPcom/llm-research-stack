from fastapi import FastAPI
from pydantic import BaseModel
from ..rag.rag_chain import RAGChain

app = FastAPI(title="Binance LLM Research Stack API")
rag = RAGChain()

class ChatRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
def ingest():
    # Minimal example: ingest sample docs
    rag.ingest(collection="academy", docs_dir="data/sample_docs")
    return {"ok": True}

@app.post("/chat")
def chat(req: ChatRequest):
    answer, refs = rag.answer(req.question, collection="academy")
    return {"answer": answer, "references": refs}
