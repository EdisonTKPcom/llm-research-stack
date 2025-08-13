from typing import List, Tuple
import os
from .retriever import Retriever

# The generator here is a placeholder. Wire your model/provider of choice.
def simple_generator(prompt: str) -> str:
    # In practice, call OpenAI, local HF model, or vendor LLM here.
    # Keeping it offline-friendly for the template.
    return "This is a stubbed answer based on retrieved context. Replace with an LLM call."

SYSTEM_PROMPT = """You are a helpful, risk-aware assistant for a crypto exchange.
Use the provided context only. If unsure, say you don't know.

"""

class RAGChain:
    def __init__(self):
        self.retriever = Retriever()

    def ingest(self, collection: str, docs_dir: str):
        from .ingest import ingest as do_ingest
        do_ingest(collection=collection, docs_dir=docs_dir)

    def compose_prompt(self, question: str, contexts: List[Tuple[str, str]]) -> str:
        ctx = "\n\n".join([f"[{i+1}] {c[1]}" for i, c in enumerate(contexts)])
        user = f"Question: {question}\n\nContext:\n{ctx}\n\nAnswer:"
        return SYSTEM_PROMPT + user

    def answer(self, question: str, collection: str = "academy"):
        ctxs = self.retriever.search(question, collection=collection, top_k=3)
        prompt = self.compose_prompt(question, ctxs)
        ans = simple_generator(prompt)
        refs = [c[0] for c in ctxs]
        return ans, refs
