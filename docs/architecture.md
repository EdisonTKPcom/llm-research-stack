# Architecture

- **FastAPI backend** exposes `/chat`, `/ingest`, `/health`.
- **RAG** uses Qdrant vector store; simple retriever -> prompt composer -> LLM.
- **SFT** and **RLHF** directories hold training templates with `transformers` and `trl`.
- **Agent** orchestrates tools (market data, retrieval, simple stats).
