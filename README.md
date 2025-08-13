# Binance LLM Research Stack

A GitHub-ready template to showcase research and engineering capabilities for a role focusing on **LLMs, RAG, SFT, reward modeling/RLHF, evaluation**, and **AI agent** use-cases in crypto/fintech.

> This repository is a working scaffold: you can run a local RAG demo, start an API, and extend with SFT/RLHF experiments. It’s designed to be easily forked for interviews and coding tasks.

---

## Features

- **RAG pipeline** with Qdrant vector store, document ingestion, retriever, and chat endpoint.
- **Knowledge‑grounded dialogue** prompts and guardrails.
- **SFT** (LoRA PEFT) training skeleton with Hugging Face `transformers` + `datasets`.
- **Reward modeling + PPO (RLHF)** templates using `trl`.
- **Evaluation harness** (faithfulness/answer quality) with classic metrics; plug for RAGAS.
- **Crypto research agent** that can call tools (market data, simple analytics) and route to RAG/LLM.
- **FastAPI** backend + (optional) **Streamlit** reference UI.
- **Solid CI** (ruff, black, mypy, pytest) and docs via **MkDocs**.
- **Devcontainers & Docker Compose** for one‑command local spin‑up (API + Qdrant).

> ⚠️ This template ships with small sample docs only. Replace with compliant, internal docs when appropriate.

---

## Quickstart

```bash
# Python 3.10+ recommended
python -m venv .venv && source .venv/bin/activate

# Install (CPU)
pip install -r requirements.txt

# Start vector DB (docker required)
docker compose -f docker/compose.yaml up -d qdrant

# Ingest sample docs
python -m binance_llm.rag.ingest --collection academy --docs data/sample_docs

# Run API
uvicorn binance_llm.api.main:app --reload --port 8000
```

Open http://localhost:8000/docs for interactive API, then call `POST /chat`.

---

## Repo Layout

```
binance-llm-research-stack/
├── binance_llm/               # Python package
│   ├── api/                   # FastAPI app
│   ├── rag/                   # Retrieval Augmented Generation pipeline
│   ├── agents/                # Tool-using agents for crypto research
│   ├── sft/                   # Supervised fine-tuning templates
│   ├── rlhf/                  # Reward modeling + PPO training
│   └── evaluation/            # Eval harness and metrics
├── data/sample_docs/          # Small seed corpus for RAG
├── notebooks/                 # Demo notebooks
├── docs/                      # MkDocs documentation
├── .github/workflows/         # CI & Docs pipelines
├── docker/                    # Dockerfiles and compose
├── tests/                     # Unit tests
├── requirements.txt
├── pyproject.toml
└── mkdocs.yml
```

---

## What to Build Next

- Replace sample docs with **Binance product/infra docs** (if permitted) to demo knowledge grounding.
- Add **finetuned adapters** for the target domain using `binance_llm/sft/train_sft.py`.
- Train a lightweight **reward model** for answer helpfulness, integrate PPO loop.
- Extend the **agent** with more tools (on-chain data, risk heuristics, compliance flags).

---

## Security & Safety

See [`SECURITY.md`](SECURITY.md) for reporting and [`docs/safety.md`](docs/safety.md) for model guardrails and misuse prevention.
