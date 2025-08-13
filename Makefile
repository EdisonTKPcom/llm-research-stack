install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

lint:
	ruff check .
	black --check .
	mypy binance_llm

fmt:
	black .
	ruff check --fix .

test:
	pytest -q

run-api:
	uvicorn binance_llm.api.main:app --reload --port 8000

start-qdrant:
	docker compose -f docker/compose.yaml up -d qdrant

ingest:
	python -m binance_llm.rag.ingest --collection academy --docs data/sample_docs
