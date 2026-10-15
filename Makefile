SHELL := /bin/bash
.DEFAULT_GOAL := help

INDEX ?= heading-400
RETRIEVER ?= vector
RAG := uv run rag

export RAG_DATABASE_URL ?= postgresql://rag:rag@localhost:5433/rag
export RAG_TEST_DATABASE_URL ?= postgresql://rag:rag@localhost:5433/postgres

.PHONY: help setup setup-ci db migrate discover corpus-init fetch verify ingest build-index evaluate gate publish serve status test test-unit lint format split validate ablation confirm-test clean

help:
	@echo "setup         install pinned dependencies including the embedding model extras"
	@echo "db            start PostgreSQL 16 with pgvector and wait until healthy"
	@echo "ingest        migrate, download the corpus by manifest, verify, parse and chunk"
	@echo "build-index   embed chunks and build the index for INDEX=$(INDEX)"
	@echo "evaluate      score INDEX with RETRIEVER on the dev split, then run the gate"
	@echo "publish       switch the served index to the latest passing gate decision"
	@echo "serve         run the API on port 8000"
	@echo "test          lint and run all tests"

setup:
	uv sync --locked --extra models --extra generation

setup-ci:
	uv sync --locked

db:
	docker compose up -d --wait

migrate:
	$(RAG) db migrate

discover:
	$(RAG) corpus discover

corpus-init:
	$(RAG) corpus init

fetch:
	$(RAG) corpus fetch

verify:
	$(RAG) corpus verify

ingest: migrate fetch
	$(RAG) ingest --index $(INDEX)

build-index:
	$(RAG) index build --index $(INDEX)

validate:
	$(RAG) eval validate

split:
	$(RAG) eval split

evaluate:
	$(RAG) eval run --index $(INDEX) --retriever $(RETRIEVER)
	$(RAG) gate check --latest

gate:
	$(RAG) gate check --latest

publish:
	$(RAG) publish --latest-pass

serve:
	$(RAG) serve

status:
	$(RAG) status

ablation:
	$(RAG) report ablation

confirm-test:
	@test -n "$(DECISION)" || (echo "set DECISION=D-xxx, an entry in docs/decisions.md" && exit 1)
	$(RAG) eval confirm-test --index $(INDEX) --retriever $(RETRIEVER) --decision $(DECISION)

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test-unit:
	uv run pytest -m "not integration"

test: lint
	uv run pytest

clean:
	rm -rf .pytest_cache .ruff_cache
