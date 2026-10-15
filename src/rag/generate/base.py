import os
from collections.abc import Sequence
from typing import Protocol

import psycopg

from rag.evaluate.answer_metrics import Answer
from rag.retrieve.types import ChunkRecord

ABSTAIN_MARKER = "INSUFFICIENT_EVIDENCE"


class Generator(Protocol):
    name: str

    def generate(self, question: str, chunks: Sequence[ChunkRecord]) -> Answer: ...


def make_generator(kind: str | None = None, conn: psycopg.Connection | None = None) -> Generator:
    kind = kind or os.environ.get("RAG_GENERATOR", "extractive")
    if kind == "extractive":
        from rag.generate.extractive import ExtractiveGenerator

        return ExtractiveGenerator()
    if kind == "anthropic":
        from rag.generate.hosted import AnthropicGenerator

        if conn is None:
            raise ValueError("the hosted generator needs a database connection for its cache")
        return AnthropicGenerator.from_env(conn)
    raise ValueError(f"unknown generator {kind!r}")
