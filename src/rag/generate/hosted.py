import json
import os
import re
from collections.abc import Sequence

import psycopg

from rag.evaluate.answer_metrics import Answer
from rag.generate.base import ABSTAIN_MARKER
from rag.hashing import canonical_json, sha256_text
from rag.retrieve.types import ChunkRecord

PROMPT_VERSION = "answer-prompt-v1"
CITATION = re.compile(r"\[c(\d+)\]")

SYSTEM_PROMPT = (
    "You answer questions about the PostgreSQL manual using only the numbered passages given. "
    "Cite every claim with the passage label in square brackets, for example [c1]. "
    f"If the passages do not contain the answer, reply with exactly {ABSTAIN_MARKER} and nothing "
    "else. Do not use outside knowledge."
)


class BudgetExceeded(RuntimeError):
    pass


def build_prompt(question: str, chunks: Sequence[ChunkRecord]) -> str:
    passages = "\n\n".join(
        f"[c{i}] {' > '.join(c.heading_path)}\n{c.body}" for i, c in enumerate(chunks, start=1)
    )
    return f"Passages:\n\n{passages}\n\nQuestion: {question}"


def parse_response(text: str, chunks: Sequence[ChunkRecord]) -> Answer:
    if text.strip() == ABSTAIN_MARKER or not text.strip():
        return Answer("", (), True)
    cited: list[str] = []
    for match in CITATION.finditer(text):
        position = int(match.group(1)) - 1
        if 0 <= position < len(chunks) and chunks[position].chunk_id not in cited:
            cited.append(chunks[position].chunk_id)
    return Answer(text.strip(), tuple(cited), False)


class AnthropicGenerator:
    def __init__(
        self, conn: psycopg.Connection, model: str, token_budget: int, max_tokens: int = 400
    ) -> None:
        import anthropic

        self.conn = conn
        self.model = model
        self.token_budget = token_budget
        self.max_tokens = max_tokens
        self.name = f"anthropic:{model}:{PROMPT_VERSION}"
        self._client = anthropic.Anthropic()

    @classmethod
    def from_env(cls, conn: psycopg.Connection) -> "AnthropicGenerator":
        model = os.environ.get("RAG_GEN_MODEL", "").strip()
        if not model:
            raise ValueError("set RAG_GEN_MODEL to the pinned model id before running generation")
        budget = int(os.environ.get("RAG_GEN_TOKEN_BUDGET", "0"))
        if budget <= 0:
            raise ValueError("set RAG_GEN_TOKEN_BUDGET to a positive spending cap in tokens")
        return cls(conn, model, budget)

    def _spent(self) -> int:
        row = self.conn.execute(
            "SELECT coalesce(sum(input_tokens + output_tokens), 0) AS spent FROM answer_cache "
            "WHERE model = %s",
            (self.model,),
        ).fetchone()
        return int(row["spent"])

    def generate(self, question: str, chunks: Sequence[ChunkRecord]) -> Answer:
        prompt = build_prompt(question, chunks)
        key = sha256_text(canonical_json([PROMPT_VERSION, self.model, SYSTEM_PROMPT, prompt]))
        cached = self.conn.execute(
            "SELECT response FROM answer_cache WHERE input_sha256 = %s", (key,)
        ).fetchone()
        if cached:
            return parse_response(cached["response"]["text"], chunks)
        if self._spent() >= self.token_budget:
            raise BudgetExceeded(f"token budget of {self.token_budget} reached for {self.model}")
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        self.conn.execute(
            "INSERT INTO answer_cache (input_sha256, model, response, input_tokens, output_tokens) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            (
                key,
                self.model,
                json.dumps({"text": text}),
                message.usage.input_tokens,
                message.usage.output_tokens,
            ),
        )
        return parse_response(text, chunks)
