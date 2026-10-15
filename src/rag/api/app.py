from collections.abc import Callable, Iterator
from typing import Annotated

import psycopg
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.config import database_url
from rag.db import connect
from rag.generate.base import Generator, make_generator
from rag.index.embed import make_embedder
from rag.publish.switch import current
from rag.retrieve.retriever import Retriever
from rag.retrieve.types import fetch_chunks
from rag.specs import EmbedderSpec, RetrieverSpec


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    chunk_id: str
    section_ref: str
    heading_path: list[str]
    source_url: str


class AskResponse(BaseModel):
    answer: str
    abstained: bool
    citations: list[Citation]
    index_id: str
    retriever: str
    generator: str


def create_app(
    url: str | None = None,
    generator_factory: Callable[[psycopg.Connection], Generator] | None = None,
) -> FastAPI:
    app = FastAPI(title="rag-eval-pipeline", version="0.1.0")
    db_url = url or database_url()
    make_gen = generator_factory or (lambda conn: make_generator(conn=conn))

    def get_conn() -> Iterator[psycopg.Connection]:
        conn = connect(db_url)
        try:
            yield conn
        finally:
            conn.close()

    @app.get("/health")
    def health(conn: Annotated[psycopg.Connection, Depends(get_conn)]) -> dict:
        conn.execute("SELECT 1")
        served = current(conn)
        return {
            "status": "ok",
            "index_id": served.index_id if served else None,
            "retriever": served.retriever.get("name") if served else None,
        }

    @app.post("/ask", response_model=AskResponse)
    def ask(
        body: AskRequest, conn: Annotated[psycopg.Connection, Depends(get_conn)]
    ) -> AskResponse:
        served = current(conn)
        if served is None:
            raise HTTPException(status_code=503, detail="no index has been published yet")
        identity = conn.execute(
            "SELECT identity FROM index_versions WHERE index_id = %s", (served.index_id,)
        ).fetchone()["identity"]
        embedder = make_embedder(EmbedderSpec.from_dict(identity["embedder"]))
        spec = RetrieverSpec.from_dict(served.retriever)
        hits = Retriever(conn, served.index_id, spec, embedder).search(body.question, body.k)
        chunks = fetch_chunks(conn, [h.chunk_id for h in hits])
        generator = make_gen(conn)
        answer = generator.generate(body.question, chunks)
        by_id = {c.chunk_id: c for c in chunks}
        citations = [
            Citation(
                chunk_id=cid,
                section_ref=by_id[cid].section_ref,
                heading_path=list(by_id[cid].heading_path),
                source_url=by_id[cid].source_url,
            )
            for cid in answer.cited_chunk_ids
            if cid in by_id
        ]
        return AskResponse(
            answer=answer.text,
            abstained=answer.abstained,
            citations=citations,
            index_id=served.index_id,
            retriever=spec.name or served.retriever_key,
            generator=generator.name,
        )

    return app
