import io
import json
import time
from dataclasses import dataclass

import numpy as np
import psycopg
from psycopg import sql

from rag.db import parse_vector, vector_literal
from rag.hashing import sha256_text
from rag.index.embed import Embedder
from rag.index.identity import index_id, index_identity
from rag.ingest.load import corpus_doc_shas, ensure_chunks
from rag.specs import IndexSpec

SUPPORTED_DIM = 384


class IndexBuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class BuildResult:
    index_id: str
    created: bool
    n_chunks: int
    build_seconds: float


def partition_name(iid: str) -> str:
    return f"embeddings_{iid}"


def _cached_embeddings(
    conn: psycopg.Connection, embedder: Embedder, texts: list[str]
) -> np.ndarray:
    model_key = embedder.spec.model_key
    shas = [sha256_text(t) for t in texts]
    rows = conn.execute(
        "SELECT text_sha256, embedding::text AS v FROM embedding_cache "
        "WHERE model_key = %s AND text_sha256 = ANY(%s)",
        (model_key, sorted(set(shas))),
    ).fetchall()
    known = {r["text_sha256"]: parse_vector(r["v"]) for r in rows}
    missing = sorted({s: t for s, t in zip(shas, texts, strict=True) if s not in known}.items())
    batch = embedder.spec.batch_size
    for start in range(0, len(missing), batch):
        part = missing[start : start + batch]
        vectors = embedder.embed_documents([t for _, t in part])
        with conn.transaction(), conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO embedding_cache (model_key, text_sha256, embedding) "
                "VALUES (%s, %s, %s::vector) ON CONFLICT DO NOTHING",
                [
                    (model_key, s, vector_literal(v))
                    for (s, _), v in zip(part, vectors, strict=True)
                ],
            )
        for (s, _), v in zip(part, vectors, strict=True):
            known[s] = np.asarray(v, dtype=np.float32)
    if not texts:
        return np.zeros((0, embedder.spec.dim), dtype=np.float32)
    return np.stack([known[s] for s in shas])


def build_index(
    conn: psycopg.Connection, corpus_sha256: str, spec: IndexSpec, embedder: Embedder
) -> BuildResult:
    if spec.embedder.dim != SUPPORTED_DIM:
        raise IndexBuildError(f"schema stores vector({SUPPORTED_DIM}); got dim {spec.embedder.dim}")
    if embedder.spec.identity() != spec.embedder.identity():
        raise IndexBuildError("embedder does not match the index spec")
    identity = index_identity(corpus_sha256, spec)
    iid = index_id(identity)
    existing = conn.execute(
        "SELECT n_chunks, build_seconds FROM index_versions WHERE index_id = %s", (iid,)
    ).fetchone()
    if existing:
        return BuildResult(iid, False, existing["n_chunks"], existing["build_seconds"])

    doc_shas = corpus_doc_shas(conn, corpus_sha256)
    if not doc_shas:
        raise IndexBuildError(f"corpus {corpus_sha256} is not registered; run ingest first")
    started = time.perf_counter()
    for doc_sha in doc_shas:
        ensure_chunks(conn, doc_sha, spec.chunker)
    rows = conn.execute(
        "SELECT chunk_id, embed_text FROM chunks WHERE chunker_key = %s AND doc_sha256 = ANY(%s) "
        "ORDER BY chunk_id",
        (spec.chunker.key, doc_shas),
    ).fetchall()
    if not rows:
        raise IndexBuildError("no chunks to index")
    vectors = _cached_embeddings(conn, embedder, [r["embed_text"] for r in rows])

    part = sql.Identifier(partition_name(iid))
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (iid,))
        if conn.execute("SELECT 1 FROM index_versions WHERE index_id = %s", (iid,)).fetchone():
            return BuildResult(iid, False, len(rows), 0.0)
        conn.execute(
            sql.SQL(
                "CREATE TABLE {} (LIKE embeddings INCLUDING DEFAULTS INCLUDING CONSTRAINTS, "
                "CHECK (index_id = {}))"
            ).format(part, sql.Literal(iid))
        )
        buffer = io.StringIO()
        for row, vec in zip(rows, vectors, strict=True):
            buffer.write(f"{iid}\t{row['chunk_id']}\t{vector_literal(vec)}\n")
        with (
            conn.cursor() as cur,
            cur.copy(
                sql.SQL("COPY {} (index_id, chunk_id, embedding) FROM STDIN").format(part)
            ) as copy,
        ):
            copy.write(buffer.getvalue())
        conn.execute(
            sql.SQL(
                "CREATE INDEX ON {} USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = {}, ef_construction = {})"
            ).format(part, sql.Literal(spec.hnsw_m), sql.Literal(spec.hnsw_ef_construction))
        )
        build_seconds = round(time.perf_counter() - started, 3)
        conn.execute(
            "INSERT INTO index_versions (index_id, identity, corpus_sha256, chunker_key, "
            "embed_model, embed_revision, dim, params, n_chunks, build_seconds) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                iid,
                json.dumps(identity),
                corpus_sha256,
                spec.chunker.key,
                spec.embedder.model,
                spec.embedder.revision,
                spec.embedder.dim,
                json.dumps(spec.params()),
                len(rows),
                build_seconds,
            ),
        )
        conn.execute(
            sql.SQL("ALTER TABLE embeddings ATTACH PARTITION {} FOR VALUES IN ({})").format(
                part, sql.Literal(iid)
            )
        )
        conn.execute(sql.SQL("GRANT SELECT ON {} TO rag_reader").format(part))
        conn.execute(sql.SQL("ANALYZE {}").format(part))
    return BuildResult(iid, True, len(rows), build_seconds)
