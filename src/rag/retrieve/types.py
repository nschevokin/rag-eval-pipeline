from collections.abc import Sequence
from dataclasses import dataclass

import psycopg


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    section_ref: str
    section_refs: tuple[str, ...]
    heading_path: tuple[str, ...]
    body: str
    embed_text: str
    source_url: str


def fetch_chunks(conn: psycopg.Connection, chunk_ids: Sequence[str]) -> list[ChunkRecord]:
    if not chunk_ids:
        return []
    rows = conn.execute(
        "SELECT c.chunk_id, c.section_ref, c.section_refs, c.heading_path, c.body, c.embed_text, "
        "d.source_url FROM chunks c JOIN documents d ON d.doc_sha256 = c.doc_sha256 "
        "WHERE c.chunk_id = ANY(%s)",
        (list(chunk_ids),),
    ).fetchall()
    by_id = {
        r["chunk_id"]: ChunkRecord(
            chunk_id=r["chunk_id"],
            section_ref=r["section_ref"],
            section_refs=tuple(r["section_refs"]),
            heading_path=tuple(r["heading_path"]),
            body=r["body"],
            embed_text=r["embed_text"],
            source_url=r["source_url"],
        )
        for r in rows
    }
    return [by_id[c] for c in chunk_ids if c in by_id]
