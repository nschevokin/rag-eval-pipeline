import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import psycopg

from rag.hashing import canonical_json, sha256_text
from rag.ingest.chunk import chunk_sections
from rag.ingest.manifest import Manifest, ManifestFile, verify_corpus
from rag.ingest.parse import PARSER_VERSION, Section, parse_html
from rag.specs import ChunkerSpec


@dataclass
class IngestReport:
    corpus_sha256: str
    documents_added: int = 0
    sections_added: int = 0
    chunks_added: dict[str, int] = field(default_factory=dict)

    @property
    def rows_added(self) -> int:
        return self.documents_added + self.sections_added + sum(self.chunks_added.values())


def register_corpus(conn: psycopg.Connection, manifest: Manifest) -> str:
    corpus_sha = manifest.corpus_sha256
    with conn.transaction():
        inserted = conn.execute(
            "INSERT INTO corpora (corpus_sha256, name, manifest) VALUES (%s, %s, %s) "
            "ON CONFLICT DO NOTHING RETURNING corpus_sha256",
            (corpus_sha, manifest.corpus, json.dumps(manifest.to_dict())),
        ).fetchone()
        if inserted:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO corpus_documents (corpus_sha256, doc_sha256, name) "
                    "VALUES (%s, %s, %s)",
                    [(corpus_sha, f.sha256, f.name) for f in manifest.files],
                )
    return corpus_sha


def _has_receipt(conn: psycopg.Connection, doc_sha: str, stage: str, key: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM ingest_receipts WHERE doc_sha256 = %s AND stage = %s AND stage_key = %s",
        (doc_sha, stage, key),
    ).fetchone()
    return row is not None


def section_id(doc_sha: str, ord_: int) -> str:
    return sha256_text(canonical_json([PARSER_VERSION, doc_sha, ord_]))[:32]


def ensure_sections(
    conn: psycopg.Connection, entry: ManifestFile, raw_dir: Path
) -> tuple[int, int]:
    if _has_receipt(conn, entry.sha256, "sections", PARSER_VERSION):
        return 0, 0
    html = (raw_dir / entry.name).read_text(encoding="utf-8", errors="replace")
    page = parse_html(html, entry.name)
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (entry.sha256,))
        if _has_receipt(conn, entry.sha256, "sections", PARSER_VERSION):
            return 0, 0
        doc = conn.execute(
            "INSERT INTO documents (doc_sha256, name, source_url, title, n_bytes) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING RETURNING doc_sha256",
            (entry.sha256, entry.name, entry.url, page.title, entry.bytes),
        ).fetchone()
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO sections (section_id, doc_sha256, parser_version, ord, section_ref, "
                "section_refs, heading_path, body) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                [
                    (
                        section_id(entry.sha256, s.ord),
                        entry.sha256,
                        PARSER_VERSION,
                        s.ord,
                        s.ref,
                        list(s.refs),
                        list(s.heading_path),
                        s.body,
                    )
                    for s in page.sections
                ],
            )
        conn.execute(
            "INSERT INTO ingest_receipts (doc_sha256, stage, stage_key, n_rows) "
            "VALUES (%s, 'sections', %s, %s)",
            (entry.sha256, PARSER_VERSION, len(page.sections)),
        )
    return (1 if doc else 0), len(page.sections)


def load_sections(conn: psycopg.Connection, doc_sha: str) -> list[Section]:
    rows = conn.execute(
        "SELECT ord, section_ref, section_refs, heading_path, body FROM sections "
        "WHERE doc_sha256 = %s AND parser_version = %s ORDER BY ord",
        (doc_sha, PARSER_VERSION),
    ).fetchall()
    return [
        Section(
            r["ord"],
            r["section_ref"],
            tuple(r["section_refs"]),
            tuple(r["heading_path"]),
            r["body"],
        )
        for r in rows
    ]


def ensure_chunks(conn: psycopg.Connection, doc_sha: str, spec: ChunkerSpec) -> int:
    if _has_receipt(conn, doc_sha, "chunks", spec.key):
        return 0
    chunks = chunk_sections(spec, doc_sha, load_sections(conn, doc_sha))
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (doc_sha + spec.key,))
        if _has_receipt(conn, doc_sha, "chunks", spec.key):
            return 0
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (chunk_id, chunker_key, doc_sha256, ord, section_ref, "
                "section_refs, heading_path, n_tokens, body, embed_text) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [
                    (
                        c.chunk_id,
                        spec.key,
                        doc_sha,
                        c.ord,
                        c.section_ref,
                        list(c.section_refs),
                        list(c.heading_path),
                        c.n_tokens,
                        c.body,
                        c.embed_text,
                    )
                    for c in chunks
                ],
            )
        conn.execute(
            "INSERT INTO ingest_receipts (doc_sha256, stage, stage_key, n_rows) "
            "VALUES (%s, 'chunks', %s, %s)",
            (doc_sha, spec.key, len(chunks)),
        )
    return len(chunks)


def ingest(
    conn: psycopg.Connection,
    manifest: Manifest,
    raw_dir: Path,
    chunkers: Iterable[ChunkerSpec],
) -> IngestReport:
    verify_corpus(manifest, raw_dir)
    chunkers = list(chunkers)
    report = IngestReport(corpus_sha256=register_corpus(conn, manifest))
    for entry in manifest.files:
        docs, sections = ensure_sections(conn, entry, raw_dir)
        report.documents_added += docs
        report.sections_added += sections
        for spec in chunkers:
            added = ensure_chunks(conn, entry.sha256, spec)
            report.chunks_added[spec.key] = report.chunks_added.get(spec.key, 0) + added
    return report


def corpus_doc_shas(conn: psycopg.Connection, corpus_sha: str) -> list[str]:
    rows = conn.execute(
        "SELECT doc_sha256 FROM corpus_documents WHERE corpus_sha256 = %s ORDER BY name",
        (corpus_sha,),
    ).fetchall()
    return [r["doc_sha256"] for r in rows]


def corpus_section_refs(conn: psycopg.Connection, corpus_sha: str) -> set[str]:
    rows = conn.execute(
        "SELECT DISTINCT unnest(s.section_refs) AS ref FROM sections s "
        "JOIN corpus_documents cd ON cd.doc_sha256 = s.doc_sha256 "
        "WHERE cd.corpus_sha256 = %s AND s.parser_version = %s",
        (corpus_sha, PARSER_VERSION),
    ).fetchall()
    return {r["ref"] for r in rows}
