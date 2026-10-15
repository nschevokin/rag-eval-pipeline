import pytest

import rag.ingest.load as load
from rag.db import connect
from rag.ingest.load import corpus_section_refs, ingest

from conftest import FIXTURE_INDEX

TABLES = ("documents", "sections", "chunks", "ingest_receipts", "corpora", "corpus_documents")


def snapshot(conn) -> dict:
    out = {}
    for table in TABLES:
        cols = {
            "documents": "doc_sha256, name, source_url, title, n_bytes",
            "sections": "section_id, doc_sha256, ord, section_ref, section_refs, body",
            "chunks": "chunk_id, chunker_key, doc_sha256, ord, section_refs, body",
            "ingest_receipts": "doc_sha256, stage, stage_key, n_rows",
            "corpora": "corpus_sha256, name, manifest",
            "corpus_documents": "corpus_sha256, doc_sha256, name",
        }[table]
        out[table] = conn.execute(f"SELECT {cols} FROM {table} ORDER BY 1, 2").fetchall()
    return out


def test_second_ingestion_adds_zero_rows(conn, fixture_corpus):
    manifest, raw = fixture_corpus
    first = ingest(conn, manifest, raw, [FIXTURE_INDEX.chunker])
    assert first.documents_added == 3
    assert first.rows_added > 3
    second = ingest(conn, manifest, raw, [FIXTURE_INDEX.chunker])
    assert second.rows_added == 0


def test_killed_run_recovers_to_the_clean_state(
    conn, fixture_corpus, monkeypatch, admin_url, template_db
):
    manifest, raw = fixture_corpus
    real_parse = load.parse_html
    calls = {"n": 0}

    def crash_on_second(html, name):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("simulated kill")
        return real_parse(html, name)

    monkeypatch.setattr(load, "parse_html", crash_on_second)
    with pytest.raises(KeyboardInterrupt):
        ingest(conn, manifest, raw, [FIXTURE_INDEX.chunker])
    monkeypatch.setattr(load, "parse_html", real_parse)
    ingest(conn, manifest, raw, [FIXTURE_INDEX.chunker])
    recovered = snapshot(conn)

    import uuid

    from psycopg.conninfo import make_conninfo

    name = f"rag_clean_{uuid.uuid4().hex[:8]}"
    import psycopg

    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{name}" TEMPLATE "{template_db}"')
    try:
        with connect(make_conninfo(admin_url, dbname=name)) as clean:
            ingest(clean, manifest, raw, [FIXTURE_INDEX.chunker])
            assert snapshot(clean) == recovered
    finally:
        with psycopg.connect(admin_url, autocommit=True) as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def test_ingestion_refuses_a_tampered_corpus(conn, fixture_corpus):
    from rag.ingest.manifest import CorpusVerificationError

    manifest, raw = fixture_corpus
    (raw / "search.html").write_text("tampered")
    with pytest.raises(CorpusVerificationError):
        ingest(conn, manifest, raw, [FIXTURE_INDEX.chunker])
    assert conn.execute("SELECT count(*) AS n FROM documents").fetchone()["n"] == 0


def test_section_refs_include_ancestors(conn, fixture_corpus):
    manifest, raw = fixture_corpus
    ingest(conn, manifest, raw, [FIXTURE_INDEX.chunker])
    refs = corpus_section_refs(conn, manifest.corpus_sha256)
    assert {
        "storage#storage-wal",
        "storage#storage-wal-checkpoints",
        "search#search-partial",
    } <= refs
