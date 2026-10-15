from dataclasses import replace

import pytest

from rag.index.build import IndexBuildError, build_index
from rag.index.embed import HashingEmbedder
from rag.retrieve.retriever import Retriever
from rag.retrieve.types import fetch_chunks

from conftest import FIXTURE_INDEX, VECTOR
from helpers import ingest_and_build


def test_rebuilding_the_same_inputs_is_a_no_op(conn, fixture_corpus):
    manifest, raw = fixture_corpus
    first = ingest_and_build(conn, manifest, raw)
    assert first.created and first.n_chunks > 0
    again = build_index(
        conn, manifest.corpus_sha256, FIXTURE_INDEX, HashingEmbedder(FIXTURE_INDEX.embedder)
    )
    assert not again.created
    assert again.index_id == first.index_id
    rows = conn.execute("SELECT count(*) AS n FROM index_versions").fetchone()["n"]
    assert rows == 1
    stored = conn.execute(
        "SELECT count(*) AS n FROM embeddings WHERE index_id = %s", (first.index_id,)
    ).fetchone()["n"]
    assert stored == first.n_chunks


def test_changing_an_input_builds_a_new_index_with_recorded_build_time(conn, fixture_corpus):
    manifest, raw = fixture_corpus
    first = ingest_and_build(conn, manifest, raw)
    other_spec = replace(FIXTURE_INDEX, chunker=replace(FIXTURE_INDEX.chunker, target_tokens=60))
    second = ingest_and_build(conn, manifest, raw, other_spec)
    assert second.created and second.index_id != first.index_id
    rows = conn.execute("SELECT index_id, build_seconds FROM index_versions").fetchall()
    assert {r["index_id"] for r in rows} == {first.index_id, second.index_id}
    assert all(r["build_seconds"] >= 0 for r in rows)


def test_each_index_version_gets_its_own_hnsw_partition(conn, fixture_corpus):
    manifest, raw = fixture_corpus
    result = ingest_and_build(conn, manifest, raw)
    index = conn.execute(
        "SELECT indexdef FROM pg_indexes WHERE tablename = %s AND indexdef LIKE %s",
        (f"embeddings_{result.index_id}", "%hnsw%"),
    ).fetchone()
    assert index is not None


def test_embedder_must_match_the_spec(conn, fixture_corpus):
    manifest, raw = fixture_corpus
    ingest_and_build(conn, manifest, raw)
    other = replace(FIXTURE_INDEX.embedder, revision="v9")
    with pytest.raises(IndexBuildError, match="does not match"):
        build_index(conn, manifest.corpus_sha256, FIXTURE_INDEX, HashingEmbedder(other))


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("how often does a checkpoint run", "storage#storage-wal-checkpoints"),
        ("deadlock_timeout default", "locking#locking-deadlocks"),
        ("index for full-text search on word lists", "search#search-inverted"),
    ],
)
def test_hand_picked_queries_find_their_section_in_the_top_5(conn, fixture_corpus, query, expected):
    manifest, raw = fixture_corpus
    result = ingest_and_build(conn, manifest, raw)
    retriever = Retriever(conn, result.index_id, VECTOR, HashingEmbedder(FIXTURE_INDEX.embedder))
    hits = retriever.search(query, 5)
    refs = [c.section_ref for c in fetch_chunks(conn, [h.chunk_id for h in hits])]
    assert expected in refs
