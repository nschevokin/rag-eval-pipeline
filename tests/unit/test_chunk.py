from dataclasses import replace

import pytest

from rag.ingest.chunk import chunk_sections, windows
from rag.ingest.parse import parse_html
from rag.ingest.tokens import count_tokens
from rag.specs import ChunkerSpec

from conftest import FIXTURE_CORPUS, HEADING_SMALL

DOC = "a" * 64


def sections(name: str = "storage.html"):
    return parse_html((FIXTURE_CORPUS / name).read_text(encoding="utf-8"), name).sections


def all_sections():
    return [s for p in sorted(FIXTURE_CORPUS.glob("*.html")) for s in sections(p.name)]


def test_windows_cover_every_token_with_overlap():
    assert windows(0, 10, 2) == []
    assert windows(5, 10, 2) == [(0, 5)]
    assert windows(25, 10, 2) == [(0, 10), (8, 18), (16, 25)]


def test_same_input_twice_gives_identical_chunk_ids():
    first = chunk_sections(HEADING_SMALL, DOC, sections())
    second = chunk_sections(HEADING_SMALL, DOC, sections())
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)


def test_chunker_parameters_change_the_ids():
    base = chunk_sections(HEADING_SMALL, DOC, sections())
    other = chunk_sections(replace(HEADING_SMALL, overlap=5), DOC, sections())
    assert {c.chunk_id for c in base}.isdisjoint({c.chunk_id for c in other})


def test_no_heading_chunk_crosses_a_top_level_heading():
    secs = all_sections()
    body_of = {s.ref: s.body for s in secs}
    chunks = chunk_sections(HEADING_SMALL, DOC, secs)
    assert len(chunks) > len(secs)
    for c in chunks:
        assert c.section_refs[-1] == c.section_ref
        assert c.body in body_of[c.section_ref]


def test_chunks_respect_the_token_target():
    for c in chunk_sections(HEADING_SMALL, DOC, all_sections()):
        assert 0 < c.n_tokens <= HEADING_SMALL.target_tokens
        assert count_tokens(c.body) == c.n_tokens


def test_heading_path_is_prepended_for_embedding_only():
    chunk = chunk_sections(HEADING_SMALL, DOC, sections())[1]
    assert chunk.embed_text.startswith("Chapter 1. Storage Engine > 1.1. Pages\n")
    assert not chunk.body.startswith("Chapter 1")


def test_fixed_windows_record_every_section_they_touch():
    spec = ChunkerSpec(
        kind="fixed", version="fixed-v1", target_tokens=60, overlap=10, prepend_heading=False
    )
    chunks = chunk_sections(spec, DOC, sections())
    assert any(len({r for r in c.section_refs if "#" in r}) > 2 for c in chunks)
    assert chunks[0].embed_text == chunks[0].body


def test_invalid_spec_is_rejected():
    with pytest.raises(ValueError):
        ChunkerSpec(kind="heading", version="v", target_tokens=10, overlap=10, prepend_heading=True)
