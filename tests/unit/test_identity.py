from dataclasses import replace

from rag.index.identity import index_id, index_identity

from conftest import FIXTURE_INDEX

CORPUS = "c" * 64


def iid(spec=FIXTURE_INDEX, corpus=CORPUS):
    return index_id(index_identity(corpus, spec))


def test_same_inputs_give_the_same_id():
    assert iid() == iid()
    assert len(iid()) == 16


def test_preset_name_is_not_part_of_the_identity():
    assert iid(replace(FIXTURE_INDEX, name="renamed")) == iid()


def test_any_input_change_gives_a_new_id():
    spec = FIXTURE_INDEX
    variants = [
        iid(corpus="d" * 64),
        iid(replace(spec, chunker=replace(spec.chunker, target_tokens=41))),
        iid(replace(spec, chunker=replace(spec.chunker, version="chunker-v2"))),
        iid(replace(spec, embedder=replace(spec.embedder, revision="v2"))),
        iid(replace(spec, embedder=replace(spec.embedder, model="other"))),
        iid(replace(spec, embedder=replace(spec.embedder, query_prefix="q: "))),
        iid(replace(spec, hnsw_m=32)),
        iid(replace(spec, hnsw_ef_construction=128)),
    ]
    assert iid() not in variants
    assert len(set(variants)) == len(variants)


def test_batch_size_does_not_change_the_identity():
    spec = replace(FIXTURE_INDEX, embedder=replace(FIXTURE_INDEX.embedder, batch_size=8))
    assert iid(spec) == iid()
