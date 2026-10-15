from dataclasses import replace

import pytest

from rag.db import discover_migrations
from rag.evaluate.runner import run_key
from rag.specs import RetrieverSpec, load_experiments

from conftest import ROOT, VECTOR


def test_experiment_presets_load():
    experiments = load_experiments(ROOT / "experiments.toml")
    assert experiments.index().name == "heading-400"
    assert experiments.index().embedder.revision
    assert {"vector", "fulltext", "hybrid"} <= set(experiments.retrievers)
    assert len(experiments.indexes) + len(experiments.retrievers) - 1 >= 6


def test_reranker_requires_a_pinned_revision():
    with pytest.raises(ValueError, match="pin rerank_revision"):
        RetrieverSpec(name="r", mode="hybrid", rerank_model="some/cross-encoder")


def test_reranker_switch_changes_the_run_identity():
    hybrid = RetrieverSpec(name="hybrid", mode="hybrid")
    reranked = replace(hybrid, rerank_model="m", rerank_revision="abc", rerank_top=20)
    assert hybrid.key != reranked.key
    assert run_key("i", hybrid, "dev", "q", None) != run_key("i", reranked, "dev", "q", None)


def test_retriever_name_is_a_label_not_identity():
    assert replace(VECTOR, name="renamed").key == VECTOR.key


def test_migration_files_are_numbered_without_gaps():
    assert [m.version for m in discover_migrations(ROOT / "migrations")] == [1, 2, 3]
