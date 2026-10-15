import json
from dataclasses import replace

import pytest

from rag.evaluate.confirm import TestRunRefused, confirm_on_test
from rag.evaluate.questions import load_question_file, make_split, write_split
from rag.evaluate.report import ablation_markdown
from rag.evaluate.runner import run_evaluation
from rag.generate.extractive import ExtractiveGenerator
from rag.index.embed import HashingEmbedder
from rag.specs import RetrieverSpec

from conftest import FIXTURE_INDEX, FIXTURE_QUESTIONS, VECTOR
from helpers import evaluate_dev, ingest_and_build

FULLTEXT = RetrieverSpec(name="fulltext", mode="fulltext")
HYBRID = RetrieverSpec(name="hybrid", mode="hybrid")


def test_running_the_same_index_twice_gives_identical_scores(conn, fixture_corpus, tmp_path):
    manifest, raw = fixture_corpus
    built = ingest_and_build(conn, manifest, raw)
    first = evaluate_dev(conn, tmp_path, built.index_id, VECTOR)
    second = evaluate_dev(conn, tmp_path, built.index_id, VECTOR)
    assert first.metrics == second.metrics
    assert first.run_key == second.run_key
    assert first.eval_id != second.eval_id
    evidence = json.loads(first.evidence_path.read_text())
    assert evidence["metrics"] == first.metrics
    assert evidence["labels"] == {"index": FIXTURE_INDEX.name, "retriever": "vector"}
    stored = conn.execute(
        "SELECT count(*) AS n FROM eval_results WHERE eval_id = %s", (first.eval_id,)
    ).fetchone()["n"]
    assert stored == 12


@pytest.mark.parametrize("retriever", [VECTOR, FULLTEXT, HYBRID], ids=lambda r: r.name)
def test_every_retrieval_mode_produces_scores(conn, fixture_corpus, tmp_path, retriever):
    manifest, raw = fixture_corpus
    built = ingest_and_build(conn, manifest, raw)
    run = evaluate_dev(conn, tmp_path, built.index_id, retriever)
    assert 0.0 <= run.metrics["recall_at_5"] <= 1.0
    assert set(run.metrics["by_type"]) == {"factual", "multi_section", "paraphrase"}
    assert run.metrics["n_answerable"] == 10


def test_answers_are_scored_when_a_generator_is_given(conn, fixture_corpus, tmp_path):
    manifest, raw = fixture_corpus
    built = ingest_and_build(conn, manifest, raw)
    questions = load_question_file(FIXTURE_QUESTIONS)
    run = run_evaluation(
        conn,
        index_id=built.index_id,
        retriever_spec=HYBRID,
        questions=questions,
        split="dev",
        embedder=HashingEmbedder(FIXTURE_INDEX.embedder),
        evidence_dir=tmp_path / "evidence",
        generator=ExtractiveGenerator(),
    )
    for metric in (
        "key_fact_coverage",
        "citation_validity",
        "false_abstention",
        "abstention_accuracy",
    ):
        assert 0.0 <= run.metrics[metric] <= 1.0
    assert run.evidence_path.name.endswith("-answers.json")


def test_ablation_table_is_generated_from_evidence(conn, fixture_corpus, tmp_path):
    manifest, raw = fixture_corpus
    built = ingest_and_build(conn, manifest, raw)
    other = ingest_and_build(
        conn,
        manifest,
        raw,
        replace(
            FIXTURE_INDEX, name="fx-60", chunker=replace(FIXTURE_INDEX.chunker, target_tokens=60)
        ),
    )
    evaluate_dev(conn, tmp_path, built.index_id, VECTOR)
    evaluate_dev(conn, tmp_path, built.index_id, FULLTEXT)
    evaluate_dev(conn, tmp_path, other.index_id, VECTOR)
    table = ablation_markdown(tmp_path / "evidence")
    assert table.count(f"`{built.index_id}`") == 2
    assert f"`{other.index_id}`" in table


def test_test_set_runs_once_per_decision(conn, fixture_corpus, tmp_path):
    manifest, raw = fixture_corpus
    built = ingest_and_build(conn, manifest, raw)
    questions = load_question_file(FIXTURE_QUESTIONS)
    split_path = tmp_path / "split.json"
    write_split(split_path, make_split(questions, seed=3, dev_fraction=0.5))
    decisions = tmp_path / "decisions.md"
    decisions.write_text("# Decisions\n\n## D-010 Choose vector retrieval\n\nreason\n")
    kwargs = dict(
        questions_path=FIXTURE_QUESTIONS,
        split_path=split_path,
        decisions_md=decisions,
        evidence_dir=tmp_path / "evidence",
        index_id=built.index_id,
        retriever_spec=VECTOR,
        embedder=HashingEmbedder(FIXTURE_INDEX.embedder),
    )
    with pytest.raises(TestRunRefused, match="not a heading"):
        confirm_on_test(conn, decision="D-011", **kwargs)
    result = confirm_on_test(conn, decision="D-010", **kwargs)
    assert result.evidence_path.name.startswith("test-")
    with pytest.raises(TestRunRefused, match="already authorised"):
        confirm_on_test(conn, decision="D-010", **kwargs)
    history = json.loads((tmp_path / "evidence" / "test_confirmation.json").read_text())
    assert [h["decision"] for h in history] == ["D-010"]
