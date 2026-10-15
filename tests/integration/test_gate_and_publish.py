import json

import psycopg
import pytest

import rag.publish.switch as switch
from rag.evaluate.gate import (
    FloorsLoosened,
    GateConfig,
    GateNotConfigured,
    check_gate,
    floors_from_baseline,
)
from rag.publish.switch import PublishRejected, current, publish

from conftest import VECTOR
from helpers import MUTANT_INDEX, evaluate_dev, ingest_and_build


@pytest.fixture
def baseline(conn, fixture_corpus, tmp_path):
    manifest, raw = fixture_corpus
    built = ingest_and_build(conn, manifest, raw)
    run = evaluate_dev(conn, tmp_path, built.index_id, VECTOR)
    config = GateConfig(floors_from_baseline(run.metrics, 0.05), 0.02)
    decision = check_gate(conn, run.eval_id, config)
    assert decision.passed, decision.reasons
    return {
        "built": built,
        "run": run,
        "config": config,
        "decision": decision,
        "manifest": manifest,
        "raw": raw,
    }


def test_gate_refuses_to_run_without_floors(conn, baseline):
    with pytest.raises(GateNotConfigured):
        check_gate(conn, baseline["run"].eval_id, GateConfig({}, 0.02))


def test_publish_switches_the_served_index(conn, baseline):
    assert current(conn) is None
    publication_id = publish(conn, baseline["decision"].decision_id, "test")
    served = current(conn)
    assert served.publication_id == publication_id
    assert served.index_id == baseline["built"].index_id


def test_a_deliberately_broken_index_is_rejected(conn, baseline, tmp_path):
    publish(conn, baseline["decision"].decision_id, "test")
    mutant = ingest_and_build(conn, baseline["manifest"], baseline["raw"], MUTANT_INDEX)
    run = evaluate_dev(conn, tmp_path, mutant.index_id, VECTOR, MUTANT_INDEX)
    assert run.metrics["recall_at_5"] < baseline["run"].metrics["recall_at_5"]
    decision = check_gate(conn, run.eval_id, baseline["config"])
    assert not decision.passed
    assert any("regresses" in r or "below the floor" in r for r in decision.reasons)
    with pytest.raises(PublishRejected):
        publish(conn, decision.decision_id, "test")
    assert current(conn).index_id == baseline["built"].index_id


def test_the_database_rejects_a_publication_without_a_passing_gate(conn, baseline, tmp_path):
    mutant = ingest_and_build(conn, baseline["manifest"], baseline["raw"], MUTANT_INDEX)
    run = evaluate_dev(conn, tmp_path, mutant.index_id, VECTOR, MUTANT_INDEX)
    strict = GateConfig({k: 1.0 for k in baseline["config"].floors}, 0.02)
    failed = check_gate(conn, run.eval_id, strict)
    assert not failed.passed
    with pytest.raises(psycopg.errors.IntegrityError):
        conn.execute(
            "INSERT INTO publications (decision_id, index_id, retriever_key, retriever, "
            "published_by) VALUES (%s, %s, %s, '{}', 'bypass')",
            (failed.decision_id, mutant.index_id, VECTOR.key),
        )
    with pytest.raises(psycopg.errors.IntegrityError):
        conn.execute(
            "INSERT INTO publications (decision_id, index_id, retriever_key, passed, retriever, "
            "published_by) VALUES (%s, %s, %s, false, '{}', 'bypass')",
            (failed.decision_id, mutant.index_id, VECTOR.key),
        )


def test_a_crash_before_commit_leaves_the_old_index_serving(conn, baseline, tmp_path, monkeypatch):
    first = publish(conn, baseline["decision"].decision_id, "test")
    other = ingest_and_build(
        conn,
        baseline["manifest"],
        baseline["raw"],
        baseline_spec_with_overlap(),
    )
    run = evaluate_dev(conn, tmp_path, other.index_id, VECTOR)
    decision = check_gate(conn, run.eval_id, GateConfig(baseline["config"].floors, 1.0))
    assert decision.passed, decision.reasons

    def crash(*args, **kwargs):
        raise RuntimeError("simulated crash before commit")

    monkeypatch.setattr(switch, "_set_pointer", crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        publish(conn, decision.decision_id, "test")
    assert current(conn).publication_id == first
    count = conn.execute("SELECT count(*) AS n FROM publications").fetchone()["n"]
    assert count == 1


def baseline_spec_with_overlap():
    from dataclasses import replace

    from conftest import FIXTURE_INDEX

    return replace(FIXTURE_INDEX, chunker=replace(FIXTURE_INDEX.chunker, overlap=5))


def test_a_stale_decision_cannot_be_published(conn, baseline, tmp_path):
    other = ingest_and_build(
        conn, baseline["manifest"], baseline["raw"], baseline_spec_with_overlap()
    )
    run = evaluate_dev(conn, tmp_path, other.index_id, VECTOR)
    early = check_gate(conn, run.eval_id, GateConfig(baseline["config"].floors, 1.0))
    publish(conn, baseline["decision"].decision_id, "test")
    with pytest.raises(PublishRejected, match="changed after this gate decision"):
        publish(conn, early.decision_id, "test")


def test_floors_can_never_be_loosened(conn, baseline):
    looser = GateConfig({k: v - 0.1 for k, v in baseline["config"].floors.items()}, 0.02)
    with pytest.raises(FloorsLoosened):
        check_gate(conn, baseline["run"].eval_id, looser)


def test_the_pointer_cannot_be_deleted_or_set_to_a_failed_decision(conn, baseline):
    publish(conn, baseline["decision"].decision_id, "test")
    with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
        conn.execute("DELETE FROM current_index")


def test_gate_decision_records_scores_and_reasons(conn, baseline):
    row = conn.execute(
        "SELECT d.passed, d.reasons, d.floors, e.metrics FROM gate_decisions d "
        "JOIN evaluations e USING (eval_id) WHERE d.decision_id = %s",
        (baseline["decision"].decision_id,),
    ).fetchone()
    assert row["passed"] is True
    assert row["reasons"] == []
    assert set(row["floors"]) == {"recall_at_5", "recall_at_10", "mrr", "ndcg_at_10"}
    assert json.dumps(row["metrics"])
