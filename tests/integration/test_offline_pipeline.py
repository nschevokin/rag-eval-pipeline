from fastapi.testclient import TestClient

from rag.api.app import create_app
from rag.evaluate.gate import GateConfig, check_gate, floors_from_baseline
from rag.generate.extractive import ExtractiveGenerator
from rag.publish.switch import publish

from conftest import VECTOR
from helpers import evaluate_dev, ingest_and_build


def test_end_to_end_offline_pipeline_serves_cited_answers(conn, db_url, fixture_corpus, tmp_path):
    manifest, raw = fixture_corpus
    built = ingest_and_build(conn, manifest, raw)
    run = evaluate_dev(conn, tmp_path, built.index_id, VECTOR)
    assert run.metrics["recall_at_10"] >= 0.5
    decision = check_gate(
        conn, run.eval_id, GateConfig(floors_from_baseline(run.metrics, 0.05), 0.02)
    )
    assert decision.passed
    publish(conn, decision.decision_id, "ci")

    client = TestClient(create_app(db_url, generator_factory=lambda _: ExtractiveGenerator()))
    health = client.get("/health").json()
    assert health == {"status": "ok", "index_id": built.index_id, "retriever": "vector"}

    body = client.post("/ask", json={"question": "How often does a checkpoint run by default?"})
    assert body.status_code == 200
    answer = body.json()
    assert answer["index_id"] == built.index_id
    assert not answer["abstained"]
    assert "5 minutes" in answer["answer"]
    assert answer["citations"]
    assert answer["citations"][0]["source_url"].startswith("https://example.test/")

    unrelated = client.post("/ask", json={"question": "Who won the football world cup?"}).json()
    assert unrelated["abstained"]
    assert unrelated["citations"] == []


def test_api_reports_when_nothing_is_published(db_url):
    client = TestClient(create_app(db_url, generator_factory=lambda _: ExtractiveGenerator()))
    assert client.get("/health").json()["index_id"] is None
    assert client.post("/ask", json={"question": "anything"}).status_code == 503
    assert client.post("/ask", json={"question": ""}).status_code == 422
