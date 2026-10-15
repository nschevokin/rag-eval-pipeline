import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg

from rag.evaluate.answer_metrics import answer_scores
from rag.evaluate.metrics import RETRIEVAL_METRICS, mean_scores, percentile, retrieval_scores
from rag.evaluate.questions import Question, question_set_sha256
from rag.generate.base import Generator
from rag.hashing import short_id
from rag.index.embed import Embedder
from rag.retrieve.rerank import Reranker
from rag.retrieve.retriever import Retriever
from rag.retrieve.types import fetch_chunks
from rag.specs import RetrieverSpec

K_MAX = 20
ANSWER_CONTEXT = 5
ANSWER_METRICS = ("key_fact_coverage", "citation_validity", "false_abstention")


@dataclass(frozen=True)
class RunResult:
    eval_id: int
    run_key: str
    metrics: dict[str, Any]
    timing: dict[str, float]
    evidence_path: Path


def run_key(
    index_id: str, retriever: RetrieverSpec, split: str, qset_sha: str, generator: str | None
) -> str:
    return short_id(
        {
            "index_id": index_id,
            "retriever": retriever.identity(),
            "split": split,
            "question_set_sha256": qset_sha,
            "k_max": K_MAX,
            "generator": generator,
        },
        16,
    )


def evidence_file(
    evidence_dir: Path, split: str, index_id: str, retriever_key: str, generator: str | None
) -> Path:
    suffix = "" if generator is None else "-answers"
    return evidence_dir / "evals" / f"{split}-{index_id}-{retriever_key}{suffix}.json"


def run_evaluation(
    conn: psycopg.Connection,
    *,
    index_id: str,
    retriever_spec: RetrieverSpec,
    questions: Sequence[Question],
    split: str,
    embedder: Embedder,
    evidence_dir: Path,
    reranker: Reranker | None = None,
    generator: Generator | None = None,
    labels: dict[str, str] | None = None,
) -> RunResult:
    if split not in ("dev", "test"):
        raise ValueError("split must be dev or test")
    if not questions:
        raise ValueError("no questions to evaluate")
    index_row = conn.execute(
        "SELECT build_seconds, n_chunks FROM index_versions WHERE index_id = %s", (index_id,)
    ).fetchone()
    if index_row is None:
        raise LookupError(f"index {index_id} does not exist; build it first")

    qset_sha = question_set_sha256(questions)
    gen_name = generator.name if generator else None
    key = run_key(index_id, retriever_spec, split, qset_sha, gen_name)
    retriever = Retriever(conn, index_id, retriever_spec, embedder, reranker)

    per_question: list[dict[str, Any]] = []
    latencies: list[float] = []
    for q in sorted(questions, key=lambda item: item.id):
        started = time.perf_counter()
        hits = retriever.search(q.question, K_MAX)
        latencies.append((time.perf_counter() - started) * 1000)
        chunks = fetch_chunks(conn, [h.chunk_id for h in hits])
        ranked_refs = [set(c.section_refs) for c in chunks]
        row: dict[str, Any] = {
            "id": q.id,
            "type": q.type,
            "retrieved": [c.chunk_id for c in chunks],
            "retrieved_sections": [c.section_ref for c in chunks[:10]],
            "scores": retrieval_scores(ranked_refs, q.gold_sections) if q.answerable else {},
        }
        if generator is not None:
            answer = generator.generate(q.question, chunks[:ANSWER_CONTEXT])
            context = chunks[:ANSWER_CONTEXT]
            row["answer"] = {
                "text": answer.text,
                "cited": list(answer.cited_chunk_ids),
                "abstained": answer.abstained,
            }
            row["scores"].update(
                answer_scores(
                    answer,
                    answerable=q.answerable,
                    key_facts=q.key_facts,
                    retrieved_ids={c.chunk_id for c in context},
                    chunk_refs={c.chunk_id: c.section_refs for c in context},
                    gold=q.gold_sections,
                )
            )
        per_question.append(row)

    answerable = [r["scores"] for r in per_question if r["type"] != "unanswerable"]
    metrics: dict[str, Any] = dict(mean_scores(answerable, RETRIEVAL_METRICS))
    by_type: dict[str, dict[str, float]] = {}
    for qtype in sorted({r["type"] for r in per_question if r["type"] != "unanswerable"}):
        rows = [r["scores"] for r in per_question if r["type"] == qtype]
        by_type[qtype] = {**mean_scores(rows, RETRIEVAL_METRICS), "n": len(rows)}
    metrics["by_type"] = by_type
    metrics["n_answerable"] = len(answerable)
    if generator is not None:
        metrics.update(mean_scores(answerable, ANSWER_METRICS))
        unanswerable = [r["scores"] for r in per_question if r["type"] == "unanswerable"]
        if unanswerable:
            metrics["abstention_accuracy"] = mean_scores(unanswerable, ["abstained_correctly"])[
                "abstained_correctly"
            ]
        metrics["generator"] = gen_name
    timing = {
        "p50_ms": round(percentile(latencies, 0.5), 2),
        "p95_ms": round(percentile(latencies, 0.95), 2),
        "build_seconds": float(index_row["build_seconds"]),
    }

    with conn.transaction():
        eval_id = conn.execute(
            "INSERT INTO evaluations (run_key, index_id, retriever_key, retriever, split, "
            "question_set_sha256, n_questions, metrics, timing) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING eval_id",
            (
                key,
                index_id,
                retriever_spec.key,
                json.dumps(retriever_spec.as_dict()),
                split,
                qset_sha,
                len(per_question),
                json.dumps(metrics),
                json.dumps(timing),
            ),
        ).fetchone()["eval_id"]
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO eval_results (eval_id, question_id, retrieved, metrics) "
                "VALUES (%s, %s, %s, %s)",
                [(eval_id, r["id"], r["retrieved"], json.dumps(r["scores"])) for r in per_question],
            )

    evidence = {
        "run_key": key,
        "index_id": index_id,
        "retriever_key": retriever_spec.key,
        "retriever": retriever_spec.as_dict(),
        "labels": labels or {},
        "split": split,
        "question_set_sha256": qset_sha,
        "n_questions": len(per_question),
        "n_chunks": index_row["n_chunks"],
        "metrics": metrics,
        "timing": timing,
        "per_question": [{k: v for k, v in r.items() if k != "retrieved"} for r in per_question],
    }
    path = evidence_file(evidence_dir, split, index_id, retriever_spec.key, gen_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return RunResult(eval_id, key, metrics, timing, path)
