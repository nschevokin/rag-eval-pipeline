import json
from dataclasses import dataclass
from typing import Any

import psycopg

PUBLISH_LOCK = 7_420_003


class PublishRejected(RuntimeError):
    pass


@dataclass(frozen=True)
class Served:
    publication_id: int
    index_id: str
    retriever_key: str
    retriever: dict[str, Any]
    decision_id: int
    published_at: str


def current(conn: psycopg.Connection) -> Served | None:
    row = conn.execute(
        "SELECT ci.publication_id, ci.index_id, ci.retriever_key, p.retriever, p.decision_id, "
        "p.published_at::text AS published_at FROM current_index ci "
        "JOIN publications p ON p.publication_id = ci.publication_id"
    ).fetchone()
    return Served(**row) if row else None


def latest_passing_decision(conn: psycopg.Connection) -> int | None:
    row = conn.execute(
        "SELECT d.decision_id FROM gate_decisions d "
        "LEFT JOIN publications p ON p.decision_id = d.decision_id "
        "WHERE d.passed AND p.publication_id IS NULL ORDER BY d.decision_id DESC LIMIT 1"
    ).fetchone()
    return row["decision_id"] if row else None


def _set_pointer(
    conn: psycopg.Connection, publication_id: int, index_id: str, retriever_key: str
) -> None:
    conn.execute(
        "INSERT INTO current_index (singleton, publication_id, index_id, retriever_key) "
        "VALUES (true, %s, %s, %s) ON CONFLICT (singleton) DO UPDATE SET "
        "publication_id = EXCLUDED.publication_id, index_id = EXCLUDED.index_id, "
        "retriever_key = EXCLUDED.retriever_key, switched_at = now()",
        (publication_id, index_id, retriever_key),
    )


def publish(conn: psycopg.Connection, decision_id: int, published_by: str) -> int:
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (PUBLISH_LOCK,))
        decision = conn.execute(
            "SELECT d.decision_id, d.index_id, d.retriever_key, d.passed, d.served_publication_id, "
            "e.retriever FROM gate_decisions d JOIN evaluations e ON e.eval_id = d.eval_id "
            "WHERE d.decision_id = %s",
            (decision_id,),
        ).fetchone()
        if decision is None:
            raise PublishRejected(f"gate decision {decision_id} does not exist")
        if not decision["passed"]:
            raise PublishRejected(f"gate decision {decision_id} rejected this index")
        served = current(conn)
        served_id = served.publication_id if served else None
        if served_id != decision["served_publication_id"]:
            raise PublishRejected(
                "the served index changed after this gate decision; run the gate again"
            )
        publication_id = conn.execute(
            "INSERT INTO publications (decision_id, index_id, retriever_key, retriever, "
            "published_by) VALUES (%s, %s, %s, %s, %s) RETURNING publication_id",
            (
                decision_id,
                decision["index_id"],
                decision["retriever_key"],
                json.dumps(decision["retriever"]),
                published_by,
            ),
        ).fetchone()["publication_id"]
        _set_pointer(conn, publication_id, decision["index_id"], decision["retriever_key"])
    return publication_id
