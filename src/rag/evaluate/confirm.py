import json
import re
from datetime import UTC, datetime
from pathlib import Path

import psycopg

from rag.evaluate.questions import _grant_test_access, load_test
from rag.evaluate.runner import RunResult, run_evaluation
from rag.generate.base import Generator
from rag.index.embed import Embedder
from rag.retrieve.rerank import Reranker
from rag.specs import RetrieverSpec

DECISION_ID = re.compile(r"^D-\d{3}$")


class TestRunRefused(PermissionError):
    __test__ = False


def confirmation_path(evidence_dir: Path) -> Path:
    return evidence_dir / "test_confirmation.json"


def _check_decision(decisions_md: Path, decision: str, used: list[str]) -> None:
    if not DECISION_ID.match(decision):
        raise TestRunRefused(f"decision id must look like D-007, got {decision!r}")
    text = decisions_md.read_text(encoding="utf-8") if decisions_md.exists() else ""
    if not re.search(rf"^#+\s*{re.escape(decision)}\b", text, flags=re.MULTILINE):
        raise TestRunRefused(f"{decision} is not a heading in {decisions_md}")
    if decision in used:
        raise TestRunRefused(
            f"{decision} already authorised a test run; record a new decision to run it again"
        )


def confirm_on_test(
    conn: psycopg.Connection,
    *,
    decision: str,
    questions_path: Path,
    split_path: Path,
    decisions_md: Path,
    evidence_dir: Path,
    index_id: str,
    retriever_spec: RetrieverSpec,
    embedder: Embedder,
    reranker: Reranker | None = None,
    generator: Generator | None = None,
    labels: dict[str, str] | None = None,
) -> RunResult:
    record_path = confirmation_path(evidence_dir)
    history = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else []
    _check_decision(decisions_md, decision, [entry["decision"] for entry in history])
    access = _grant_test_access(decision)
    questions = load_test(questions_path, split_path, access)
    result = run_evaluation(
        conn,
        index_id=index_id,
        retriever_spec=retriever_spec,
        questions=questions,
        split="test",
        embedder=embedder,
        evidence_dir=evidence_dir,
        reranker=reranker,
        generator=generator,
        labels=labels,
    )
    history.append(
        {
            "decision": decision,
            "eval_id": result.eval_id,
            "run_key": result.run_key,
            "index_id": index_id,
            "retriever": retriever_spec.as_dict(),
            "metrics": result.metrics,
            "evidence": result.evidence_path.name,
            "ran_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return result
