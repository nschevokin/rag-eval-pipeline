import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg

from rag.evaluate.metrics import RETRIEVAL_METRICS


class GateNotConfigured(RuntimeError):
    pass


class FloorsLoosened(RuntimeError):
    pass


@dataclass(frozen=True)
class GateConfig:
    floors: dict[str, float]
    regression_tolerance: float


@dataclass(frozen=True)
class GateDecision:
    decision_id: int
    passed: bool
    reasons: list[str]


def load_gate_config(path: Path) -> GateConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    floors = {k: float(v) for k, v in raw.get("floors", {}).items()}
    unknown = set(floors) - set(RETRIEVAL_METRICS)
    if unknown:
        raise GateNotConfigured(f"unknown metrics in floors: {sorted(unknown)}")
    return GateConfig(floors=floors, regression_tolerance=float(raw["regression_tolerance"]))


def render_gate_config(config: GateConfig, baseline_eval_id: int | None = None) -> str:
    lines = [f"regression_tolerance = {config.regression_tolerance}", ""]
    if baseline_eval_id is not None:
        lines.append(f"baseline_eval_id = {baseline_eval_id}")
        lines.append("")
    lines.append("[floors]")
    lines.extend(
        f"{name} = {config.floors[name]}" for name in RETRIEVAL_METRICS if name in config.floors
    )
    return "\n".join(lines) + "\n"


def floors_from_baseline(scores: Mapping[str, float], margin: float) -> dict[str, float]:
    return {m: round(max(0.0, float(scores[m]) - margin), 3) for m in RETRIEVAL_METRICS}


def judge(
    scores: Mapping[str, float],
    config: GateConfig,
    baseline: Mapping[str, float] | None,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for metric in RETRIEVAL_METRICS:
        value = float(scores[metric])
        floor = config.floors.get(metric)
        if floor is not None and value < floor:
            reasons.append(f"{metric} {value:.4f} is below the floor {floor:.4f}")
        if baseline is not None:
            limit = float(baseline[metric]) - config.regression_tolerance
            if value < limit - 1e-12:
                reasons.append(
                    f"{metric} {value:.4f} regresses more than {config.regression_tolerance:.2f} "
                    f"below the served index ({float(baseline[metric]):.4f})"
                )
    return not reasons, reasons


def _served(conn: psycopg.Connection) -> dict[str, Any] | None:
    return conn.execute(
        "SELECT ci.publication_id, d.eval_id FROM current_index ci "
        "JOIN publications p ON p.publication_id = ci.publication_id "
        "JOIN gate_decisions d ON d.decision_id = p.decision_id"
    ).fetchone()


def _check_not_loosened(conn: psycopg.Connection, floors: Mapping[str, float]) -> None:
    previous = conn.execute(
        "SELECT floors FROM gate_decisions ORDER BY decision_id DESC LIMIT 1"
    ).fetchone()
    if previous is None:
        return
    loosened = [
        f"{metric}: {floors.get(metric)} < {old}"
        for metric, old in previous["floors"].items()
        if floors.get(metric) is None or floors[metric] < float(old)
    ]
    if loosened:
        raise FloorsLoosened("gate floors may never be loosened: " + "; ".join(loosened))


def check_gate(conn: psycopg.Connection, eval_id: int, config: GateConfig) -> GateDecision:
    if set(config.floors) != set(RETRIEVAL_METRICS):
        raise GateNotConfigured(
            "eval/gate.toml has no floors for every retrieval metric yet; set them from the "
            "baseline with `rag gate set-floors` (ticket R12)"
        )
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(7420002)")
        _check_not_loosened(conn, config.floors)
        ev = conn.execute(
            "SELECT eval_id, index_id, retriever_key, split, question_set_sha256, metrics "
            "FROM evaluations WHERE eval_id = %s",
            (eval_id,),
        ).fetchone()
        if ev is None:
            raise LookupError(f"evaluation {eval_id} not found")
        if ev["split"] != "dev":
            raise ValueError("the gate only judges dev-set evaluations")

        served = _served(conn)
        baseline_scores = None
        baseline_eval_id = None
        extra: list[str] = []
        if served is not None:
            base = conn.execute(
                "SELECT eval_id, question_set_sha256, metrics FROM evaluations WHERE eval_id = %s",
                (served["eval_id"],),
            ).fetchone()
            baseline_eval_id = base["eval_id"]
            if base["question_set_sha256"] != ev["question_set_sha256"]:
                extra.append(
                    "the served index was judged on a different question set; re-evaluate it on "
                    "the current dev set before comparing"
                )
            else:
                baseline_scores = base["metrics"]

        passed, reasons = judge(ev["metrics"], config, baseline_scores)
        reasons = extra + reasons
        passed = passed and not extra
        decision_id = conn.execute(
            "INSERT INTO gate_decisions (eval_id, index_id, retriever_key, passed, reasons, "
            "floors, regression_tolerance, served_publication_id, baseline_eval_id) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING decision_id",
            (
                ev["eval_id"],
                ev["index_id"],
                ev["retriever_key"],
                passed,
                json.dumps(reasons),
                json.dumps(config.floors),
                config.regression_tolerance,
                served["publication_id"] if served else None,
                baseline_eval_id,
            ),
        ).fetchone()["decision_id"]
    return GateDecision(decision_id, passed, reasons)
