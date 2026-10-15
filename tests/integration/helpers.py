from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import psycopg

from rag.evaluate.questions import load_dev, make_split, write_split
from rag.evaluate.runner import RunResult, run_evaluation
from rag.index.build import BuildResult, build_index
from rag.index.embed import HashingEmbedder
from rag.ingest.load import ingest
from rag.ingest.manifest import Manifest
from rag.specs import IndexSpec, RetrieverSpec

from conftest import FIXTURE_INDEX, FIXTURE_QUESTIONS, HASHING

SHUFFLED = replace(HASHING, model="hashing-bow-shuffled")
MUTANT_INDEX = replace(FIXTURE_INDEX, name="fx-mutant", embedder=SHUFFLED)


class ShuffledEmbedder(HashingEmbedder):
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        vectors = super().embed_documents(texts)
        rng = np.random.default_rng(0)
        return vectors[rng.permutation(len(vectors))]


def ingest_and_build(
    conn: psycopg.Connection, manifest: Manifest, raw: Path, spec: IndexSpec = FIXTURE_INDEX
) -> BuildResult:
    ingest(conn, manifest, raw, [spec.chunker])
    embedder = (
        ShuffledEmbedder(spec.embedder)
        if spec.embedder == SHUFFLED
        else HashingEmbedder(spec.embedder)
    )
    return build_index(conn, manifest.corpus_sha256, spec, embedder)


def fixture_split(tmp_path: Path) -> Path:
    from rag.evaluate.questions import load_question_file

    path = tmp_path / "split.json"
    if not path.exists():
        questions = load_question_file(FIXTURE_QUESTIONS)
        write_split(path, make_split(questions, seed=1, dev_fraction=1.0))
    return path


def evaluate_dev(
    conn: psycopg.Connection,
    tmp_path: Path,
    index_id: str,
    retriever: RetrieverSpec,
    spec: IndexSpec = FIXTURE_INDEX,
) -> RunResult:
    questions = load_dev(FIXTURE_QUESTIONS, fixture_split(tmp_path))
    embedder = HashingEmbedder(spec.embedder)
    return run_evaluation(
        conn,
        index_id=index_id,
        retriever_spec=retriever,
        questions=questions,
        split="dev",
        embedder=embedder,
        evidence_dir=tmp_path / "evidence",
        labels={"index": spec.name, "retriever": retriever.name},
    )
