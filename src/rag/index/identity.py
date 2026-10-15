from typing import Any

from rag.hashing import short_id
from rag.specs import IndexSpec

IDENTITY_SCHEMA = 1


def index_identity(corpus_sha256: str, spec: IndexSpec) -> dict[str, Any]:
    return {
        "schema": IDENTITY_SCHEMA,
        "corpus_sha256": corpus_sha256,
        "chunker": {**spec.chunker.as_dict(), "key": spec.chunker.key},
        "embedder": spec.embedder.identity(),
        "params": spec.params(),
    }


def index_id(identity: dict[str, Any]) -> str:
    return short_id(identity, 16)
