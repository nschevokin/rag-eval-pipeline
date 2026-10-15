from collections.abc import Sequence
from typing import Protocol

from rag.retrieve.types import ChunkRecord, Hit


class Reranker(Protocol):
    def rerank(self, query: str, chunks: Sequence[ChunkRecord]) -> list[Hit]: ...


class CrossEncoderReranker:
    def __init__(self, model: str, revision: str, device: str = "cpu") -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model, revision=revision, device=device)

    def rerank(self, query: str, chunks: Sequence[ChunkRecord]) -> list[Hit]:
        if not chunks:
            return []
        scores = self._model.predict(
            [(query, c.embed_text) for c in chunks], show_progress_bar=False
        )
        paired = sorted(
            zip(chunks, scores, strict=True), key=lambda item: (-float(item[1]), item[0].chunk_id)
        )
        return [Hit(c.chunk_id, float(s)) for c, s in paired]


_CACHE: dict[tuple[str, str], Reranker] = {}


def make_reranker(model: str, revision: str) -> Reranker:
    key = (model, revision)
    if key not in _CACHE:
        _CACHE[key] = CrossEncoderReranker(model, revision)
    return _CACHE[key]
