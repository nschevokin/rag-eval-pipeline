from collections.abc import Sequence

from rag.retrieve.types import Hit


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> list[Hit]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [Hit(chunk_id, score) for chunk_id, score in ordered]
