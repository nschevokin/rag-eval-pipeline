import math
from collections.abc import Collection, Mapping, Sequence

RETRIEVAL_METRICS = ("recall_at_5", "recall_at_10", "mrr", "ndcg_at_10")

RankedRefs = Sequence[Collection[str]]


def _matches(refs: Collection[str], gold: Collection[str]) -> set[str]:
    return {g for g in gold if g in refs}


def recall_at_k(ranked: RankedRefs, gold: Collection[str], k: int) -> float:
    return 1.0 if any(_matches(refs, gold) for refs in ranked[:k]) else 0.0


def reciprocal_rank(ranked: RankedRefs, gold: Collection[str]) -> float:
    for rank, refs in enumerate(ranked, start=1):
        if _matches(refs, gold):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: RankedRefs, gold: Collection[str], k: int = 10) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    covered: set[str] = set()
    dcg = 0.0
    for rank, refs in enumerate(ranked[:k], start=1):
        new = _matches(refs, gold_set) - covered
        if new:
            covered |= new
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold_set), k) + 1))
    return dcg / ideal


def retrieval_scores(ranked: RankedRefs, gold: Collection[str]) -> dict[str, float]:
    return {
        "recall_at_5": recall_at_k(ranked, gold, 5),
        "recall_at_10": recall_at_k(ranked, gold, 10),
        "mrr": reciprocal_rank(ranked, gold),
        "ndcg_at_10": ndcg_at_k(ranked, gold, 10),
    }


def mean_scores(rows: Sequence[Mapping[str, float]], names: Sequence[str]) -> dict[str, float]:
    if not rows:
        return {}
    return {name: round(sum(r[name] for r in rows) / len(rows), 6) for name in names}


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
