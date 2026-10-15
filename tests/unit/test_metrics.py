import math

import pytest

from rag.evaluate.metrics import (
    mean_scores,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
    retrieval_scores,
)

A, B, C = "p#a", "p#b", "p#c"

EXAMPLE = [
    ([{A}, {"x"}, {"y"}], {A}),
    ([{"x"}, {"y"}, {A}], {A}),
    ([{"x"}] * 7 + [{A}], {A}),
    ([{"x"}] * 12, {A}),
    ([{A}, {"x"}, {B}], {A, B}),
]


def test_recall_hand_computed():
    assert [recall_at_k(r, g, 5) for r, g in EXAMPLE] == [1, 1, 0, 0, 1]
    assert [recall_at_k(r, g, 10) for r, g in EXAMPLE] == [1, 1, 1, 0, 1]


def test_mrr_hand_computed():
    assert [reciprocal_rank(r, g) for r, g in EXAMPLE] == [1.0, 1 / 3, 1 / 8, 0.0, 1.0]
    mean = (1 + 1 / 3 + 1 / 8 + 0 + 1) / 5
    rows = [retrieval_scores(r, g) for r, g in EXAMPLE]
    assert mean_scores(rows, ["mrr"])["mrr"] == pytest.approx(round(mean, 6))


def test_ndcg_hand_computed():
    assert ndcg_at_k(*EXAMPLE[0]) == 1.0
    assert ndcg_at_k(*EXAMPLE[1]) == pytest.approx(1 / math.log2(4))
    assert ndcg_at_k(*EXAMPLE[2]) == pytest.approx(1 / math.log2(9))
    assert ndcg_at_k(*EXAMPLE[3]) == 0.0
    ideal = 1 + 1 / math.log2(3)
    assert ndcg_at_k(*EXAMPLE[4]) == pytest.approx((1 + 1 / math.log2(4)) / ideal)


def test_ndcg_counts_each_gold_section_once():
    ranked = [{A}, {A}, {A}]
    assert ndcg_at_k(ranked, {A, B}) == pytest.approx(1 / (1 + 1 / math.log2(3)))


def test_ancestor_refs_match_child_chunks():
    ranked = [{"p", "p#a", "p#a-1"}]
    assert recall_at_k(ranked, {"p#a"}, 5) == 1.0
    assert recall_at_k(ranked, {"p#b"}, 5) == 0.0


def test_multi_gold_recall_needs_only_one_hit():
    assert recall_at_k([{C}], {A, C}, 5) == 1.0


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([10], 0.95) == 10
    assert percentile([], 0.5) == 0.0
