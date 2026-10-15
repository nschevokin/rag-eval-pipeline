import pytest

from rag.retrieve.fusion import reciprocal_rank_fusion


def test_rrf_matches_hand_computed_order():
    vector = ["a", "b", "c", "d"]
    fulltext = ["c", "a", "e"]
    fused = reciprocal_rank_fusion([vector, fulltext], k=60)
    expected = {
        "a": 1 / 61 + 1 / 62,
        "b": 1 / 62,
        "c": 1 / 63 + 1 / 61,
        "d": 1 / 64,
        "e": 1 / 63,
    }
    assert [h.chunk_id for h in fused] == ["a", "c", "b", "e", "d"]
    for hit in fused:
        assert hit.score == pytest.approx(expected[hit.chunk_id])


def test_rrf_breaks_ties_by_chunk_id():
    fused = reciprocal_rank_fusion([["y", "x"], ["x", "y"]], k=60)
    assert [h.chunk_id for h in fused] == ["x", "y"]


def test_rrf_with_one_empty_ranking_keeps_the_other_order():
    fused = reciprocal_rank_fusion([["b", "a"], []])
    assert [h.chunk_id for h in fused] == ["b", "a"]
