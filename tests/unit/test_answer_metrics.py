from rag.evaluate.answer_metrics import (
    Answer,
    answer_scores,
    citation_validity,
    fact_present,
    key_fact_coverage,
)

GOLD = {"locking#locking-deadlocks"}
REFS = {
    "c1": ("locking", "locking#locking", "locking#locking-deadlocks"),
    "c2": ("search", "search#search", "search#search-tree"),
}
RETRIEVED = {"c1", "c2"}
FACTS = ["1 second", "aborted"]


def score(answer: Answer, answerable: bool = True) -> dict:
    return answer_scores(
        answer,
        answerable=answerable,
        key_facts=FACTS if answerable else [],
        retrieved_ids=RETRIEVED,
        chunk_refs=REFS,
        gold=GOLD if answerable else set(),
    )


def test_right_answer_scores_full_marks():
    right = Answer(
        "After 1 second of waiting the check runs and one transaction is aborted [c1].",
        ("c1",),
        False,
    )
    assert score(right) == {
        "key_fact_coverage": 1.0,
        "citation_validity": 1.0,
        "false_abstention": 0.0,
    }


def test_wrong_answer_loses_coverage():
    wrong = Answer("The check runs after 5 seconds.", ("c1",), False)
    assert score(wrong)["key_fact_coverage"] == 0.0


def test_uncited_answer_scores_zero_citation_validity():
    uncited = Answer("After 1 second one transaction is aborted.", (), False)
    result = score(uncited)
    assert result["key_fact_coverage"] == 1.0
    assert result["citation_validity"] == 0.0


def test_invented_citation_scores_zero():
    invented = Answer("1 second, then aborted.", ("c99",), False)
    assert citation_validity(invented, RETRIEVED, REFS, GOLD) == 0.0


def test_citing_retrieved_but_irrelevant_chunk_scores_zero():
    off_topic = Answer("1 second, then aborted.", ("c2",), False)
    assert citation_validity(off_topic, RETRIEVED, REFS, GOLD) == 0.0
    mixed = Answer("1 second, then aborted.", ("c1", "c2"), False)
    assert citation_validity(mixed, RETRIEVED, REFS, GOLD) == 0.5


def test_overconfident_answer_to_unanswerable_question_fails():
    overconfident = Answer("You can configure slots with ledger_slot().", ("c2",), False)
    assert score(overconfident, answerable=False) == {"abstained_correctly": 0.0}
    assert score(Answer("", (), True), answerable=False) == {"abstained_correctly": 1.0}


def test_abstaining_on_answerable_question_is_a_false_abstention():
    result = score(Answer("", (), True))
    assert result == {"key_fact_coverage": 0.0, "citation_validity": 0.0, "false_abstention": 1.0}


def test_fact_matching_is_robust_to_case_and_word_order_but_not_numbers():
    assert fact_present("FOR UPDATE", "use select ... for update here")
    assert fact_present("log directory grow", "the log directory will grow")
    assert not fact_present("1 GB", "after 10 GB of log")
    assert key_fact_coverage(Answer("1 second", (), False), FACTS) == 0.5
