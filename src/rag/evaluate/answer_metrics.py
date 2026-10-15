import re
import unicodedata
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from rag.text import FACT_STOPWORDS

WORD = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
NUMBER = re.compile(r"\d+(?:\.\d+)?")


@dataclass(frozen=True)
class Answer:
    text: str
    cited_chunk_ids: tuple[str, ...]
    abstained: bool


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(text.split())


def fact_present(fact: str, answer_text: str) -> bool:
    fact_n, answer_n = normalize(fact), normalize(answer_text)
    if fact_n and fact_n in answer_n:
        return True
    answer_words = set(WORD.findall(answer_n))
    fact_numbers = NUMBER.findall(fact_n)
    if any(n not in set(NUMBER.findall(answer_n)) for n in fact_numbers):
        return False
    content = [w for w in WORD.findall(fact_n) if w not in FACT_STOPWORDS]
    return bool(content) and all(w in answer_words for w in content)


def key_fact_coverage(answer: Answer, key_facts: Sequence[str]) -> float:
    if not key_facts or answer.abstained:
        return 0.0
    return sum(fact_present(f, answer.text) for f in key_facts) / len(key_facts)


def citation_validity(
    answer: Answer,
    retrieved_ids: Collection[str],
    chunk_refs: Mapping[str, Collection[str]],
    gold: Collection[str],
) -> float:
    if not answer.cited_chunk_ids:
        return 0.0
    valid = 0
    for cid in answer.cited_chunk_ids:
        refs = chunk_refs.get(cid, ())
        if cid in retrieved_ids and any(g in refs for g in gold):
            valid += 1
    return valid / len(answer.cited_chunk_ids)


def answer_scores(
    answer: Answer,
    *,
    answerable: bool,
    key_facts: Sequence[str],
    retrieved_ids: Collection[str],
    chunk_refs: Mapping[str, Collection[str]],
    gold: Collection[str],
) -> dict[str, float]:
    if not answerable:
        return {"abstained_correctly": 1.0 if answer.abstained else 0.0}
    return {
        "key_fact_coverage": key_fact_coverage(answer, key_facts),
        "citation_validity": 0.0
        if answer.abstained
        else citation_validity(answer, retrieved_ids, chunk_refs, gold),
        "false_abstention": 1.0 if answer.abstained else 0.0,
    }
