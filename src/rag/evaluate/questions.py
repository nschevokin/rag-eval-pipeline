import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from rag.hashing import canonical_json, sha256_text

QUESTION_TYPES = ("factual", "multi_section", "paraphrase", "unanswerable")
DEFAULT_SEED = 20261014
DEV_FRACTION = 0.3


class QuestionSetError(ValueError):
    pass


class TestSetLocked(PermissionError):
    __test__ = False


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    gold_sections: tuple[str, ...]
    key_facts: tuple[str, ...]
    type: str

    @property
    def answerable(self) -> bool:
        return self.type != "unanswerable"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["gold_sections"] = list(self.gold_sections)
        data["key_facts"] = list(self.key_facts)
        return data


def load_question_file(path: Path) -> list[Question]:
    questions = []
    problems = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            q = Question(
                id=str(raw["id"]),
                question=str(raw["question"]).strip(),
                gold_sections=tuple(raw.get("gold_sections", [])),
                key_facts=tuple(raw.get("key_facts", [])),
                type=str(raw["type"]),
            )
        except (KeyError, json.JSONDecodeError) as exc:
            problems.append(f"line {lineno}: {exc!r}")
            continue
        questions.append(q)
    ids = [q.id for q in questions]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        problems.append(f"duplicate ids: {duplicates}")
    if problems:
        raise QuestionSetError("; ".join(problems))
    return questions


def question_set_sha256(questions: Iterable[Question]) -> str:
    return sha256_text(
        canonical_json(sorted((q.to_dict() for q in questions), key=lambda d: d["id"]))
    )


def validate_questions(questions: Sequence[Question], known_refs: set[str]) -> list[str]:
    problems = []
    for q in questions:
        if q.type not in QUESTION_TYPES:
            problems.append(f"{q.id}: unknown type {q.type!r}")
        if not q.question:
            problems.append(f"{q.id}: empty question")
        if q.answerable:
            if not q.gold_sections:
                problems.append(f"{q.id}: answerable question without gold sections")
            if not q.key_facts:
                problems.append(f"{q.id}: answerable question without key facts")
            if q.type == "multi_section" and len(set(q.gold_sections)) < 2:
                problems.append(f"{q.id}: multi_section question needs two or more gold sections")
        elif q.gold_sections:
            problems.append(f"{q.id}: unanswerable question must not have gold sections")
        for ref in q.gold_sections:
            if ref not in known_refs:
                problems.append(f"{q.id}: gold section {ref!r} not found in the corpus")
    return problems


def type_mix(questions: Sequence[Question]) -> dict[str, float]:
    total = len(questions) or 1
    return {t: round(sum(q.type == t for q in questions) / total, 3) for t in QUESTION_TYPES}


def make_split(
    questions: Sequence[Question],
    seed: int = DEFAULT_SEED,
    dev_fraction: float = DEV_FRACTION,
    existing: dict | None = None,
) -> dict:
    ids = {q.id for q in questions}
    assigned_dev = set(existing["dev"]) & ids if existing else set()
    assigned_test = set(existing["test"]) & ids if existing else set()
    dev, test = set(assigned_dev), set(assigned_test)
    rng = random.Random(seed)
    for qtype in QUESTION_TYPES:
        of_type = sorted(q.id for q in questions if q.type == qtype)
        fresh = [i for i in of_type if i not in assigned_dev and i not in assigned_test]
        rng.shuffle(fresh)
        target_dev = round(len(of_type) * dev_fraction)
        have_dev = sum(1 for i in of_type if i in dev)
        take = max(0, min(len(fresh), target_dev - have_dev))
        dev.update(fresh[:take])
        test.update(fresh[take:])
    return {
        "seed": seed,
        "dev_fraction": dev_fraction,
        "questions_sha256": question_set_sha256(questions),
        "dev": sorted(dev),
        "test": sorted(test),
    }


def write_split(path: Path, split: dict, *, extend: bool = False) -> None:
    if path.exists() and not extend:
        raise FileExistsError(f"{path} is frozen; use --extend to assign newly added questions")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")


def load_split(path: Path, questions: Sequence[Question]) -> dict:
    split = json.loads(path.read_text(encoding="utf-8"))
    if split["questions_sha256"] != question_set_sha256(questions):
        raise QuestionSetError(
            "questions changed since the split was made; run `rag eval split --extend` and record "
            "a decision entry"
        )
    overlap = set(split["dev"]) & set(split["test"])
    if overlap:
        raise QuestionSetError(f"dev and test overlap: {sorted(overlap)}")
    return split


def load_dev(questions_path: Path, split_path: Path) -> list[Question]:
    questions = load_question_file(questions_path)
    split = load_split(split_path, questions)
    dev_ids = set(split["dev"])
    return [q for q in questions if q.id in dev_ids]


_TEST_ACCESS_TOKEN = object()


class TestSetAccess:
    __test__ = False

    def __init__(self, token: object, decision: str) -> None:
        if token is not _TEST_ACCESS_TOKEN:
            raise TestSetLocked("test-set access is granted only by rag.evaluate.confirm")
        self.decision = decision


def _grant_test_access(decision: str) -> TestSetAccess:
    return TestSetAccess(_TEST_ACCESS_TOKEN, decision)


def load_test(questions_path: Path, split_path: Path, access: TestSetAccess) -> list[Question]:
    if not isinstance(access, TestSetAccess):
        raise TestSetLocked("the test set is locked; use `rag eval confirm-test`")
    questions = load_question_file(questions_path)
    split = load_split(split_path, questions)
    test_ids = set(split["test"])
    return [q for q in questions if q.id in test_ids]
