import ast
import json
from pathlib import Path

import pytest

from rag.evaluate.questions import (
    Question,
    QuestionSetError,
    TestSetAccess,
    TestSetLocked,
    load_dev,
    load_question_file,
    load_split,
    load_test,
    make_split,
    type_mix,
    validate_questions,
    write_split,
)

from conftest import FIXTURE_QUESTIONS, ROOT


def synthetic(n_per_type: dict[str, int]) -> list[Question]:
    out = []
    for qtype, n in n_per_type.items():
        for i in range(n):
            gold = () if qtype == "unanswerable" else ("p#a", "p#b")
            facts = () if qtype == "unanswerable" else ("x",)
            out.append(Question(f"{qtype}-{i:03d}", f"q {qtype} {i}", gold, facts, qtype))
    return out


MIX = {"factual": 50, "multi_section": 20, "paraphrase": 20, "unanswerable": 10}


def test_split_has_no_overlap_and_covers_everything():
    questions = synthetic(MIX)
    split = make_split(questions, seed=7)
    assert set(split["dev"]).isdisjoint(split["test"])
    assert set(split["dev"]) | set(split["test"]) == {q.id for q in questions}
    assert len(split["dev"]) == 30


def test_split_keeps_type_proportions():
    questions = synthetic(MIX)
    split = make_split(questions, seed=7)
    by_id = {q.id: q for q in questions}
    dev = type_mix([by_id[i] for i in split["dev"]])
    test = type_mix([by_id[i] for i in split["test"]])
    for qtype in MIX:
        assert dev[qtype] == pytest.approx(test[qtype], abs=0.02)


def test_split_is_deterministic_for_a_seed():
    questions = synthetic(MIX)
    assert make_split(questions, seed=7) == make_split(questions, seed=7)
    assert make_split(questions, seed=7)["dev"] != make_split(questions, seed=8)["dev"]


def test_extending_keeps_existing_assignments(tmp_path):
    first = synthetic({"factual": 10, "paraphrase": 10})
    split = make_split(first, seed=7)
    more = first + synthetic({"multi_section": 10})
    extended = make_split(more, seed=7, existing=split)
    assert set(split["dev"]) <= set(extended["dev"])
    assert set(split["test"]) <= set(extended["test"])
    assert len(extended["dev"]) == 9


def test_split_file_is_frozen(tmp_path):
    path = tmp_path / "split.json"
    write_split(path, make_split(synthetic(MIX)))
    with pytest.raises(FileExistsError):
        write_split(path, make_split(synthetic(MIX), seed=1))


def write_questions(path: Path, questions: list[Question]) -> None:
    path.write_text("\n".join(json.dumps(q.to_dict()) for q in questions) + "\n")


def test_dev_loader_refuses_questions_changed_after_the_split(tmp_path):
    questions = synthetic(MIX)
    qpath, spath = tmp_path / "q.jsonl", tmp_path / "split.json"
    write_questions(qpath, questions)
    write_split(spath, make_split(questions))
    assert len(load_dev(qpath, spath)) == 30
    write_questions(qpath, [*questions, Question("new", "q", ("p#a",), ("x",), "factual")])
    with pytest.raises(QuestionSetError, match="changed since the split"):
        load_dev(qpath, spath)


def test_test_set_is_locked_without_a_grant(tmp_path):
    questions = synthetic(MIX)
    qpath, spath = tmp_path / "q.jsonl", tmp_path / "split.json"
    write_questions(qpath, questions)
    write_split(spath, make_split(questions))
    with pytest.raises(TestSetLocked):
        load_test(qpath, spath, object())
    with pytest.raises(TestSetLocked):
        TestSetAccess(object(), "D-001")


def test_only_the_confirm_module_can_open_the_test_split():
    allowed = {
        ROOT / "src/rag/evaluate/questions.py",
        ROOT / "src/rag/evaluate/confirm.py",
    }
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, ast.Attribute):
                names.append(node.attr)
            elif isinstance(node, ast.alias):
                names.append(node.name)
            touches = {"load_test", "_grant_test_access", "TestSetAccess"} & set(names)
            if touches and path not in allowed:
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} {sorted(touches)}")
    assert not offenders, "tuning code touches the test split: " + ", ".join(offenders)


def test_split_loader_rejects_overlap(tmp_path):
    questions = synthetic({"factual": 4})
    split = make_split(questions)
    split["test"].append(split["dev"][0])
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split))
    with pytest.raises(QuestionSetError, match="overlap"):
        load_split(path, questions)


def test_validation_catches_bad_questions():
    good_refs = {"p#a", "p#b"}
    bad = [
        Question("1", "q", ("p#zzz",), ("x",), "factual"),
        Question("2", "q", ("p#a",), (), "factual"),
        Question("3", "q", ("p#a",), ("x",), "multi_section"),
        Question("4", "q", ("p#a",), (), "unanswerable"),
        Question("5", "q", ("p#a",), ("x",), "trivia"),
    ]
    problems = validate_questions(bad, good_refs)
    assert len(problems) == 5
    assert validate_questions(synthetic(MIX), good_refs) == []


def test_fixture_question_file_loads():
    questions = load_question_file(FIXTURE_QUESTIONS)
    assert len(questions) == 12
    assert type_mix(questions)["unanswerable"] > 0
