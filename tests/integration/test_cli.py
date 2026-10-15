import json
import shutil

import pytest

from rag.cli import main

from conftest import FIXTURE_QUESTIONS, ROOT, make_fixture_manifest

EXPERIMENTS = """
[defaults]
index = "fx"
retriever = "vector"

[embedders.hashing]
kind = "hashing"
model = "hashing-bow"
revision = "v1"
dim = 384

[chunkers.small]
kind = "heading"
version = "chunker-v1"
target_tokens = 40
overlap = 10
prepend_heading = true

[indexes.fx]
chunker = "small"
embedder = "hashing"

[retrievers.vector]
mode = "vector"
candidates = 20
"""


@pytest.fixture
def cli_root(tmp_path, db_url, monkeypatch):
    root = tmp_path / "repo"
    (root / "corpus").mkdir(parents=True)
    (root / "eval").mkdir()
    (root / "docs").mkdir()
    shutil.copytree(ROOT / "migrations", root / "migrations")
    shutil.copy(ROOT / "eval" / "gate.toml", root / "eval" / "gate.toml")
    shutil.copy(FIXTURE_QUESTIONS, root / "eval" / "questions.jsonl")
    (root / "experiments.toml").write_text(EXPERIMENTS)
    make_fixture_manifest(tmp_path / "build")
    shutil.copy(tmp_path / "build" / "manifest.json", root / "corpus" / "manifest.json")
    shutil.copytree(tmp_path / "build" / "raw", root / "data" / "raw")
    monkeypatch.setenv("RAG_ROOT", str(root))
    monkeypatch.setenv("RAG_DATA_DIR", "data")
    monkeypatch.setenv("RAG_DATABASE_URL", db_url)
    return root


def test_cli_runs_the_whole_lifecycle(cli_root, capsys):
    main(["db", "migrate"])
    main(["corpus", "verify"])
    main(["ingest"])
    main(["index", "build"])
    main(["index", "build"])
    assert "no-op" in capsys.readouterr().out
    main(["eval", "validate"])
    main(["eval", "split"])
    split = json.loads((cli_root / "eval" / "split.json").read_text())
    assert set(split["dev"]).isdisjoint(split["test"])
    main(["eval", "run"])
    with pytest.raises(SystemExit):
        main(["gate", "check", "--latest"])
    capsys.readouterr()
    main(["gate", "set-floors", "--eval-id", "1", "--margin", "0.05"])
    assert "[floors]" in (cli_root / "eval" / "gate.toml").read_text()
    main(["gate", "check", "--latest"])
    assert "PASS" in capsys.readouterr().out
    main(["publish", "--latest-pass", "--by", "ci"])
    main(["status"])
    assert "serving index" in capsys.readouterr().out
    main(["report", "ablation"])
    assert "`" in (cli_root / "evidence" / "ablation.md").read_text()
    evidence = list((cli_root / "evidence" / "evals").glob("dev-*.json"))
    assert len(evidence) == 1


def test_cli_refuses_to_loosen_floors(cli_root):
    main(["db", "migrate"])
    main(["ingest"])
    main(["index", "build"])
    main(["eval", "split"])
    main(["eval", "run"])
    main(["gate", "set-floors", "--eval-id", "1", "--margin", "0.0"])
    with pytest.raises(SystemExit, match="refusing to loosen"):
        main(["gate", "set-floors", "--eval-id", "1", "--margin", "0.2"])
