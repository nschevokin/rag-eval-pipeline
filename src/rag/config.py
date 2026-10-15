import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path
    data_dir: Path

    @property
    def migrations(self) -> Path:
        return self.root / "migrations"

    @property
    def manifest(self) -> Path:
        return self.root / "corpus" / "manifest.json"

    @property
    def sources(self) -> Path:
        return self.root / "corpus" / "sources.txt"

    @property
    def chapters(self) -> Path:
        return self.root / "corpus" / "chapters.txt"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def questions(self) -> Path:
        return self.root / "eval" / "questions.jsonl"

    @property
    def split(self) -> Path:
        return self.root / "eval" / "split.json"

    @property
    def gate(self) -> Path:
        return self.root / "eval" / "gate.toml"

    @property
    def experiments(self) -> Path:
        return self.root / "experiments.toml"

    @property
    def evidence(self) -> Path:
        return self.root / "evidence"

    @property
    def decisions(self) -> Path:
        return self.root / "docs" / "decisions.md"


def database_url() -> str:
    return os.environ.get("RAG_DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag")


def default_paths() -> Paths:
    root = Path(os.environ.get("RAG_ROOT", Path.cwd())).resolve()
    data_dir = Path(os.environ.get("RAG_DATA_DIR", root / "data"))
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    return Paths(root=root, data_dir=data_dir)
