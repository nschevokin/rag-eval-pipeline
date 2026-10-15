import os
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from psycopg.conninfo import make_conninfo

from rag.db import connect, migrate
from rag.ingest.manifest import Manifest, build_manifest
from rag.specs import ChunkerSpec, EmbedderSpec, IndexSpec, RetrieverSpec

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_CORPUS = FIXTURES / "corpus"
FIXTURE_QUESTIONS = FIXTURES / "questions.jsonl"

HASHING = EmbedderSpec(kind="hashing", model="hashing-bow", revision="v1", dim=384, batch_size=256)
HEADING_SMALL = ChunkerSpec(
    kind="heading", version="chunker-v1", target_tokens=40, overlap=10, prepend_heading=True
)
FIXTURE_INDEX = IndexSpec(name="fx-heading-40", chunker=HEADING_SMALL, embedder=HASHING)
VECTOR = RetrieverSpec(name="vector", mode="vector", candidates=20, ef_search=40)


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "tests/integration/" in str(item.fspath).replace(os.sep, "/"):
            item.add_marker(pytest.mark.integration)


def local_fetcher(url: str) -> bytes:
    return (FIXTURE_CORPUS / url.rsplit("/", 1)[-1]).read_bytes()


def make_fixture_manifest(tmp_path: Path) -> tuple[Manifest, Path]:
    raw = tmp_path / "raw"
    urls = [f"https://example.test/docs/{p.name}" for p in sorted(FIXTURE_CORPUS.glob("*.html"))]
    manifest = build_manifest(
        urls,
        corpus="fixture",
        license="test",
        source="fixture",
        raw_dir=raw,
        out=tmp_path / "manifest.json",
        fetcher=local_fetcher,
    )
    return manifest, raw


@pytest.fixture
def fixture_corpus(tmp_path: Path) -> tuple[Manifest, Path]:
    return make_fixture_manifest(tmp_path)


@pytest.fixture(scope="session")
def admin_url() -> str:
    url = os.environ.get("RAG_TEST_DATABASE_URL")
    if not url:
        pytest.skip("RAG_TEST_DATABASE_URL not set")
    return url


def _admin(url: str) -> psycopg.Connection:
    return psycopg.connect(url, autocommit=True)


@pytest.fixture(scope="session")
def template_db(admin_url: str) -> Iterator[str]:
    name = f"rag_tmpl_{uuid.uuid4().hex[:8]}"
    with _admin(admin_url) as admin:
        admin.execute(f'CREATE DATABASE "{name}"')
    with connect(make_conninfo(admin_url, dbname=name)) as conn:
        migrate(conn, ROOT / "migrations")
    yield name
    with _admin(admin_url) as admin:
        admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture
def db_url(admin_url: str, template_db: str) -> Iterator[str]:
    name = f"rag_t_{uuid.uuid4().hex[:10]}"
    with _admin(admin_url) as admin:
        admin.execute(f'CREATE DATABASE "{name}" TEMPLATE "{template_db}"')
    yield make_conninfo(admin_url, dbname=name)
    with _admin(admin_url) as admin:
        admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture
def empty_db_url(admin_url: str) -> Iterator[str]:
    name = f"rag_e_{uuid.uuid4().hex[:10]}"
    with _admin(admin_url) as admin:
        admin.execute(f'CREATE DATABASE "{name}"')
    yield make_conninfo(admin_url, dbname=name)
    with _admin(admin_url) as admin:
        admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture
def conn(db_url: str) -> Iterator[psycopg.Connection]:
    c = connect(db_url)
    yield c
    c.close()


@pytest.fixture
def repo_copy(tmp_path: Path) -> Path:
    target = tmp_path / "repo"
    target.mkdir()
    for name in ("migrations", "eval", "experiments.toml"):
        src = ROOT / name
        if src.is_dir():
            shutil.copytree(src, target / name)
        else:
            shutil.copy(src, target / name)
    return target
