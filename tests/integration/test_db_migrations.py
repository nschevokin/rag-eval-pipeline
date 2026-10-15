import shutil

import psycopg
import pytest

from rag.db import MigrationError, connect, migrate

from conftest import ROOT


def test_migrations_apply_once_and_are_idempotent(empty_db_url):
    with connect(empty_db_url) as conn:
        assert migrate(conn, ROOT / "migrations") == [1, 2, 3]
        assert migrate(conn, ROOT / "migrations") == []
        ext = conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
        assert ext is not None


def test_a_changed_migration_is_refused(empty_db_url, tmp_path):
    directory = tmp_path / "migrations"
    shutil.copytree(ROOT / "migrations", directory)
    with connect(empty_db_url) as conn:
        migrate(conn, directory)
        path = directory / "003_grants.sql"
        path.write_text(path.read_text() + "\n-- edited\n")
        with pytest.raises(MigrationError, match="forward-only"):
            migrate(conn, directory)


def test_vector_of_384_numbers_round_trips(conn):
    values = [i / 384 for i in range(384)]
    literal = "[" + ",".join(str(v) for v in values) + "]"
    row = conn.execute("SELECT %s::vector(384)::text AS v", (literal,)).fetchone()
    back = [float(x) for x in row["v"].strip("[]").split(",")]
    assert len(back) == 384
    assert back == pytest.approx(values, abs=1e-6)


def test_append_only_tables_refuse_updates_and_deletes(conn):
    conn.execute(
        "INSERT INTO corpora (corpus_sha256, name, manifest) VALUES (%s, 'x', '{}')", ("a" * 64,)
    )
    with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
        conn.execute("UPDATE corpora SET name = 'y'")
    with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
        conn.execute("DELETE FROM corpora")
