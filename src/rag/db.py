import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psycopg
from psycopg.rows import dict_row

from rag.hashing import sha256_file

MIGRATION_NAME = re.compile(r"^(\d{3})_[a-z0-9_]+\.sql$")
MIGRATION_LOCK = 7_420_001


class MigrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Migration:
    version: int
    path: Path
    sha256: str


def connect(url: str) -> psycopg.Connection:
    return psycopg.connect(url, autocommit=True, row_factory=dict_row)


def vector_literal(vec: Sequence[float] | np.ndarray) -> str:
    values = np.asarray(vec, dtype=np.float32).astype(float).tolist()
    return "[" + ",".join(repr(v) for v in values) + "]"


def parse_vector(text: str) -> np.ndarray:
    return np.array([float(x) for x in text.strip("[]").split(",")], dtype=np.float32)


def discover_migrations(directory: Path) -> list[Migration]:
    found = []
    for path in sorted(directory.glob("*.sql")):
        match = MIGRATION_NAME.match(path.name)
        if not match:
            raise MigrationError(f"bad migration file name: {path.name}")
        found.append(Migration(int(match.group(1)), path, sha256_file(path)))
    versions = [m.version for m in found]
    if versions != list(range(1, len(found) + 1)):
        raise MigrationError(f"migrations must be numbered 001.. without gaps, got {versions}")
    return found


def migrate(conn: psycopg.Connection, directory: Path) -> list[int]:
    migrations = discover_migrations(directory)
    applied_now: list[int] = []
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (MIGRATION_LOCK,))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version integer PRIMARY KEY,
                filename text NOT NULL,
                sha256 text NOT NULL,
                applied_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        applied = {
            row["version"]: row
            for row in conn.execute("SELECT version, filename, sha256 FROM schema_migrations")
        }
        known = {m.version for m in migrations}
        missing = sorted(set(applied) - known)
        if missing:
            raise MigrationError(f"applied migrations missing from disk: {missing}")
        for migration in migrations:
            row = applied.get(migration.version)
            if row is not None:
                if row["sha256"] != migration.sha256:
                    raise MigrationError(
                        f"{migration.path.name} changed after it was applied; "
                        "migrations are forward-only, add a new file instead"
                    )
                continue
            conn.execute(migration.path.read_text(encoding="utf-8"))
            conn.execute(
                "INSERT INTO schema_migrations (version, filename, sha256) VALUES (%s, %s, %s)",
                (migration.version, migration.path.name, migration.sha256),
            )
            applied_now.append(migration.version)
    return applied_now
