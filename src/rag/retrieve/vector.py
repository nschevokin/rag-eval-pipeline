import re

import numpy as np
import psycopg
from psycopg import sql

from rag.db import vector_literal
from rag.retrieve.types import Hit

INDEX_ID = re.compile(r"^[0-9a-f]{16}$")


def vector_search(
    conn: psycopg.Connection, index_id: str, query_vector: np.ndarray, k: int, ef_search: int = 40
) -> list[Hit]:
    if not INDEX_ID.match(index_id):
        raise ValueError(f"bad index id {index_id!r}")
    literal = vector_literal(query_vector)
    query = sql.SQL(
        "SELECT chunk_id, 1 - (embedding <=> %s::vector) AS score FROM {} "
        "ORDER BY embedding <=> %s::vector LIMIT %s"
    ).format(sql.Identifier(f"embeddings_{index_id}"))
    with conn.transaction():
        conn.execute(
            sql.SQL("SET LOCAL hnsw.ef_search = {}").format(sql.Literal(max(int(ef_search), k)))
        )
        rows = conn.execute(query, (literal, literal, k)).fetchall()
    return [Hit(r["chunk_id"], float(r["score"])) for r in rows]
