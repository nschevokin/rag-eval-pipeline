import psycopg

from rag.retrieve.types import Hit

FULLTEXT_SQL = """
SELECT c.chunk_id, ts_rank_cd(c.tsv, q.query, 32) AS score
FROM index_versions iv
JOIN corpus_documents cd ON cd.corpus_sha256 = iv.corpus_sha256
JOIN chunks c ON c.doc_sha256 = cd.doc_sha256 AND c.chunker_key = iv.chunker_key
CROSS JOIN (
    SELECT nullif(replace(plainto_tsquery('english', %(q)s)::text, ' & ', ' | '), '')::tsquery
        AS query
) q
WHERE iv.index_id = %(index_id)s AND q.query IS NOT NULL AND c.tsv @@ q.query
ORDER BY score DESC, c.chunk_id
LIMIT %(k)s
"""


def fulltext_search(conn: psycopg.Connection, index_id: str, query: str, k: int) -> list[Hit]:
    rows = conn.execute(FULLTEXT_SQL, {"q": query, "index_id": index_id, "k": k}).fetchall()
    return [Hit(r["chunk_id"], float(r["score"])) for r in rows]
