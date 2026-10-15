import psycopg

from rag.index.embed import Embedder
from rag.retrieve.fulltext import fulltext_search
from rag.retrieve.fusion import reciprocal_rank_fusion
from rag.retrieve.rerank import Reranker, make_reranker
from rag.retrieve.types import Hit, fetch_chunks
from rag.retrieve.vector import vector_search
from rag.specs import RetrieverSpec


class Retriever:
    def __init__(
        self,
        conn: psycopg.Connection,
        index_id: str,
        spec: RetrieverSpec,
        embedder: Embedder,
        reranker: Reranker | None = None,
    ) -> None:
        self.conn = conn
        self.index_id = index_id
        self.spec = spec
        self.embedder = embedder
        if spec.rerank_model and reranker is None:
            reranker = make_reranker(spec.rerank_model, spec.rerank_revision)
        self.reranker = reranker if spec.rerank_model else None

    def _vector(self, query: str, k: int) -> list[Hit]:
        vec = self.embedder.embed_query(query)
        return vector_search(self.conn, self.index_id, vec, k, self.spec.ef_search)

    def search(self, query: str, k: int) -> list[Hit]:
        depth = max(k, self.spec.candidates)
        if self.spec.mode == "vector":
            hits = self._vector(query, depth)
        elif self.spec.mode == "fulltext":
            hits = fulltext_search(self.conn, self.index_id, query, depth)
        else:
            vec_ids = [h.chunk_id for h in self._vector(query, depth)]
            fts_ids = [h.chunk_id for h in fulltext_search(self.conn, self.index_id, query, depth)]
            hits = reciprocal_rank_fusion([vec_ids, fts_ids], self.spec.rrf_k)
        if self.reranker is not None:
            top = hits[: self.spec.rerank_top or depth]
            reranked = self.reranker.rerank(
                query, fetch_chunks(self.conn, [h.chunk_id for h in top])
            )
            hits = reranked + hits[len(top) :]
        return hits[:k]
