import hashlib
import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from rag.specs import EmbedderSpec
from rag.text import STOPWORDS

WORD = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    spec: EmbedderSpec

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    def __init__(self, spec: EmbedderSpec) -> None:
        self.spec = spec

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.spec.dim, dtype=np.float32)
        for word in WORD.findall(text.lower()):
            if word in STOPWORDS:
                continue
            digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            vec[value % self.spec.dim] += 1.0 if (value >> 63) & 1 else -1.0
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            vec[0] = 1.0
            return vec
        return vec / norm

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts]) if texts else np.zeros((0, self.spec.dim))

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(self.spec.query_prefix + text)


class SentenceTransformerEmbedder:
    def __init__(self, spec: EmbedderSpec, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.spec = spec
        self._model = SentenceTransformer(spec.model, revision=spec.revision, device=device)
        dimension = getattr(self._model, "get_embedding_dimension", None)
        actual = (dimension or self._model.get_sentence_embedding_dimension)()
        if actual != spec.dim:
            raise ValueError(f"{spec.model} produces {actual} dimensions, spec says {spec.dim}")

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._model.encode(
            list(texts),
            batch_size=self.spec.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.spec.query_prefix + text])[0]


_CACHE: dict[str, Embedder] = {}


def make_embedder(spec: EmbedderSpec) -> Embedder:
    key = f"{spec.kind}:{spec.model_key}:{spec.query_prefix}"
    if key not in _CACHE:
        if spec.kind == "hashing":
            _CACHE[key] = HashingEmbedder(spec)
        elif spec.kind == "sentence-transformers":
            _CACHE[key] = SentenceTransformerEmbedder(spec)
        else:
            raise ValueError(f"unknown embedder kind {spec.kind!r}")
    return _CACHE[key]
