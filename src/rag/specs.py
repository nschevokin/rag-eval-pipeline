import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from rag.hashing import short_id

ChunkerKind = Literal["heading", "fixed"]
EmbedderKind = Literal["sentence-transformers", "hashing"]
RetrieverMode = Literal["vector", "fulltext", "hybrid"]


@dataclass(frozen=True)
class ChunkerSpec:
    kind: ChunkerKind
    version: str
    target_tokens: int
    overlap: int
    prepend_heading: bool

    def __post_init__(self) -> None:
        if self.kind not in ("heading", "fixed"):
            raise ValueError(f"unknown chunker kind {self.kind!r}")
        if self.target_tokens <= 0 or not 0 <= self.overlap < self.target_tokens:
            raise ValueError("chunker needs target_tokens > overlap >= 0")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def key(self) -> str:
        return f"{self.version}:{short_id(self.as_dict(), 12)}"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkerSpec":
        return cls(
            kind=data["kind"],
            version=data["version"],
            target_tokens=int(data["target_tokens"]),
            overlap=int(data["overlap"]),
            prepend_heading=bool(data["prepend_heading"]),
        )


@dataclass(frozen=True)
class EmbedderSpec:
    kind: EmbedderKind
    model: str
    revision: str
    dim: int
    query_prefix: str = ""
    batch_size: int = 32

    def identity(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "model": self.model,
            "revision": self.revision,
            "dim": self.dim,
            "query_prefix": self.query_prefix,
        }

    @property
    def model_key(self) -> str:
        return f"{self.model}@{self.revision}"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbedderSpec":
        return cls(
            kind=data["kind"],
            model=data["model"],
            revision=data["revision"],
            dim=int(data["dim"]),
            query_prefix=data.get("query_prefix", ""),
            batch_size=int(data.get("batch_size", 32)),
        )


@dataclass(frozen=True)
class IndexSpec:
    name: str
    chunker: ChunkerSpec
    embedder: EmbedderSpec
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64

    def params(self) -> dict[str, Any]:
        return {"hnsw": {"m": self.hnsw_m, "ef_construction": self.hnsw_ef_construction}}


@dataclass(frozen=True)
class RetrieverSpec:
    name: str
    mode: RetrieverMode
    candidates: int = 20
    ef_search: int = 40
    rrf_k: int = 60
    rerank_model: str = ""
    rerank_revision: str = ""
    rerank_top: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ("vector", "fulltext", "hybrid"):
            raise ValueError(f"unknown retriever mode {self.mode!r}")
        if self.rerank_model and not self.rerank_revision:
            raise ValueError(
                f"retriever {self.name!r}: pin rerank_revision before using a reranker"
            )

    def identity(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("name")
        return data

    @property
    def key(self) -> str:
        return short_id(self.identity(), 12)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrieverSpec":
        return cls(
            name=data.get("name", ""),
            mode=data["mode"],
            candidates=int(data.get("candidates", 20)),
            ef_search=int(data.get("ef_search", 40)),
            rrf_k=int(data.get("rrf_k", 60)),
            rerank_model=data.get("rerank_model", ""),
            rerank_revision=data.get("rerank_revision", ""),
            rerank_top=int(data.get("rerank_top", 0)),
        )


@dataclass(frozen=True)
class Experiments:
    chunkers: dict[str, ChunkerSpec]
    embedders: dict[str, EmbedderSpec]
    indexes: dict[str, IndexSpec]
    retrievers: dict[str, dict[str, Any]] = field(default_factory=dict)
    default_index: str = ""
    default_retriever: str = ""

    def index(self, name: str | None = None) -> IndexSpec:
        key = name or self.default_index
        if key not in self.indexes:
            raise KeyError(f"no index preset {key!r}; known: {sorted(self.indexes)}")
        return self.indexes[key]

    def retriever(self, name: str | None = None) -> RetrieverSpec:
        key = name or self.default_retriever
        if key not in self.retrievers:
            raise KeyError(f"no retriever preset {key!r}; known: {sorted(self.retrievers)}")
        return RetrieverSpec.from_dict({"name": key, **self.retrievers[key]})


def load_experiments(path: Path) -> Experiments:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    chunkers = {k: ChunkerSpec.from_dict(v) for k, v in raw.get("chunkers", {}).items()}
    embedders = {k: EmbedderSpec.from_dict(v) for k, v in raw.get("embedders", {}).items()}
    indexes = {
        name: IndexSpec(
            name=name,
            chunker=chunkers[cfg["chunker"]],
            embedder=embedders[cfg["embedder"]],
            hnsw_m=int(cfg.get("hnsw_m", 16)),
            hnsw_ef_construction=int(cfg.get("hnsw_ef_construction", 64)),
        )
        for name, cfg in raw.get("indexes", {}).items()
    }
    defaults = raw.get("defaults", {})
    return Experiments(
        chunkers=chunkers,
        embedders=embedders,
        indexes=indexes,
        retrievers=dict(raw.get("retrievers", {})),
        default_index=defaults.get("index", ""),
        default_retriever=defaults.get("retriever", ""),
    )
