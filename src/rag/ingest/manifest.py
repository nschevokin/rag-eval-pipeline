import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from rag.hashing import canonical_json, sha256_bytes, sha256_file, sha256_text

Fetcher = Callable[[str], bytes]

USER_AGENT = "rag-eval-pipeline/0.1 (portfolio project; one-off corpus download)"


class CorpusVerificationError(RuntimeError):
    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("corpus verification failed:\n  " + "\n  ".join(problems))


@dataclass(frozen=True)
class ManifestFile:
    name: str
    url: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class Manifest:
    corpus: str
    license: str
    source: str
    files: tuple[ManifestFile, ...]

    @property
    def corpus_sha256(self) -> str:
        entries = sorted(
            ({"name": f.name, "sha256": f.sha256} for f in self.files), key=lambda e: e["name"]
        )
        return sha256_text(canonical_json(entries))

    def to_dict(self) -> dict:
        return {
            "corpus": self.corpus,
            "license": self.license,
            "source": self.source,
            "files": [asdict(f) for f in self.files],
        }


def load_manifest(path: Path) -> Manifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    files = tuple(ManifestFile(**f) for f in data["files"])
    names = [f.name for f in files]
    if len(set(names)) != len(names):
        raise ValueError("manifest has duplicate file names")
    return Manifest(data["corpus"], data["license"], data["source"], files)


def http_fetcher(timeout: float = 30.0) -> Fetcher:
    client = httpx.Client(
        timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    )

    def fetch(url: str) -> bytes:
        response = client.get(url)
        response.raise_for_status()
        return response.content

    return fetch


def file_name_for(url: str) -> str:
    name = Path(urlparse(url).path).name
    if not name:
        raise ValueError(f"cannot derive a file name from {url}")
    return name


def read_sources(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def build_manifest(
    urls: Iterable[str],
    *,
    corpus: str,
    license: str,
    source: str,
    raw_dir: Path,
    out: Path,
    fetcher: Fetcher,
) -> Manifest:
    urls = list(urls)
    if not urls:
        raise ValueError("no source URLs; run `make discover` to fill corpus/sources.txt")
    if out.exists():
        raise FileExistsError(
            f"{out} exists and is never edited; a new corpus needs a new manifest"
        )
    files = []
    for url in urls:
        data = fetcher(url)
        name = file_name_for(url)
        _write_atomic(raw_dir / name, data)
        files.append(ManifestFile(name=name, url=url, sha256=sha256_bytes(data), bytes=len(data)))
    manifest = Manifest(corpus, license, source, tuple(files))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8")
    return manifest


def fetch_corpus(manifest: Manifest, raw_dir: Path, fetcher: Fetcher) -> list[str]:
    downloaded = []
    problems = []
    for f in manifest.files:
        target = raw_dir / f.name
        if target.exists() and sha256_file(target) == f.sha256:
            continue
        data = fetcher(f.url)
        digest = sha256_bytes(data)
        if digest != f.sha256:
            problems.append(f"{f.name}: downloaded sha256 {digest} != manifest {f.sha256}")
            continue
        _write_atomic(target, data)
        downloaded.append(f.name)
    if problems:
        raise CorpusVerificationError(problems)
    return downloaded


def verify_corpus(manifest: Manifest, raw_dir: Path) -> None:
    problems = []
    for f in manifest.files:
        target = raw_dir / f.name
        if not target.exists():
            problems.append(f"{f.name}: missing")
            continue
        digest = sha256_file(target)
        if digest != f.sha256:
            problems.append(f"{f.name}: sha256 {digest} != manifest {f.sha256}")
    if problems:
        raise CorpusVerificationError(problems)
