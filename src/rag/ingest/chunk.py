from collections.abc import Sequence
from dataclasses import dataclass

from rag.hashing import canonical_json, sha256_text
from rag.ingest.parse import Section
from rag.ingest.tokens import token_spans
from rag.specs import ChunkerSpec

HEADING_SEPARATOR = " > "


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    ord: int
    section_ref: str
    section_refs: tuple[str, ...]
    heading_path: tuple[str, ...]
    n_tokens: int
    body: str
    embed_text: str


def windows(n_tokens: int, size: int, overlap: int) -> list[tuple[int, int]]:
    if n_tokens == 0:
        return []
    out = []
    start = 0
    step = size - overlap
    while True:
        end = min(start + size, n_tokens)
        out.append((start, end))
        if end == n_tokens:
            return out
        start += step


def chunk_id(spec: ChunkerSpec, doc_sha256: str, section_ref: str, start_char: int) -> str:
    return sha256_text(canonical_json([spec.key, doc_sha256, section_ref, start_char]))[:32]


def _embed_text(spec: ChunkerSpec, heading_path: Sequence[str], body: str) -> str:
    if spec.prepend_heading and heading_path:
        return HEADING_SEPARATOR.join(heading_path) + "\n" + body
    return body


def _heading_chunks(spec: ChunkerSpec, doc_sha256: str, sections: Sequence[Section]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        spans = token_spans(section.body)
        for start, end in windows(len(spans), spec.target_tokens, spec.overlap):
            first_char, last_char = spans[start][0], spans[end - 1][1]
            body = section.body[first_char:last_char]
            chunks.append(
                Chunk(
                    chunk_id=chunk_id(spec, doc_sha256, section.ref, first_char),
                    ord=len(chunks),
                    section_ref=section.ref,
                    section_refs=tuple(section.refs),
                    heading_path=tuple(section.heading_path),
                    n_tokens=end - start,
                    body=body,
                    embed_text=_embed_text(spec, section.heading_path, body),
                )
            )
    return chunks


def _fixed_chunks(spec: ChunkerSpec, doc_sha256: str, sections: Sequence[Section]) -> list[Chunk]:
    pieces: list[str] = []
    ranges: list[tuple[int, int, Section]] = []
    cursor = 0
    for section in sections:
        if pieces:
            pieces.append("\n\n")
            cursor += 2
        pieces.append(section.body)
        ranges.append((cursor, cursor + len(section.body), section))
        cursor += len(section.body)
    stream = "".join(pieces)
    spans = token_spans(stream)
    chunks: list[Chunk] = []
    for start, end in windows(len(spans), spec.target_tokens, spec.overlap):
        first_char, last_char = spans[start][0], spans[end - 1][1]
        touched = [s for lo, hi, s in ranges if lo < last_char and hi > first_char]
        first = touched[0]
        refs: list[str] = []
        for s in touched:
            for ref in s.refs:
                if ref not in refs:
                    refs.append(ref)
        body = stream[first_char:last_char]
        chunks.append(
            Chunk(
                chunk_id=chunk_id(spec, doc_sha256, first.ref, first_char),
                ord=len(chunks),
                section_ref=first.ref,
                section_refs=tuple(refs),
                heading_path=tuple(first.heading_path),
                n_tokens=end - start,
                body=body,
                embed_text=_embed_text(spec, first.heading_path, body),
            )
        )
    return chunks


def chunk_sections(spec: ChunkerSpec, doc_sha256: str, sections: Sequence[Section]) -> list[Chunk]:
    if spec.kind == "heading":
        return _heading_chunks(spec, doc_sha256, sections)
    return _fixed_chunks(spec, doc_sha256, sections)
