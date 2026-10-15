import re
from collections.abc import Sequence

from rag.evaluate.answer_metrics import Answer
from rag.retrieve.types import ChunkRecord
from rag.text import STOPWORDS

SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
WORD = re.compile(r"[a-z0-9]+")


def _content_words(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 1}


class ExtractiveGenerator:
    name = "extractive-v1"

    def __init__(self, min_overlap: float = 0.5, max_sentences: int = 3) -> None:
        self.min_overlap = min_overlap
        self.max_sentences = max_sentences

    def generate(self, question: str, chunks: Sequence[ChunkRecord]) -> Answer:
        wanted = _content_words(question)
        if not wanted or not chunks:
            return Answer("", (), True)
        scored: list[tuple[float, int, int, str]] = []
        for c_index, chunk in enumerate(chunks):
            for s_index, sentence in enumerate(SENTENCE.split(chunk.body)):
                words = _content_words(sentence)
                if not words:
                    continue
                overlap = len(words & wanted) / len(wanted)
                scored.append((overlap, c_index, s_index, sentence.strip()))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        if not scored or scored[0][0] < self.min_overlap:
            return Answer("", (), True)
        picked = scored[: self.max_sentences]
        cited: list[str] = []
        for _, c_index, _, _ in picked:
            cid = chunks[c_index].chunk_id
            if cid not in cited:
                cited.append(cid)
        text = " ".join(sentence for *_, sentence in picked)
        return Answer(text, tuple(cited), False)
