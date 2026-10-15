import re

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def token_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in TOKEN_RE.finditer(text)]


def count_tokens(text: str) -> int:
    return sum(1 for _ in TOKEN_RE.finditer(text))
