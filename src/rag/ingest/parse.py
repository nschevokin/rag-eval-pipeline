import re
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

PARSER_VERSION = "parser-v1"

SECTION_CLASSES = frozenset(
    {"chapter", "appendix", "part", "preface", "reference", "refentry", "simplesect", "section"}
    | {f"sect{i}" for i in range(1, 6)}
)
DROP_CLASSES = frozenset(
    {
        "navheader",
        "navfooter",
        "toc",
        "titlepage",
        "list-of-tables",
        "list-of-figures",
        "list-of-examples",
    }
)
DROP_TAGS = frozenset({"script", "style", "noscript", "head"})
HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "pre",
        "li",
        "ul",
        "ol",
        "dl",
        "dt",
        "dd",
        "table",
        "tr",
        "thead",
        "tbody",
        "blockquote",
        "br",
        "figure",
        "caption",
        *HEADING_TAGS,
    }
)
CELL_TAGS = frozenset({"td", "th"})
WS = re.compile(r"[ \t\r\f\v]+")
SLUG = re.compile(r"[^a-z0-9]+")


@dataclass
class Section:
    ord: int
    ref: str
    refs: tuple[str, ...]
    heading_path: tuple[str, ...]
    body: str


@dataclass
class ParsedPage:
    title: str
    sections: list[Section]
    all_refs: frozenset[str]


@dataclass
class _Frame:
    ref: str
    refs: tuple[str, ...]
    path: tuple[str, ...]
    parts: list[str] = field(default_factory=list)


def _classes(tag: Tag) -> set[str]:
    return set(tag.get("class") or [])


def _clean_inline(text: str) -> str:
    return WS.sub(" ", text.replace("\n", " "))


def _heading_text(tag: Tag) -> str:
    heading = tag.find(HEADING_TAGS)
    if heading is None:
        return ""
    return " ".join(heading.get_text(" ").split())


def _slug(text: str) -> str:
    return SLUG.sub("-", text.lower()).strip("-") or "section"


def _finish(parts: list[str]) -> str:
    lines = [ln.strip() for ln in "".join(parts).split("\n")]
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out).strip()


def parse_html(html: str, page_name: str) -> ParsedPage:
    soup = BeautifulSoup(html, "html.parser")
    stem = Path(page_name).stem
    title_tag = soup.find("title")
    root = soup.find(id="docContent") or soup.body or soup
    top_title = _heading_text(root) if isinstance(root, Tag) else ""
    title = top_title or (title_tag.get_text(" ").strip() if title_tag else stem)

    frames: list[_Frame] = []
    root_frame = _Frame(ref=stem, refs=(stem,), path=(title,))
    frames.append(root_frame)

    def visit(node: Tag, frame: _Frame, in_pre: bool) -> None:
        for child in node.children:
            if isinstance(child, Comment):
                continue
            if isinstance(child, NavigableString):
                text = str(child)
                frame.parts.append(text if in_pre else _clean_inline(text))
                continue
            if not isinstance(child, Tag):
                continue
            if child.name in DROP_TAGS or _classes(child) & DROP_CLASSES:
                continue
            if child.name == "div" and _classes(child) & SECTION_CLASSES:
                heading = _heading_text(child) or child.get("id") or "untitled"
                anchor = (child.get("id") or _slug(heading)).lower()
                ref = f"{stem}#{anchor}"
                parent_path = () if frame is root_frame else frame.path
                new = _Frame(ref=ref, refs=(*frame.refs, ref), path=(*parent_path, heading))
                frames.append(new)
                visit(child, new, in_pre)
                continue
            is_pre = in_pre or child.name == "pre"
            if child.name in BLOCK_TAGS:
                frame.parts.append("\n")
            visit(child, frame, is_pre)
            if child.name in BLOCK_TAGS:
                frame.parts.append("\n")
            elif child.name in CELL_TAGS:
                frame.parts.append(" | ")

    if isinstance(root, Tag):
        visit(root, root_frame, False)

    sections: list[Section] = []
    all_refs: set[str] = set()
    for frame in frames:
        all_refs.update(frame.refs)
        body = _finish(frame.parts)
        if not body:
            continue
        path = frame.path or (title,)
        sections.append(Section(len(sections), frame.ref, frame.refs, path, body))
    return ParsedPage(title=title, sections=sections, all_refs=frozenset(all_refs))
