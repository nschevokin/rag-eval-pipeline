import json
from dataclasses import asdict

from rag.ingest.parse import parse_html

from conftest import FIXTURE_CORPUS, FIXTURES

GOLDEN = FIXTURES / "golden" / "parsed_sections.json"


def parse_fixture_pages() -> dict:
    out = {}
    for path in sorted(FIXTURE_CORPUS.glob("*.html")):
        page = parse_html(path.read_text(encoding="utf-8"), path.name)
        out[path.name] = {
            "title": page.title,
            "all_refs": sorted(page.all_refs),
            "sections": [
                {**asdict(s), "refs": list(s.refs), "heading_path": list(s.heading_path)}
                for s in page.sections
            ],
        }
    return out


def test_parser_matches_golden_output():
    assert parse_fixture_pages() == json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_every_section_has_ref_heading_path_and_order():
    for page in parse_fixture_pages().values():
        assert [s["ord"] for s in page["sections"]] == list(range(len(page["sections"])))
        for s in page["sections"]:
            assert s["ref"] == s["refs"][-1]
            assert s["heading_path"]
            assert s["body"].strip()


def test_navigation_and_toc_are_dropped_but_code_and_tables_kept():
    pages = parse_fixture_pages()
    text = "\n".join(s["body"] for p in pages.values() for s in p["sections"])
    assert "Prev" not in text
    assert "1.1. Pages\n1.2." not in text
    assert "SELECT page_size, header_bytes\nFROM ledger_page_info('orders');" in text
    assert "ledger_advisory_xact_lock | transaction" in text


def test_section_refs_use_lowercase_anchors():
    refs = parse_fixture_pages()["storage.html"]["all_refs"]
    assert "storage#storage-wal-checkpoints" in refs
