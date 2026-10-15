import json

import pytest

from rag.ingest.manifest import (
    CorpusVerificationError,
    build_manifest,
    fetch_corpus,
    load_manifest,
    verify_corpus,
)

from conftest import local_fetcher, make_fixture_manifest


def test_manifest_round_trips_and_verifies(tmp_path):
    manifest, raw = make_fixture_manifest(tmp_path)
    loaded = load_manifest(tmp_path / "manifest.json")
    assert loaded == manifest
    assert len(loaded.corpus_sha256) == 64
    verify_corpus(loaded, raw)


def test_changing_one_byte_fails_verification_with_the_file_name(tmp_path):
    manifest, raw = make_fixture_manifest(tmp_path)
    target = raw / "locking.html"
    data = bytearray(target.read_bytes())
    data[100] ^= 0x01
    target.write_bytes(bytes(data))
    with pytest.raises(CorpusVerificationError) as err:
        verify_corpus(manifest, raw)
    assert len(err.value.problems) == 1
    assert err.value.problems[0].startswith("locking.html:")


def test_missing_file_fails_verification(tmp_path):
    manifest, raw = make_fixture_manifest(tmp_path)
    (raw / "search.html").unlink()
    with pytest.raises(CorpusVerificationError, match=r"search\.html: missing"):
        verify_corpus(manifest, raw)


def test_manifest_is_never_overwritten(tmp_path):
    make_fixture_manifest(tmp_path)
    with pytest.raises(FileExistsError):
        build_manifest(
            ["https://example.test/docs/search.html"],
            corpus="x",
            license="x",
            source="x",
            raw_dir=tmp_path / "raw",
            out=tmp_path / "manifest.json",
            fetcher=local_fetcher,
        )


def test_fetch_rejects_a_download_that_does_not_match(tmp_path):
    manifest, _ = make_fixture_manifest(tmp_path)
    with pytest.raises(CorpusVerificationError, match=r"storage\.html"):
        fetch_corpus(manifest, tmp_path / "fresh", lambda url: b"tampered upstream")


def test_fetch_downloads_only_missing_files(tmp_path):
    manifest, raw = make_fixture_manifest(tmp_path)
    (raw / "search.html").unlink()
    assert fetch_corpus(manifest, raw, local_fetcher) == ["search.html"]
    verify_corpus(manifest, raw)


def test_corpus_hash_ignores_order_and_urls(tmp_path):
    manifest, _ = make_fixture_manifest(tmp_path)
    data = json.loads((tmp_path / "manifest.json").read_text())
    data["files"].reverse()
    for f in data["files"]:
        f["url"] = f["url"].replace("example.test", "mirror.test")
    (tmp_path / "other.json").write_text(json.dumps(data))
    assert load_manifest(tmp_path / "other.json").corpus_sha256 == manifest.corpus_sha256
