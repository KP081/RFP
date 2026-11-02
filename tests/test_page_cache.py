from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from app.services import page_cache as C
from app.services.document_parser import DocumentPage


@pytest.fixture
def env(tmp_path, monkeypatch):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4 same bytes")

    monkeypatch.setattr(
        C, "get_settings", lambda: NS(data_dir=tmp_path / "data", parse_cache=True)
    )

    calls = []

    def fake_extract_pages(path, tables=False, max_pages=None):
        calls.append({"tables": tables, "max_pages": max_pages})
        pages = [
            DocumentPage(page_number=i, text=f"text of page {i}") for i in range(1, 6)
        ]
        return pages[:max_pages] if max_pages else pages

    monkeypatch.setattr(C, "extract_pages", fake_extract_pages)
    return NS(pdf=pdf, calls=calls, tmp=tmp_path)


def test_second_read_comes_from_the_cache(env):
    first = C.load_pages(env.pdf, tables=True)
    second = C.load_pages(env.pdf, tables=True)

    assert len(env.calls) == 1  # parsed once
    assert [(p.page_number, p.text) for p in second] == [
        (p.page_number, p.text) for p in first
    ]
    assert len(list((env.tmp / "data" / "parsed").glob("*.json"))) == 1


def test_tables_on_and_off_are_cached_separately(env):
    C.load_pages(env.pdf, tables=True)
    C.load_pages(env.pdf, tables=False)

    assert len(env.calls) == 2


def test_a_different_file_is_not_served_from_another_files_cache(env):
    other = env.tmp / "b.pdf"
    other.write_bytes(b"%PDF-1.4 different bytes")

    C.load_pages(env.pdf)
    C.load_pages(other)

    assert len(env.calls) == 2


def test_a_partial_read_uses_the_cache_when_it_exists_and_is_never_stored(env):
    C.load_pages(env.pdf, max_pages=2)  # no cache yet: read directly
    assert len(env.calls) == 1 and not (env.tmp / "data" / "parsed").exists()

    C.load_pages(env.pdf)  # full read: stored
    partial = C.load_pages(env.pdf, max_pages=2)  # served from the cache

    assert [p.page_number for p in partial] == [1, 2] and len(env.calls) == 2


def test_an_unreadable_cache_file_is_parsed_again(env):
    C.load_pages(env.pdf)
    next(iter((env.tmp / "data" / "parsed").glob("*.json"))).write_text(
        "{not json", encoding="utf-8"
    )

    pages = C.load_pages(env.pdf)

    assert len(pages) == 5 and len(env.calls) == 2


def test_cache_can_be_switched_off(env, monkeypatch):
    monkeypatch.setattr(
        C, "get_settings", lambda: NS(data_dir=env.tmp / "data", parse_cache=False)
    )

    C.load_pages(env.pdf)
    C.load_pages(env.pdf)

    assert len(env.calls) == 2 and not (env.tmp / "data" / "parsed").exists()


def test_the_parser_version_is_part_of_the_file_name(env, monkeypatch):
    C.load_pages(env.pdf)
    monkeypatch.setattr(C, "PARSER_VERSION", "2")

    C.load_pages(env.pdf)

    assert len(env.calls) == 2
