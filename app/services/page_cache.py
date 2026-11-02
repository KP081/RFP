"""
Parsed-pages cache: reading a 400-page PDF with tables takes 50-80 seconds, and the result is the same
every time for the same file. The pages are stored once, keyed by the file's content hash:

    data/parsed/<hash>__tables-1__v1.json

Only the PARSED pages are cached (document_parser output). The text cleaner runs after the cache: it is
fast, and its rules are still being tuned. Delete data/parsed/ to start fresh, or bump PARSER_VERSION
when document_parser changes what it returns (the old files are then simply not used).

A partial read (max_pages, used for the identifier check) is served from the cache when it exists, and is
otherwise read directly and NOT stored: a cache file always holds the whole document.
"""

import hashlib
import json
import logging
from pathlib import Path

from app.core.config import get_settings
from app.services.document_parser import DocumentPage, extract_pages

logger = logging.getLogger(__name__)

PARSER_VERSION = "1"


def file_hash(file_path: Path) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _cache_file(file_path: Path, tables: bool) -> Path:
    name = f"{file_hash(file_path)[:24]}__tables-{int(tables)}__v{PARSER_VERSION}.json"

    return get_settings().data_dir / "parsed" / name


def _read(target: Path) -> list[DocumentPage] | None:
    try:
        rows = json.loads(target.read_text(encoding="utf-8"))
        return [
            DocumentPage(page_number=r["page_number"], text=r["text"]) for r in rows
        ]
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning(
            "Parsed-pages cache file %s is unreadable; parsing again", target
        )
        return None


def _write(target: Path, pages: list[DocumentPage]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = target.with_suffix(".tmp")
    rows = [{"page_number": p.page_number, "text": p.text} for p in pages]
    tmp_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(target)


def load_pages(
    file_path: Path, tables: bool = False, max_pages: int | None = None
) -> list[DocumentPage]:
    """extract_pages() with the cache in front of it."""
    if not get_settings().parse_cache:
        return extract_pages(file_path, tables=tables, max_pages=max_pages)

    target = _cache_file(file_path, tables)

    if target.exists():
        pages = _read(target)

        if pages is not None:
            logger.info(
                "Parsed pages from cache: %s (%d pages)", target.name, len(pages)
            )
            return pages[:max_pages] if max_pages is not None else pages

    pages = extract_pages(file_path, tables=tables, max_pages=max_pages)

    if max_pages is None:
        _write(target, pages)
        logger.info("Parsed pages stored: %s (%d pages)", target.name, len(pages))

    return pages
