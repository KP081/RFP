"""
Compare plain page reading (pypdf) with table-aware reading (pdfplumber).

    uv run python -m scripts.compare_table_parsing
    uv run python -m scripts.compare_table_parsing --pages 280 355
    uv run python -m scripts.compare_table_parsing "path/to/other.pdf" --no-stats --pages 294

Part 1 (stats, skipped with --no-stats): tables / no-text pages and time for the whole PDF.
Part 2 (side by side): only the requested pages, read both ways, so YOU can judge.
With --no-stats only the requested pages are read (seconds, not a full pass).
Default pages are table-heavy pages of the Bhopal-Kota RFP (personnel, marks, BOQ).
"""

import argparse
import time
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

from app.services.document_parser import (
    TABLE_START,
    _page_text_with_tables,
    extract_pages_from_pdf,
    extract_pages_from_pdf_with_tables,
)

PDF = Path("uploads/c41b1645-96b3-46ae-b0d2-4830f272d1fe.pdf")
DEFAULT_PAGES = [280, 287, 355, 78, 282]
NO_TEXT_CHARS = 30  # same limit as extraction_service


def print_stats(pdf_path: Path) -> tuple[list, list]:
    started = time.time()
    plain = extract_pages_from_pdf(pdf_path)
    plain_seconds = time.time() - started

    started = time.time()
    structured = extract_pages_from_pdf_with_tables(pdf_path)
    structured_seconds = time.time() - started

    table_pages = [p.page_number for p in structured if TABLE_START in p.text]
    empty_pages = [p.page_number for p in structured if len(p.text) < NO_TEXT_CHARS]

    print("=" * 80)
    print(f"pages                : {len(structured)}")
    print(f"pages with tables    : {len(table_pages)}  first 30: {table_pages[:30]}")
    print(f"pages with no text   : {len(empty_pages)}  {empty_pages[:30]}")
    print(
        f"plain reading        : {plain_seconds:.1f}s, {sum(len(p.text) for p in plain):,} chars"
    )
    print(
        f"table-aware reading  : {structured_seconds:.1f}s, {sum(len(p.text) for p in structured):,} chars"
    )

    return plain, structured


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", nargs="?", type=Path, default=PDF)
    parser.add_argument("--pages", type=int, nargs="*", default=DEFAULT_PAGES)
    parser.add_argument(
        "--no-stats", action="store_true", help="read only the requested pages"
    )
    parser.add_argument(
        "--chars", type=int, default=1500, help="max chars shown per page"
    )
    args = parser.parse_args()

    plain = structured = None
    if not args.no_stats:
        plain, structured = print_stats(args.pdf)

    reader = PdfReader(args.pdf)
    total = len(reader.pages)

    with pdfplumber.open(args.pdf) as pdf:
        for number in args.pages:
            if not 1 <= number <= total:
                print(f"\npage {number} does not exist (PDF has {total} pages)")
                continue

            if plain is not None:
                old_text = plain[number - 1].text
                new_text = structured[number - 1].text
            else:
                old_text = (reader.pages[number - 1].extract_text() or "").strip()
                page = pdf.pages[number - 1]
                new_text = _page_text_with_tables(page)
                page.flush_cache()

            print("\n" + "=" * 80)
            print(f"PAGE {number} - PLAIN (pypdf)")
            print("-" * 80)
            print(old_text[: args.chars])
            print("\n" + "=" * 80)
            print(f"PAGE {number} - TABLE-AWARE (pdfplumber)")
            print("-" * 80)
            print(new_text[: args.chars])


if __name__ == "__main__":
    main()
