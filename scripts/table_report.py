"""
Quality SIGNALS of table-aware reading, for any number of PDFs (not an accuracy test).

    uv run python -m scripts.table_report                       # every PDF in uploads/
    uv run python -m scripts.table_report uploads/a.pdf uploads/b.pdf
    uv run python -m scripts.table_report --samples 8

For each PDF it prints counts that point at problems WITHOUT knowing the right answer:
  tiny tables      : tables with 1-2 rows (often fragments or false detections)
  duplicate cells  : a row that holds the same long text twice (nested-box artifact)
  no-text pages    : pages the text reader cannot read at all
  table share      : how much of the text now sits inside [TABLE] blocks
Compare the numbers ACROSS different RFPs. A problem that shows up in only one RFP is a
sign that a rule was tuned for that RFP. Sample lines tell you which pages to open.
"""

import argparse
import re
import time
from pathlib import Path

from app.services.document_parser import (
    TABLE_END,
    TABLE_START,
    extract_pages_from_pdf_with_tables,
)

NO_TEXT_CHARS = 30
MIN_DUPLICATE_LENGTH = 6  # shorter cells ("5%", "0") may legitimately repeat


def table_blocks(text: str) -> list[list[str]]:
    """All [TABLE] blocks of a page text, each as a list of row lines."""
    blocks, current = [], None

    for line in text.splitlines():
        if line == TABLE_START:
            current = []
        elif line == TABLE_END and current is not None:
            blocks.append(current)
            current = None
        elif current is not None:
            current.append(line)

    return blocks


def duplicate_cells(row: str) -> list[str]:
    """Cells that appear twice in one row and look like real text (not '5%' or '0')."""
    seen, repeated = set(), []

    for cell in (c.strip().lower() for c in row.split(" | ")):
        if len(cell) >= MIN_DUPLICATE_LENGTH and re.search(r"[a-z]", cell):
            if cell in seen and cell not in repeated:
                repeated.append(cell)
            seen.add(cell)

    return repeated


def analyze(pages) -> dict:
    stats = {
        "pages": len(pages),
        "no_text_pages": [p.page_number for p in pages if len(p.text) < NO_TEXT_CHARS],
        "pages_with_tables": 0,
        "tables": 0,
        "rows": 0,
        "table_chars": 0,
        "all_chars": sum(len(p.text) for p in pages),
        "tiny_tables": [],  # (page, first row)
        "duplicate_rows": [],  # (page, row)
    }

    for page in pages:
        blocks = table_blocks(page.text)
        stats["pages_with_tables"] += bool(blocks)

        for block in blocks:
            stats["tables"] += 1
            stats["rows"] += len(block)
            stats["table_chars"] += sum(len(row) for row in block)

            if len(block) <= 2:
                stats["tiny_tables"].append(
                    (page.page_number, block[0][:90] if block else "")
                )

            for row in block:
                if duplicate_cells(row):
                    stats["duplicate_rows"].append((page.page_number, row[:140]))

    return stats


def report(name: str, stats: dict, seconds: float, samples: int) -> None:
    share = stats["table_chars"] / stats["all_chars"] if stats["all_chars"] else 0
    pages = stats["pages"]

    print("=" * 80)
    print(name)
    print(f"  pages               : {pages}  ({seconds:.0f}s)")
    print(
        f"  pages with tables   : {stats['pages_with_tables']} ({stats['pages_with_tables'] / max(pages, 1):.0%})"
    )
    print(f"  tables / rows       : {stats['tables']} / {stats['rows']}")
    print(f"  table share of text : {share:.0%}")
    print(
        f"  no-text pages       : {len(stats['no_text_pages'])}  {stats['no_text_pages'][:15]}"
    )
    print(f"  tiny tables (<=2)   : {len(stats['tiny_tables'])}")
    for page, row in stats["tiny_tables"][:samples]:
        print(f"      p.{page}: {row}")
    print(f"  duplicate-cell rows : {len(stats['duplicate_rows'])}")
    for page, row in stats["duplicate_rows"][:samples]:
        print(f"      p.{page}: {row}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdfs", nargs="*", type=Path)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    pdfs = args.pdfs or sorted(Path("data/RFPs-Sample").glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDFs given and none found in uploads/")

    for pdf in pdfs:
        started = time.time()
        pages = extract_pages_from_pdf_with_tables(pdf)
        report(pdf.name, analyze(pages), time.time() - started, args.samples)


if __name__ == "__main__":
    main()
