"""
DISCOVERY for the text cleaner (Step 2): what kind of noise do your RFPs really contain?

    uv run python -m scripts.discover_noise                       # every PDF in data/RFPs-Sample
    uv run python -m scripts.discover_noise "data/RFPs-Sample/RFP.pdf" --samples 6
    uv run python -m scripts.discover_noise --dir uploads --plain  # faster: plain pypdf text
    uv run python -m scripts.discover_noise --samples 5 --out noise_report.txt   # save as UTF-8

It writes no rules. It only SHOWS, per PDF, the candidates for each cleaning step of the flow:
  1. repeated edge lines   : first/last lines that repeat on many pages (running headers/footers)
  2. page-number lines     : "Page 5 of 68", "- 5 -", a bare "5" ...
  3. digital signatures    : "Digitally signed by ..." blocks
  4. TOC dot leaders       : "Introduction ........ 3"
  5. reference-number lines: lines with RFP / NIT / Tender ... No. (to design the identifier capture)
Read the output ACROSS PDFs: a pattern that appears in several RFPs is worth a rule, a pattern
in a single RFP is probably a quirk of that document.
"""

import argparse
import contextlib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from app.services.document_parser import (
    TABLE_END,
    TABLE_START,
    extract_pages_from_pdf,
    extract_pages_from_pdf_with_tables,
)

EDGE_LINES = 3  # how many lines at the top / bottom of a page count as "edge"
MIN_PAGE_SHARE = 0.15  # an edge line must repeat on this share of pages ...
MIN_PAGES = 4  # ... and on at least this many pages

PAGE_NUMBER_PATTERNS = {
    "page N of M": re.compile(r"^\s*page\s+\d+\s*(of|/)\s*\d+\s*$", re.I),
    "page N": re.compile(r"^\s*page\s*[-:]?\s*\d+\s*$", re.I),
    "- N -": re.compile(r"^\s*[-–—]\s*\d+\s*[-–—]\s*$"),
    "bare number": re.compile(r"^\s*\d{1,4}\s*$"),
    "N of M": re.compile(r"^\s*\d+\s+of\s+\d+\s*$", re.I),
}
SIGNATURE = re.compile(
    r"digitally\s+signed|digital\s+signature|signature\s+valid", re.I
)
DOT_LEADER = re.compile(r"(\.{5,}|(\. ){5,}|-{5,}|_{5,})\s*\d{1,4}\s*$")
REFERENCE_LINE = re.compile(
    r"\b(rfp|nit|tender|bid|enquiry|reference|ref)\b.{0,30}\b(no\.?|number|id|code)\b",
    re.I,
)


def lines_outside_tables(text: str) -> list[str]:
    """Non-empty lines of a page that are not inside a [TABLE] block."""
    lines, inside = [], False

    for line in text.splitlines():
        if line == TABLE_START:
            inside = True
        elif line == TABLE_END:
            inside = False
        elif not inside and line.strip():
            lines.append(line.strip())

    return lines


def shape(line: str) -> str:
    """Line -> comparable shape: lower case, digits as '#', spaces collapsed."""
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line.lower())).strip()


def repeated_edge_lines(pages) -> list[tuple[str, str, int, str]]:
    """(position, shape, pages, example) for edge lines that repeat on many pages."""
    seen: dict[tuple[str, str], set[int]] = defaultdict(set)
    example: dict[tuple[str, str], str] = {}

    for page in pages:
        lines = lines_outside_tables(page.text)
        edges = [("top", line) for line in lines[:EDGE_LINES]]
        edges += [("bottom", line) for line in lines[-EDGE_LINES:]]

        for position, line in edges:
            key = (position, shape(line))
            seen[key].add(page.page_number)
            example.setdefault(key, line)

    needed = max(MIN_PAGES, int(MIN_PAGE_SHARE * len(pages)))
    found = [
        (position, key, len(numbers), example[(position, key)])
        for (position, key), numbers in seen.items()
        if len(numbers) >= needed and len(key) > 2
    ]

    return sorted(found, key=lambda item: -item[2])


def scan(pages) -> dict:
    page_numbers = Counter()
    page_number_examples: dict[str, str] = {}
    signatures, leaders, references = [], [], []

    for page in pages:
        lines = lines_outside_tables(page.text)

        for index, line in enumerate(lines):
            for name, pattern in PAGE_NUMBER_PATTERNS.items():
                if pattern.match(line):
                    page_numbers[name] += 1
                    page_number_examples.setdefault(name, line)

            if SIGNATURE.search(line):
                following = lines[index + 1] if index + 1 < len(lines) else ""
                signatures.append((page.page_number, line[:100], following[:100]))

            if DOT_LEADER.search(line):
                leaders.append((page.page_number, line[:100]))

            if (
                page.page_number <= 40
                and REFERENCE_LINE.search(line)
                and len(line) < 200
            ):
                following = lines[index + 1] if index + 1 < len(lines) else ""
                references.append((page.page_number, line[:150], following[:80]))

    return {
        "edges": repeated_edge_lines(pages),
        "page_numbers": page_numbers,
        "page_number_examples": page_number_examples,
        "signatures": signatures,
        "leaders": leaders,
        "references": references,
    }


def report(name: str, pages, found: dict, samples: int) -> None:
    print("=" * 80)
    print(f"{name}  ({len(pages)} pages)")

    print(f"\n  1. REPEATED EDGE LINES (>= {MIN_PAGES} pages): {len(found['edges'])}")
    for position, key, count, text in found["edges"][:samples]:
        print(f"     {position:<6} {count:>4} pages  {text[:100]!r}")

    print("\n  2. PAGE-NUMBER LINES (anywhere on the page, outside tables)")
    if not found["page_numbers"]:
        print("     none")
    for kind, count in found["page_numbers"].most_common():
        print(
            f"     {kind:<12} {count:>5}  e.g. {found['page_number_examples'][kind]!r}"
        )

    print(f"\n  3. DIGITAL SIGNATURE LINES: {len(found['signatures'])}")
    for page, line, following in found["signatures"][:samples]:
        print(f"     p.{page}: {line!r}  ->  {following!r}")

    pages_with_leaders = sorted({page for page, _ in found["leaders"]})
    print(
        f"\n  4. TOC DOT-LEADER LINES: {len(found['leaders'])} on {len(pages_with_leaders)} pages {pages_with_leaders[:12]}"
    )
    for page, line in found["leaders"][: min(samples, 3)]:
        print(f"     p.{page}: {line!r}")

    print(f"\n  5. REFERENCE-NUMBER LINES (first 40 pages): {len(found['references'])}")
    for page, line, following in found["references"][: samples + 4]:
        print(f"     p.{page}: {line!r}  ->  {following!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdfs", nargs="*", type=Path)
    parser.add_argument("--dir", type=Path, default=Path("data/RFPs-Sample"))
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument(
        "--plain", action="store_true", help="plain pypdf text (faster)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="write the report to this file as UTF-8 (use this instead of '>' on Windows)",
    )
    args = parser.parse_args()

    pdfs = args.pdfs or sorted(args.dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs given and none found in {args.dir}/")

    read = extract_pages_from_pdf if args.plain else extract_pages_from_pdf_with_tables

    def run() -> None:
        for pdf in pdfs:
            pages = read(pdf)
            report(pdf.name, pages, scan(pages), args.samples)
            sys.stdout.flush()

    if args.out:
        # Windows: '>' redirects with the cp1252 code page and crashes on characters such as
        # U+2010. Writing the file ourselves as UTF-8 avoids that.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with (
            args.out.open("w", encoding="utf-8") as handle,
            contextlib.redirect_stdout(handle),
        ):
            run()
        print(f"saved: {args.out}")
    else:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        run()


if __name__ == "__main__":
    main()
