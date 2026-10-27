"""
Show the lines around a pattern, as the table-aware parser reads them (table rows included).

    uv run python -m scripts.find_lines "data/RFPs-Sample/RFP_DPR.pdf" --pattern "Digitally Signed by" --pages 9 --context 5
    uv run python -m scripts.find_lines a.pdf b.pdf --identifiers --pages 1-15 --out ids.txt

--identifiers is a preset for RFP / NIT / Tender number labels ("RFP No.", "Tender Number",
"NIT No", "RFP Ref. No.", "RFP Document No." ...). Use it to see how the numbers are really
written in your RFPs, in normal text AND inside tables, before writing any capture rule.
--pages takes numbers and ranges: 9   1-15   1-5 10 280-282   (only those pages are read: fast).
"""

import argparse
import contextlib
import re
import sys
from pathlib import Path

import pdfplumber

from app.services.document_parser import TABLE_END, TABLE_START, _page_text_with_tables

IDENTIFIER_LABEL = (
    r"\b(?:rfp|nit|tender|bid|enquiry|e-?tender|reference|ref)\b[^|\n]{0,40}?"
    r"\b(?:no|number|id|code|ref)\b\.?"
)


def parse_pages(values: list[str] | None, total: int) -> list[int]:
    if not values:
        return list(range(1, total + 1))

    pages: list[int] = []
    for value in values:
        if "-" in value:
            start, end = value.split("-", 1)
            pages.extend(range(int(start), int(end) + 1))
        else:
            pages.append(int(value))

    return [p for p in pages if 1 <= p <= total]


def annotate(text: str) -> list[tuple[str, bool]]:
    """Page text -> [(line, inside_table)]"""
    lines, inside = [], False

    for line in text.splitlines():
        if line == TABLE_START:
            inside = True
        elif line == TABLE_END:
            inside = False
        else:
            lines.append((line, inside))

    return lines


def search(
    pdf_path: Path,
    pattern: re.Pattern,
    pages: list[str] | None,
    context: int,
    limit: int,
) -> None:
    print("=" * 80)
    print(pdf_path.name)

    found = 0

    with pdfplumber.open(pdf_path) as pdf:
        for number in parse_pages(pages, len(pdf.pages)):
            page = pdf.pages[number - 1]
            lines = annotate(_page_text_with_tables(page))
            page.flush_cache()

            for index, (line, inside) in enumerate(lines):
                if not pattern.search(line):
                    continue

                found += 1
                if found > limit:
                    print(
                        f"  ... more matches not shown (limit {limit}); use --limit or --pages"
                    )
                    return

                print(
                    f"\n  p.{number} line {index + 1}{'  [in table]' if inside else ''}"
                )
                for position in range(
                    max(0, index - context), min(len(lines), index + context + 1)
                ):
                    mark = ">" if position == index else " "
                    print(f"   {mark} {position + 1:>3}: {lines[position][0][:150]}")

    if not found:
        print("  no match")


def run(args: argparse.Namespace) -> None:
    regex = IDENTIFIER_LABEL if args.identifiers else args.pattern
    if not regex:
        raise SystemExit("give --pattern or --identifiers")

    pattern = re.compile(regex, re.IGNORECASE)

    for pdf in args.pdfs:
        search(pdf, pattern, args.pages, args.context, args.limit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument("--pattern", help="regular expression (case-insensitive)")
    parser.add_argument(
        "--identifiers",
        action="store_true",
        help="preset: RFP / NIT / Tender number labels",
    )
    parser.add_argument(
        "--pages", nargs="*", help="page numbers or ranges, e.g. 9  1-15"
    )
    parser.add_argument(
        "--context", type=int, default=2, help="lines shown before and after a match"
    )
    parser.add_argument("--limit", type=int, default=40, help="max matches per PDF")
    parser.add_argument(
        "--out",
        type=Path,
        help="write to this file as UTF-8 (use instead of '>' on Windows)",
    )
    args = parser.parse_args()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with (
            args.out.open("w", encoding="utf-8") as handle,
            contextlib.redirect_stdout(handle),
        ):
            run(args)
        print(f"saved: {args.out}")
    else:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        run(args)


if __name__ == "__main__":
    main()
