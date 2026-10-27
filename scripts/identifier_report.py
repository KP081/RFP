"""
What RFP / NIT / tender numbers does the capture find in your RFPs? Review it before trusting it.

    uv run python -m scripts.identifier_report --out ids_report.txt
    uv run python -m scripts.identifier_report "data/RFPs-Sample/rfp_4_0.pdf" --pages 1-40 --show-rejected
    uv run python -m scripts.identifier_report --dir uploads --all

Reads only the first pages (default 1-30: the numbers are on the cover, in the data sheet or in a
header; fast) with the table-aware parser, on the RAW text (before cleaning), like the real pipeline.
Per PDF it prints the best number per kind and, with --all, every candidate. --show-rejected lists
the labels that were found but whose value was refused, which is where a missed number hides.
Compare with what you know: Bhopal ...efile-299153, Siliguri NHIDCL/DPR/Siliguri-Guwahati/2025-26,
Gandhidham 03/2026.
"""

import argparse
import contextlib
import sys
from pathlib import Path

import pdfplumber

from app.services.document_parser import DocumentPage, _page_text_with_tables
from app.services.identifier_capture import capture_identifiers


def parse_range(value: str, total: int) -> list[int]:
    start, _, end = value.partition("-")
    last = int(end) if end else int(start)
    return [p for p in range(int(start), last + 1) if 1 <= p <= total]


def read_pages(path: Path, page_range: str) -> list[DocumentPage]:
    pages = []

    with pdfplumber.open(path) as pdf:
        for number in parse_range(page_range, len(pdf.pages)):
            page = pdf.pages[number - 1]
            try:
                text = _page_text_with_tables(page)
            except Exception:
                text = (page.extract_text() or "").strip()
            page.flush_cache()
            pages.append(DocumentPage(page_number=number, text=text))

    return pages


def describe(candidate) -> str:
    return (
        f"{candidate.value!r}  (p.{candidate.page}, label {candidate.label!r}, "
        f"{candidate.source}, strength {candidate.strength}, seen {candidate.occurrences}x)"
    )


def report(path: Path, args: argparse.Namespace) -> None:
    pages = read_pages(path, args.pages)
    result = capture_identifiers(pages)

    print("=" * 80)
    print(f"{path.name}  (pages {args.pages})")

    for kind in ("rfp", "nit", "tender_code"):
        found = result.best[kind]
        print(f"  {kind:<12}: {describe(found) if found else '-'}")

    if args.all:
        print(f"\n  all candidates ({len(result.candidates)}):")
        for candidate in result.candidates:
            print(f"     {candidate.kind:<11} {describe(candidate)}")

    if args.show_rejected:
        print(f"\n  labels found but value refused ({len(result.rejected)}), first 12:")
        for page, label, snippet in result.rejected[:12]:
            print(f"     p.{page} {label!r} -> {snippet!r}")


def run(args: argparse.Namespace) -> None:
    pdfs = args.pdfs or sorted(args.dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs given and none found in {args.dir}/")

    for pdf in pdfs:
        report(pdf, args)
        sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdfs", nargs="*", type=Path)
    parser.add_argument("--dir", type=Path, default=Path("data/RFPs-Sample"))
    parser.add_argument("--pages", default="1-30", help="page range, e.g. 1-30")
    parser.add_argument(
        "--all",
        action="store_true",
        help="list every candidate, not only the best per kind",
    )
    parser.add_argument("--show-rejected", action="store_true")
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
