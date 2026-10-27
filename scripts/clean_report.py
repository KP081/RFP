"""
What does the text cleaner remove from your RFPs? Review it before trusting it.

    uv run python -m scripts.clean_report --out clean_report.txt
    uv run python -m scripts.clean_report "data/RFPs-Sample/RFP.pdf" --samples 12
    uv run python -m scripts.clean_report --dir uploads --plain --out clean_report.txt

For every PDF: text size before/after and, per rule, how many lines were removed. The samples are
spread over the whole document (first, last and in between). READ THE REMOVED LINES: a line that
looks like real content (a heading, a clause, a value) is a false positive and the rule needs a guard.
Compare across PDFs: a rule that is clean on 12 RFPs and wrong on one tells you where to look.
"""

import argparse
import contextlib
import sys
from collections import Counter
from pathlib import Path

from app.services.document_parser import (
    extract_pages_from_pdf,
    extract_pages_from_pdf_with_tables,
)
from app.services.text_cleaner import clean_pages, line_key

RULES = ["signature", "toc", "page_stamp", "page_number", "repeated"]


def spread(items: list, count: int) -> list:
    """`count` items spread over the whole list (not just the first ones), so a rule that is only
    wrong on page 200 still shows up in the review."""
    if len(items) <= count:
        return items

    step = (len(items) - 1) / (count - 1)
    return [items[round(i * step)] for i in range(count)]


def report(name: str, pages, samples: int) -> None:
    result = clean_pages(pages)
    before = sum(len(p.text) for p in pages)
    after = sum(len(p.text) for p in result.pages)
    emptied = [
        new.page_number
        for old, new in zip(pages, result.pages)
        if old.text and not new.text
    ]

    print("=" * 80)
    print(f"{name}  ({len(pages)} pages)")
    print(
        f"  text: {before:,} -> {after:,} chars  ({(before - after) / max(before, 1):.1%} removed)"
    )
    print(f"  pages emptied by cleaning: {len(emptied)} {emptied[:10]}")

    for rule in RULES:
        items = result.removed.get(rule, [])
        print(f"\n  [{rule}] {len(items)} lines on {len({p for p, _ in items})} pages")

        if rule == "repeated":
            by_key = Counter(line_key(text) for _, text in items)
            example = {line_key(text): text for _, text in items}
            for key, count in by_key.most_common(samples):
                print(f"     {count:>4}x  {example[key][:110]!r}")
        elif rule == "page_number":
            # a real page number is "printed number - page index" = the same offset on many pages
            offsets = Counter(int(text) - page for page, text in items)
            print(
                f"     offsets (printed number - page index): {dict(offsets.most_common(4))}"
            )
            for page, text in spread(items, samples):
                print(f"     p.{page}: {text[:110]!r}")
        else:
            for page, text in spread(items, samples):
                print(f"     p.{page}: {text[:110]!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdfs", nargs="*", type=Path)
    parser.add_argument("--dir", type=Path, default=Path("data/RFPs-Sample"))
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument(
        "--plain", action="store_true", help="plain pypdf text (faster)"
    )
    parser.add_argument(
        "--out", type=Path, help="write the report to this file as UTF-8"
    )
    args = parser.parse_args()

    pdfs = args.pdfs or sorted(args.dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs given and none found in {args.dir}/")

    read = extract_pages_from_pdf if args.plain else extract_pages_from_pdf_with_tables

    def run() -> None:
        for pdf in pdfs:
            report(pdf.name, read(pdf), args.samples)
            sys.stdout.flush()

    if args.out:
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
