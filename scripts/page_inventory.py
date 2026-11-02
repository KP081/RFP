"""
How many pages of each PDF are scanned, image or table pages? Decides what a hybrid text+image
reading could save, and what the text path loses. Free: no API call.

    uv run python -m scripts.page_inventory                          # every PDF in uploads/
    uv run python -m scripts.page_inventory uploads/a.pdf uploads/b.pdf
    uv run python -m scripts.page_inventory --min-image-share 0.2    # only images >= 20% of the page

Per PDF (a page can be in more than one group):
  no-text pages : fewer than 30 characters of text (scanned page, blank page, full-page picture)
  image pages   : has text AND an image that covers >= --min-image-share of the page (chart, map,
                  stamped form). Logos and stamps are smaller than that and are ignored.
  table pages   : the table-aware reader found a [TABLE] block
Then the page count a hybrid reading would send as real page images.
"""

import argparse
import time
from pathlib import Path

import pdfplumber

from app.services.document_parser import TABLE_START, _page_text_with_tables

NO_TEXT_CHARS = 30  # same limit as extraction_service


def inventory(pdf_path: Path, min_image_share: float) -> dict:
    found = {"pages": 0, "chars": 0, "no_text": [], "image": [], "table": []}

    with pdfplumber.open(pdf_path) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            area = float(page.width) * float(page.height)
            big_image = any(
                (im["x1"] - im["x0"]) * (im["bottom"] - im["top"]) / area
                >= min_image_share
                for im in page.images
            )

            try:
                text = _page_text_with_tables(page)
            except Exception:
                text = (page.extract_text() or "").strip()
            page.flush_cache()

            found["pages"] += 1
            found["chars"] += len(text)

            if len(text) < NO_TEXT_CHARS:
                found["no_text"].append(number)
            elif big_image:
                found["image"].append(number)

            if TABLE_START in text:
                found["table"].append(number)

    return found


def share(count: int, total: int) -> str:
    return f"{count} ({count / max(total, 1):.0%})"


def report(path: Path, found: dict, seconds: float) -> None:
    total = found["pages"]
    vision = sorted(set(found["no_text"]) | set(found["image"]))
    with_tables = sorted(set(vision) | set(found["table"]))

    print("=" * 80)
    print(f"{path.name}  ({seconds:.0f}s)")
    print(f"  pages                    : {total}   text chars: {found['chars']:,}")
    print(
        f"  no-text pages            : {share(len(found['no_text']), total)}  {found['no_text'][:25]}"
    )
    print(
        f"  image pages (with text)  : {share(len(found['image']), total)}  {found['image'][:25]}"
    )
    print(
        f"  table pages              : {share(len(found['table']), total)}  first 25: {found['table'][:25]}"
    )
    print(
        f"  hybrid, scanned+image    : {share(len(vision), total)} pages go to Gemini as images"
    )
    print(
        f"  hybrid, + table pages    : {share(len(with_tables), total)} pages go to Gemini as images"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdfs", nargs="*", type=Path)
    parser.add_argument("--dir", type=Path, default=Path("data/RFPs-Sample"))
    parser.add_argument("--min-image-share", type=float, default=0.10)
    args = parser.parse_args()

    pdfs = args.pdfs or sorted(args.dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs given and none found in {args.dir}/")

    for pdf in pdfs:
        started = time.time()
        report(pdf, inventory(pdf, args.min_image_share), time.time() - started)


if __name__ == "__main__":
    main()
