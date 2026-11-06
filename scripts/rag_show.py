"""
Show what the extraction RAG flow sends for given pages (free, no LLM).

    uv run python -m scripts.rag_show 78 79 285
    uv run python -m scripts.rag_show 78 --text
    uv run python -m scripts.rag_show 96 --pdf uploads/<file>.pdf --budget 120000

Per page: its chunks, and which of them are in the evidence. With --text the text of the chunks that
ARE sent is printed, so you can see what the model really read (a table without its header, a table
cut after the rows that matter ...).
"""

import argparse
from pathlib import Path

from app.core.config import get_settings
from app.services.evidence_selector import select_evidence
from app.services.page_cache import load_pages
from app.services.rag_chunker import chunk_pages_for_extraction
from app.services.text_cleaner import clean_pages

PDF = Path("uploads/c41b1645-96b3-46ae-b0d2-4830f272d1fe.pdf")


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("pages", type=int, nargs="+")
    parser.add_argument("--pdf", type=Path, default=PDF)
    parser.add_argument("--budget", type=int, default=settings.extract_rag_budget_chars)
    parser.add_argument(
        "--text", action="store_true", help="print the text of the sent chunks"
    )
    parser.add_argument(
        "--chars", type=int, default=1500, help="max chars printed per chunk"
    )
    args = parser.parse_args()

    pages = load_pages(args.pdf, tables=settings.extract_tables)
    if settings.extract_clean_text:
        pages = clean_pages(pages).pages

    chunks = chunk_pages_for_extraction(pages, settings.extract_rag_chunk_chars)
    sent = {c.index for c in select_evidence(chunks, args.budget).chunks}

    for number in args.pages:
        here = [c for c in chunks if c.page_number == number]
        print(
            f"\n=== p.{number}: {len(here)} chunk(s), sent {sum(c.index in sent for c in here)}/{len(here)}"
        )

        for chunk in here:
            state = "SENT" if chunk.index in sent else "not sent"
            print(f"  chunk {chunk.index:>4}  {len(chunk.text):>5} chars  {state}")

            if args.text and chunk.index in sent:
                print("    " + chunk.text[: args.chars].replace("\n", "\n    "))


if __name__ == "__main__":
    main()
