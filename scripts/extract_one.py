"""
Run ONE extraction end to end (no API, no DB) and save the JSON.

    uv run python -m scripts.extract_one uploads/<file>.pdf
    uv run python -m scripts.extract_one uploads/<file>.pdf --mode text

Prints tokens, estimated cost, time and review flags, so we can judge
quality / cost / speed of `pdf` vs `text` context on real RFPs.
"""

import argparse
import json
from pathlib import Path

from app.services.extraction_service import ExtractionService

# List prices per 1M tokens for gemini-3.5-flash-lite found on third-party price
# pages. CHECK Google's own pricing page and edit these two numbers if they differ.
PRICE_IN_PER_M = 0.30
PRICE_OUT_PER_M = 2.50


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--mode", choices=["pdf", "text"], default=None)
    parser.add_argument("--out", type=Path, default=Path("data/extractions"))
    args = parser.parse_args()

    result = ExtractionService().extract(args.pdf, mode=args.mode)
    usage = result.usage

    # thinking tokens (if any) are billed as output tokens
    cost = (
        usage.prompt_tokens * PRICE_IN_PER_M
        + (usage.output_tokens + usage.thinking_tokens) * PRICE_OUT_PER_M
    ) / 1_000_000

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"{args.pdf.stem}__{result.mode}.json"
    target.write_text(
        json.dumps(
            result.extraction.model_dump(mode="json"), indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    print(f"mode            : {result.mode}")
    print(f"time            : {result.seconds:.0f}s")
    print(f"calls / repairs : {usage.calls} / {result.repairs}")
    print(f"prompt tokens   : {usage.prompt_tokens:,}")
    print(f"output tokens   : {usage.output_tokens:,}")
    print(f"thinking tokens : {usage.thinking_tokens:,}")
    print(f"est. cost       : ${cost:.3f}")
    print(f"saved           : {target}")

    flags = result.extraction.meta.review_flags
    print(f"review flags    : {len(flags)}")
    for flag in flags:
        print(f"  - {flag}")


if __name__ == "__main__":
    main()
