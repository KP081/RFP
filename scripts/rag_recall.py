"""
FREE retrieval eval for the extraction RAG flow: does the selected evidence contain the PAGES that
hold the facts? No LLM call, no embeddings.

    uv run python -m scripts.rag_recall
    uv run python -m scripts.rag_recall --only bhopal --budgets 80000 120000 160000 --show-missing
    uv run python -m scripts.rag_recall --only bhopal --budgets 120000 --min-scores 0.5 --explain 96 285

The pages to find come from eval/references/*.json (every check has a "page"). Trap checks (the
document does NOT state the value) are left out: not finding their page is fine. The text comes from
the parse cache (data/parsed), cleaned like text mode. Per setting it prints how big the evidence is,
how many check pages it contains, and how much of the text ON those pages was kept (a table cut in
half shows up there, not in the page count).

--min-scores compares the threshold a chunk needs to use its group's protected share (0 = old
behaviour, groups fill their share with anything). --explain PAGE ... shows, for the LAST budget and
min-score, why a page was or was not picked: per group the best score on that page, its rank among all
chunks, and whether the group picked it. Rank far down = scoring problem; rank high but not picked =
budget problem.

HOLD-OUT references are measured, never used to tune queries or keywords.
"""

import argparse
from collections import defaultdict
from pathlib import Path

from app.core.config import get_settings
from app.services.evidence_selector import MIN_SCORE, score_groups, select_evidence
from app.services.page_cache import load_pages
from app.services.rag_chunker import chunk_pages_for_extraction
from app.services.text_cleaner import clean_pages
from scripts.eval_extraction import load_references


def explain(chunks, selection, pages: list[int]) -> None:
    scores = score_groups(chunks)
    final = {c.index for c in selection.chunks}

    for page in pages:
        indexes = [c.index for c in chunks if c.page_number == page]
        if not indexes:
            print(f"    explain p.{page}: no text on this page")
            continue

        print(
            f"    explain p.{page}: {len(indexes)} chunk(s), in final evidence: "
            f"{sum(i in final for i in indexes)}/{len(indexes)}"
        )

        for name, group_scores in scores.items():
            best_index = max(indexes, key=lambda i: group_scores[i])
            best = group_scores[best_index]
            rank = (
                f"#{1 + sum(1 for value in group_scores if value > best)}"
                if best > 0
                else "-"
            )
            picked = best_index in selection.groups.get(name, [])
            print(
                f"       {name:<15} best {best:4.2f}  rank {rank:<5} picked by group: {'yes' if picked else 'no'}"
            )


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="reference name contains this")
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="*",
        default=[40_000, 80_000, settings.extract_rag_budget_chars],
    )
    parser.add_argument("--min-scores", type=float, nargs="*", default=[0.0, MIN_SCORE])
    parser.add_argument(
        "--chunk-chars", type=int, default=settings.extract_rag_chunk_chars
    )
    parser.add_argument("--show-missing", action="store_true")
    parser.add_argument("--include-traps", action="store_true")
    parser.add_argument("--explain", type=int, nargs="*", default=[])
    args = parser.parse_args()

    totals = defaultdict(lambda: [0, 0])  # (min_score, budget) -> [found, wanted]

    for ref in load_references(args.only):
        pdf = Path(ref["pdf"])
        if not pdf.exists():
            print(f"\n[{ref['name']}] skipped: {pdf} not found")
            continue

        pages = load_pages(pdf, tables=settings.extract_tables)
        if settings.extract_clean_text:
            pages = clean_pages(pages).pages

        chunks = chunk_pages_for_extraction(pages, args.chunk_chars)
        page_chars: dict[int, int] = defaultdict(int)
        for chunk in chunks:
            page_chars[chunk.page_number] += len(chunk.text)

        wanted: dict[int, list[str]] = defaultdict(list)
        for check in ref["checks"]:
            if check.get("page") and (args.include_traps or not check.get("trap")):
                wanted[check["page"]].append(check["path"])

        hold = (
            "  (HOLD-OUT: measure only, do not tune)"
            if "HOLD-OUT" in ref.get("about", "")
            else ""
        )
        total = sum(len(c.text) for c in chunks)
        print(
            f"\n=== {ref['name']}: {len(pages)} pages, {len(chunks)} chunks, {total:,} chars{hold}"
        )

        for min_score in args.min_scores:
            for budget in args.budgets:
                selection = select_evidence(chunks, budget, min_score=min_score)
                kept: dict[int, int] = defaultdict(int)
                for chunk in selection.chunks:
                    kept[chunk.page_number] += len(chunk.text)

                found = [p for p in wanted if kept.get(p)]
                missing = {p: paths for p, paths in wanted.items() if not kept.get(p)}
                on_pages = sum(page_chars[p] for p in wanted)
                kept_on_pages = sum(kept.get(p, 0) for p in wanted)
                totals[(min_score, budget)][0] += len(found)
                totals[(min_score, budget)][1] += len(wanted)

                print(
                    f"  min {min_score:.1f} budget {budget:>8,}: {selection.chars:>8,} chars "
                    f"(~{selection.chars // 4:,} tokens, {selection.chars / max(total, 1):.0%} of doc) "
                    f"| pages found {len(found)}/{len(wanted)} | text kept on them {kept_on_pages / max(on_pages, 1):.0%}"
                )

                if args.show_missing:
                    for page, paths in sorted(missing.items())[:15]:
                        print(f"       missing p.{page}: {paths[:3]}")

        if args.explain:
            print(
                f"  --- explain (budget {args.budgets[-1]:,}, min-score {args.min_scores[-1]})"
            )
            explain(
                chunks,
                select_evidence(
                    chunks, args.budgets[-1], min_score=args.min_scores[-1]
                ),
                args.explain,
            )

    if totals:
        print("\n" + "=" * 70)
        for (min_score, budget), (found, wanted) in totals.items():
            print(
                f"min {min_score:.1f} budget {budget:>8,}: {found}/{wanted} check pages found "
                f"({found / max(wanted, 1):.0%})"
            )


if __name__ == "__main__":
    main()
