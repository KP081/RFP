"""
Show HOW pdfplumber sees the tables of one page (for debugging misaligned rows).

    uv run python -m scripts.debug_table_page 355
    uv run python -m scripts.debug_table_page 355 --find "5,00,000" "3,50,000"
    uv run python -m scripts.debug_table_page 355 --from-row 20 --max-rows 14
    uv run python -m scripts.debug_table_page 355 --from-row 27 --max-rows 2 --cells
    uv run python -m scripts.debug_table_page 355 --pdf uploads/<file>.pdf

Prints, per table: the bounding box, and every row with its top/bottom (in points from
the page top) and its non-empty cells. With --find it also prints the position of the
given words, so you can check whether a value lies INSIDE the row it should belong to.
Saves data/debug/page_<n>.png with the detected cell lines drawn on the page.
"""

import argparse
from pathlib import Path

import pdfplumber

PDF = Path("uploads/c41b1645-96b3-46ae-b0d2-4830f272d1fe.pdf")
OUT = Path("data/debug")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("page", type=int)
    parser.add_argument("--pdf", type=Path, default=PDF)
    parser.add_argument("--find", nargs="*", default=[])
    parser.add_argument(
        "--cells", action="store_true", help="print x/y box of every non-empty cell"
    )
    parser.add_argument("--from-row", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=14)
    args = parser.parse_args()

    with pdfplumber.open(args.pdf) as pdf:
        page = pdf.pages[args.page - 1]
        tables = page.find_tables()
        print(
            f"page {args.page}: size {page.width:.0f}x{page.height:.0f}, tables: {len(tables)}"
        )

        for number, table in enumerate(tables, start=1):
            x0, top, x1, bottom = table.bbox
            print(
                f"\nTABLE {number}: bbox x {x0:.0f}-{x1:.0f}, y {top:.0f}-{bottom:.0f}, rows {len(table.rows)}"
            )

            shown_rows = 0
            for index, (row, values) in enumerate(
                zip(table.rows, table.extract(x_tolerance=5))
            ):
                if index < args.from_row:
                    continue
                if shown_rows >= args.max_rows:
                    print("  ...")
                    break
                shown_rows += 1
                _, row_top, _, row_bottom = row.bbox
                present = sum(1 for cell in row.cells if cell is not None)
                shown = [(i, v.replace("\n", " ")) for i, v in enumerate(values) if v]
                print(
                    f"  row {index:>2}  y {row_top:6.1f}-{row_bottom:6.1f}  "
                    f"cells {present}/{len(row.cells)}  {shown}"
                )

                if args.cells:
                    for i, (box, value) in enumerate(zip(row.cells, values)):
                        if box is not None and value:
                            cx0, ctop, cx1, cbottom = box
                            print(
                                f"        cell {i}: x {cx0:6.1f}-{cx1:6.1f}  "
                                f"y {ctop:6.1f}-{cbottom:6.1f}  {value.replace(chr(10), ' ')[:40]!r}"
                            )

        if args.find:
            print("\nWORD POSITIONS")
            for word in page.extract_words(x_tolerance=5):
                if any(target in word["text"] for target in args.find):
                    print(
                        f"  {word['text']!r:>14}  x {word['x0']:.0f}-{word['x1']:.0f}  "
                        f"y {word['top']:.1f}-{word['bottom']:.1f}"
                    )

        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / f"page_{args.page}.png"
        page.to_image(resolution=110).debug_tablefinder().save(target)
        print(f"\nsaved image: {target}")


if __name__ == "__main__":
    main()
