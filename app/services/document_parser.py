import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

logger = logging.getLogger(__name__)

TABLE_START = "[TABLE]"
TABLE_END = "[/TABLE]"
MIN_SEGMENT_HEIGHT = 2  # points; thinner gaps cannot hold a text line
TABLE_CELL_X_TOLERANCE = 5  # default 3 splits widely spaced digits: "12" -> "1 2"


@dataclass
class DocumentPage:
    page_number: int
    text: str


def extract_pages_from_pdf(file_path: Path) -> list[DocumentPage]:
    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        pages.append(
            DocumentPage(
                page_number=page_number,
                text=text.strip(),
            )
        )

    return pages


# ---------- table-aware PDF reading (pdfplumber) ----------


# A cell that is ONLY single digits separated by spaces ("1 2", "1 1") is one number whose
# digits sit far apart in the PDF. Safety net for gaps larger than TABLE_CELL_X_TOLERANCE.
_SPLIT_DIGITS = re.compile(r"\d(?: \d){1,9}")


def _clean_cell(cell: str | None) -> str:
    text = re.sub(r"\s+", " ", cell or "").strip()

    if _SPLIT_DIGITS.fullmatch(text):
        text = text.replace(" ", "")

    return text


def _render_rows(rows: list[list[str]]) -> str:
    """Rows -> one line per row, cells joined with ' | '.

    Empty cells are dropped (same rule as the DOCX reader): in messy PDF tables they are
    mostly merged-cell or nested-box leftovers, and keeping them only adds " |  |  | " noise.
    """
    lines = []

    for row in rows:
        cells = [cell for cell in row if cell]

        if cells:
            lines.append(" | ".join(cells))

    if not lines:
        return ""

    return "\n".join([TABLE_START, *lines, TABLE_END])


def _is_fragment(row, previous_bbox) -> bool:
    """A 'fragment row' is a sliver that sits INSIDE the previous row and has real cells in
    only a few columns. It appears when a cell has an extra line drawn inside it (RFP page 355:
    the Rate value sits in a half-height box). It belongs to the row above, not to a new row.

    Why the 'few cells' test: a row below a row-span cell also sits inside the previous row's
    box, but it still has cells in most columns, so it must stay a real row.
    """
    _, top, _, bottom = row.bbox
    _, previous_top, _, previous_bottom = previous_bbox

    inside = top > previous_top + 0.5 and bottom <= previous_bottom + 0.5
    present = sum(1 for cell in row.cells if cell is not None)

    return inside and present * 2 < len(row.cells)


def _inside(inner: tuple, outer: tuple, tolerance: float = 1.0) -> bool:
    """Is box `inner` (x0, top, x1, bottom) fully inside box `outer`?"""
    return (
        inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _already_counted(text: str, box: tuple, owners: list[tuple[tuple, str]]) -> bool:
    """True if these characters were already read by a bigger cell that contains this box.

    Real table cells never overlap. When a nested box is drawn inside a cell, pdfplumber
    returns BOTH cells and both extract the same characters. Two conditions must hold:
    the box lies inside the owner's box AND the owner's text contains this text. Equal
    values in two separate cells (5% | 5%) fail the first condition and are kept.
    """
    key = _squash(text)

    return any(
        owner_box != box and _inside(box, owner_box) and key in _squash(owner_text)
        for owner_box, owner_text in owners
    )


def _table_rows(table) -> list[list[str]]:
    """Table -> rows of clean cell text.

    1. A cell inside another cell of the same row is dropped (its text is already there).
    2. A fragment row is folded into the row it belongs to; fragment text that a cell of
       that row already holds (by position, see _already_counted) is not added again.
    """
    values = table.extract(x_tolerance=TABLE_CELL_X_TOLERANCE)
    merged: list[tuple[tuple, list[str], list[tuple[tuple, str]]]] = []

    for row, raw in zip(table.rows, values):
        cells = [_clean_cell(cell) for cell in raw]
        owners = [
            (box, text)
            for box, text in zip(row.cells, cells)
            if box is not None and text
        ]

        for index, box in enumerate(row.cells):
            if (
                box is not None
                and cells[index]
                and _already_counted(cells[index], box, owners)
            ):
                cells[index] = ""

        if merged and _is_fragment(row, merged[-1][0]):
            target, target_owners = merged[-1][1], merged[-1][2]

            for index, text in enumerate(cells):
                box = row.cells[index]

                if text and not (
                    box is not None and _already_counted(text, box, target_owners)
                ):
                    target[index] = f"{target[index]} {text}".strip()

            continue

        merged.append((row.bbox, cells, owners))

    return [cells for _, cells, _ in merged]


def _has_columns(rows: list[list[str]]) -> bool:
    """A table needs at least one row with two or more filled cells.

    A single boxed heading or a one-column list has no column structure; wrapping it in
    [TABLE] would only mislead, so such boxes are read as plain text.
    """
    return any(sum(1 for cell in row if cell) >= 2 for row in rows)


def _drop_nested_tables(tables: list) -> list:
    """Remove tables that lie fully inside another table (small boxes drawn inside cells).

    Their text is already part of the outer table; keeping them would repeat it as
    extra [TABLE] blocks with broken fragments.
    """
    kept: list = []

    for table in tables:
        x0, top, x1, bottom = table.bbox

        nested = any(
            x0 >= ox0 - 1
            and top >= otop - 1
            and x1 <= ox1 + 1
            and bottom <= obottom + 1
            for ox0, otop, ox1, obottom in (outer.bbox for outer in kept)
        )

        if not nested:
            kept.append(table)

    return kept


def _page_text_with_tables(page) -> str:
    """Page text in reading order, tables rendered as rows.

    Idea: find the table boxes, then walk the page top to bottom:
    text above table 1, table 1, text between 1 and 2, table 2, ... text below.
    Text inside a table box is NOT read as plain text (the table rows already hold it).
    """
    tables = _drop_nested_tables(sorted(page.find_tables(), key=lambda t: t.bbox[1]))

    if not tables:
        return (page.extract_text() or "").strip()

    parts: list[str] = []
    cursor = 0.0  # y position (from page top) up to which we have already read

    def read_box(x0: float, top: float, x1: float, bottom: float) -> None:
        x0, x1 = max(x0, 0.0), min(x1, float(page.width))
        top, bottom = max(top, 0.0), min(bottom, float(page.height))

        if bottom - top < MIN_SEGMENT_HEIGHT or x1 - x0 < MIN_SEGMENT_HEIGHT:
            return

        text = (page.crop((x0, top, x1, bottom)).extract_text() or "").strip()

        if text:
            parts.append(text)

    def read_gap(top: float, bottom: float) -> None:
        read_box(0.0, top, float(page.width), bottom)

    for table in tables:
        x0, top, x1, bottom = table.bbox

        read_gap(cursor, top)

        rows = _table_rows(table)

        if _has_columns(rows):
            rendered = _render_rows(rows)
            if rendered:
                parts.append(rendered)
        else:
            read_box(
                x0, top, x1, bottom
            )  # boxed heading / list: keep the original lines

        cursor = max(cursor, bottom)

    read_gap(cursor, float(page.height))

    return "\n".join(parts).strip()


def extract_pages_from_pdf_with_tables(file_path: Path) -> list[DocumentPage]:
    pages = []

    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            try:
                text = _page_text_with_tables(page)
            except Exception:
                # one bad page must not kill a 400-page document
                logger.warning(
                    "Table-aware read failed on page %d, using plain text", page_number
                )
                text = (page.extract_text() or "").strip()
            finally:
                page.flush_cache()  # keeps memory flat on big PDFs

            pages.append(DocumentPage(page_number=page_number, text=text))

    return pages


# ---------- DOCX ----------


def _iter_docx_blocks(document):
    """Paragraphs or tables gives as a order"""
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]

        if tag == "p":
            yield Paragraph(child, document).text.strip()

        elif tag == "tbl":
            table = Table(child, document)

            for row in table.rows:
                cells = []

                for cell in row.cells:
                    cell_text = cell.text.strip()

                    if cell_text and (not cells or cells[-1] != cell_text):
                        cells.append(cell_text)

                if cells:
                    yield " | ".join(cells)


def extract_pages_from_docx(file_path: Path) -> list[DocumentPage]:
    document = Document(file_path)

    blocks = [block for block in _iter_docx_blocks(document) if block]
    text = "\n\n".join(blocks)

    if not text:
        return []

    return [DocumentPage(page_number=1, text=text)]


def extract_pages(file_path: Path, tables: bool = False) -> list[DocumentPage]:
    extension = file_path.suffix.lower()

    if extension == ".pdf":
        if tables:
            return extract_pages_from_pdf_with_tables(file_path)
        return extract_pages_from_pdf(file_path)

    if extension == ".docx":
        return extract_pages_from_docx(file_path)

    raise ValueError(f"Unsupported file type: {extension}")


def extract_text(file_path: Path) -> str:
    pages = extract_pages(file_path)

    return "\n\n".join(page.text for page in pages)
