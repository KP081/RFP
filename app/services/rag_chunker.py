"""
Evidence chunks for the extraction RAG flow (flow doc, stage 5: indexing).

Input: cleaned pages (text_cleaner output; tables are [TABLE] ... [/TABLE] rows).
Output: small chunks with these rules:

- a chunk never crosses a page, so every chunk has ONE page number (the eval references are per page);
- a [TABLE] block stays whole when it fits; a long table is split BETWEEN rows, and the table's first
  row (the header) is repeated at the top of every continuation chunk, so a marks column is never
  read without its header;
- no overlap: the selector adds the neighbouring chunks instead (a table that continues on the next
  page is the next chunk).

chunker.py is a different thing: it splits plain text by characters for the Q&A index.
"""

from dataclasses import dataclass

from app.services.document_parser import TABLE_END, TABLE_START, DocumentPage

DEFAULT_CHUNK_CHARS = 1500


@dataclass(frozen=True)
class EvidenceChunk:
    index: int  # position in the document: 0, 1, 2 ... in page order
    page_number: int
    text: str


def _split_paragraph(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    parts: list[str] = []
    current = ""

    for line in text.split("\n"):
        while len(line) > max_chars:  # one very long line: cut at a space
            cut = line.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars

            if current:
                parts.append(current)
                current = ""

            parts.append(line[:cut])
            line = line[cut:].lstrip()

        if not line:
            continue

        if current and len(current) + 1 + len(line) > max_chars:
            parts.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line

    if current:
        parts.append(current)

    return parts


def _wrap(rows: list[str]) -> str:
    return "\n".join([TABLE_START, *rows, TABLE_END])


def _split_table(rows: list[str], max_chars: int) -> list[str]:
    if not rows:
        return []

    if len(_wrap(rows)) <= max_chars:
        return [_wrap(rows)]

    header = rows[0]
    groups: list[list[str]] = []
    current = [header]

    for row in rows[1:]:
        # a group is closed only when it already holds a data row (header + 1 row)
        if len(current) >= 2 and len(_wrap([*current, row])) > max_chars:
            groups.append(current)
            current = [header, row]
        else:
            current.append(row)

    groups.append(current)

    return [_wrap(group) for group in groups]


def _units(text: str, max_chars: int) -> list[str]:
    """Page text -> units that are never cut again: paragraphs and table pieces."""
    units: list[str] = []
    lines = text.split("\n")
    paragraph: list[str] = []
    position = 0

    def flush() -> None:
        if paragraph:
            units.extend(_split_paragraph("\n".join(paragraph), max_chars))
            paragraph.clear()

    while position < len(lines):
        line = lines[position]

        if line.strip() == TABLE_START:
            flush()
            end = position + 1
            rows: list[str] = []

            while end < len(lines) and lines[end].strip() != TABLE_END:
                if lines[end].strip():
                    rows.append(lines[end])
                end += 1

            units.extend(_split_table(rows, max_chars))
            position = end + 1
            continue

        if line.strip():
            paragraph.append(line)
        else:
            flush()

        position += 1

    flush()
    return units


def chunk_pages_for_extraction(
    pages: list[DocumentPage], max_chars: int = DEFAULT_CHUNK_CHARS
) -> list[EvidenceChunk]:
    if max_chars < 200:
        raise ValueError("max_chars must be at least 200")

    chunks: list[EvidenceChunk] = []

    def emit(page_number: int, text: str) -> None:
        chunks.append(EvidenceChunk(len(chunks), page_number, text.strip()))

    for page in pages:
        text = page.text.strip()

        if not text:
            continue

        current = ""

        for unit in _units(text, max_chars):
            if current and len(current) + 2 + len(unit) > max_chars:
                emit(page.page_number, current)
                current = unit
            else:
                current = f"{current}\n\n{unit}" if current else unit

        if current:
            emit(page.page_number, current)

    return chunks
