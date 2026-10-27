"""
Basic text cleaning (flow Stage 3, step 5.1), page by page.

What it removes, and why each rule is built the way it is (rules come from reading 13 real RFPs):

  signature   a stamp: a line that STARTS with "Digitally signed by" and has a detail line
              (Date:, Reason:, Location:, a timestamp) within the next 4 lines. Everything up to that
              last detail line goes (the signer's name often sits on its own line in between).
              Without a detail line it is prose ("digitally signed by the authorised signatory")
              and stays. "digitally signed modified bid" in the middle of a line is never touched.
  toc         a line with some title text, 5+ dots and a page number at the end ("Scope ..... 9").
  page_stamp  a whole line like "Page 5 of 68", "pg. 5", "- 5 -", "5 of 68", ANYWHERE on the page
              (one RFP prints the stamp at the top of some pages and at the bottom of others).
  page_number a bare number ("23"), only as the first or last line of a page AND only if the numbers
              form a page sequence: number - page_index is the same on at least 3 pages. A real page
              number grows by one per page; a lone "5" at the bottom of one page is content (a list
              number, a table value). A bare number in the middle of a page is never removed.
  repeated    a running header/footer: a first/last line of the page that repeats EXACTLY on many
              pages. One copy is kept (the first), the others are removed. Only the page marker
              inside a line is ignored ("... pg. 2", "Page 3 of 18 ..."), so stamps glued to text
              are caught ("IIITB/EHRC/2022/IT-01, 25-Oct-2022 pg. 2"). Other digits must match
              too: "Annexure 1", "Annexure 2" ... are different headings, not one repeated header.

Not touched: everything inside [TABLE] ... [/TABLE] (a table row "1" is not a page number, a
repeated table header is not boilerplate). Pages keep their page_number and their position, even
when a page becomes empty.

Not done here on purpose (flow 5.2 and 5.3): boilerplate-section pruning and near-duplicate
removal. They can delete real requirements, so they come later, after the extraction eval exists.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from app.services.document_parser import TABLE_END, TABLE_START, DocumentPage

EDGE_LINES = (
    3  # how many lines at the top / bottom of a page can be a running header/footer
)
MIN_REPEAT_PAGES = 4  # a header/footer must repeat on at least this many pages ...
MIN_REPEAT_SHARE = 0.15  # ... and on at least this share of all pages
MAX_HEADER_LENGTH = 200
MIN_KEY_LENGTH = (
    15  # shorter lines ("Date:", "Signature:", "****") are too generic to call a header
)

PAGE_STAMP = re.compile(
    r"""^\s*(?:
        page\s*[-:]?\s*\d+(?:\s*(?:of|/)\s*\d+)?   # Page 5 | Page 5 of 68 | Page 5/68
      | pg\.?\s*\d+                                # pg. 5
      | [-–—]\s*\d+\s*[-–—]                        # - 5 -
      | \d+\s+of\s+\d+                             # 5 of 68
    )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)
_MARKER = r"(?:page|pg\.?)\s*[-:]?\s*\d+(?:\s*(?:of|/)\s*\d+)?"
# only at the START or END of a line: a stamp glued to header text. "see page 5 of the annexure"
# inside a sentence is not a stamp.
PAGE_MARKER = re.compile(rf"^{_MARKER}|\b{_MARKER}$", re.IGNORECASE)
BARE_NUMBER = re.compile(r"^\s*(\d{1,4})\s*$")
TOC_LINE = re.compile(r"[A-Za-z].*(?:\.{5,}|(?:\.\s){5,})\s*\d{1,4}\s*$")
SIGNATURE_START = re.compile(r"^\s*digitally\s+signed\s+by\b", re.IGNORECASE)
SIGNATURE_DETAIL = re.compile(
    r"^\s*(?:date|reason|location)\s*:|^\s*\d{4}[./-]\d{1,2}[./-]\d{1,2}|^\s*\d{1,2}:\d{2}(?::\d{2})?\b",
    re.IGNORECASE,
)
SIGNATURE_MAX_DETAIL_LINES = 4
MIN_SEQUENCE_PAGES = (
    3  # bare numbers count as page numbers when this many pages share an offset
)


@dataclass
class _Line:
    text: str
    in_table: bool
    rule: str | None = None  # set when the line is removed

    @property
    def keep_candidate(self) -> bool:
        """An ordinary line that no rule has removed yet (never a table line)."""
        return not self.in_table and self.rule is None and bool(self.text.strip())


@dataclass
class CleanResult:
    pages: list[DocumentPage]
    removed: dict[str, list[tuple[int, str]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def counts(self) -> dict[str, int]:
        return {rule: len(items) for rule, items in self.removed.items()}


def line_key(line: str) -> str:
    """Line -> comparison key: lower case, spaces collapsed, a page marker at the start or
    end of the line as 'page #'.

    Only that marker is normalised. Every other digit stays, so two different numbered
    headings never collapse into one key.
    """
    text = re.sub(r"\s+", " ", line.lower()).strip()
    return PAGE_MARKER.sub("page #", text)


def _split_lines(text: str) -> list[_Line]:
    lines, inside = [], False

    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        text_line = raw.rstrip()

        if text_line == TABLE_START:
            inside = True
            lines.append(_Line(text_line, True))
        elif text_line == TABLE_END:
            lines.append(_Line(text_line, True))
            inside = False
        else:
            lines.append(_Line(text_line, inside))

    return lines


def _looks_like_a_name(text: str) -> bool:
    """A short line without sentence punctuation: the signer's name inside a stamp."""
    return (
        len(text) <= 60
        and len(text.split()) <= 6
        and not text.rstrip().endswith((".", ":", ";", ","))
    )


def _mark_signatures(lines: list[_Line]) -> None:
    for index, line in enumerate(lines):
        if not (line.keep_candidate and SIGNATURE_START.match(line.text)):
            continue

        last_detail = None

        for follow in range(
            index + 1, min(len(lines), index + 1 + SIGNATURE_MAX_DETAIL_LINES)
        ):
            candidate = lines[follow]

            if not candidate.keep_candidate:
                break

            if SIGNATURE_DETAIL.match(candidate.text):
                last_detail = follow
            elif not _looks_like_a_name(candidate.text):
                break

        if last_detail is None:
            continue  # no detail line: this is a sentence, not a stamp

        for position in range(index, last_detail + 1):
            lines[position].rule = "signature"


def _mark_toc(lines: list[_Line]) -> None:
    for line in lines:
        if line.keep_candidate and TOC_LINE.search(line.text):
            line.rule = "toc"


def _mark_page_stamps(lines: list[_Line]) -> None:
    for line in lines:
        if line.keep_candidate and PAGE_STAMP.match(line.text):
            line.rule = "page_stamp"


def _bare_number_candidates(lines: list[_Line]) -> list[tuple[_Line, int]]:
    """Bare numbers that are the first or the last line of the page."""
    candidates = [line for line in lines if line.keep_candidate]

    if not candidates:
        return []

    edges = {id(candidates[0]): candidates[0], id(candidates[-1]): candidates[-1]}
    found = []

    for line in edges.values():
        match = BARE_NUMBER.match(line.text)

        if match:
            found.append((line, int(match.group(1))))

    return found


def _mark_page_number_sequences(all_lines: list[list[_Line]]) -> None:
    """Remove bare edge numbers that form a page sequence (same number - page index offset)."""
    found: list[tuple[int, _Line]] = []  # (offset, line)

    for page_index, lines in enumerate(all_lines, start=1):
        for line, number in _bare_number_candidates(lines):
            found.append((number - page_index, line))

    pages_per_offset = Counter(offset for offset, _ in found)

    for offset, line in found:
        if pages_per_offset[offset] >= MIN_SEQUENCE_PAGES:
            line.rule = "page_number"


def _edge_lines(lines: list[_Line]) -> list[_Line]:
    """First and last lines of the page (no overlap on short pages)."""
    candidates = [line for line in lines if line.keep_candidate]
    count = len(candidates)

    if count >= 2 * EDGE_LINES:
        return candidates[:EDGE_LINES] + candidates[-EDGE_LINES:]

    half = count // 2
    return candidates[:half] + candidates[half:][-EDGE_LINES:]


def _mark_repeated_edges(all_lines: list[list[_Line]]) -> None:
    pages_with_key: dict[str, set[int]] = defaultdict(set)

    for page_index, lines in enumerate(all_lines):
        for line in _edge_lines(lines):
            if len(line.text) <= MAX_HEADER_LENGTH:
                pages_with_key[line_key(line.text)].add(page_index)

    needed = max(MIN_REPEAT_PAGES, int(MIN_REPEAT_SHARE * len(all_lines)))
    repeated = {
        key
        for key, pages in pages_with_key.items()
        if len(pages) >= needed and len(key) >= MIN_KEY_LENGTH
    }
    kept: set[str] = set()

    for lines in all_lines:
        for line in _edge_lines(lines):
            key = line_key(line.text)

            if key not in repeated or len(line.text) > MAX_HEADER_LENGTH:
                continue

            if key in kept:
                line.rule = "repeated"
            else:
                kept.add(key)  # first occurrence stays


def _rebuild(lines: list[_Line]) -> str:
    text = "\n".join(line.text for line in lines if line.rule is None)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_pages(pages: list[DocumentPage]) -> CleanResult:
    all_lines = [_split_lines(page.text) for page in pages]

    for lines in all_lines:
        _mark_signatures(lines)
        _mark_toc(lines)
        _mark_page_stamps(lines)

    _mark_page_number_sequences(all_lines)
    _mark_repeated_edges(
        all_lines
    )  # after stamps are gone, so edges are the real header/footer

    result = CleanResult(pages=[])

    for page, lines in zip(pages, all_lines):
        for line in lines:
            if line.rule:
                result.removed[line.rule].append((page.page_number, line.text))

        result.pages.append(
            DocumentPage(page_number=page.page_number, text=_rebuild(lines))
        )

    return result
