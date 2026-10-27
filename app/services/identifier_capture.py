"""
Capture the RFP / NIT / tender numbers a document states explicitly, BEFORE the text is cleaned.

Why before cleaning: the number is often only in a running header/footer ("IIITB/EHRC/2022/IT-01 ... pg. 5"),
and the cleaner removes those. Why at all: an LLM can mix up a number with a nearby date, package
number or portal id, so the code finds what the document itself labels as the number and uses it to
check (or fill) the model's answer.

Built from how real RFPs write it (14 local RFPs + ~25 public ones: India NHAI/NHIDCL/MoRTH NITs,
GeM, World Bank style RFPs, Nepal, Kenya, Pakistan, Afghanistan, US):

    RFP Document No.: RFP/ Bhopal-Kota/2025-26/efile-299153        label, then value
    Tender Reference Number | RFP/ Bhopal- Kota/2025- 26/...        value in the next table cell
    RFP Number | <long title>, IIITB/EHRC/2022/IT- 01, 25-Oct-2022  value buried in a long cell
    NIT No. 10/NHAI/RO-BNG/24016/Tender/2018     Dated:18.06.2018   value ends where "Dated" starts
    No. NHIDCL/Mizoram/DPR/Lunglei bypass/2022-23  Dated: ...       bare "No.", spaces INSIDE the value
    RFP #10-019   |   RFP No.: [PTC/T-220/2024]   |   RFP NO. TNT/002/2022 - 2023
    RFP / Bid Number: GEM/2024/BOB/CAT/01 (Gem Portal Bid ID: GEM/2024/B/5429752)

Three kinds are kept apart on purpose: rfp (RFP / tender / bid / reference number), nit (only when
the document says NIT / Notice Inviting Tender / Tender Notice) and tender_code (portal ids such as
"Tender ID", "Bid ID"). An RFP number is never copied into the NIT field.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field

from app.services.document_parser import TABLE_END, TABLE_START, DocumentPage

MAX_VALUE_LENGTH = 100
MAX_VALUE_WORDS = (
    8  # a value with more words is a title with a number inside: look for the token
)

_LABEL = re.compile(
    r"""
    (?P<label>
        \b(?:notice\s+inviting\s+(?:re-?)?tender|tender\s+notice|nit)\s*(?:no|number|ref(?:erence)?)\b\.?
      | \b(?:gem\s+portal\s+)?(?:e-?tender|tender|bid)\s*(?:id|code)\b
      | \b(?:rfp|rfq|rfx|rfb|rft|eoi|itt|itb|request\s+for\s+proposals?|proposal|solicitation|
            tender|bid|enquiry|package|contract)
        (?:\s*/\s*(?:bid|tender))?
        (?:\s+(?:document|reference|ref|identification|control|notice))*\.?
        \s*(?:no\b|number\b|num\b|nr\b|\#)\.?
      | \b(?:reference|ref)\.?\s*(?:no\b|number\b)\.?
      | \b(?:f|file)\.?\s*no\b\.?
      | (?<![A-Za-z])no\.
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# where the value stops: "Dated", "(hereinafter", a closing quote/bracket, a pipe, 3+ spaces,
# or a "(" that is not part of the identifier ("Road/ (Pir Ki Gali-Tunnel)/2017" keeps its bracket)
_TERMINATOR = re.compile(
    r"\s+(?:dated?|dt)\b|(?<![/\-])\s+\(|\s{3,}|[|;\]”\"’]|\s+(?:issued|hereinafter)\b",
    re.IGNORECASE,
)
_OPENERS = " \t:-–—.#=,([{“\"'‘"
_TOKEN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.&]*(?:\s?[/\-–]\s?[A-Za-z0-9][A-Za-z0-9_.&]*)+"
)
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
_DATE = re.compile(
    rf"""^\s*(?:
        \d{{1,2}}[./-]\d{{1,2}}[./-]\d{{2,4}}
      | \d{{4}}[./-]\d{{1,2}}[./-]\d{{1,2}}
      | \d{{1,2}}(?:st|nd|rd|th)?[\s./-]*{_MONTH}[\s./,-]*\d{{2,4}}
      | {_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{2,4}}
    )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)
_DOUBLED = re.compile(r"\S{6,}")


@dataclass(frozen=True)
class Candidate:
    kind: str  # "rfp" | "nit" | "tender_code"
    value: str  # normalised: spaces around "/" and "-" removed
    raw: str  # as the text had it
    label: str
    page: int
    source: str  # "line" | "table" | "next_line"
    strength: int  # 3 explicit RFP/NIT/ID label, 2 tender/bid/reference label, 1 bare "No." / "F. No."
    occurrences: int = 0


@dataclass
class IdentifierCapture:
    candidates: list[Candidate] = field(default_factory=list)
    best: dict[str, Candidate | None] = field(default_factory=dict)
    squashed_text: str = ""
    rejected: list[tuple[int, str, str]] = field(
        default_factory=list
    )  # (page, label, text after it): for debugging

    def value(self, kind: str) -> str | None:
        found = self.best.get(kind)
        return found.value if found else None


# ---------- small helpers ----------


def squash(text: str) -> str:
    """Lower case, no whitespace: makes "Bhopal- Kota", "Bhopal-\\nKota" and "Bhopal-Kota" equal."""
    return re.sub(r"\s+", "", text).lower()


def normalise(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip().replace("–", "-").replace("—", "-")
    value = re.sub(r"\s*([/\-])\s*", r"\1", value)
    return value.strip(" .,;:")


def _undouble(text: str) -> str:
    """Some PDFs print bold text twice ("GGEEMM//22002244"): collapse tokens made of doubled characters."""

    def fix(match: re.Match) -> str:
        token = match.group()
        if len(token) % 2 == 0 and token[0::2] == token[1::2]:
            return token[0::2]
        return token

    return _DOUBLED.sub(fix, text)


def _classify(label: str) -> tuple[str, int]:
    text = re.sub(r"\s+", " ", label.lower()).strip()

    if re.search(r"\bnit\b|notice inviting|tender notice", text):
        return "nit", 3
    if re.search(r"(?:tender|bid)\s*(?:id|code)", text):
        return "tender_code", 3
    if re.match(r"(?:rfp|rfq|rfx|rfb|rft|eoi|itt|itb|request for proposal)", text):
        return "rfp", 3
    if re.match(
        r"(?:tender|bid|enquiry|solicitation|proposal|package|contract|reference|ref)",
        text,
    ):
        return "rfp", 2
    return "rfp", 1


def _is_valid(value: str, strength: int) -> bool:
    if not 3 <= len(value) <= MAX_VALUE_LENGTH or not re.search(r"\d", value):
        return False
    if _DATE.match(value) or "@" in value or value.lower().startswith(("http", "www")):
        return False
    if strength <= 1:  # a bare "No." needs a strongly identifier-like shape
        return value.count("/") >= 2

    compact = re.sub(r"\s", "", value)
    digits = re.sub(r"\D", "", compact)

    if re.search(r"[/_\-]", compact):
        return True

    if (
        " " in value
    ):  # words with spaces need separators ("/", "-", "_") to count as one identifier
        return False

    return len(compact) >= 6 and (
        bool(re.search(r"[A-Za-z]", compact)) or len(digits) >= 6
    )


def _balance_brackets(value: str) -> str:
    if value.count("(") > value.count(")"):
        value = value[: value.rindex("(")]
    while value.count(")") > value.count("("):
        value = value[:-1] if value.endswith(")") else value.replace(")", "", 1)
    return value.strip()


def _looks_like_one_identifier(value: str) -> bool:
    """Spaces may sit INSIDE an identifier ("Lunglei bypass/2022-23", "Mughal Road/(Pir Ki Gali-Tunnel)/2017")
    but two or more plain words in a row ("Software Development Services") are a sentence or a title."""
    if "," in value:
        return False

    value = re.sub(
        r"\([^)]*\)", " ", value
    )  # words inside brackets ("(Pir Ki Gali-Tunnel)") do not count
    run = 0
    for word in value.split():
        if re.search(r"[\d/_&\-]", word):
            run = 0
        else:
            run += 1
            if run >= 2:
                return False

    return True


def _value_from(rest: str, strength: int) -> tuple[str, str] | None:
    """Text after a label -> (raw, normalised) or None. Two tries, see the module docstring."""
    text = _undouble(rest).lstrip(_OPENERS)

    if re.match(
        r"(?:dated?|dt)\b", text, re.IGNORECASE
    ):  # "Reference No. dated 25-10-2022": no number given
        return None

    cut = _TERMINATOR.split(text, maxsplit=1)[0].strip().rstrip(".,:)}")
    cut = _balance_brackets(cut)

    if (
        cut
        and len(cut.split()) <= MAX_VALUE_WORDS
        and _looks_like_one_identifier(cut)
        and _is_valid(normalise(cut), strength)
    ):
        return cut, normalise(cut)

    # a long cell: "<title words>, IIITB/EHRC/2022/IT- 01, 25-Oct-2022" -> pick the identifier-shaped token
    for token in _TOKEN.finditer(text):
        candidate = normalise(token.group())
        if _is_valid(candidate, strength):
            return token.group(), candidate

    return None


# ---------- reading the pages ----------


def _page_items(text: str) -> list[tuple[str, list[str] | None]]:
    """Page text -> [(line, None)] for normal lines, [(row text, cells)] for table rows.

    A normal line that ends with '-' or '/' is joined to the next one: "RFP/ Bhopal-" + "Kota/2025-26/..."
    """
    raw, inside = [], False

    for line in text.replace("\r\n", "\n").split("\n"):
        stripped = line.strip()

        if stripped == TABLE_START:
            inside = True
        elif stripped == TABLE_END:
            inside = False
        elif stripped:
            raw.append((stripped, inside))

    items: list[tuple[str, list[str] | None]] = []
    index = 0

    while index < len(raw):
        line, in_table = raw[index]

        if in_table:
            items.append((line, [cell.strip() for cell in line.split(" | ")]))
            index += 1
            continue

        while (
            line.endswith(("-", "/")) and index + 1 < len(raw) and not raw[index + 1][1]
        ):
            index += 1
            line += raw[index][0]

        items.append((line, None))
        index += 1

    return items


def _scan(
    text: str,
    next_text: str | None,
    page: int,
    source: str,
    found: list[Candidate],
    rejected: list[tuple[int, str, str]],
) -> None:
    for match in _LABEL.finditer(text):
        kind, strength = _classify(match.group("label"))
        rest = text[match.end() :]
        origin = source

        if not rest.strip(
            _OPENERS
        ):  # label at the end of the line / cell: the value is next
            if next_text is None:
                continue
            rest, origin = next_text, "next_line"

        value = _value_from(rest, strength)

        if not value:
            rejected.append(
                (
                    page,
                    re.sub(r"\s+", " ", match.group("label")).strip(),
                    rest.strip()[:100],
                )
            )
        else:
            raw, normal = value
            found.append(
                Candidate(
                    kind=kind,
                    value=normal,
                    raw=raw.strip(),
                    label=re.sub(r"\s+", " ", match.group("label")).strip(),
                    page=page,
                    source=origin,
                    strength=strength,
                )
            )


def capture_identifiers(pages: list[DocumentPage]) -> IdentifierCapture:
    """Find explicitly labelled RFP / NIT / tender numbers. Call it on the RAW pages (before cleaning)."""
    squashed_text = squash("\n".join(page.text for page in pages))
    found: list[Candidate] = []
    rejected: list[tuple[int, str, str]] = []

    for page in pages:
        items = _page_items(page.text)

        for position, (line, cells) in enumerate(items):
            if cells is None:
                following = (
                    items[position + 1][0]
                    if position + 1 < len(items) and items[position + 1][1] is None
                    else None
                )
                _scan(line, following, page.page_number, "line", found, rejected)
            else:
                for index, cell in enumerate(cells):
                    following = next((c for c in cells[index + 1 :] if c), None)
                    _scan(cell, following, page.page_number, "table", found, rejected)

    # one entry per distinct value (per kind): keep the strongest label, count how often it appears
    groups: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for candidate in found:
        groups[(candidate.kind, squash(candidate.value))].append(candidate)

    candidates = []
    for (_, key), members in groups.items():
        top = max(members, key=lambda c: (c.strength, -c.page))
        candidates.append(
            Candidate(**{**top.__dict__, "occurrences": squashed_text.count(key)})
        )

    best: dict[str, Candidate | None] = {}
    for kind in ("rfp", "nit", "tender_code"):
        of_kind = [c for c in candidates if c.kind == kind]
        best[kind] = max(
            of_kind,
            key=lambda c: (c.strength, min(c.occurrences, 20), -c.page),
            default=None,
        )

    candidates.sort(key=lambda c: (c.kind, -c.strength, -c.occurrences, c.page))
    return IdentifierCapture(
        candidates=candidates, best=best, squashed_text=squashed_text, rejected=rejected
    )


# ---------- using it on the model's answer ----------


def reconcile(
    model_value: str | None, capture: IdentifierCapture, kind: str
) -> tuple[str | None, str | None]:
    """The model's value + what the document labels -> (value to keep, review note or None).

    - model said nothing, the document labels one      -> use the labelled one
    - model value = labelled value (ignoring spaces)   -> keep
    - model value is the labelled value plus words
      (a title around the number), or only a part of it -> the labelled value (cleaner)
    - model value is not in the document at all        -> the labelled value if there is one, else keep + flag
    - model value is in the document, differs from the
      labelled one                                     -> keep the model's, flag it (documents often
                                                          hold several numbers: RFP, package, portal)
    """
    found = capture.best.get(kind)
    wanted = squash(found.value) if found else None

    if not model_value or not model_value.strip():
        if found:
            return (
                found.value,
                f"{kind}: not returned by the model, taken from the label '{found.label}' on page {found.page}",
            )
        return None, None

    ours = squash(model_value)
    grounded = ours in capture.squashed_text

    if not found:
        return (
            model_value,
            None
            if grounded
            else f"{kind}: '{model_value}' was not found in the document text",
        )

    if ours == wanted:
        return model_value, None

    if wanted in ours or (ours in wanted and len(ours) >= 4):
        return (
            found.value,
            f"{kind}: reduced to the labelled number '{found.value}' (page {found.page})",
        )

    if not grounded:
        return (
            found.value,
            f"{kind}: model value '{model_value}' is not in the document; used '{found.value}' from '{found.label}'",
        )

    return (
        model_value,
        f"{kind}: model value differs from the number labelled '{found.label}' on page {found.page}: '{found.value}'",
    )
