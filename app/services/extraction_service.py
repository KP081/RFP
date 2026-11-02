import hashlib
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import math

from app.core.config import get_settings
from app.core.prompts import load_prompt
from app.schemas.extraction import (
    SCHEMA_VERSION,
    ExtractionMeta,
    PersonnelCategory,
    TenderExtraction,
    TenderExtractionLLM,
    ThresholdBasis,
    ThresholdKind,
    is_iso_date,
)
from app.services.identifier_capture import (
    capture_identifiers,
    normalise,
    reconcile,
    squash,
)
from app.services.llm_service import LLMService, Usage
from app.services.page_cache import load_pages
from app.services.text_cleaner import clean_pages

logger = logging.getLogger(__name__)

DATE_FIELDS = (
    "published",
    "download_start",
    "download_end",
    "clarification_end",
    "submission_start",
    "submission_end",
    "hard_copy_deadline",
    "technical_opening",
    "financial_opening",
)

NO_TEXT_CHARS = 30  # a page with fewer characters has no usable text layer

# threshold kinds that are money (staff_strength / count are plain numbers: no unit fix)
MONEY_KINDS = (
    ThresholdKind.TURNOVER,
    ThresholdKind.NET_WORTH,
    ThresholdKind.PROJECT_VALUE,
)

# Bump it when the pipeline around the model changes (text reading, cleaning, identifier checks):
# it is part of the cache key, so an old stored result is not served for a new pipeline.
PIPELINE_VERSION = "2"


@dataclass
class ExtractionResult:
    extraction: TenderExtraction
    usage: Usage
    repairs: int
    mode: str
    seconds: float


class ExtractionService:
    """One Gemini call per RFP. Context = the PDF itself ("pdf") or parsed text ("text")."""

    def __init__(self, llm_service: LLMService | None = None):
        self.settings = get_settings()
        self.llm_service = llm_service or LLMService()

    def extract(self, file_path: Path, mode: str | None = None) -> ExtractionResult:
        mode = self._resolve_mode(file_path, mode)
        template = load_prompt(self.settings.extract_prompt_file)
        started = time.time()
        flags: list[str] = []
        uploaded = None
        raw_pages: list = []  # RAW text (before cleaning): the numbers are checked against it

        try:
            if mode == "pdf":
                uploaded = self.llm_service.upload_file(file_path)
                parts = [uploaded]
                raw_pages = self._leading_pages(file_path)
            else:
                pages = load_pages(file_path, tables=self.settings.extract_tables)
                raw_pages = pages[: self.settings.identifier_scan_pages]

                empty = [p.page_number for p in pages if len(p.text) < NO_TEXT_CHARS]
                if empty:
                    flags.append(
                        f"{len(empty)} page(s) have no text layer (scanned?) and were "
                        f"not read in text mode: {empty[:20]}"
                    )

                if self.settings.extract_clean_text:
                    cleaned = clean_pages(pages)
                    logger.info("Text cleaner removed: %s", cleaned.counts())
                    pages = cleaned.pages

                parts = [self._pages_to_text(pages)]

            result = self.llm_service.generate_structured(
                template, TenderExtractionLLM, parts=parts
            )
        finally:
            if uploaded is not None:
                self.llm_service.delete_file(uploaded)

        flags.extend(self._check_identifiers(result.data, raw_pages))
        flags.extend(fill_derived(result.data))
        flags.extend(consistency_flags(result.data))

        meta = ExtractionMeta(
            schema_version=SCHEMA_VERSION,
            needs_review=bool(flags),
            review_flags=flags,
            model=self.llm_service.model,
            prompt_hash=hashlib.sha256(template.encode("utf-8")).hexdigest()[:12],
            extracted_at=datetime.now(timezone.utc).isoformat(),
        )

        extraction = TenderExtraction.model_validate(
            {
                **result.data.model_dump(mode="json"),
                "meta": meta.model_dump(mode="json"),
            }
        )

        return ExtractionResult(
            extraction=extraction,
            usage=result.usage,
            repairs=result.repairs,
            mode=mode,
            seconds=time.time() - started,
        )

    def fingerprint(self, file_path: Path, mode: str | None = None) -> str:
        """Everything that changes the result besides the file itself (used as a cache key)."""
        template = load_prompt(self.settings.extract_prompt_file)
        prompt_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()[:12]
        mode = self._resolve_mode(file_path, mode)

        return f"{SCHEMA_VERSION}|{self.llm_service.model}|{prompt_hash}|p{PIPELINE_VERSION}|{mode}"

    def _resolve_mode(self, file_path: Path, mode: str | None) -> str:
        mode = mode or self.settings.extract_context

        if mode not in ("pdf", "text"):
            raise ValueError(f"Unknown extraction mode: {mode}")

        # Gemini reads PDFs natively; DOCX goes through the text parser.
        if file_path.suffix.lower() != ".pdf":
            return "text"

        return mode

    @staticmethod
    def _pages_to_text(pages) -> str:
        return "\n\n".join(f"=== PAGE {p.page_number} ===\n{p.text}" for p in pages)

    def _leading_pages(self, file_path: Path) -> list:
        """pdf mode: Gemini reads the file itself, so the first pages are read locally only to
        check the RFP / NIT numbers. A failure here must never fail the extraction."""
        if not self.settings.identifier_capture:
            return []

        try:
            return load_pages(
                file_path,
                tables=self.settings.extract_tables,
                max_pages=self.settings.identifier_scan_pages,
            )
        except Exception:
            logger.warning(
                "Could not read the first pages for the identifier check", exc_info=True
            )
            return []

    def _check_identifiers(self, doc: TenderExtractionLLM, pages: list) -> list[str]:
        """Check the model's rfp_number / nit_number against the labels the document itself prints.

        Returns review flags only for real doubts (a different grounded number, or a number that is
        nowhere in the text). Automatic fixes (filled in, shortened to the labelled number) are logged.
        """
        flags: list[str] = []
        identity = doc.identity
        # nothing readable (a scanned PDF): "not found in the text" would be a false alarm
        readable = any(p.text.strip() for p in pages)
        capture = (
            capture_identifiers(pages)
            if readable and self.settings.identifier_capture
            else None
        )

        for field, kind in (("rfp_number", "rfp"), ("nit_number", "nit")):
            value = getattr(identity, field)

            if capture is not None:
                value, note = reconcile(value, capture, kind)

                if note:
                    logger.info("identifier check: %s", note)

                    if "differs" in note or "was not found" in note:
                        flags.append(note)

            setattr(identity, field, _tidy_reference(value))

        # an RFP number is never a NIT number (the model sometimes copies one into the other)
        if (
            identity.nit_number
            and identity.rfp_number
            and squash(identity.nit_number) == squash(identity.rfp_number)
        ):
            logger.info(
                "identifier check: nit_number was a copy of rfp_number, cleared"
            )
            identity.nit_number = None

        return flags


def _tidy_reference(value: str | None) -> str | None:
    """'RFP/ Bhopal- Kota/2025-26/' -> 'RFP/Bhopal-Kota/2025-26' (spaces around / and -, trailing /)."""
    if not value or not value.strip():
        return None

    return normalise(value).rstrip("/ ") or None


# ---------- derived values and free consistency checks (no LLM) ----------


_TECHNICAL = re.compile(r"\b(technical|quality)\b", re.IGNORECASE)
_FINANCIAL = re.compile(r"\b(financial|price|commercial)\b", re.IGNORECASE)
_UNIT_AMOUNT = re.compile(
    r"(?<![\d,])(?<!\d\.)(\d[\d,]*(?:\.\d+)?)\s*(crores?|cr\b\.?|lakhs?|lacs?)",
    re.IGNORECASE,
)
_UNIT_FACTOR = {
    "cr": 1e7,
    "crore": 1e7,
    "crores": 1e7,
    "lakh": 1e5,
    "lakhs": 1e5,
    "lac": 1e5,
    "lacs": 1e5,
}


def _amount_in_text(text: str | None) -> tuple[float, float] | None:
    """'Rs.5.00 Crores' -> (5.0, 50000000.0): the bare number, and the number with its unit applied."""
    match = _UNIT_AMOUNT.search(text or "")

    if not match:
        return None

    number = float(match.group(1).replace(",", ""))
    unit = match.group(2).lower().rstrip(".")

    return number, number * _UNIT_FACTOR[unit]


def fix_money_units(doc: TenderExtractionLLM) -> list[str]:
    """The model sometimes copies the bare number ("5.00" from "Rs.5.00 Crores") into a money field that
    must be in plain rupees. When raw_text names the unit and value is exactly the bare number, apply the
    unit. Only fixes this one case; every other value is left alone. Returns notes for review_flags.
    """
    notes: list[str] = []

    def check(obj, label: str) -> None:
        found = _amount_in_text(obj.raw_text)

        if obj.value is None or found is None:
            return

        bare, full = found

        if math.isclose(obj.value, bare, rel_tol=1e-9) and not math.isclose(bare, full):
            notes.append(
                f"{label}: value {obj.value:g} is the bare number of '{obj.raw_text}'; "
                f"unit applied -> {full:,.0f}"
            )
            obj.value = full

    for item in doc.qualification.financial:
        if item.basis == ThresholdBasis.ABSOLUTE_AMOUNT and item.kind in MONEY_KINDS:
            check(item, f"qualification.financial[{item.label}]")

    check(doc.project.estimated_contract_value, "project.estimated_contract_value")
    check(doc.project.project_cost, "project.project_cost")
    check(doc.securities.document_fee, "securities.document_fee")

    return notes


def fill_derived(doc: TenderExtractionLLM) -> list[str]:
    """Fixes done by code, never by the model. Returns notes for meta.review_flags.

    1. Totals are computed by code: if the document prints a total the model may have copied
       it (kept; consistency_flags compares it with the roles), otherwise we fill the sum of
       the non-conditional key roles.
    2. The model sometimes files the technical or financial weight under weights.other.
       If technical_pct / financial_pct is empty, move that entry up (and say so).
    """
    notes: list[str] = fix_money_units(doc)

    section = doc.key_personnel
    if section.key_person_months_total is None:
        months = [
            role.person_months
            for role in section.roles
            if role.category == PersonnelCategory.KEY
            and not role.is_conditional
            and role.person_months is not None
        ]
        if months:
            section.key_person_months_total = round(sum(months), 2)

    weights = doc.evaluation.weights
    kept = []
    for item in weights.other:
        if weights.technical_pct is None and _TECHNICAL.search(item.name):
            weights.technical_pct = item.weight_pct
            notes.append(f"weights: moved '{item.name}' from other to technical_pct")
        elif weights.financial_pct is None and _FINANCIAL.search(item.name):
            weights.financial_pct = item.weight_pct
            notes.append(f"weights: moved '{item.name}' from other to financial_pct")
        else:
            kept.append(item)
    weights.other = kept

    return notes


def consistency_flags(doc: TenderExtractionLLM) -> list[str]:
    """Review flags only: a flag means 'look at this', not 'this is wrong'."""
    flags: list[str] = []

    # money without a currency
    if _has_money(doc) and not doc.identity.currency:
        flags.append("money amounts found but identity.currency is not set")

    # dates
    for name in DATE_FIELDS:
        value = getattr(doc.dates, name)
        if not is_iso_date(value):
            flags.append(f"dates.{name} is not a clean date: {value!r}")

    if not is_iso_date(doc.dates.pre_bid.date):
        flags.append(
            f"dates.pre_bid.date is not a clean date: {doc.dates.pre_bid.date!r}"
        )

    start, opening = doc.dates.submission_end, doc.dates.technical_opening
    if (
        start
        and opening
        and is_iso_date(start)
        and is_iso_date(opening)
        and opening < start
    ):
        flags.append(
            f"technical_opening ({opening}) is before submission_end ({start})"
        )

    # combined-score weights
    weights = doc.evaluation.weights
    parts = [weights.technical_pct, weights.financial_pct]
    parts += [item.weight_pct for item in weights.other]
    known = [p for p in parts if p is not None]
    if len(known) >= 2 and abs(sum(known) - 100) > 0.01:
        flags.append(f"evaluation weights add up to {sum(known):g}, not 100")

    # key person-months
    section = doc.key_personnel
    months = [
        role.person_months
        for role in section.roles
        if role.category == PersonnelCategory.KEY
        and not role.is_conditional
        and role.person_months is not None
    ]
    total = section.key_person_months_total
    if total is not None and months and abs(sum(months) - total) > 0.01:
        flags.append(
            f"key person-months: roles add up to {sum(months):g}, stated total {total:g}"
        )

    # payment schedule
    stages = doc.contract.payment_stages
    percents = [stage.percent for stage in stages]
    if stages and all(value is not None for value in percents):
        if abs(sum(percents) - 100) > 0.01:
            flags.append(f"payment stages add up to {sum(percents):g}%, not 100%")

    # marks: sub-criteria vs criterion
    for criterion in doc.evaluation.criteria:
        subs = [sub.max_points for sub in criterion.sub_criteria]
        if (
            criterion.max_points is not None
            and subs
            and all(value is not None for value in subs)
            and abs(sum(subs) - criterion.max_points) > 0.01
        ):
            flags.append(
                f"evaluation criterion '{criterion.name}': sub-criteria add up to "
                f"{sum(subs):g}, max is {criterion.max_points:g}"
            )

    return flags


def _has_money(doc: TenderExtractionLLM) -> bool:
    amounts = [
        doc.project.estimated_contract_value,
        doc.project.project_cost,
        doc.securities.document_fee,
    ]
    if any(a.value is not None or a.value_with_tax is not None for a in amounts):
        return True

    securities = (
        doc.securities.bid_security,
        doc.securities.performance_security,
        doc.securities.additional_performance_security,
    )
    if any(sec.amount is not None for sec in securities):
        return True

    if any(
        t.basis == ThresholdBasis.ABSOLUTE_AMOUNT and t.value is not None
        for t in doc.qualification.financial
    ):
        return True

    return any(
        r.rate_per_month is not None or r.cost is not None
        for r in doc.key_personnel.roles
    )
