import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings
from app.core.prompts import load_prompt
from app.schemas.extraction import (
    SCHEMA_VERSION,
    ExtractionMeta,
    PersonnelCategory,
    TenderExtraction,
    TenderExtractionLLM,
    ThresholdBasis,
    is_iso_date,
)
from app.services.document_parser import extract_pages
from app.services.llm_service import LLMService, Usage

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

        try:
            if mode == "pdf":
                uploaded = self.llm_service.upload_file(file_path)
                parts = [uploaded]
            else:
                pages = extract_pages(file_path)
                parts = [self._pages_to_text(pages)]

                empty = [p.page_number for p in pages if len(p.text) < NO_TEXT_CHARS]
                if empty:
                    flags.append(
                        f"{len(empty)} page(s) have no text layer (scanned?) and were "
                        f"not read in text mode: {empty[:20]}"
                    )

            result = self.llm_service.generate_structured(
                template, TenderExtractionLLM, parts=parts
            )
        finally:
            if uploaded is not None:
                self.llm_service.delete_file(uploaded)

        fill_derived(result.data)
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


# ---------- derived values and free consistency checks (no LLM) ----------


def fill_derived(doc: TenderExtractionLLM) -> None:
    """Totals are computed by code, never by the model.

    If the document prints a total the model may have copied it (kept, and
    consistency_flags compares it with the roles). If it is null we fill the sum
    of the non-conditional key roles.
    """
    section = doc.key_personnel

    if section.key_person_months_total is not None:
        return

    months = [
        role.person_months
        for role in section.roles
        if role.category == PersonnelCategory.KEY
        and not role.is_conditional
        and role.person_months is not None
    ]

    if months:
        section.key_person_months_total = round(sum(months), 2)


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
