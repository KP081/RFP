"""
Universal tender / RFP extraction schema (v1).

Design (based on research):
- A universal CORE contains common information like identity, dates, securities, eligibility, and evaluation. These are common across consultancy RFPs, IT/SI RFPs, and works/HAM bids.
- Family-specific details go into GENERIC building blocks like `Threshold`, `ExperienceCriterion`, and `Criterion`, instead of separate fields. This allows the schema to support new RFP types without breaking.
- If something does not fit into any block, it goes into `additional_conditions`. This helps us identify which fields should be added to the schema in the future.

Rules for every field:
- `None` or an empty list means "not found in the document." Never guess.
- Fill document-specific defaults (such as "maximum 4 JV members") only if they are explicitly mentioned in the document.
- Store amounts as INR numbers (1 lakh = 1e5, 1 crore = 1e7) and keep the original text alongside them.
- Store dates in ISO `YYYY-MM-DD` format. If the LLM returns dates in `dd.mm.yyyy` format, `normalize_date()` will handle the conversion.

For Gemini: Do not use recursive models or `dict[str, Any]`. Also, do not use `extra="forbid"` in LLM-facing models because it can cause errors with `additionalProperties`.

Strict validation will happen after extraction, not in the schema.
"""

import re
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "1.1"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%d/%m/%Y",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d, %Y",
)


def normalize_date(value: object) -> str | None:
    """'25.03.2026, 05:00 PM' -> '2026-03-25'. Parse na ho to raw text wapas
    (taaki data na jaye); baad me review flag lag jayega."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    # remove time : "25.03.2026, 05:00 PM" -> "25.03.2026"
    text = re.split(r"[,\s]+(?=\d{1,2}:\d{2})", text)[0].strip(" ,")

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue

    return text


def is_iso_date(value: str | None) -> bool:
    if value is None:
        return True
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------
# enums
# --------------------------------------------------------------------------


class DocumentFamily(StrEnum):
    CONSULTANCY = "consultancy_rfp"  # DPR, AE, IE, PMC, financial, ...
    IT_SERVICES = "it_services_rfp"  # system integrator, software, website
    WORKS = "works_bid"  # EPC / HAM / BOT / item-rate works
    GOODS = "goods_supply"
    OTHER = "other"


class DocumentKind(StrEnum):
    TENDER_DOCUMENT = "tender_document"  # asli RFP / tender
    CORRIGENDUM = "corrigendum"  # addendum / corrigendum: sirf badlav, poora data nahi
    OTHER = "other"


class AssignmentType(StrEnum):
    """Closed set filter/analytics ke liye. Kuch na mile to OTHER + assignment_type_text."""

    DPR = "dpr"
    FEASIBILITY_STUDY = "feasibility_study"
    AUTHORITY_ENGINEER = "authority_engineer"
    INDEPENDENT_ENGINEER = "independent_engineer"
    PROJECT_MANAGEMENT = "project_management"
    CONSTRUCTION_SUPERVISION = "construction_supervision"
    SURVEY_INVESTIGATION = "survey_investigation"
    ROAD_SAFETY_AUDIT = "road_safety_audit"
    FINANCIAL_ADVISORY = "financial_advisory"
    SYSTEM_INTEGRATOR = "system_integrator"
    SOFTWARE_DEVELOPMENT = "software_development"
    WORKS_CONSTRUCTION = "works_construction"
    OTHER = "other"


class SelectionMethod(StrEnum):
    QCBS = "qcbs"
    QBS = "qbs"
    FBS = "fbs"
    LCS = "lcs"
    CQS = "cqs"
    SSS = "sss"
    L1_LOWEST_PRICE = "l1_lowest_price"
    BID_PARAMETER = "bid_parameter"  # HAM/BOT jaise bids
    OTHER = "other"


class Basis(StrEnum):
    STATED = "stated"  # document me seedha likha hai
    INFERRED = "inferred"  # formula/weights se nikala (e.g. S = ST*Tw + SF*Fw)


class ContractForm(StrEnum):
    TIME_BASED = "time_based"
    LUMP_SUM = "lump_sum"
    UNIT_RATE = "unit_rate"
    PERCENTAGE = "percentage"
    ANNUITY = "annuity"
    OTHER = "other"


class SecurityStatus(StrEnum):
    REQUIRED = "required"
    NIL = "nil"  # document saaf likhta hai ki nahi chahiye
    DECLARATION_ALLOWED = "declaration_allowed"  # bid securing declaration chalega
    NOT_STATED = "not_stated"  # document me zikr hi nahi


class ThresholdBasis(StrEnum):
    ABSOLUTE_INR = "absolute_inr"
    PCT_OF_ESTIMATED_COST = "pct_of_estimated_cost"  # consultancy/contract cost ka %
    PCT_OF_PROJECT_COST = "pct_of_project_cost"  # civil/EPC cost ka %
    MULTIPLE_OF_PROJECT_LENGTH = "multiple_of_project_length"
    COUNT = "count"
    OTHER = "other"


class ThresholdKind(StrEnum):
    TURNOVER = "turnover"
    NET_WORTH = "net_worth"
    STAFF_STRENGTH = "staff_strength"
    PROJECT_VALUE = "project_value"
    OTHER = "other"


class Aggregation(StrEnum):
    AVERAGE = "average"
    EACH_YEAR = "each_year"
    TOTAL = "total"
    AT_LEAST_ONE_YEAR = "at_least_one_year"


class PersonnelCategory(StrEnum):
    KEY = "key"
    SUB_PROFESSIONAL = "sub_professional"
    SUPPORT = "support"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# --------------------------------------------------------------------------
# reusable building blocks
# --------------------------------------------------------------------------


class Amount(BaseModel):
    value_inr: float | None = Field(
        default=None,
        ge=0,
        description="Amount in INR rupees (lakh=1e5, crore=1e7). null if not stated.",
    )
    raw_text: str | None = Field(
        default=None,
        description="Exact wording from the document, e.g. 'Rs 11,64,00,510'.",
    )
    includes_tax: bool | None = Field(
        default=None,
        description="True if figure includes GST/taxes, False if it excludes them, null if unclear.",
    )


class Security(BaseModel):
    status: SecurityStatus = SecurityStatus.NOT_STATED
    amount_inr: float | None = Field(default=None, ge=0)
    percent: float | None = Field(default=None, ge=0, le=100)
    percent_basis: str | None = Field(
        default=None, description="What the percent applies to, e.g. 'contract price'."
    )
    is_minimum: bool | None = Field(
        default=None,
        description="True if the percent/amount is a floor ('minimum 5%').",
    )
    bidder_quoted: bool | None = Field(
        default=None,
        description="True if the bidder quotes the security amount in the bid.",
    )
    exemptions: list[str] = Field(
        default_factory=list, description="Who is exempt, e.g. ['MSME', 'Startup']."
    )
    forms_accepted: list[str] = Field(
        default_factory=list,
        description="e.g. ['bank guarantee', 'surety bond', 'DD'].",
    )


class Threshold(BaseModel):
    """Ek numeric eligibility limit: turnover, net worth, staff strength ..."""

    kind: ThresholdKind = ThresholdKind.OTHER
    label: str = Field(description="e.g. 'average annual turnover', 'net worth'.")
    value: float | None = Field(default=None, description="Numeric limit as written.")
    basis: ThresholdBasis = ThresholdBasis.OTHER
    period_years: int | None = Field(
        default=None, description="Look-back window in years."
    )
    aggregation: Aggregation | None = None
    counted_from: str | None = Field(
        default=None,
        description="Which turnover counts, e.g. 'consultancy only', 'ICT only'.",
    )
    raw_text: str | None = None


class ExperienceCriterion(BaseModel):
    """Generic experience requirement. DPR/AE/IT/works sab isi me aate hain."""

    category: str = Field(
        description="Short label, e.g. 'DPR preparation', 'project supervision', 'website development'."
    )
    lookback_years: int | None = None
    min_count: int | None = Field(
        default=None, description="Minimum number of projects."
    )
    min_single_value_inr: float | None = Field(default=None, ge=0)
    min_aggregate_value_inr: float | None = Field(default=None, ge=0)
    min_aggregate_value_multiple_of_cost: float | None = Field(
        default=None,
        description="Aggregate value as multiple of estimated project cost (2 = 2x).",
    )
    min_aggregate_length_multiple: float | None = Field(
        default=None,
        description="Aggregate length as multiple of project length (2 = 2x).",
    )
    min_single_length_pct: float | None = Field(
        default=None,
        description="Single project length as % of project length (e.g. 40).",
    )
    client_types: list[str] = Field(
        default_factory=list, description="e.g. ['Government', 'PSU']."
    )
    exclusions: list[str] = Field(default_factory=list)
    raw_text: str | None = None


class AdjustmentFactor(BaseModel):
    """Multiplication/enhancement factors, e.g. 2-lane experience x0.4."""

    condition: str
    factor: float


class Certification(BaseModel):
    name: str = Field(description="e.g. 'CMMI', 'ISO 27001', 'INFRACON registration'.")
    min_level: str | None = Field(default=None, description="e.g. 'Level 3'.")
    required: bool | None = Field(
        default=None, description="True=mandatory, False=only scored."
    )


class Condition(BaseModel):
    """Outlet: jo cheez kisi typed field me fit nahi hoti."""

    category: str = Field(
        description="One or two words, e.g. 'local office', 'security audit'."
    )
    text: str = Field(description="The condition in one short sentence.")


# --------------------------------------------------------------------------
# sections (Each section's model defines the response schema for its LLM call.)
# --------------------------------------------------------------------------


class Contact(BaseModel):
    name: str | None = None
    designation: str | None = None
    email: str | None = None
    phone: str | None = None
    address: str | None = Field(
        default=None, description="Address for communication / physical submission."
    )


class Identity(BaseModel):
    tender_code: str | None = None
    rfp_number: str | None = None
    nit_number: str | None = None
    title: str | None = None
    title_full: str | None = None
    authority_name: str | None = None
    client: str | None = Field(
        default=None, description="Ministry / parent body if different."
    )
    funding_agency: str | None = Field(
        default=None, description="e.g. 'World Bank', 'JICA'."
    )
    tender_type: str | None = Field(
        default=None, description="e.g. 'domestic', 'international'."
    )
    assignment_type: AssignmentType | None = None
    assignment_type_text: str | None = Field(
        default=None,
        description="Assignment as the document names it, e.g. 'Authority's Engineer'.",
    )
    sector: str | None = None
    selection_method: SelectionMethod | None = None
    selection_method_basis: Basis | None = Field(
        default=None,
        description="'inferred' if only a combined-score formula is given and the method is not named.",
    )
    contract_form: ContractForm | None = None
    no_of_covers: int | None = None
    eportal_url: str | None = None
    contact: Contact = Field(default_factory=Contact)


class PackageItem(BaseModel):
    package_id: str | None = None
    name: str | None = None
    state: str | None = None
    length_km: float | None = Field(default=None, ge=0)
    project_cost_inr: float | None = Field(default=None, ge=0)
    estimated_value_inr: float | None = Field(default=None, ge=0)
    duration_months: float | None = Field(default=None, ge=0)


class Project(BaseModel):
    description: str | None = None
    states: list[str] = Field(default_factory=list)
    routes: list[str] = Field(
        default_factory=list,
        description="Full route/stretch names including destination.",
    )
    length_km: float | None = Field(default=None, ge=0)
    lanes: str | None = Field(
        default=None, description="e.g. '4-lane', '2-lane with paved shoulders'."
    )
    project_mode: str | None = Field(
        default=None,
        description="Mode of the project being serviced: EPC, HAM, BOT, DBFOT ... null if not decided.",
    )
    estimated_contract_value: Amount = Field(
        default_factory=Amount,
        description="Estimated cost of THIS assignment/contract (e.g. consultancy cost).",
    )
    project_cost: Amount = Field(
        default_factory=Amount,
        description="Cost of the underlying project (e.g. civil cost of the highway). Different from contract value.",
    )
    packages: list[PackageItem] = Field(
        default_factory=list,
        description="ONLY if the document lists 2 or more packages/lots. Leave empty for a single package.",
        max_length=20,
    )


class PreBid(BaseModel):
    required: bool | None = None
    date: str | None = None
    place: str | None = None

    @field_validator("date", mode="before")
    @classmethod
    def _date(cls, v: object) -> str | None:
        return normalize_date(v)


class Dates(BaseModel):
    published: str | None = None
    download_start: str | None = None
    download_end: str | None = None
    clarification_end: str | None = None
    submission_start: str | None = None
    submission_end: str | None = Field(
        default=None, description="Online submission deadline."
    )
    hard_copy_deadline: str | None = Field(
        default=None,
        description="Physical submission deadline, only if different from online.",
    )
    technical_opening: str | None = None
    financial_opening: str | None = None
    validity_days: int | None = Field(
        default=None, ge=0, description="Proposal/bid validity in days."
    )
    validity_basis: str | None = Field(
        default=None,
        description="Counted from when, e.g. 'last date of submission', 'bid opening'.",
    )
    pre_bid: PreBid = Field(default_factory=PreBid)

    @field_validator(
        "published",
        "download_start",
        "download_end",
        "clarification_end",
        "submission_start",
        "submission_end",
        "hard_copy_deadline",
        "technical_opening",
        "financial_opening",
        mode="before",
    )
    @classmethod
    def _dates(cls, v: object) -> str | None:
        return normalize_date(v)


class Securities(BaseModel):
    document_fee: Amount = Field(default_factory=Amount)
    bid_security: Security = Field(default_factory=Security)
    performance_security: Security = Field(default_factory=Security)
    additional_performance_security: Security = Field(default_factory=Security)


class JointVenture(BaseModel):
    allowed: bool | None = None
    max_members: int | None = Field(default=None, ge=1)
    lead_min_share_pct: float | None = Field(default=None, ge=0, le=100)
    member_min_share_pct: float | None = Field(default=None, ge=0, le=100)
    lead_min_criteria_pct: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Lead must meet this % of the eligibility criteria.",
    )
    partner_min_criteria_pct: float | None = Field(default=None, ge=0, le=100)
    partner_restriction: str | None = Field(
        default=None, description="e.g. 'international firm only'."
    )
    joint_and_several_liability: bool | None = None
    mou_required: bool | None = None


class Subcontracting(BaseModel):
    allowed: bool | None = None
    max_pct: float | None = Field(default=None, ge=0, le=100)
    scope: str | None = None
    prior_approval_required: bool | None = None


class Eligibility(BaseModel):
    entity_types: list[str] = Field(default_factory=list)
    registrations: list[str] = Field(
        default_factory=list, description="e.g. GSTIN, PAN, EPF, ESI, INFRACON."
    )
    certifications: list[Certification] = Field(default_factory=list)
    min_years_in_business: int | None = None
    exclusions: list[str] = Field(
        default_factory=list,
        description="Short phrases: 'blacklisted', 'insolvent', 'conflict of interest' ...",
    )
    restricted_country_rule: bool | None = Field(
        default=None,
        description="True if land-border / reciprocity restriction is mentioned.",
    )
    startup_relaxation: bool | None = None
    participation_limit: str | None = Field(
        default=None, description="e.g. 'only one bid per consultant'."
    )
    jv: JointVenture = Field(default_factory=JointVenture)
    subcontracting: Subcontracting = Field(default_factory=Subcontracting)
    associate_rules: str | None = Field(
        default=None, description="Rules for associate firms / associate key personnel."
    )


class BidCapacity(BaseModel):
    required: bool | None = None
    formula: str | None = None
    description: str | None = None


class Qualification(BaseModel):
    financial: list[Threshold] = Field(default_factory=list)
    experience: list[ExperienceCriterion] = Field(default_factory=list)
    adjustment_factors: list[AdjustmentFactor] = Field(default_factory=list)
    bid_capacity: BidCapacity = Field(default_factory=BidCapacity)


# ---- evaluation: 3 fixed levels (criteria -> sub_criteria -> bands), no recursion ----


class ScoreBand(BaseModel):
    range_text: str = Field(
        description="Band as written, e.g. '>10 to <=20', '5 projects'."
    )
    points: float | None = None


class SubCriterion(BaseModel):
    name: str
    max_points: float | None = None
    bands: list[ScoreBand] = Field(default_factory=list)


class Criterion(BaseModel):
    name: str
    max_points: float | None = None
    sub_criteria: list[SubCriterion] = Field(default_factory=list)


class NamedWeight(BaseModel):
    name: str
    weight_pct: float = Field(ge=0, le=100)


class EvaluationWeights(BaseModel):
    technical_pct: float | None = Field(default=None, ge=0, le=100)
    financial_pct: float | None = Field(default=None, ge=0, le=100)
    other: list[NamedWeight] = Field(
        default_factory=list,
        description="e.g. DPR rating, PBG quote in newer NHAI RFPs.",
    )


class Evaluation(BaseModel):
    weights: EvaluationWeights = Field(default_factory=EvaluationWeights)
    formula_text: str | None = None
    min_technical_score: float | None = None
    stages: list[str] = Field(
        default_factory=list,
        description="e.g. ['pre-qualification', 'technical', 'commercial'].",
    )
    criteria: list[Criterion] = Field(default_factory=list)
    per_expert_rule: str | None = Field(
        default=None,
        description="Rule on individual key experts, e.g. 'each CV must score >=70%'.",
    )


class KeyPersonnel(BaseModel):
    role: str
    category: PersonnelCategory = PersonnelCategory.KEY
    count: int | None = None
    person_months: float | None = Field(default=None, ge=0)
    site_pct: float | None = Field(default=None, ge=0, le=100)
    office_pct: float | None = Field(default=None, ge=0, le=100)
    qualification: str | None = None
    certifications: list[str] = Field(default_factory=list)
    location: str | None = Field(
        default=None,
        description="e.g. 'Site', 'Design Office', 'Site and Design Office'.",
    )
    responsibilities: str | None = Field(
        default=None, description="One short sentence, only if stated."
    )
    min_experience_years: float | None = Field(
        default=None,
        description="ONLY if the document states a hard minimum. Do not copy scoring-band thresholds here.",
    )
    technical_weight_points: float | None = Field(
        default=None,
        description="This role's share of the technical score (e.g. 14 of 60).",
    )
    individual_max_points: float | None = Field(
        default=None, description="Max marks of the individual CV scale (e.g. 100)."
    )
    min_score_pct: float | None = Field(default=None, ge=0, le=100)
    rate_inr_per_month: float | None = Field(default=None, ge=0)
    cost_inr: float | None = Field(default=None, ge=0)
    is_conditional: bool = False
    condition: str | None = None


class KeyPersonnelSection(BaseModel):
    summary: str | None = Field(
        default=None,
        description="At most 2 sentences: how many key / sub-professional / support staff and total person-months.",
    )
    roles: list[KeyPersonnel] = Field(
        default_factory=list,
        description="Key personnel first. Sub-professional/support staff only if listed with roles.",
    )
    key_person_months_total: float | None = Field(default=None, ge=0)
    all_staff_person_months_total: float | None = Field(default=None, ge=0)
    rules: str | None = Field(
        default=None,
        description="Replacement / deployment rules in one or two sentences.",
    )


class Phase(BaseModel):
    name: str = Field(
        description="e.g. 'construction', 'warranty', 'O&M', 'maintenance'."
    )
    months: float | None = Field(default=None, ge=0)


class PaymentStage(BaseModel):
    stage: str
    percent: float | None = Field(default=None, ge=0, le=100)


class LiquidatedDamages(BaseModel):
    rate_text: str | None = Field(
        default=None, description="e.g. '0.05% of contract price per day'."
    )
    max_pct: float | None = Field(default=None, ge=0, le=100)


class ContractTerms(BaseModel):
    duration_months: float | None = Field(default=None, ge=0)
    commencement: str | None = Field(
        default=None, description="e.g. 'within 30 days of signing'."
    )
    price_escalation: str | None = Field(
        default=None, description="e.g. '5% per year after month 12'."
    )
    phases: list[Phase] = Field(default_factory=list)
    payment_stages: list[PaymentStage] = Field(default_factory=list)
    liquidated_damages: LiquidatedDamages = Field(default_factory=LiquidatedDamages)


# --------------------------------------------------------------------------
# system block (LLM can't put, backend put it)
# --------------------------------------------------------------------------


class ExtractionMeta(BaseModel):
    schema_version: str = SCHEMA_VERSION
    confidence: Confidence | None = None
    needs_review: bool = False
    review_flags: list[str] = Field(default_factory=list)
    pages_used: dict[str, list[int]] = Field(default_factory=dict)
    model: str | None = None
    prompt_hash: str | None = None
    extracted_at: str | None = None


# --------------------------------------------------------------------------
# root
# --------------------------------------------------------------------------


class TenderExtractionLLM(BaseModel):
    document_kind: DocumentKind = DocumentKind.TENDER_DOCUMENT
    family: DocumentFamily = DocumentFamily.OTHER
    identity: Identity = Field(default_factory=Identity)
    project: Project = Field(default_factory=Project)
    dates: Dates = Field(default_factory=Dates)
    securities: Securities = Field(default_factory=Securities)
    eligibility: Eligibility = Field(default_factory=Eligibility)
    qualification: Qualification = Field(default_factory=Qualification)
    evaluation: Evaluation = Field(default_factory=Evaluation)
    key_personnel: KeyPersonnelSection = Field(default_factory=KeyPersonnelSection)
    contract: ContractTerms = Field(default_factory=ContractTerms)
    additional_conditions: list[Condition] = Field(default_factory=list, max_length=10)


class TenderExtraction(TenderExtractionLLM):
    meta: ExtractionMeta = Field(default_factory=ExtractionMeta)


# Section name -> model. ExtractionService will loop through this mapping,
# and each model will define the response_schema for that section's LLM call.
SECTION_MODELS: dict[str, type[BaseModel]] = {
    "identity": Identity,
    "project": Project,
    "dates": Dates,
    "securities": Securities,
    "eligibility": Eligibility,
    "qualification": Qualification,
    "evaluation": Evaluation,
    "key_personnel": KeyPersonnelSection,
    "contract": ContractTerms,
}
