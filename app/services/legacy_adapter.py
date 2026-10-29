"""
v1.2 TenderExtraction  ->  the HR/tender app's `RfpExtractionResponse` (PascalCase, DD-MM-YYYY).

Pure code: no LLM call, no pydantic, no imports from this project. It takes a dict (or any object
with `model_dump`) and returns a dict, so the ONE file can be copied into the other app and used in
place of `extract_rfp()`:

    from app.services.legacy_adapter import to_legacy
    data = to_legacy(extraction)          # extraction = TenderExtraction or its JSON dict

Why an adapter and not a schema change: extraction, cache fingerprint and eval stay on v1.2; only the
last step changes shape. The router of the other app (preview/save/NIT/audits/MD review) stays as is.

What the router really reads (so these keys must keep these exact names):
  EligibilityCriteriaJson.eligibility_criteria_amount          crore, bare number
  EligibilityCriteriaJson.eligibility_criteria_condition
  .financial_capacity.min_annual_turnover_crore / min_pct_of_estimated_cost / percentage_requirement_type
  .technical_capacity.eligible_assignment_count / scoring_rules / lookback_years /
                      similar_assignment_min_length_km / similar_assignment_category
  .jv_conditions.max_jv_members_total / lead_member_qualification_pct /
                 substantial_member_qualification_pct / lead_member_min_share_pct / each_member_min_share_pct

Rules kept from v1.2: a value that the document does not state stays None. Nothing is guessed.
Derived values: max_jv_partner_other_than_lead = max_members - 1 (v1.2 `max_members` is the total incl. lead).
Derived values that also say so in `ExtractionReviewFlags`: km thresholds computed from "% of project
length", EstimatedValue chosen between net / with tax or taken from the project cost.
"""

import re
from datetime import datetime
from typing import Any

ADAPTER_VERSION = "1"

# every key this adapter can write at top level (the extraction-time keys of RfpExtractionResponse
# plus two additive ones). Estimate / NIT keys (EstimatedValueSource, Nit*, ...) are NOT here:
# the other app's router fills them later.
LEGACY_KEYS = (
    "TenderCode",
    "NitNumber",
    "TenderTitle",
    "TenderTitleFull",
    "AuthorityName",
    "RfpNumber",
    "Client",
    "ConsultancyType",
    "ProjectSector",
    "ProjectMode",
    "ProjectDescription",
    "State",
    "Route",
    "Routes",
    "LengthKm",
    "EstimatedValue",
    "BidSecurityAmount",
    "Deadline",
    "BidSubmissionDate",
    "TechnicalBidOpeningDate",
    "PreBidMeetingDate",
    "PreBidMeetingRequired",
    "PreBidMeetingPlace",
    "Status",
    "EPortalUrl",
    "BidDocumentBlobPath",
    "EligibilityCriteriaJson",
    "PersonnelSummary",
    "RequiredKeyPersonnel",
    "ConditionalKeyPersonnel",
    "extraction_confidence",
    "missing_fields",
    "ExtractionReviewFlags",
    "ExtractionSource",
)

# keys that count for `missing_fields` / confidence (the others are app-supplied or never in a document)
_SCORED_KEYS = (
    "TenderCode",
    "TenderTitle",
    "TenderTitleFull",
    "AuthorityName",
    "RfpNumber",
    "Client",
    "ConsultancyType",
    "ProjectSector",
    "ProjectDescription",
    "State",
    "Route",
    "LengthKm",
    "EstimatedValue",
    "BidSecurityAmount",
    "BidSubmissionDate",
    "TechnicalBidOpeningDate",
    "PreBidMeetingDate",
    "PreBidMeetingPlace",
    "EPortalUrl",
    "EligibilityCriteriaJson",
    "RequiredKeyPersonnel",
)

_CRORE = 10_000_000

_ASSIGNMENT_TO_CONSULTANCY = {
    "dpr": "DPR",
    "feasibility_study": "DPR/Feasibility",
    "authority_engineer": "Authority Engineer",
    "independent_engineer": "Independent Engineer (IE)",
    "project_management": "PMC",
    "construction_supervision": "Supervision",
}

_SECTOR_WORDS = (
    ("Road/Highway", ("highway", "road", "expressway", "corridor")),
    ("Railway", ("railway", "rail ")),
    ("Metro", ("metro",)),
    ("Bridge", ("bridge",)),
    ("Water", ("water", "irrigation", "sewer")),
    ("Airport", ("airport", "aviation")),
    ("Urban", ("urban", "smart city", "municipal")),
)

_PROJECT_MODES = {
    "dbfot": "DBFOT",
    "epc": "EPC",
    "ham": "HAM",
    "bot": "BOT",
    "budget": "Budget",
    "hybrid": "Hybrid",
}


# ---------------------------------------------------------------- small helpers


def _dump(extraction: Any) -> dict:
    if hasattr(extraction, "model_dump"):
        return extraction.model_dump(mode="json")
    return extraction


def _d(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _l(value: Any) -> list:
    return value if isinstance(value, list) else []


def _compact(data: dict) -> dict:
    """Drop None and empty list/dict values (keeps False and 0)."""
    return {k: v for k, v in data.items() if v is not None and v != [] and v != {}}


def _clean_ref(value: Any) -> str | None:
    """Reference number: no spaces around / and -, no trailing separator ("2025-26/" -> "2025-26")."""
    if not value:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip().replace("–", "-").replace("—", "-")
    text = re.sub(r"\s*([/\-])\s*", r"\1", text)
    return text.strip(" ./,;:") or None


def _group_indian(number: int) -> str:
    digits = str(abs(number))
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    parts: list[str] = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join([*parts, tail])


def _is_inr(currency: str | None) -> bool:
    return currency in (None, "", "INR")


def _money(amount: Any, currency: str | None) -> str | None:
    """INR -> '₹ 9,86,44,500' (the router parses this); other currency -> 'USD 325,000'."""
    if amount is None:
        return None
    if _is_inr(currency):
        return f"₹ {'-' if amount < 0 else ''}{_group_indian(int(round(amount)))}"
    whole = float(amount).is_integer()
    return f"{currency} {amount:,.0f}" if whole else f"{currency} {amount:,.2f}"


def _crore(amount: Any, currency: str | None) -> float | None:
    """Money -> crore. Only for INR: converting another currency to crore would be wrong."""
    if amount is None or not _is_inr(currency):
        return None
    return round(amount / _CRORE, 6)


def _ddmmyyyy(value: Any) -> str | None:
    """ISO date -> DD-MM-YYYY. A placeholder text the parser could not turn into a date -> None."""
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").strftime("%d-%m-%Y")
    except ValueError:
        return None


def _slug(text: str, limit: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return slug[:limit] or "item"


def _unique(key: str, taken: dict) -> str:
    candidate, number = key, 2
    while candidate in taken:
        candidate = f"{key}_{number}"
        number += 1
    return candidate


# ---------------------------------------------------------------- title (stop-gap, see note below)

_TITLE_PREFIX = re.compile(
    r"^\s*(?:request\s+for\s+proposals?\s*(?:\(rfp\))?\s*(?:for\s+)?"
    r"|consultancy\s+services?\s+(?:for\s+)?(?:the\s+)?"
    r"|selection\s+of\s+(?:an?\s+)?"
    r"|appointment\s+of\s+(?:an?\s+)?)+",
    re.IGNORECASE,
)
_LENGTH_PAREN = re.compile(
    r"\(\s*total\s+length\s*(?:\(approx\.?\))?\s*:?\s*[\d.,]+\s*km\s*\)", re.IGNORECASE
)
_STATE_TAIL = re.compile(r"\s+in\s+the\s+states?\s+of\s+.+$", re.IGNORECASE)
_ACRONYMS = {
    "DPR",
    "NH",
    "NHAI",
    "NHIDCL",
    "PMC",
    "EPC",
    "HAM",
    "BOT",
    "IE",
    "AE",
    "MPRDC",
    "MSRDC",
    "PWD",
    "AIP",
    "ITI",
    "SPV",
    "RFP",
    "NIT",
    "PPP",
    "ATP",
    "ADA",
    "PM-SETU",
    "MORTH",
    "GST",
    "UP",
    "MP",
    "RRTS",
}
_SMALL_WORDS = {
    "of",
    "for",
    "the",
    "and",
    "in",
    "to",
    "a",
    "an",
    "on",
    "at",
    "by",
    "with",
    "from",
}


def _cap_word(word: str, first: bool) -> str:
    core = re.sub(r"[^\w&-]", "", word)
    if core.upper() in _ACRONYMS or (word.startswith("(") and 2 <= len(core) <= 5):
        return word.upper()
    if core.lower() == "km":
        return word.lower()
    if not first and core.lower() in _SMALL_WORDS:
        return word.lower()
    return "-".join(part.capitalize() for part in word.split("-"))


def _short_title(title: str | None) -> str | None:
    """Readable display title from the full one. The v1.2 schema has no short title, so this is a
    stop-gap: strip 'Consultancy Services for', the '(Total length ...)' bracket and a trailing
    'in the State(s) of ...', fix 'Bhopal- Kota', and un-shout ALL CAPS. TenderTitleFull keeps the exact text."""
    if not title:
        return None
    text = re.sub(r"\s+", " ", title).strip()
    text = _LENGTH_PAREN.sub("", text)
    text = _TITLE_PREFIX.sub("", text)
    text = _STATE_TAIL.sub("", text)
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "-", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,.;:") or title.strip()

    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) >= 0.7:
        text = " ".join(_cap_word(w, i == 0) for i, w in enumerate(text.split(" ")))
    return text[:1].upper() + text[1:]


# ---------------------------------------------------------------- identity / project


def _consultancy_type(identity: dict) -> str | None:
    kind = identity.get("assignment_type")
    text = (identity.get("assignment_type_text") or "").lower()
    if "general consultant" in text:
        return "General Consultant (GC)"
    if kind is None:
        return None
    return _ASSIGNMENT_TO_CONSULTANCY.get(kind, "Other")


def _sector(value: str | None) -> str | None:
    if not value:
        return None
    low = f"{value.lower()} "
    for legacy, words in _SECTOR_WORDS:
        if any(word in low for word in words):
            return legacy
    return "Other"


def _project_mode(value: str | None) -> str | None:
    if not value:
        return None
    low = value.lower()
    for word, legacy in _PROJECT_MODES.items():
        if re.search(rf"\b{word}\b", low):
            return legacy
    return "Other"


def _routes(project: dict) -> tuple[str | None, list[dict] | None]:
    names = [r for r in _l(project.get("routes")) if r]
    summary = " – ".join(names) if names else None

    packages = _l(project.get("packages"))
    items = None
    if len(packages) >= 2:
        items = [
            _compact(
                {
                    "name": p.get("name") or p.get("package_id"),
                    "length_km": p.get("length_km"),
                    "configuration": project.get("lanes"),
                }
            )
            for p in packages
        ]
        if not summary:
            summary = (
                "; ".join(str(i.get("name")) for i in items if i.get("name")) or None
            )
    return summary, items


def _estimated_value(doc: dict, flags: list[str]) -> str | None:
    """Consultancy contract value (or project cost when that is the only figure) as one money string.
    Net of tax by default; the with-tax figure goes in a review flag so nothing is hidden."""
    currency = _d(doc.get("identity")).get("currency")
    project = _d(doc.get("project"))
    contract, cost = (
        _d(project.get("estimated_contract_value")),
        _d(project.get("project_cost")),
    )

    pct_of_project = any(
        t.get("basis") == "pct_of_project_cost"
        for t in _l(_d(doc.get("qualification")).get("financial"))
    )
    order = (cost, contract) if pct_of_project else (contract, cost)

    for amount in order:
        net, gross = amount.get("value"), amount.get("value_with_tax")
        chosen = net if net is not None else gross
        if chosen is None:
            continue
        if net is not None and gross is not None and gross != net:
            flags.append(
                f"EstimatedValue is the figure without tax ({_money(net, currency)}); with tax it is {_money(gross, currency)}"
            )
        if amount is cost and order[0] is not cost:
            flags.append(
                "EstimatedValue is the project cost; the document states no contract value"
            )
        return _money(chosen, currency)
    return None


def _bid_security(securities: dict, currency: str | None) -> str | None:
    sec = _d(securities.get("bid_security"))
    if sec.get("amount") is not None:
        return _money(sec["amount"], currency)
    if sec.get("percent") is not None:
        base = f" of {sec['percent_basis']}" if sec.get("percent_basis") else ""
        return f"{sec['percent']:g}%{base}"
    return None


# ---------------------------------------------------------------- eligibility


def _financial_block(
    financial: list[dict], currency: str | None
) -> tuple[dict, float | None, str | None]:
    """-> (financial_capacity dict, eligibility_criteria_amount in crore, eligibility_criteria_condition)."""
    turnovers = [t for t in financial if t.get("kind") == "turnover"]
    main = next(
        (t for t in turnovers if t.get("value") is not None),
        turnovers[0] if turnovers else None,
    )

    block: dict = {
        "min_annual_turnover_crore": None,
        "min_pct_of_estimated_cost": None,
        "percentage_requirement_type": None,
        "percentage_requirement_base": None,
        "lookback_years": None,
        "requirement_basis": None,
        "lead_member_only": None,
        "notes": None,
    }
    amount = condition = None

    if main:
        basis, value = main.get("basis"), main.get("value")
        block["lookback_years"] = main.get("period_years")
        block["notes"] = main.get("raw_text")
        block["percentage_requirement_type"] = (
            "turnover"  # the other app sets it for every turnover rule
        )
        block["requirement_basis"] = {
            "average": "annual_average",
            "total": "total_over_period",
        }.get(main.get("aggregation"), main.get("aggregation"))
        if basis == "absolute_amount":
            amount = _crore(value, currency)
            block["min_annual_turnover_crore"] = amount
        elif (
            basis in ("pct_of_estimated_cost", "pct_of_project_cost")
            and value is not None
        ):
            block["min_pct_of_estimated_cost"] = value
            block["percentage_requirement_base"] = (
                "Estimated Project Cost"
                if basis == "pct_of_estimated_cost"
                else "Project Cost"
            )

        period = (
            f", last {main['period_years']} years" if main.get("period_years") else ""
        )
        how = f", {main['aggregation']}" if main.get("aggregation") else ""
        text = (
            f"{main.get('label') or 'turnover'} ({period.lstrip(', ')}{how})".replace(
                "( ", "("
            )
            .replace("()", "")
            .strip()
        )
        if main.get("raw_text"):
            text = f"{text}: {main['raw_text']}".strip(": ")
        condition = text or None

    others = [t for t in financial if t is not main]
    if others:
        block["other_thresholds"] = [
            _compact(
                {
                    "kind": t.get("kind"),
                    "label": t.get("label"),
                    "value": t.get("value"),
                    "value_crore": _crore(t.get("value"), currency)
                    if t.get("kind") in ("turnover", "net_worth", "project_value")
                    and t.get("basis") == "absolute_amount"
                    else None,
                    "basis": t.get("basis"),
                    "period_years": t.get("period_years"),
                    "aggregation": t.get("aggregation"),
                    "raw_text": t.get("raw_text"),
                }
            )
            for t in others
        ]
    return block, amount, condition


def _scoring_rules(evaluation: dict) -> dict | None:
    criteria = _l(evaluation.get("criteria"))
    if not criteria:
        return None
    rules: dict = {}
    for criterion in criteria:
        subs: dict = {}
        for sub in _l(criterion.get("sub_criteria")):
            entry = _compact(
                {
                    "name": sub.get("name"),
                    "max": sub.get("max_points"),
                    "thresholds": [
                        _compact(
                            {"range": b.get("range_text"), "points": b.get("points")}
                        )
                        for b in _l(sub.get("bands"))
                    ],
                }
            )
            subs[_unique(_slug(sub.get("name", "")), subs)] = entry
        rules[_unique(_slug(criterion.get("name", "")), rules)] = _compact(
            {
                "name": criterion.get("name"),
                "max_points": criterion.get("max_points"),
                "weight": criterion.get("weight"),
                "is_pass_fail": criterion.get("is_pass_fail"),
                "sub_criteria": subs,
            }
        )
    return rules


def _technical_block(doc: dict, flags: list[str]) -> dict:
    experience = _l(_d(doc.get("qualification")).get("experience"))
    evaluation = _d(doc.get("evaluation"))
    length = _d(doc.get("project")).get("length_km")
    adjustments = _l(_d(doc.get("qualification")).get("adjustment_factors"))
    currency = _d(doc.get("identity")).get("currency")

    pure_count = next(
        (
            e
            for e in experience
            if e.get("min_count")
            and not e.get("min_single_length_pct")
            and not e.get("min_aggregate_length_multiple")
            and e.get("min_single_value") is None
            and e.get("min_aggregate_value") is None
        ),
        None,
    )

    first = experience[0] if experience else {}
    multiple, pct = (
        first.get("min_aggregate_length_multiple"),
        first.get("min_single_length_pct"),
    )
    aggregate_km = round(length * multiple, 3) if length and multiple else None
    single_km = round(length * pct / 100, 3) if length and pct else None

    # ONE clear km rule only. Several rules together (aggregate AND single, adjustment factors,
    # more than one experience line) are left null for manual review: same rule as the other app's prompt.
    one_rule = (
        len(experience) == 1
        and not adjustments
        and (aggregate_km is None) != (single_km is None)
    )
    similar_km = (
        (aggregate_km if aggregate_km is not None else single_km) if one_rule else None
    )
    if similar_km is not None:
        flags.append(
            f"technical_capacity.similar_assignment_min_length_km={similar_km:g} is computed from the project length"
        )

    scoring = _scoring_rules(evaluation)
    return _compact(
        {
            "eligible_assignment_count": pure_count.get("min_count")
            if pure_count
            else None,
            "scoring_based": True if scoring else None,
            "scoring_rules": scoring,
            "max_total_marks": evaluation.get("total_points"),
            "min_qualifying_marks": evaluation.get("min_technical_score"),
            "min_value_crore": _crore(first.get("min_single_value"), currency),
            "eligible_project_categories": [
                e["category"] for e in experience if e.get("category")
            ],
            "lookback_years": first.get("lookback_years"),
            "similar_assignment_min_length_km": similar_km,
            "similar_assignment_category": first.get("category")
            if similar_km is not None
            else None,
            "minimum_aggregate_length_km": aggregate_km,
            "minimum_single_project_length_km": single_km,
            "description": first.get("raw_text"),
        }
    )


def _jv_block(eligibility: dict) -> dict:
    jv = _d(eligibility.get("jv"))
    members = jv.get(
        "max_members"
    )  # read as the TOTAL incl. lead; partners besides the lead = members - 1

    rules = []
    if jv.get("partner_restriction"):
        rules.append(jv["partner_restriction"])
    if jv.get("joint_and_several_liability"):
        rules.append("All JV members are jointly and severally liable.")
    if jv.get("mou_required"):
        rules.append("A JV agreement / MoU is required.")

    return {
        "allowed": jv.get("allowed"),
        "note": jv.get("partner_restriction"),
        "max_jv_members_total": members,
        "max_jv_partner_other_than_lead": members - 1 if members else None,
        "lead_member_min_share_pct": jv.get("lead_min_share_pct"),
        "each_member_min_share_pct": jv.get("member_min_share_pct"),
        "lead_member_qualification_pct": jv.get("lead_min_criteria_pct"),
        "substantial_member_qualification_pct": jv.get("partner_min_criteria_pct"),
        "substantial_member_min_share_pct": None,
        "structural_rules": rules,
        "jv_partner_type": jv.get("partner_restriction"),
        "joint_and_several_liability": jv.get("joint_and_several_liability"),
        "mou_required": jv.get("mou_required"),
    }


def _eligibility_json(doc: dict, flags: list[str]) -> dict:
    eligibility = _d(doc.get("eligibility"))
    qualification = _d(doc.get("qualification"))
    evaluation = _d(doc.get("evaluation"))
    currency = _d(doc.get("identity")).get("currency")

    financial, amount, condition = _financial_block(
        _l(qualification.get("financial")), currency
    )
    sub = _d(eligibility.get("subcontracting"))
    experience = _l(qualification.get("experience"))

    other = [
        f"{c.get('category')}: {c.get('text')}"
        for c in _l(doc.get("additional_conditions"))
    ]
    if eligibility.get("associate_rules"):
        other.append(f"Associates: {eligibility['associate_rules']}")
    if eligibility.get("startup_relaxation"):
        other.append("Relaxation for start-ups is allowed.")

    result = {
        "eligibility_criteria_amount": amount,  # ALWAYS present, as the router expects (null if none)
        "eligibility_criteria_condition": condition,
        "technical_capacity": _technical_block(doc, flags),
        "financial_capacity": financial,
        "jv_conditions": _jv_block(eligibility),
        "sub_contracting": _compact(
            {
                "allowed": sub.get("allowed"),
                "max_value_pct": sub.get("max_pct"),
                "scope": sub.get("scope"),
                "prior_approval_required": sub.get("prior_approval_required"),
            }
        ),
        "experience_requirements": [
            _compact(
                {
                    "category": e.get("category"),
                    "lookback_years": e.get("lookback_years"),
                    "min_count": e.get("min_count"),
                    "min_aggregate_length_multiple": e.get(
                        "min_aggregate_length_multiple"
                    ),
                    "min_single_length_pct": e.get("min_single_length_pct"),
                    "min_single_value": e.get("min_single_value"),
                    "min_aggregate_value": e.get("min_aggregate_value"),
                    "min_aggregate_value_multiple_of_cost": e.get(
                        "min_aggregate_value_multiple_of_cost"
                    ),
                    "client_types": e.get("client_types"),
                    "exclusions": e.get("exclusions"),
                    "raw_text": e.get("raw_text"),
                }
            )
            for e in experience
        ],
        "multiplication_factors": {
            a.get("condition"): a.get("factor")
            for a in _l(qualification.get("adjustment_factors"))
        },
        "residual_bid_capacity": _compact(_d(qualification.get("bid_capacity"))),
        "registration_requirements": [
            *_l(eligibility.get("registrations")),
            *[
                c.get("name")
                for c in _l(eligibility.get("certifications"))
                if c.get("name")
            ],
        ],
        "blacklisting_restrictions": "; ".join(_l(eligibility.get("exclusions")))
        or None,
        "land_border_restrictions": (
            "Bidders from countries sharing a land border with India must be registered with the competent authority."
            if eligibility.get("restricted_country_rule")
            else None
        ),
        "participation_restriction": eligibility.get("participation_limit"),
        "other_conditions": other,
        "evaluation": _compact(
            {
                "weights": _compact(
                    {
                        "technical_pct": _d(evaluation.get("weights")).get(
                            "technical_pct"
                        ),
                        "financial_pct": _d(evaluation.get("weights")).get(
                            "financial_pct"
                        ),
                        "other": _l(_d(evaluation.get("weights")).get("other")),
                    }
                ),
                "formula_text": evaluation.get("formula_text"),
                "min_technical_score": evaluation.get("min_technical_score"),
                "total_points": evaluation.get("total_points"),
                "rating_scale_max": evaluation.get("rating_scale_max"),
                "bid_parameter": evaluation.get("bid_parameter"),
                "stages": evaluation.get("stages"),
                "per_expert_rule": evaluation.get("per_expert_rule"),
            }
        ),
    }
    return result


# ---------------------------------------------------------------- personnel


def _plain_number(value: float) -> str:
    """6000000.0 -> '6000000'. Digits only: HrTenderRole.CostBudget may be a numeric column."""
    return str(int(value)) if float(value).is_integer() else f"{value:.2f}"


def _role_budget(role: dict) -> tuple[str | None, bool]:
    """-> (cost_budget, derived). The other app stores the role's TOTAL amount (rate x months), as plain digits.
    Uses the document's own amount (`cost`); when that was not extracted it is rate_per_month x person_months."""
    if role.get("cost") is not None:
        return _plain_number(role["cost"]), False
    rate, months = role.get("rate_per_month"), role.get("person_months")
    if rate is not None and months is not None:
        return _plain_number(rate * months), True
    return None, False


def _role(role: dict, budget: str | None) -> dict:
    rate, cost = role.get("rate_per_month"), role.get("cost")
    return {
        "role": role.get("role"),
        "qualification": role.get("qualification"),
        "experience_years": role.get("min_experience_years"),
        "cost_budget": budget,
        "location": role.get("location"),
        "responsibilities": role.get("responsibilities"),
        "min_site_days": None,  # v1.2 has no such field; site_pct / person_months are in `eligibility`
        "count": role.get("count"),
        "certifications": _l(role.get("certifications")) or None,
        "skills": None,
        "domain_expertise": None,
        "employment_type": None,
        "reporting_requirements": None,
        "language_requirements": None,
        "travel_requirements": None,
        "eligibility": _compact(
            {
                "category": role.get("category")
                if role.get("category") != "key"
                else None,
                "person_months": role.get("person_months"),
                "site_pct": role.get("site_pct"),
                "office_pct": role.get("office_pct"),
                "marking_points": role.get("technical_weight_points"),
                "individual_max_points": role.get("individual_max_points"),
                "min_score_pct": role.get("min_score_pct"),
                "rate_per_month": rate,
                "cost": cost,
                "condition": role.get("condition"),
            }
        )
        or None,
    }


def _personnel_summary(roles: list[dict]) -> str | None:
    parts = []
    for r in roles:
        text = f"{r['role']}" + (f" (×{r['count']})" if r.get("count") else "")
        if r.get("experience_years") is not None:
            text += f" Min {r['experience_years']:g} yrs"
        if r.get("location"):
            text += f" [{r['location']}]"
        parts.append(text)
    return ", ".join(parts) or None


# ---------------------------------------------------------------- main


def _confidence(result: dict, needs_review: bool) -> str:
    found = sum(1 for key in _SCORED_KEYS if result.get(key) not in (None, "", [], {}))
    share = found / len(_SCORED_KEYS)
    level = 2 if share >= 0.8 else 1 if share >= 0.5 else 0
    if needs_review:
        level = min(level, 1)
    return ("low", "medium", "high")[level]


def to_legacy(extraction: Any) -> dict:
    """v1.2 TenderExtraction (model or dict) -> RfpExtractionResponse dict. See the module docstring."""
    doc = _dump(extraction)
    identity, project = _d(doc.get("identity")), _d(doc.get("project"))
    dates, securities = _d(doc.get("dates")), _d(doc.get("securities"))
    meta = _d(doc.get("meta"))
    currency = identity.get("currency")
    flags: list[str] = list(_l(meta.get("review_flags")))

    rfp_number = _clean_ref(identity.get("rfp_number"))
    route, route_items = _routes(project)
    portal = identity.get("eportal_url")

    roles = [r for r in _l(_d(doc.get("key_personnel")).get("roles")) if r.get("role")]
    derived_budgets = 0

    def build(row: dict) -> dict:
        nonlocal derived_budgets
        budget, derived = _role_budget(row)
        derived_budgets += derived
        return _role(row, budget)

    required = [
        build(r)
        for r in roles
        if not r.get("is_conditional") and r.get("category", "key") == "key"
    ]
    conditional = [build(r) for r in roles if r.get("is_conditional")]
    if derived_budgets:
        flags.append(
            f"cost_budget = rate_per_month x person_months for {derived_budgets} role(s) (the document's own amount was not extracted)"
        )
    skipped = [
        r
        for r in roles
        if not r.get("is_conditional") and r.get("category", "key") != "key"
    ]
    if skipped:
        flags.append(
            f"{len(skipped)} sub-professional/support role(s) are not in RequiredKeyPersonnel"
        )

    submission = _ddmmyyyy(dates.get("submission_end"))
    pre_bid = _d(dates.get("pre_bid"))

    result = {
        "TenderCode": rfp_number
        or _clean_ref(identity.get("tender_code"))
        or _clean_ref(identity.get("nit_number")),
        "NitNumber": _clean_ref(identity.get("nit_number")),
        "TenderTitle": _short_title(
            identity.get("title") or identity.get("title_full")
        ),
        "TenderTitleFull": identity.get("title_full") or identity.get("title"),
        "AuthorityName": identity.get("authority_name"),
        "RfpNumber": rfp_number,
        "Client": identity.get("client") or identity.get("authority_name"),
        "ConsultancyType": _consultancy_type(identity),
        "ProjectSector": _sector(identity.get("sector")),
        "ProjectMode": _project_mode(project.get("project_mode")),
        "ProjectDescription": project.get("description"),
        "State": ", ".join(_l(project.get("states"))) or None,
        "Route": route,
        "Routes": route_items,
        "LengthKm": project.get("length_km"),
        "EstimatedValue": _estimated_value(doc, flags),
        "BidSecurityAmount": _bid_security(securities, currency),
        "Deadline": submission,
        "BidSubmissionDate": submission,
        "TechnicalBidOpeningDate": _ddmmyyyy(dates.get("technical_opening")),
        "PreBidMeetingDate": _ddmmyyyy(pre_bid.get("date")),
        "PreBidMeetingRequired": pre_bid.get("required"),
        "PreBidMeetingPlace": pre_bid.get("place"),
        "Status": None,  # never inferred
        "EPortalUrl": portal
        if portal and str(portal).lower().startswith("http")
        else None,
        "BidDocumentBlobPath": None,  # set by the app
        "EligibilityCriteriaJson": _eligibility_json(doc, flags),
        "PersonnelSummary": _personnel_summary(required),
        "RequiredKeyPersonnel": required or None,
        "ConditionalKeyPersonnel": conditional or None,
    }

    if meta.get("needs_review") and not flags:
        flags.append("the extraction was marked needs_review")

    result["extraction_confidence"] = _confidence(
        result, bool(meta.get("needs_review"))
    )
    result["missing_fields"] = [
        key for key in _SCORED_KEYS if result.get(key) in (None, "", [], {})
    ]
    result["ExtractionReviewFlags"] = flags
    result["ExtractionSource"] = _compact(
        {
            "engine": "rfp-rag",
            "schema_version": meta.get("schema_version"),
            "model": meta.get("model"),
            "prompt_hash": meta.get("prompt_hash"),
            "extracted_at": meta.get("extracted_at"),
            "adapter_version": ADAPTER_VERSION,
        }
    )
    return result
