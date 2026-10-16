import json

import pytest
from pydantic import ValidationError

from app.schemas.extraction import (
    SECTION_MODELS,
    Amount,
    Criterion,
    Dates,
    DelayDamage,
    Security,
    SecurityStatus,
    TenderExtraction,
    TenderExtractionLLM,
    is_iso_date,
    normalize_date,
)


def test_empty_extraction_is_valid():
    # If an RFP gives nothing, the object must still be valid (everything "not found").
    doc = TenderExtraction()

    assert doc.identity.title is None
    assert doc.key_personnel.roles == []
    assert doc.securities.bid_security.status == SecurityStatus.NOT_STATED


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("25.03.2026", "2026-03-25"),
        ("25-03-2026", "2026-03-25"),
        ("25/03/2026", "2026-03-25"),
        ("25.03.2026, 05:00 PM", "2026-03-25"),
        ("2026-03-25", "2026-03-25"),
        ("25 March 2026", "2026-03-25"),
        ("January 28, 2020", "2020-01-28"),
        (None, None),
        ("", None),
    ],
)
def test_normalize_date(raw, expected):
    assert normalize_date(raw) == expected


def test_unparseable_date_keeps_raw_text_and_is_detectable():
    value = normalize_date("To be notified")

    assert value == "To be notified"
    assert not is_iso_date(value)


def test_dates_model_normalizes_on_input():
    dates = Dates(submission_end="25.03.2026, 05:00 PM", technical_opening="27.03.2026")

    assert dates.submission_end == "2026-03-25"
    assert dates.technical_opening == "2026-03-27"


def test_percent_bounds_are_enforced():
    with pytest.raises(ValidationError):
        Security(percent=150)


def test_roundtrip_json():
    # Build with model_validate (attribute assignment skips validators).
    doc = TenderExtraction.model_validate(
        {
            "identity": {"title": "DPR for 4-laning"},
            "dates": {"submission_end": "25.03.2026"},
        }
    )

    again = TenderExtraction.model_validate(json.loads(doc.model_dump_json()))

    assert again == doc
    assert again.dates.submission_end == "2026-03-25"


# ---------- Gemini structured-output compatibility (static checks) ----------


def _walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


@pytest.mark.parametrize("name, model", list(SECTION_MODELS.items()))
def test_section_schema_has_no_recursion(name, model):
    # A recursive schema = a $ref that points to itself. Gemini often rejects it.
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})

    def refs_in(node):
        return {
            n["$ref"].split("/")[-1]
            for n in _walk(node)
            if isinstance(n, dict) and "$ref" in n
        }

    for def_name, definition in defs.items():
        assert def_name not in refs_in(definition), f"{name}: {def_name} is recursive"


@pytest.mark.parametrize("name, model", list(SECTION_MODELS.items()))
def test_section_schema_has_no_open_dicts_or_forbid(name, model):
    schema = model.model_json_schema()

    for node in _walk(schema):
        # dict[str, Any] => additionalProperties: {...}; extra=forbid => additionalProperties: false
        assert "additionalProperties" not in node, f"{name}: additionalProperties found"


def test_llm_root_schema_has_no_open_dicts():
    # The ROOT has `meta.pages_used: dict`, so only the LLM-facing root may go to Gemini.
    for node in _walk(TenderExtractionLLM.model_json_schema()):
        assert "additionalProperties" not in node

    assert "meta" not in TenderExtractionLLM.model_fields
    assert "meta" in TenderExtraction.model_fields


# ---------- v1.1: fields that came from the audit ----------


def test_closed_sets_never_hold_a_value_outside_the_set():
    from app.schemas.extraction import AssignmentType, Threshold

    # An unknown value is coerced (other / default), never stored as is.
    assert Threshold(label="turnover", aggregation="sometimes").aggregation is None

    doc = TenderExtraction.model_validate(
        {"identity": {"assignment_type": "random_thing"}}
    )
    assert doc.identity.assignment_type == AssignmentType.OTHER


def test_threshold_kind_defaults_to_other():
    from app.schemas.extraction import Threshold, ThresholdKind

    assert Threshold(label="x").kind == ThresholdKind.OTHER


def test_security_can_express_minimum_quoted_pbg_with_msme_exemption():
    sec = Security(
        status=SecurityStatus.REQUIRED,
        percent=5,
        percent_basis="higher of 5% of contract price and the quoted PBG",
        is_minimum=True,
        bidder_quoted=True,
        exemptions=["MSME"],
    )

    assert sec.is_minimum and sec.bidder_quoted and sec.exemptions == ["MSME"]


def test_personnel_defaults_and_totals():
    from app.schemas.extraction import KeyPersonnel, PersonnelCategory

    person = KeyPersonnel(role="Team Leader")

    assert person.category == PersonnelCategory.KEY
    assert person.certifications == []
    assert TenderExtraction().key_personnel.key_person_months_total is None


def test_packages_limit_and_corrigendum_kind():
    from app.schemas.extraction import DocumentKind, PackageItem

    too_many = [PackageItem(package_id=str(i)).model_dump() for i in range(21)]

    with pytest.raises(ValidationError):
        TenderExtraction.model_validate({"project": {"packages": too_many}})

    doc = TenderExtraction.model_validate({"document_kind": "corrigendum"})
    assert doc.document_kind == DocumentKind.CORRIGENDUM


# ---------- v1.2 ----------


def test_amount_keeps_net_and_with_tax_separately():
    fee = Amount(
        value=5000, value_with_tax=5900, raw_text="Rs.5,000 + 18% GST (Rs. 5,900)"
    )

    assert fee.value == 5000
    assert fee.value_with_tax == 5900
    assert Amount().value is None and Amount().value_with_tax is None


def test_money_is_currency_neutral_and_document_has_a_currency():
    doc = TenderExtraction.model_validate(
        {
            "identity": {"currency": "USD"},
            "project": {"estimated_contract_value": {"value": 325000}},
        }
    )

    assert doc.identity.currency == "USD"
    assert doc.project.estimated_contract_value.value == 325000

    # the old INR-only names must be gone
    dumped = json.dumps(TenderExtraction().model_dump())
    assert "_inr" not in dumped


def test_ppp_family_exists():
    doc = TenderExtraction.model_validate({"family": "ppp_partnership"})

    assert doc.family.value == "ppp_partnership"


def test_null_object_or_list_means_default_not_error():
    # Real failure: the model wrote null for project.project_cost / estimated_contract_value.
    doc = TenderExtractionLLM.model_validate_json(
        '{"project": {"estimated_contract_value": null, "project_cost": null, "states": null},'
        ' "securities": {"bid_security": null}, "identity": {"contact": null, "title": null}}'
    )

    assert doc.project.project_cost.value is None
    assert doc.project.estimated_contract_value.raw_text is None
    assert doc.project.states == []
    assert doc.securities.bid_security.status.value == "not_stated"
    assert doc.identity.contact.email is None and doc.identity.title is None


def test_amount_accepts_a_plain_number_or_text():
    doc = TenderExtractionLLM.model_validate(
        {"project": {"estimated_contract_value": 325000, "project_cost": "Rs 14.86 Cr"}}
    )

    assert doc.project.estimated_contract_value.value == 325000
    assert doc.project.project_cost.raw_text == "Rs 14.86 Cr"
    assert doc.project.project_cost.value is None


def test_null_tolerance_does_not_hide_real_errors():
    with pytest.raises(ValidationError):
        TenderExtractionLLM.model_validate(
            {"securities": {"bid_security": {"percent": 150}}}
        )

    with pytest.raises(ValidationError):
        TenderExtractionLLM.model_validate({"dates": {"validity_days": -1}})


def test_nil_security_can_still_say_declaration_is_permitted():
    sec = Security(status=SecurityStatus.NIL, amount=0, declaration_permitted=True)

    assert sec.status == SecurityStatus.NIL and sec.declaration_permitted is True


def test_cost_plus_fixed_fee_contract_form():
    doc = TenderExtraction.model_validate(
        {"identity": {"contract_form": "cost_plus_fixed_fee"}}
    )

    assert doc.identity.contract_form.value == "cost_plus_fixed_fee"


def test_rated_and_pass_fail_criteria():
    scored = Criterion(name="Qualifications & Experience", max_points=20, weight=20)
    gate = Criterion(name="Completeness of Response", is_pass_fail=True)

    assert scored.weight == 20 and scored.is_pass_fail is None
    assert gate.is_pass_fail is True and gate.max_points is None

    doc = TenderExtraction.model_validate(
        {
            "evaluation": {
                "rating_scale_max": 5,
                "total_points": 100,
                "bid_parameter": "share %",
            }
        }
    )
    assert doc.evaluation.rating_scale_max == 5


def test_delay_damages_can_hold_two_different_clauses():
    doc = TenderExtraction.model_validate(
        {
            "contract": {
                "delay_damages": [
                    {
                        "kind": "liquidated_damages",
                        "rate_text": "1/2% of monthly bill per week",
                    },
                    {"kind": "penalty", "rate_text": "0.05% per day", "max_pct": 5},
                ]
            }
        }
    )

    kinds = [d.kind.value for d in doc.contract.delay_damages]
    assert kinds == ["liquidated_damages", "penalty"]
    assert isinstance(doc.contract.delay_damages[0], DelayDamage)

    with pytest.raises(ValidationError):
        DelayDamage(max_pct=150)


def test_unknown_enum_value_becomes_other_or_default_not_error():
    # Real failure: identity.contract_form got a value that is not in the closed set.
    doc = TenderExtractionLLM.model_validate(
        {
            "identity": {
                "contract_form": "time_based_with_lump_sum",
                "selection_method": "QBS",
            },
            "securities": {"bid_security": {"status": "maybe"}},
            "key_personnel": {"roles": [{"role": "TL", "category": "Key Personnel"}]},
            "family": "something odd",
        }
    )

    assert doc.identity.contract_form.value == "other"  # set has "other"
    assert doc.identity.selection_method.value == "qbs"  # case / spacing fixed
    assert (
        doc.securities.bid_security.status.value == "not_stated"
    )  # no "other": default
    assert doc.key_personnel.roles[0].category.value == "key"  # default
    assert doc.family.value == "other"


def test_valid_enum_values_are_untouched():
    doc = TenderExtractionLLM.model_validate(
        {
            "identity": {"contract_form": "cost_plus_fixed_fee"},
            "securities": {"bid_security": {"status": "nil"}},
        }
    )

    assert doc.identity.contract_form.value == "cost_plus_fixed_fee"
    assert doc.securities.bid_security.status.value == "nil"
