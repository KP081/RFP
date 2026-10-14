import json

import pytest
from pydantic import ValidationError

from app.schemas.extraction import (
    SECTION_MODELS,
    Dates,
    Security,
    SecurityStatus,
    TenderExtraction,
    is_iso_date,
    normalize_date,
)


def test_empty_extraction_is_valid():
    # Koi RFP kuch nahi deta to bhi object valid hona chahiye (sab "not found").
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
    # model_validate se banao (attribute assign karne pe validators nahi chalte).
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
    # Recursive schema = $ref jo khud ko hi point kare. Gemini ise aksar reject karta hai.
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})

    def refs_in(node):
        return {
            n["$ref"].split("/")[-1] for n in _walk(node) if isinstance(n, dict) and "$ref" in n
        }

    for def_name, definition in defs.items():
        assert def_name not in refs_in(definition), f"{name}: {def_name} is recursive"


@pytest.mark.parametrize("name, model", list(SECTION_MODELS.items()))
def test_section_schema_has_no_open_dicts_or_forbid(name, model):
    schema = model.model_json_schema()

    for node in _walk(schema):
        # dict[str, Any] => additionalProperties: {...}; extra=forbid => additionalProperties: false
        assert "additionalProperties" not in node, f"{name}: additionalProperties found"


# ---------- v1.1: audit se aaye fields ----------


def test_closed_sets_reject_unknown_values():
    from app.schemas.extraction import Threshold

    with pytest.raises(ValidationError):
        Threshold(label="turnover", aggregation="sometimes")

    with pytest.raises(ValidationError):
        TenderExtraction.model_validate({"identity": {"assignment_type": "random_thing"}})


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