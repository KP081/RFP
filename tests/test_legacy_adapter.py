"""Offline tests for app/services/legacy_adapter.py (no API calls).

Fixtures: the four saved v1.2 outputs in tests/fixtures/extractions/ (copied from data/extractions/).
"""

import json
import re
from pathlib import Path

import pytest

from app.services.legacy_adapter import LEGACY_KEYS, _short_title, to_legacy

FIXTURES = Path(__file__).parent / "fixtures" / "extractions"


def _key(text: str) -> str:
    """'FINAL RFP Lone Pine ...' and 'FINAL_RFP_Lone_Pine ...' must match: compare letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def load(name_part: str) -> dict:
    wanted = _key(name_part)
    path = next((p for p in sorted(FIXTURES.glob("*.json")) if wanted in _key(p.name)), None)
    assert path is not None, f"no fixture matching '{name_part}' in {FIXTURES}"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bhopal():
    return to_legacy(load("a9378dcd"))


@pytest.fixture(scope="module")
def siliguri():
    return to_legacy(load("rfp_4_0_1"))


@pytest.fixture(scope="module")
def lone_pine():
    return to_legacy(load("Lone_Pine"))


@pytest.fixture(scope="module")
def gandhidham():
    return to_legacy(load("Gandhidham"))


def elig(result: dict) -> dict:
    return result["EligibilityCriteriaJson"]


# ---- copy of the other app's `_parse_currency_to_crore` (router), to prove our money strings parse ----
_RUPEES_PER_CRORE = 10_000_000


def router_parse_currency_to_crore(value):
    if value is None:
        return None
    s = str(value).strip()
    m = re.search(r"([\d,]+(?:\.\d+)?)\s*(?:[Cc]r(?:ore)?|CR)\b", s)
    if m:
        return float(m.group(1).replace(",", ""))
    cleaned = re.sub(r"₹|Rs\.?|\s", "", s).replace(",", "")
    try:
        return float(cleaned) / _RUPEES_PER_CRORE
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------- shape


@pytest.mark.parametrize("name", ["a9378dcd", "rfp_4_0_1", "Lone_Pine", "Gandhidham"])
def test_only_known_keys_and_json_safe(name):
    result = to_legacy(load(name))

    assert set(result) <= set(LEGACY_KEYS)
    json.dumps(result)  # no objects that cannot be serialised


def test_empty_extraction_does_not_crash_and_stays_null():
    result = to_legacy({})

    assert result["RfpNumber"] is None and result["EstimatedValue"] is None
    assert result["RequiredKeyPersonnel"] is None
    assert elig(result)["eligibility_criteria_amount"] is None  # the key is always there
    assert "eligibility_criteria_amount" in elig(result)
    assert result["extraction_confidence"] == "low"


def test_accepts_a_pydantic_like_object():
    class Fake:
        def model_dump(self, mode="json"):
            return {"identity": {"rfp_number": "A/1/2025"}}

    assert to_legacy(Fake())["RfpNumber"] == "A/1/2025"


# ---------------------------------------------------------------- identity and dates


def test_reference_numbers_are_cleaned(bhopal, siliguri):
    assert bhopal["RfpNumber"] == "RFP/Bhopal-Kota/2025-26/efile-299153"
    assert bhopal["TenderCode"] == bhopal["RfpNumber"]  # the full reference, not only 'efile-299153'
    assert siliguri["RfpNumber"] == "NHIDCL/DPR/Siliguri-Guwahati/2025-26"  # trailing '/' gone


def test_nit_number_is_never_filled_from_the_rfp_number(bhopal):
    assert bhopal["NitNumber"] is None


def test_dates_are_dd_mm_yyyy_and_placeholders_become_null(bhopal, gandhidham):
    assert bhopal["BidSubmissionDate"] == bhopal["Deadline"] == "25-03-2026"
    assert bhopal["TechnicalBidOpeningDate"] == "27-03-2026"
    assert gandhidham["BidSubmissionDate"] is None and gandhidham["TechnicalBidOpeningDate"] is None


def test_unparseable_date_text_is_null():
    result = to_legacy({"dates": {"submission_end": "To be notified"}})

    assert result["BidSubmissionDate"] is None


def test_pre_bid_fields(siliguri, bhopal):
    assert siliguri["PreBidMeetingDate"] == "19-06-2025" and siliguri["PreBidMeetingPlace"] == "NHIDCL-HQ"
    assert bhopal["PreBidMeetingRequired"] is True and bhopal["PreBidMeetingDate"] is None


def test_status_is_never_inferred(bhopal, lone_pine):
    assert bhopal["Status"] is None and lone_pine["Status"] is None


def test_portal_must_be_a_url(gandhidham, bhopal):
    assert gandhidham["EPortalUrl"] is None  # "nProcure"
    assert bhopal["EPortalUrl"].startswith("https://")


def test_classification(bhopal, gandhidham):
    assert bhopal["ConsultancyType"] == "DPR" and bhopal["ProjectSector"] == "Road/Highway"
    assert gandhidham["ConsultancyType"] == "Other" and gandhidham["ProjectSector"] == "Other"


def test_state_route_length(bhopal, siliguri):
    assert bhopal["State"] == "Madhya Pradesh, Rajasthan" and bhopal["LengthKm"] == 340.0
    assert siliguri["Route"] == "Siliguri – Guwahati – Jorabat"


@pytest.mark.parametrize(
    "full, expected",
    [
        (
            "CONSULTANCY SERVICES FOR PREPARATION OF DETAILED PROJECT REPORT (DPR) FOR 4-LANING OF THE "
            "BHOPAL- KOTA (TOTAL LENGTH (APPROX): 340 KM) IN THE STATE OF MADHYA PRADESH AND RAJASTHAN",
            "Preparation of Detailed Project Report (DPR) for 4-Laning of the Bhopal-Kota",
        ),
        (
            "Request for Proposal (RFP) For Selection of an Anchor Industry Partner (AIP) for Upgradation of "
            "ITI Clusters Gandhidham under the PM-SETU in the State of Gujarat",
            "Anchor Industry Partner (AIP) for Upgradation of ITI Clusters Gandhidham under the PM-SETU",
        ),
        ("Lone Pine Sidewalk Construction and ADA Improvement Project", "Lone Pine Sidewalk Construction and ADA Improvement Project"),
        (None, None),
    ],
)
def test_short_title_is_a_readable_stop_gap(full, expected):
    assert _short_title(full) == expected


def test_full_title_is_kept_verbatim(bhopal):
    assert bhopal["TenderTitleFull"].startswith("CONSULTANCY SERVICES FOR PREPARATION")


# ---------------------------------------------------------------- money


def test_money_strings_survive_the_routers_parser(gandhidham, siliguri, lone_pine):
    assert router_parse_currency_to_crore(gandhidham["EstimatedValue"]) == pytest.approx(241.0)
    assert router_parse_currency_to_crore(gandhidham["BidSecurityAmount"]) == pytest.approx(7.23)
    assert router_parse_currency_to_crore(siliguri["BidSecurityAmount"]) == pytest.approx(0.02)
    assert router_parse_currency_to_crore(lone_pine["EstimatedValue"]) is None  # USD: the router must not treat it as rupees


def test_other_currency_is_not_converted(lone_pine):
    assert lone_pine["EstimatedValue"] == "USD 325,000"
    assert elig(lone_pine)["eligibility_criteria_amount"] is None


def test_estimated_value_is_net_of_tax_and_the_other_figure_is_flagged():
    result = to_legacy(
        {
            "identity": {"currency": "INR"},
            "project": {"estimated_contract_value": {"value": 98644500, "value_with_tax": 116400510}},
        }
    )

    assert result["EstimatedValue"] == "₹ 9,86,44,500"
    assert any("11,64,00,510" in flag for flag in result["ExtractionReviewFlags"])


def test_estimated_value_falls_back_to_project_cost_and_says_so(gandhidham):
    assert gandhidham["EstimatedValue"] == "₹ 2,41,00,00,000"
    assert any("project cost" in flag for flag in gandhidham["ExtractionReviewFlags"])


def test_percent_of_project_cost_rule_prefers_the_project_cost():
    result = to_legacy(
        {
            "identity": {"currency": "INR"},
            "project": {
                "estimated_contract_value": {"value": 1_000_000},
                "project_cost": {"value": 2_000_000_000},
            },
            "qualification": {"financial": [{"kind": "turnover", "basis": "pct_of_project_cost", "value": 3}]},
        }
    )

    assert router_parse_currency_to_crore(result["EstimatedValue"]) == pytest.approx(200.0)


def test_bid_security_amount_or_percent_or_nothing(bhopal):
    assert bhopal["BidSecurityAmount"] is None  # status nil, no amount
    result = to_legacy({"securities": {"bid_security": {"percent": 3, "percent_basis": "estimated cost"}}})
    assert result["BidSecurityAmount"] == "3% of estimated cost"


# ---------------------------------------------------------------- what the router reads in EligibilityCriteriaJson


def test_criteria_amount_is_a_bare_crore_number(bhopal, siliguri, gandhidham):
    assert elig(bhopal)["eligibility_criteria_amount"] == 10.0
    assert elig(siliguri)["eligibility_criteria_amount"] == 5.0
    assert elig(gandhidham)["eligibility_criteria_amount"] == 500.0
    assert "Rs.10.00 Crores" in elig(bhopal)["eligibility_criteria_condition"]
    assert elig(bhopal)["financial_capacity"]["percentage_requirement_type"] == "turnover"
    assert elig(bhopal)["financial_capacity"]["min_pct_of_estimated_cost"] is None  # absolute rule: no percent


def test_percentage_rule_sets_the_percent_and_leaves_the_amount_to_the_router():
    result = to_legacy(
        {
            "identity": {"currency": "INR"},
            "qualification": {
                "financial": [
                    {
                        "kind": "turnover",
                        "label": "average annual turnover",
                        "basis": "pct_of_estimated_cost",
                        "value": 150,
                        "period_years": 3,
                        "aggregation": "average",
                        "raw_text": "not less than 150% of estimated project cost",
                    }
                ]
            },
        }
    )
    fin = elig(result)["financial_capacity"]

    assert elig(result)["eligibility_criteria_amount"] is None
    assert fin["min_pct_of_estimated_cost"] == 150 and fin["percentage_requirement_type"] == "turnover"
    assert fin["min_annual_turnover_crore"] is None


def test_other_financial_thresholds_are_kept_not_lost(gandhidham):
    others = elig(gandhidham)["financial_capacity"]["other_thresholds"]

    assert {t["kind"] for t in others} == {"net_worth", "staff_strength"}
    assert next(t for t in others if t["kind"] == "staff_strength")["value"] == 500


def test_jv_keys_use_the_names_the_router_scans_for(bhopal, siliguri, gandhidham):
    jv = elig(bhopal)["jv_conditions"]
    assert jv["max_jv_members_total"] == 2 and jv["max_jv_partner_other_than_lead"] == 1
    assert jv["lead_member_qualification_pct"] == 60.0 and jv["substantial_member_qualification_pct"] == 40.0

    jv = elig(siliguri)["jv_conditions"]
    assert jv["lead_member_qualification_pct"] == 75.0 and jv["substantial_member_qualification_pct"] == 50.0

    # the other app's prompt invented 50 / 25 when the RFP is silent; here a silent RFP stays null
    jv = elig(gandhidham)["jv_conditions"]
    assert jv["allowed"] is True and jv["max_jv_members_total"] == 5
    assert jv["lead_member_qualification_pct"] is None and jv["substantial_member_qualification_pct"] is None


def test_no_jv_information_means_all_null(lone_pine):
    jv = elig(lone_pine)["jv_conditions"]

    assert jv["allowed"] is None and jv["max_jv_members_total"] is None and jv["max_jv_partner_other_than_lead"] is None


def test_technical_hard_count_only_for_a_pure_count_rule(bhopal, lone_pine, siliguri):
    assert "eligible_assignment_count" not in elig(bhopal)["technical_capacity"]  # length rule, not a count
    assert "eligible_assignment_count" not in elig(siliguri)["technical_capacity"]
    assert elig(lone_pine)["technical_capacity"]["eligible_assignment_count"] == 3


def test_compound_length_rules_are_not_collapsed_into_one_km_number(bhopal):
    tech = elig(bhopal)["technical_capacity"]

    assert "similar_assignment_min_length_km" not in tech  # aggregate AND single AND adjustment factors
    assert tech["minimum_aggregate_length_km"] == 340.0 and tech["minimum_single_project_length_km"] == 136.0


def test_one_clear_length_rule_gives_the_km_number_and_a_flag():
    result = to_legacy(
        {
            "project": {"length_km": 164.355},
            "qualification": {"experience": [{"category": "NH/Expressway IE/AE", "min_single_length_pct": 40}]},
        }
    )
    tech = elig(result)["technical_capacity"]

    assert tech["similar_assignment_min_length_km"] == pytest.approx(65.742)
    assert tech["similar_assignment_category"] == "NH/Expressway IE/AE"
    assert any("65.742" in flag for flag in result["ExtractionReviewFlags"])


def test_scoring_rules_keep_points_bands_and_pass_fail(gandhidham, lone_pine):
    rules = elig(gandhidham)["technical_capacity"]["scoring_rules"]
    assert rules["e_hr_plan"]["max_points"] == 15.0
    assert rules["e_hr_plan"]["sub_criteria"]["proposed_spv_ceo_managers"]["max"] == 5.0
    assert elig(gandhidham)["technical_capacity"]["min_qualifying_marks"] == 50.0

    gates = elig(lone_pine)["technical_capacity"]["scoring_rules"]
    assert gates["completeness_of_response"]["is_pass_fail"] is True


def test_scoring_rule_keys_are_unique():
    result = to_legacy({"evaluation": {"criteria": [{"name": "Experience"}, {"name": "Experience"}]}})

    assert list(elig(result)["technical_capacity"]["scoring_rules"]) == ["experience", "experience_2"]


def test_subcontracting_and_other_blocks(bhopal):
    assert elig(bhopal)["sub_contracting"]["max_value_pct"] == 10.0
    assert "INFRACON registration" in elig(bhopal)["registration_requirements"]
    assert elig(bhopal)["residual_bid_capacity"]["required"] is True
    assert any(c.startswith("Biometric Attendance") for c in elig(bhopal)["other_conditions"])


# ---------------------------------------------------------------- personnel


def test_required_and_conditional_roles_are_split(bhopal, siliguri):
    assert len(bhopal["RequiredKeyPersonnel"]) == 10 and len(bhopal["ConditionalKeyPersonnel"]) == 4
    assert siliguri["ConditionalKeyPersonnel"] is None
    assert not any(r["role"] in {c["role"] for c in bhopal["ConditionalKeyPersonnel"]} for r in bhopal["RequiredKeyPersonnel"])


def test_role_fields_and_the_catch_all(bhopal):
    team_leader = bhopal["RequiredKeyPersonnel"][0]

    assert team_leader["role"].startswith("Team Leader") and team_leader["count"] == 1
    assert team_leader["experience_years"] is None  # marking bands are not a minimum (v1.2 trap)
    assert team_leader["cost_budget"] == "6000000"  # 12 months x 5,00,000: digits only, same figure as the other app
    assert team_leader["eligibility"]["marking_points"] == 14.0 and team_leader["eligibility"]["person_months"] == 12.0


def test_hard_minimum_experience_is_kept(siliguri):
    assert siliguri["RequiredKeyPersonnel"][0]["experience_years"] == 12.0


def test_conditional_role_keeps_its_condition_and_rate(bhopal):
    role = next(r for r in bhopal["ConditionalKeyPersonnel"] if "Geophysicist" in r["role"])

    assert "tunnels" in role["eligibility"]["condition"] and role["eligibility"]["rate_per_month"] == 350000.0
    assert role["cost_budget"] is None  # no person-months stated, so no total is invented


def test_cost_budget_is_the_documents_amount_else_rate_times_months_and_it_is_flagged():
    own = to_legacy({"key_personnel": {"roles": [{"role": "A", "cost": 6000000, "rate_per_month": 1, "person_months": 1}]}})
    derived = to_legacy({"key_personnel": {"roles": [{"role": "B", "rate_per_month": 500000, "person_months": 7.5}]}})

    assert own["RequiredKeyPersonnel"][0]["cost_budget"] == "6000000"
    assert not any("cost_budget" in f for f in own["ExtractionReviewFlags"])
    assert derived["RequiredKeyPersonnel"][0]["cost_budget"] == "3750000"
    assert any("cost_budget = rate_per_month x person_months for 1 role" in f for f in derived["ExtractionReviewFlags"])


def test_every_role_value_fits_the_columns_save_role_details_writes(bhopal):
    # HrTenderRole.CostBudget may be numeric, and HrTenderRoleEligibility stores str(value) per key
    for role in bhopal["RequiredKeyPersonnel"]:
        assert role["cost_budget"] is None or role["cost_budget"].replace(".", "", 1).isdigit()
        assert isinstance(role["role"], str) and role["role"]
        assert all(isinstance(k, str) for k in (role["eligibility"] or {}))


def test_support_staff_are_not_required_key_personnel():
    result = to_legacy(
        {
            "key_personnel": {
                "roles": [
                    {"role": "Team Leader", "category": "key"},
                    {"role": "Typist", "category": "support"},
                ]
            }
        }
    )

    assert [r["role"] for r in result["RequiredKeyPersonnel"]] == ["Team Leader"]
    assert any("1 sub-professional/support" in flag for flag in result["ExtractionReviewFlags"])


def test_personnel_summary_format(siliguri):
    assert siliguri["PersonnelSummary"].startswith("Team Leader cum Sr. Highway Engineer (×1) Min 12 yrs [Site and Design Office]")


# ---------------------------------------------------------------- confidence and missing fields


def test_missing_fields_list_the_null_scored_keys(bhopal, siliguri):
    assert "EstimatedValue" in bhopal["missing_fields"] and "RfpNumber" not in bhopal["missing_fields"]
    assert siliguri["missing_fields"] == ["EstimatedValue"]
    assert siliguri["extraction_confidence"] == "high"


def test_needs_review_caps_the_confidence():
    doc = load("rfp_4_0_1")
    doc["meta"]["needs_review"] = True

    assert to_legacy(doc)["extraction_confidence"] == "medium"
    assert to_legacy(doc)["ExtractionReviewFlags"]


def test_extraction_source_is_recorded(bhopal):
    assert bhopal["ExtractionSource"]["engine"] == "rfp-rag" and bhopal["ExtractionSource"]["schema_version"] == "1.2"