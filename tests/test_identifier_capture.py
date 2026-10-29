import pytest

from app.services.document_parser import DocumentPage
from app.services.identifier_capture import (
    capture_identifiers,
    normalise,
    reconcile,
    squash,
)


def capture(*texts: str):
    return capture_identifiers(
        [DocumentPage(page_number=i + 1, text=t) for i, t in enumerate(texts)]
    )


def best(text: str, kind: str = "rfp") -> str | None:
    return capture(text).value(kind)


# ---------- the real lines from the local RFPs (the three reference files + Tele MANAS) ----------


def test_bhopal_label_line():
    assert (
        best("RFP Document No.: RFP/ Bhopal-Kota/2025-26/efile-299153")
        == "RFP/Bhopal-Kota/2025-26/efile-299153"
    )


def test_bhopal_table_row_with_wrapped_spaces_and_a_second_label_cell():
    page = (
        "[TABLE]\n"
        "Tender Reference Number | RFP/ Bhopal- Kota/2025- 26/efile-299153 | Nature of assignment: <Normal Highway/Standalone Bridge>\n"
        "Tender Type | International Competitive Bidding | Tender Category | Consultancy Services\n"
        "[/TABLE]"
    )

    assert best(page) == "RFP/Bhopal-Kota/2025-26/efile-299153"


def test_bhopal_sentence_where_the_number_is_wrapped_over_two_lines():
    page = (
        "in TIS. This Request for Proposal (RFP) Document, reference number, RFP/ Bhopal-\n"
        "Kota/2025-26/efile-299153 (hereinafter referred to as ‘the RFP Document’), gives further"
    )

    assert best(page) == "RFP/Bhopal-Kota/2025-26/efile-299153"


def test_gandhidham():
    assert (
        best(
            "ITI Clusters Gandhidham\nRFP Ref. No.: 03/2026\nDate of Issue: As per bid detail"
        )
        == "03/2026"
    )


def test_tele_manas_number_buried_in_a_long_cell():
    page = (
        "[TABLE]\n"
        "RFP Number | Software Development and Platform Support Services for National Tele-Mental Health "
        "Initiative, IIITB/EHRC/2022/IT- 01, 25-Oct-2022\n"
        "Department Name | E-Health Research Centre, IIITB\n"
        "[/TABLE]"
    )

    assert best(page) == "IIITB/EHRC/2022/IT-01"


# ---------- label forms seen in public RFPs / NITs ----------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("RFP NO. TNT/002/2022 - 2023", "TNT/002/2022-2023"),
        ("RFP No.:  [PTC/T-220/2024]", "PTC/T-220/2024"),
        ("RFP No.: IN-BRLPS-275613-CS-LCS", "IN-BRLPS-275613-CS-LCS"),
        (
            "RFP No: Consult-RFP-Sahid Park/01/2075/076",
            "Consult-RFP-Sahid Park/01/2075/076",
        ),
        ("RFP  No. :- 01/074/75", "01/074/75"),
        ("RFP Ref No: RFP-001-CRL-2022", "RFP-001-CRL-2022"),
        ("REQUEST FOR PROPOSAL GHTD RFP #10-019", "10-019"),
        ("“RFP No. M/Dind/MSME/11/2024/3”", "M/Dind/MSME/11/2024/3"),
        (
            "Tender Reference No.: NH-12037/10/2011/Border infrastructure/SARDP-NE   Dated 29th January, 2019",
            "NH-12037/10/2011/Border infrastructure/SARDP-NE",
        ),
        (
            "Bid/ Package no.: NHIDCL/DPR/J&K/Mughal Road/ (Pir Ki Gali-Tunnel)/2017",
            "NHIDCL/DPR/J&K/Mughal Road/(Pir Ki Gali-Tunnel)/2017",
        ),
        (
            "No. NHIDCL/Mizoram/DPR/Lunglei bypass/2022-23                  Dated: 15.05.2023",
            "NHIDCL/Mizoram/DPR/Lunglei bypass/2022-23",
        ),
        (
            "F. No. RO/UK/DPR/ALTERNATE HIGHWAY/KUND-SONPRAYAG/2023-24 DATED- 22.03.2023",
            "RO/UK/DPR/ALTERNATE HIGHWAY/KUND-SONPRAYAG/2023-24",
        ),
        (
            "proposal reference number\n04/2015-16 (Tender-PR Agency) Dated : 31.12.2015",
            "04/2015-16",
        ),
    ],
)
def test_label_forms(text, expected):
    assert best(text) == expected


def test_nit_is_kept_apart_from_the_rfp_number():
    result = capture(
        "NIT No. 10/NHAI/RO-BNG/24016/Tender/2018                Dated:18.06.2018\nRFP No.: A/1/2025"
    )

    assert result.value("nit") == "10/NHAI/RO-BNG/24016/Tender/2018"
    assert result.value("rfp") == "A/1/2025"


def test_an_rfp_number_is_never_copied_into_nit():
    assert capture("RFP Ref. No.: 03/2026").value("nit") is None


def test_gem_rfp_number_and_portal_bid_id_are_two_things():
    text = "RFP / Bid Number: GEM/2024/BOB/CAT/01 Date: 23rd September, 2024 (Gem Portal Bid ID: GEM/2024/B/5429752)"

    result = capture(text)

    assert result.value("rfp") == "GEM/2024/BOB/CAT/01"
    assert result.value("tender_code") == "GEM/2024/B/5429752"


def test_value_on_the_next_line_or_cell():
    assert best("RFP Ref. No.\n03/2026") == "03/2026"
    assert best(
        "[TABLE]\nRFP No. | NHIDCL/DPR/Siliguri-Guwahati/2025-26\nDated | 04-06-2025\n[/TABLE]"
    ) == ("NHIDCL/DPR/Siliguri-Guwahati/2025-26")


def test_text_printed_twice_by_the_pdf_is_collapsed():
    assert (
        best("Bid Number: GGEEMM//22002244//BB//55447777333344") == "GEM/2024/B/5477334"
    )


# ---------- things that must NOT be captured ----------


@pytest.mark.parametrize(
    "text",
    [
        "RFP Ref. No.: [insert RFP identification number]",
        "RFP NUMBER:\n[PURCHASER to enter]",
        "Proposal No: ……………………",
        "RFP No.: XXXX",
        "[TABLE]\nSr. No. | Description | Qty\n1 | Laptop | 5\n[/TABLE]",
        "Page No. 5 of the annexure",
        "No. of persons: 5",
        "Date of Issue: As per bid detail",
        "Bid No. 12",
        "the bid number of the bidders is shown in Annexure 3",
        "Reference No. dated 25-10-2022",
        "RFP No.: 25-Oct-2022",
        "RFP No.: 04/06/2025",
        "Tender ID: see the portal",
    ],
)
def test_not_a_number(text):
    result = capture(text)

    assert result.candidates == [], [c.value for c in result.candidates]


def test_a_bare_no_needs_a_strong_identifier_shape():
    assert best("No. 12/2025") is None  # too weak: one slash
    assert best("No. NHIDCL/Arunachal/DPR/Inter-corridor/2023   Dated: 23.01.2023") == (
        "NHIDCL/Arunachal/DPR/Inter-corridor/2023"
    )


# ---------- choosing between several ----------


def test_the_number_that_repeats_in_the_document_wins():
    page1 = "RFP No.: AAA/1/2025\nsome text"
    body = "\n".join(
        "As per RFP document BBB/2/2025 the bidder shall comply" for _ in range(4)
    )
    page2 = "Tender No.: BBB/2/2025\n" + body

    assert (
        capture(page1, page2).value("rfp") == "AAA/1/2025"
    )  # an explicit RFP label beats a tender label

    page3 = "Tender No.: AAA/1/2025\nsome text"
    assert (
        capture(page3, page2).value("rfp") == "BBB/2/2025"
    )  # same strength: repeats more, wins


def test_occurrences_are_counted_even_when_the_number_is_wrapped_or_spaced():
    result = capture(
        "RFP Document No.: RFP/ Bhopal-Kota/2025-26/efile-299153",
        "again RFP/ Bhopal-\nKota/2025-26/efile-299153 here and RFP/Bhopal- Kota/2025- 26/efile-299153",
    )

    assert result.best["rfp"].occurrences == 3


# ---------- reconcile with the model's answer ----------


def test_reconcile_fills_a_missing_value():
    cap = capture("RFP Ref. No.: 03/2026")

    value, note = reconcile(None, cap, "rfp")

    assert value == "03/2026" and "not returned" in note


def test_reconcile_keeps_a_matching_value_even_if_spaces_differ():
    cap = capture("RFP Document No.: RFP/ Bhopal-Kota/2025-26/efile-299153")

    assert reconcile("RFP/ Bhopal-Kota/2025-26/efile-299153", cap, "rfp") == (
        "RFP/ Bhopal-Kota/2025-26/efile-299153",
        None,
    )


def test_reconcile_reduces_a_title_around_the_number():
    cap = capture(
        "[TABLE]\nRFP Number | Software Development Services, IIITB/EHRC/2022/IT- 01, 25-Oct-2022\n[/TABLE]"
    )
    model = "Software Development Services, IIITB/EHRC/2022/IT-01, 25-Oct-2022"

    value, note = reconcile(model, cap, "rfp")

    assert value == "IIITB/EHRC/2022/IT-01" and "reduced" in note


def test_reconcile_replaces_a_value_that_is_not_in_the_document():
    cap = capture("RFP Ref. No.: 03/2026")

    value, note = reconcile("04/2027", cap, "rfp")

    assert value == "03/2026" and "not in the document" in note


def test_reconcile_keeps_a_grounded_but_different_value_and_flags_it():
    cap = capture("RFP Ref. No.: 03/2026\nPackage code NHAI/PKG/12/2026 is attached")

    value, note = reconcile("NHAI/PKG/12/2026", cap, "rfp")

    assert value == "NHAI/PKG/12/2026" and "differs" in note


def test_reconcile_flags_an_ungrounded_value_when_nothing_is_labelled():
    cap = capture("no numbers here")

    value, note = reconcile("XYZ/9/2030", cap, "rfp")

    assert value == "XYZ/9/2030" and "not found" in note
    assert reconcile(None, cap, "rfp") == (None, None)


def test_helpers():
    assert normalise("RFP/ Bhopal- Kota/2025- 26") == "RFP/Bhopal-Kota/2025-26"
    assert squash("Bhopal- Kota") == squash("Bhopal-\nKota") == "bhopal-kota"


def test_letter_header_number_before_dated():
    page = (
        "Letter of Invitation (LOI)\n"
        "NHIDCL/DPR/Siliguri-Guwahati/2025-26/ Dated :\n"
        "Dear Sir,\n"
        "Sub: Consultancy Services for Preparation of Detailed Project Report"
    )
    assert best(page) == "NHIDCL/DPR/Siliguri-Guwahati/2025-26"


def test_letter_header_number_with_dated_on_the_next_line():
    page = "NHIDCL/DPR/Siliguri-Guwahati/2025-26\nDated: 04.06.2025\nDear Sir,"
    assert best(page) == "NHIDCL/DPR/Siliguri-Guwahati/2025-26"


def test_a_labelled_line_is_not_read_twice_by_the_dated_rule():
    result = capture(
        "No. NHIDCL/Mizoram/DPR/Lunglei bypass/2022-23      Dated: 15.05.2023"
    )
    assert [c.value for c in result.candidates] == [
        "NHIDCL/Mizoram/DPR/Lunglei bypass/2022-23"
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Ref 12-3 Dated: 04.06.2025",
        "Sub: Extension of date Dated: 04.06.2025",
        "12/03/2025 Dated: 04.06.2025",
        "Dated: 04.06.2025",
        "CONTRACT # 2024-_ _ _ _ _ _",
        "RFP No.: 2024/XXXX",
        "Contract No. 146.1-Actual",
    ],
)
def test_more_things_that_are_not_a_number(text):
    assert capture(text).candidates == []


@pytest.mark.parametrize(
    "text",
    [
        "to Make in India), Order 2017 (PPP-MII Order), Order No. P-45021/2/2017-BEII",
        "issued vide letter No. P-45021/2/2017(BE-II) dated 04.06.2020 & vide letter No. P-",
        "Letter of Invitation No. X/1/2/2025",
    ],
)
def test_a_bare_no_inside_a_sentence_is_a_citation(text):
    assert capture(text).candidates == []
