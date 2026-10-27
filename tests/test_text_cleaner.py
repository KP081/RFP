from app.services.document_parser import DocumentPage
from app.services.text_cleaner import clean_pages, line_key


def body(page: int) -> list[str]:
    # enough different lines that header/footer lines are clearly at the edges
    return [
        f"Clause {page}.{n}: the consultant shall do task number {page * 10 + n}."
        for n in range(1, 9)
    ]


def pages_of(texts: list[str]) -> list[DocumentPage]:
    return [DocumentPage(page_number=i + 1, text=text) for i, text in enumerate(texts)]


def clean_text(texts: list[str]) -> list[str]:
    return [page.text for page in clean_pages(pages_of(texts)).pages]


# ---------- page stamps and page numbers ----------


def test_page_stamps_are_removed_wherever_they_are_on_the_page():
    texts = []
    for i in range(1, 7):
        lines = body(i)
        stamp = f"Page {i} of 6"
        texts.append(
            "\n".join([stamp, *lines] if i % 2 else [*lines, stamp])
        )  # top OR bottom

    for text in clean_text(texts):
        assert "Page" not in text and "Clause" in text


def test_other_stamp_shapes():
    texts = [
        "\n".join([*body(1), "pg. 4"]),
        "\n".join([*body(2), "- 5 -"]),
        "\n".join([*body(3), "6 of 9"]),
        "\n".join([*body(4), "Page 7/9"]),
    ]

    for text in clean_text(texts):
        assert text.splitlines()[-1].startswith("Clause")


def test_bare_numbers_forming_a_page_sequence_are_removed_even_with_an_offset():
    # printed numbers start at 3 on physical page 1: offset 2 on every page
    texts = ["\n".join([*body(i), str(i + 2)]) for i in range(1, 7)]

    for text in clean_text(texts):
        assert text.splitlines()[-1].startswith("Clause")


def test_a_lone_bare_number_at_the_end_of_a_page_is_content_not_a_page_number():
    # real case: one stray "5" on the last page of a 32-page RFP
    texts = ["\n".join(body(i)) for i in range(1, 7)]
    texts[5] = "\n".join([*body(6), "5"])

    assert clean_text(texts)[5].endswith("5")


def test_the_same_number_on_many_pages_is_not_a_sequence():
    texts = ["\n".join([*body(i), "5"]) for i in range(1, 8)]

    assert all(text.endswith("5") for text in clean_text(texts))


def test_bare_number_in_the_middle_of_a_page_is_kept_even_in_a_sequence():
    texts = []
    for i in range(1, 7):
        lines = body(i)
        texts.append(
            "\n".join([*lines[:4], str(i), *lines[4:], str(i)])
        )  # middle AND last line

    for i, text in enumerate(clean_text(texts), start=1):
        assert f"\n{i}\n" in text  # the middle one stays
        assert text.splitlines()[-1].startswith("Clause")  # the last one went


def test_the_text_of_a_sentence_with_page_in_it_is_kept():
    text = "\n".join([*body(1), "See page 5 of the annexure for details of the scope."])

    assert "See page 5 of the annexure" in clean_text([text])[0]


# ---------- TOC ----------


def test_toc_lines_are_removed_but_blank_fill_in_lines_are_kept():
    text = "\n".join(
        [
            "1. Introduction ........................................ 3",
            "2. Scope of Work . . . . . . . . . . . . . . . . . . 9",
            "Name of bidder: .................................",
            "Signature: ______________ 2026",
            *body(1),
        ]
    )

    result = clean_text([text])[0]

    assert "Introduction" not in result and "Scope of Work" not in result
    assert "Name of bidder" in result and "Signature" in result


# ---------- repeated headers / footers ----------


def test_running_header_is_kept_once_and_digits_do_not_break_the_match():
    texts = []
    for i in range(1, 11):
        texts.append(
            "\n".join(
                [
                    "Request for Proposal for Test Services",
                    *body(i),
                    f"Ref IT-01, 25-Oct-2022 pg. {i}",
                ]
            )
        )

    cleaned = clean_text(texts)

    assert cleaned[0].startswith("Request for Proposal for Test Services")
    assert cleaned[0].endswith("pg. 1")  # first occurrence stays
    for text in cleaned[1:]:
        assert "Request for Proposal" not in text and "pg." not in text
        assert "Clause" in text


def test_a_line_repeated_on_too_few_pages_is_not_a_header():
    texts = [
        "\n".join(["Some note that appears twice", *body(i)])
        if i < 3
        else "\n".join(body(i))
        for i in range(1, 11)
    ]

    cleaned = clean_text(texts)

    assert "Some note" in cleaned[0] and "Some note" in cleaned[1]


def test_the_same_line_in_the_middle_of_pages_is_not_a_header():
    texts = []
    for i in range(1, 11):
        lines = body(i)
        texts.append(
            "\n".join([*lines[:4], "Interim remark repeated in the middle", *lines[4:]])
        )

    assert all("Interim remark" in text for text in clean_text(texts))


def test_numbered_headings_are_not_one_repeated_header():
    # "Annexure 1" ... "Annexure 6" at the top of 6 pages share a SHAPE but are different headings
    texts = [
        "\n".join([f"Annexure {i}: Format of declaration", *body(i)])
        for i in range(1, 7)
    ]

    cleaned = clean_text(texts)

    for i, text in enumerate(cleaned, start=1):
        assert text.startswith(f"Annexure {i}:")


def test_a_sentence_that_mentions_a_page_is_not_a_repeated_header():
    # short pages: every line is an "edge" line, and the wording repeats except for the number
    texts = [
        f"Intro line of page {i}\nSee page {i} of the annexure for the scope details\nEnd line {i}"
        for i in range(1, 8)
    ]

    for text in clean_text(texts):
        assert "See page" in text


def test_short_generic_lines_repeated_at_page_ends_are_kept():
    texts = ["\n".join([*body(i), "Signature:"]) for i in range(1, 11)]

    assert all(text.endswith("Signature:") for text in clean_text(texts))


# ---------- tables are never touched ----------


def test_table_rows_are_never_removed():
    table = (
        "[TABLE]\nSr. No | Item | Marks\n1 | Experience | 30\n2 | Team | 70\n[/TABLE]"
    )
    texts = []
    for i in range(1, 11):
        texts.append(
            "\n".join(
                [
                    "Request for Proposal for Test Services",
                    *body(i),
                    table,
                    f"Page {i} of 10",
                ]
            )
        )

    for text in clean_text(texts):
        assert table in text


# ---------- signatures ----------


def test_digital_signature_block_is_removed_but_prose_about_signatures_is_kept():
    text = "\n".join(
        [
            "Digitally signed by JOHN DOE",
            "Date: 2025.06.04 11:20:01 IST",
            "Reason: Approved",
            *body(1),
            "The bidder must upload a digitally signed modified bid on the portal.",
            "Registration needs a Digital Signature Certificate (DSC).",
        ]
    )

    result = clean_text([text])[0]

    assert "JOHN DOE" not in result and "Reason" not in result
    assert "digitally signed modified bid" in result
    assert "Digital Signature Certificate" in result


def test_stamp_with_the_name_on_its_own_line_is_removed_completely():
    text = "\n".join(
        [
            "Digitally Signed by",
            "RAJ KUMAR GUPTA",
            "Date: 2025.06.04 11:20:01 IST",
            *body(1),
        ]
    )

    result = clean_text([text])[0]

    assert (
        "RAJ KUMAR" not in result
        and "Date:" not in result
        and "Digitally" not in result
    )
    assert result.startswith("Clause 1.1")


def test_a_line_that_starts_like_a_stamp_but_has_no_detail_line_is_prose():
    wrapped = "\n".join(
        [
            *body(1)[:3],
            "digitally signed by the authorised signatory of the bidder and",
            "uploaded on the portal.",
            *body(1)[3:],
        ]
    )
    bare = "\n".join(
        [
            *body(2)[:3],
            "Digitally signed by",
            "the bidder before submission.",
            *body(2)[3:],
        ]
    )

    first, second = clean_text([wrapped, bare])

    assert "digitally signed by the authorised signatory" in first
    assert "Digitally signed by\nthe bidder before submission." in second


# ---------- structure ----------


def test_pages_keep_their_numbers_even_when_empty():
    result = clean_pages(pages_of(["Page 1 of 3", "\n".join(body(2)), ""]))

    assert [p.page_number for p in result.pages] == [1, 2, 3]
    assert result.pages[0].text == ""


def test_cleaning_is_idempotent_and_reports_what_it_removed():
    texts = ["\n".join([*body(i), f"Page {i} of 8"]) for i in range(1, 9)]

    first = clean_pages(pages_of(texts))
    second = clean_pages(first.pages)

    assert [p.text for p in second.pages] == [p.text for p in first.pages]
    assert first.counts() == {"page_stamp": 8}
    assert first.removed["page_stamp"][0] == (1, "Page 1 of 8")


def test_line_key_normalises_only_the_page_marker():
    assert line_key("Page 12 of  368") == "page #"
    assert (
        line_key("IIITB/EHRC/2022/IT-01, 25-Oct-2022 pg. 12")
        == "iiitb/ehrc/2022/it-01, 25-oct-2022 page #"
    )
    assert line_key("Annexure 3") == "annexure 3"
    assert line_key("Page 3 of 18 ATP - Lone Pine") == "page # atp - lone pine"
    assert (
        line_key("RFP_APPOINTMENT of PR agency_2015-16 Page 2")
        == "rfp_appointment of pr agency_2015-16 page #"
    )
    assert (
        line_key("Refer page 5 of the annexure for details")
        == "refer page 5 of the annexure for details"
    )
