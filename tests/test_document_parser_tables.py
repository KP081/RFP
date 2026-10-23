import pytest
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Table,
    TableStyle,
)

from app.services.document_parser import (
    TABLE_END,
    TABLE_START,
    extract_pages,
    extract_pages_from_pdf,
)

GRID = TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)])


@pytest.fixture
def sample_pdf(tmp_path):
    styles = getSampleStyleSheet()
    path = tmp_path / "sample.pdf"

    story = [
        Paragraph("Section 5: Key Personnel deployment", styles["Heading2"]),
        Paragraph(
            "The consultant shall deploy the following experts.", styles["Normal"]
        ),
        Table(
            [
                ["Position", "Nos.", "Man-months"],
                ["Team Leader", "1", "12"],
                ["Bridge Engineer", "2", "7"],
            ],
            style=GRID,
        ),
        Paragraph("Note: replacement needs prior approval.", styles["Normal"]),
        PageBreak(),
        Paragraph("Page two has only plain text, no table at all.", styles["Normal"]),
        PageBreak(),
        Table(
            [
                ["Position", "Qualification"],
                ["Tunnel Expert", "B.E. Civil\nwith 15 years experience"],
            ],
            style=GRID,
        ),
    ]
    SimpleDocTemplate(str(path), pagesize=A4).build(story)
    return path


def test_table_rows_keep_their_columns(sample_pdf):
    pages = extract_pages(sample_pdf, tables=True)
    text = pages[0].text

    assert TABLE_START in text and TABLE_END in text
    assert "Position | Nos. | Man-months" in text
    assert "Team Leader | 1 | 12" in text
    assert "Bridge Engineer | 2 | 7" in text


def test_text_around_the_table_stays_in_reading_order(sample_pdf):
    text = extract_pages(sample_pdf, tables=True)[0].text

    intro = text.index("The consultant shall deploy")
    table = text.index(TABLE_START)
    note = text.index("Note: replacement")

    assert intro < table < note


def test_table_text_is_not_duplicated_as_plain_text(sample_pdf):
    text = extract_pages(sample_pdf, tables=True)[0].text

    assert text.count("Team Leader") == 1


def test_page_without_table_matches_plain_reading(sample_pdf):
    structured = extract_pages(sample_pdf, tables=True)[1].text
    plain = extract_pages_from_pdf(sample_pdf)[1].text

    assert TABLE_START not in structured
    assert " ".join(structured.split()) == " ".join(plain.split())


def test_multiline_cell_becomes_one_line(sample_pdf):
    text = extract_pages(sample_pdf, tables=True)[2].text

    assert "Tunnel Expert | B.E. Civil with 15 years experience" in text


def test_default_extract_pages_is_unchanged(sample_pdf):
    pages = extract_pages(sample_pdf)  # tables=False

    assert len(pages) == 3
    assert TABLE_START not in pages[0].text


def test_widely_spaced_digits_in_a_cell_are_not_split(tmp_path):
    # Real bug (RFP page 355): "12" came out as "1 2" because the digits sit far apart
    # in the PDF and there is no space character between them.
    from reportlab.pdfgen import canvas

    path = tmp_path / "spaced.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    xs, ys = [100, 300, 400], [700, 680, 660]
    for x in xs:
        pdf.line(x, ys[0], x, ys[-1])
    for y in ys:
        pdf.line(xs[0], y, xs[-1], y)
    pdf.setFont("Helvetica", 10)
    pdf.drawString(105, 687, "Position")
    pdf.drawString(310, 687, "SM")
    pdf.drawString(105, 667, "Team Leader")
    text = pdf.beginText(310, 667)
    text.setFont("Helvetica", 10)
    text.setCharSpace(3.5)
    text.textOut("12")
    pdf.drawText(text)
    pdf.save()

    page_text = extract_pages(path, tables=True)[0].text

    assert "Team Leader | 12" in page_text


# ---------- RFP page 355: rows with extra lines / boxes inside cells ----------


def _grid(pdf, xs, ys):
    for x in xs:
        pdf.line(x, ys[0], x, ys[-1])
    for y in ys:
        pdf.line(xs[0], y, xs[-1], y)


def test_value_in_a_half_height_box_stays_in_its_own_row(tmp_path):
    # Real bug: the Rate sat in a box with an extra line inside the cell, so pdfplumber made
    # a separate sliver row for it and "5,00,000" ended up on the line AFTER its position.
    from reportlab.pdfgen import canvas

    path = tmp_path / "fragment.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFont("Helvetica", 9)
    xs, ys = [100, 300, 400, 500], [700, 680, 640, 600]
    _grid(pdf, xs, ys)
    for x, label in zip(xs, ["Position", "Rate", "SM"]):
        pdf.drawString(x + 4, 687, label)
    pdf.drawString(104, 667, "Team Leader")
    pdf.drawString(404, 667, "12")
    pdf.drawString(104, 627, "Bridge Engineer")
    pdf.drawString(404, 627, "7")
    for y in (660, 620):  # extra line inside the Rate column only
        pdf.line(300, y, 400, y)
    pdf.drawString(304, 643, "5,00,000")
    pdf.drawString(304, 603, "3,50,000")
    pdf.save()

    text = extract_pages(path, tables=True)[0].text

    assert "Team Leader | 5,00,000 | 12" in text
    assert "Bridge Engineer | 3,50,000 | 7" in text


def test_small_box_drawn_inside_a_cell_is_not_repeated_as_its_own_table(tmp_path):
    from reportlab.pdfgen import canvas

    path = tmp_path / "nested.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFont("Helvetica", 9)
    xs, ys = [100, 300, 400], [700, 680, 640]
    _grid(pdf, xs, ys)
    pdf.drawString(104, 687, "Position")
    pdf.drawString(304, 687, "SM")
    pdf.rect(104, 650, 90, 24)  # nested box with its own middle line
    pdf.line(104, 662, 194, 662)
    pdf.drawString(108, 666, "Team")
    pdf.drawString(108, 654, "Leader")
    pdf.drawString(304, 660, "12")
    pdf.save()

    text = extract_pages(path, tables=True)[0].text

    assert text.count(TABLE_START) == 1
    assert text.count("Leader") == 1


def test_row_below_a_row_span_cell_stays_a_real_row(tmp_path):
    from reportlab.pdfgen import canvas

    path = tmp_path / "rowspan.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFont("Helvetica", 9)
    xs = [100, 200, 400, 500]
    for x in xs:
        pdf.line(x, 700, x, 600)
    for y in (700, 680, 600):
        pdf.line(100, y, 500, y)
    pdf.line(
        200, 640, 500, 640
    )  # middle line only over columns 2-3: column 1 spans two rows
    pdf.drawString(104, 687, "Stage")
    pdf.drawString(204, 687, "Item")
    pdf.drawString(404, 687, "Pct")
    pdf.drawString(104, 640, "First Stage")
    pdf.drawString(204, 660, "Inception report")
    pdf.drawString(404, 660, "2.5")
    pdf.drawString(204, 620, "Alignment report")
    pdf.drawString(404, 620, "10")
    pdf.save()

    text = extract_pages(path, tables=True)[0].text

    assert "First Stage | Inception report | 2.5" in text
    assert "Alignment report | 10" in text


def test_digits_split_farther_than_the_tolerance_are_still_joined(tmp_path):
    # Real RFP page 355: even x_tolerance=5 left "12" as "1 2". A cell made only of
    # single digits separated by spaces is joined as one number.
    from reportlab.pdfgen import canvas

    path = tmp_path / "very_spaced.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    xs, ys = [100, 300, 400], [700, 680, 660]
    _grid(pdf, xs, ys)
    pdf.setFont("Helvetica", 10)
    pdf.drawString(105, 687, "Position")
    pdf.drawString(310, 687, "SM")
    pdf.drawString(105, 667, "Team Leader")
    text = pdf.beginText(310, 667)
    text.setFont("Helvetica", 10)
    text.setCharSpace(9)
    text.textOut("12")
    pdf.drawText(text)
    pdf.save()

    page_text = extract_pages(path, tables=True)[0].text

    assert "Team Leader | 12" in page_text


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1 2", "12"),
        ("1 1", "11"),
        ("1 5 0", "150"),
        ("12", "12"),
        ("5,00,000", "5,00,000"),
        ("2.5 %", "2.5 %"),
        ("1 2 months", "1 2 months"),
        ("Team  Leader\nCum", "Team Leader Cum"),
        (None, ""),
    ],
)
def test_clean_cell(raw, expected):
    from app.services.document_parser import _clean_cell

    assert _clean_cell(raw) == expected


# ---------- overlapping cells (RFP page 355, rows 27-28) ----------
# These tests use a FAKE table that copies the geometry pdfplumber reported, so they pin
# the mechanism (a cell inside another cell + the same text) and not the RFP's words.


def _fake_table(rows):
    from types import SimpleNamespace as NS

    def make_row(bbox, cells):
        return NS(bbox=bbox, cells=[box for box, _ in cells])

    return NS(
        rows=[make_row(bbox, cells) for bbox, cells in rows],
        extract=lambda **kwargs: [[text for _, text in cells] for _, cells in rows],
    )


def _cells(**by_index):
    # 10 columns; indexes not given have no cell
    row = [(None, None)] * 10
    for index, value in by_index.items():
        row[int(index[1:])] = value
    return row


def test_text_read_twice_through_overlapping_cells_is_kept_once():
    from app.services.document_parser import _table_rows

    main = _cells(
        c0=((45.8, 624.6, 73.1, 658.9), "4"),
        c1=((73.1, 624.6, 221.0, 658.9), "Survey Engineer (7 Nos)"),
        c2=((78.1, 624.6, 216.0, 637.9), "Survey Engineer"),  # inside c1
        c5=((310.3, 624.6, 372.5, 658.9), "85,000"),
        c8=((372.5, 624.6, 492.9, 658.9), "48"),
    )
    sliver = _cells(
        c2=((78.1, 637.9, 216.0, 651.0), "(7 Nos)"),  # inside c1, text already in c1
        c6=((315.4, 637.9, 367.4, 651.0), "85,000"),  # inside c5, text already in c5
    )
    table = _fake_table(
        [
            ((45.8, 624.6, 565.4, 658.9), main),
            ((78.1, 637.9, 367.4, 651.0), sliver),
        ]
    )

    rows = _table_rows(table)

    assert len(rows) == 1
    assert [cell for cell in rows[0] if cell] == [
        "4",
        "Survey Engineer (7 Nos)",
        "85,000",
        "48",
    ]


def test_same_value_in_a_separate_cell_is_not_mistaken_for_a_duplicate():
    from app.services.document_parser import _table_rows

    # rate 12 sits in a sliver; SM is also 12 but in a different, non-containing cell
    main = _cells(
        c0=((45.0, 100.0, 70.0, 140.0), "1"),
        c1=((70.0, 100.0, 200.0, 140.0), "Engineer"),
        c8=((372.0, 100.0, 490.0, 140.0), "12"),
    )
    sliver = _cells(c6=((315.0, 125.0, 367.0, 140.0), "12"))
    table = _fake_table(
        [((45.0, 100.0, 565.0, 140.0), main), ((315.0, 125.0, 367.0, 140.0), sliver)]
    )

    rows = _table_rows(table)

    assert len(rows) == 1
    assert [cell for cell in rows[0] if cell] == ["1", "Engineer", "12", "12"]


# ---------- boxes without columns are plain text, not tables ----------


def test_boxed_heading_is_plain_text_with_its_lines(tmp_path):
    from reportlab.pdfgen import canvas

    path = tmp_path / "boxed.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFont("Helvetica", 10)
    pdf.rect(100, 700, 300, 40)
    pdf.drawString(110, 725, "Request for Proposal for Procurement")
    pdf.drawString(110, 710, "and Implementation of Services")
    pdf.drawString(100, 650, "Body text after the box.")
    pdf.save()

    text = extract_pages(path, tables=True)[0].text

    assert TABLE_START not in text
    assert (
        "Request for Proposal for Procurement\nand Implementation of Services" in text
    )
    assert text.index("Request for Proposal") < text.index("Body text")


def test_one_column_table_is_plain_text_but_two_columns_stay_a_table(tmp_path):
    from reportlab.pdfgen import canvas

    path = tmp_path / "columns.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFont("Helvetica", 10)
    _grid(pdf, [100, 300], [700, 680, 660])  # one column, two rows
    pdf.drawString(105, 687, "First point")
    pdf.drawString(105, 667, "Second point")
    _grid(pdf, [100, 300, 400], [600, 580, 560])  # two columns
    pdf.drawString(105, 587, "Item")
    pdf.drawString(305, 587, "Marks")
    pdf.drawString(105, 567, "Experience")
    pdf.drawString(305, 567, "30")
    pdf.save()

    text = extract_pages(path, tables=True)[0].text

    assert text.count(TABLE_START) == 1
    assert "Item | Marks" in text
    assert "First point" in text and "Second point" in text
    assert text.index("Second point") < text.index(TABLE_START)
