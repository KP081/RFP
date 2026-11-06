import time

import pytest

from app.services.document_parser import TABLE_END, TABLE_START, DocumentPage
from app.services.evidence_selector import (
    NOTE,
    QueryGroup,
    score_groups,
    render_evidence,
    select_evidence,
    tokens,
)
from app.services.rag_chunker import chunk_pages_for_extraction


def filler(page: int, size: int = 1800) -> str:
    # words that none of the queries use, different on every page
    words = [f"zorblat{page}x{n} quillfeather{n} munderbock{page}" for n in range(1, 200)]
    return " ".join(words)[:size]


def pages_of(special: dict[int, str], count: int = 80) -> list[DocumentPage]:
    return [
        DocumentPage(page_number=p, text=special.get(p, filler(p)))
        for p in range(1, count + 1)
    ]


# ---------------------------------------------------------------- chunker


def test_a_chunk_never_crosses_a_page_and_indexes_are_consecutive():
    pages = [DocumentPage(1, "A" * 100), DocumentPage(2, ""), DocumentPage(3, "B" * 100)]

    chunks = chunk_pages_for_extraction(pages, 400)

    assert [(c.index, c.page_number) for c in chunks] == [(0, 1), (1, 3)]


def test_a_table_that_fits_stays_whole_between_paragraphs():
    text = f"Intro line.\n\n{TABLE_START}\nPosition | Nos.\nTeam Leader | 1\n{TABLE_END}\n\nNote after."

    chunks = chunk_pages_for_extraction([DocumentPage(1, text)], 1500)

    assert len(chunks) == 1 and "Team Leader | 1" in chunks[0].text


def test_a_long_table_is_cut_between_rows_and_the_header_repeats():
    rows = [f"Role {n} | {n} | {n * 10}" for n in range(1, 41)]
    text = "\n".join([TABLE_START, "Position | Nos. | Marks", *rows, TABLE_END])

    chunks = chunk_pages_for_extraction([DocumentPage(1, text)], 400)

    assert len(chunks) > 2
    for chunk in chunks:
        lines = chunk.text.splitlines()
        assert lines[0] == TABLE_START and lines[1] == "Position | Nos. | Marks"
        assert lines[-1] == TABLE_END
    body = "\n".join(c.text for c in chunks)
    for row in rows:  # no row lost, none cut in half
        assert body.count(row) == 1


def test_a_long_paragraph_is_split_without_losing_words():
    text = " ".join(f"word{n}" for n in range(600))

    chunks = chunk_pages_for_extraction([DocumentPage(1, text)], 500)

    assert len(chunks) > 1 and all(len(c.text) <= 500 for c in chunks)
    assert " ".join(c.text for c in chunks).split() == text.split()


def test_tiny_chunk_size_is_refused():
    with pytest.raises(ValueError):
        chunk_pages_for_extraction([], 50)


# ---------------------------------------------------------------- selector


SPECIAL = {
    7: "Bid Security: the earnest money deposit is Rs 200000 in the form of a bank guarantee.",
    30: (
        f"{TABLE_START}\nKey personnel | Nos. | Man-months | Site\nTeam Leader | 1 | 12 | 50%\n{TABLE_END}"
    ),
    31: "Notes: Tunnel Design Expert is required in case tunnels are proposed, rate 500000 per month.",
    45: "Liquidated damages: penalty for delay is 0.05% of contract price per day, maximum 5%.",
    60: "Detailed marking scheme: further break-up of sub-points and maximum marks for each criteria.",
}


@pytest.fixture(scope="module")
def chunks():
    return chunk_pages_for_extraction(pages_of(SPECIAL), 1500)


def test_small_document_is_taken_whole():
    pages = [DocumentPage(1, "short text"), DocumentPage(2, "more text")]
    selection = select_evidence(chunk_pages_for_extraction(pages), 120_000)

    assert [c.page_number for c in selection.chunks] == [1, 2]


def test_pages_with_the_facts_are_selected_under_the_budget(chunks):
    budget = 20_000
    selection = select_evidence(chunks, budget)

    assert selection.chars <= budget
    assert selection.chars < selection.total_chars / 3
    for page in SPECIAL:
        assert page in selection.pages, f"page {page} missing: {selection.pages}"


def test_the_front_of_the_document_is_always_kept(chunks):
    selection = select_evidence(chunks, 20_000)

    assert 1 in selection.pages


def test_chunks_come_back_in_document_order_without_duplicates(chunks):
    indexes = [c.index for c in select_evidence(chunks, 20_000).chunks]

    assert indexes == sorted(set(indexes))


def test_neighbours_of_a_hit_are_included(chunks):
    selection = select_evidence(chunks, 20_000)

    # the personnel table is on page 30; the conditional-experts note on 31 is its neighbour
    assert 30 in selection.pages and 31 in selection.pages


def test_render_has_note_page_markers_and_omitted_marks(chunks):
    text = render_evidence(select_evidence(chunks, 20_000))

    assert text.startswith(NOTE)
    assert "=== PAGE 7 ===" in text and "=== PAGE 45 ===" in text
    assert "[... omitted ...]" in text
    assert text.index("=== PAGE 7 ===") < text.index("=== PAGE 45 ===")


def test_tokens_drop_stop_words_and_lowercase():
    assert tokens("The Consultant SHALL submit 5 copies") == ["consultant", "submit", "5", "copies"]


def test_a_400_page_document_is_selected_fast():
    pages = [DocumentPage(p, filler(p, 3000)) for p in range(1, 401)]
    pages[281] = DocumentPage(282, SPECIAL[60])
    big = chunk_pages_for_extraction(pages, 1500)

    started = time.time()
    selection = select_evidence(big, 120_000)

    assert time.time() - started < 15
    assert 282 in selection.pages and selection.chars <= 120_000


# ---------------------------------------------------------------- unused share is lent, min_score


def _chunk(index, text):
    from app.services.rag_chunker import EvidenceChunk

    return EvidenceChunk(index, index + 1, text)


def test_a_group_with_only_weak_chunks_lends_its_share_to_a_group_with_many_good_ones():
    pad = lambda n: " ".join(f"pad{n}x{k}" for k in range(150))  # ~1000 chars, unique words
    a_chunks = [_chunk(i, ("alphaword " * 12) + pad(i)) for i in range(40)]
    b_chunks = [_chunk(40, ("betaword " * 40) + pad(40))]
    b_chunks += [_chunk(41 + i, "betaword " + pad(41 + i)) for i in range(39)]  # weak: one mention
    chunks = a_chunks + b_chunks
    groups = (
        QueryGroup("A", ("alphaword",), (), 0.5),
        QueryGroup("B", ("betaword",), (), 0.5),
    )
    budget = sum(len(c.text) for c in chunks) // 4

    lend = select_evidence(chunks, budget, groups, min_score=0.5)
    keep = select_evidence(chunks, budget, groups, min_score=0.0)

    def a_chars(selection):
        return sum(len(c.text) for c in selection.chunks if c.index < 40)

    assert a_chars(lend) > a_chars(keep)  # with a threshold A may use more than its own half
    assert a_chars(lend) > 0.6 * budget


def test_score_groups_gives_one_score_per_chunk_and_group(chunks):
    scores = score_groups(chunks)

    assert set(scores) >= {"securities", "contract", "evaluation"}
    assert all(len(v) == len(chunks) for v in scores.values())
    page7 = next(c.index for c in chunks if c.page_number == 7)
    assert scores["securities"][page7] == max(scores["securities"])


def test_sub_points_header_written_with_a_space_is_a_keyword_hit():
    import re
    from app.services.evidence_selector import GROUPS

    evaluation = next(g for g in GROUPS if g.name == "evaluation")
    header = "S. No. | Description | Max Points | Sub- Points"

    assert sum(bool(re.search(p, header, re.I)) for p in evaluation.keywords) == 2


def test_a_table_introduced_by_a_lead_in_sentence_is_followed_to_its_end(monkeypatch):
    from app.services import evidence_selector as S

    rows = [f"Stage {n} | submission of item number {n} | {n}" for n in range(1, 151)]
    table = "\n".join([TABLE_START, "Stage | Description | Percent", *rows, TABLE_END])
    lead = (
        "The consultant will be paid stage-wise as a percentage of the contract price "
        "as per the payment schedule given below:"
    )
    chunks = chunk_pages_for_extraction(pages_of({30: lead + "\n" + table}, count=80), 1500)

    def on_page(selection):
        # what the contract group itself asked for (the leftover fill could add the rest later)
        return sum(1 for i in selection.groups["contract"] if chunks[i].page_number == 30)

    with_chain = select_evidence(chunks, 80_000)
    monkeypatch.setattr(S, "_table_chain", lambda chunks, i, limit=4: [])
    without = select_evidence(chunks, 80_000)

    assert on_page(with_chain) >= 5 and on_page(with_chain) > on_page(without)


def test_table_chain_only_follows_chunks_that_start_a_table():
    from app.services.evidence_selector import _table_chain
    from app.services.rag_chunker import EvidenceChunk

    def c(i, text):
        return EvidenceChunk(i, 1, text)

    chunks = [
        c(0, "Payment is made as per the schedule given below:"),
        c(1, f"{TABLE_START}\nA | 1\n{TABLE_END}"),
        c(2, f"{TABLE_START}\nA | 2\n{TABLE_END}"),
        c(3, "Plain text again."),
        c(4, f"{TABLE_START}\nB | 9\n{TABLE_END}"),
    ]

    assert _table_chain(chunks, 0) == [1, 2]
    assert _table_chain(chunks, 1) == [2]  # a table chunk continues into the next table chunk
    assert _table_chain(chunks, 3) == []  # plain text that is not a lead-in