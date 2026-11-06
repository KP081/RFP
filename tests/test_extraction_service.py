"""Offline tests (no API calls): fake Gemini client and fake LLM service."""

import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from app.core.exceptions import LLMError
from app.schemas.extraction import TenderExtractionLLM
from app.services import extraction_service as E
from app.services import llm_service as L

GOOD = json.dumps(
    {"identity": {"title": "T"}, "dates": {"submission_end": "2026-03-25"}}
)
BAD = json.dumps({"identity": {"title": 123}, "dates": {"validity_days": "many"}})


# ---------------------------------------------------------------- fakes


def reply(text, finish="STOP", prompt=100, out=50):
    return NS(
        text=text,
        usage_metadata=NS(
            prompt_token_count=prompt,
            candidates_token_count=out,
            thoughts_token_count=None,
        ),
        candidates=[NS(finish_reason=NS(name=finish))],
    )


class FakeModels:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append((contents, config))
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_service(replies):
    service = L.LLMService.__new__(L.LLMService)  # skip the real genai client
    service.client = NS(models=FakeModels(replies))
    service.model = "fake-model"
    service.temperature = 0.0
    return service


# ---------------------------------------------------------------- LLMService


def test_first_try_with_fence_puts_document_first_and_sends_no_response_schema():
    service = make_service([reply("```json\n" + GOOD + "\n```")])

    result = service.generate_structured("INSTR", TenderExtractionLLM, parts=["DOC"])

    assert result.data.identity.title == "T" and result.repairs == 0
    contents, config = service.client.models.calls[0]
    assert contents[0] == "DOC" and "JSON SCHEMA" in contents[-1]
    assert config.response_mime_type == "application/json"
    assert getattr(config, "response_schema", None) is None


def test_null_objects_pass_without_a_repair_call():
    nulls = '{"project": {"estimated_contract_value": null, "project_cost": null}}'
    service = make_service([reply(nulls)])

    result = service.generate_structured("I", TenderExtractionLLM)

    assert result.repairs == 0 and service.client.models.calls.__len__() == 1


def test_repair_call_does_not_resend_the_document():
    service = make_service([reply(BAD), reply(GOOD)])

    result = service.generate_structured("I", TenderExtractionLLM, parts=["BIGDOC"])

    assert result.repairs == 1 and result.usage.calls == 2
    repair_contents = service.client.models.calls[1][0]
    assert len(repair_contents) == 1 and "BIGDOC" not in repair_contents[0]
    assert "an object must stay an object" in repair_contents[0]


def test_persistent_invalid_output_raises_and_saves_the_raw_text(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    service = make_service([reply(BAD), reply(BAD)])

    with pytest.raises(LLMError) as error:
        service.generate_structured("I", TenderExtractionLLM)

    assert "raw output saved" in error.value.message
    assert list((tmp_path / "data" / "debug").glob("invalid_*.json"))


def test_cut_off_output_is_an_error_not_a_repair():
    service = make_service([reply('{"identity":', finish="MAX_TOKENS")])

    with pytest.raises(LLMError, match="MAX_TOKENS"):
        service.generate_structured("I", TenderExtractionLLM)

    assert len(service.client.models.calls) == 1


RATE_LIMIT = (
    "429 RESOURCE_EXHAUSTED. Please retry in 37.4s. "
    "GenerateContentInputTokensPerModelPerMinute-FreeTier"
)


def test_per_minute_rate_limit_waits_and_retries(monkeypatch):
    sleeps = []
    monkeypatch.setattr(L.time, "sleep", sleeps.append)
    service = make_service([Exception(RATE_LIMIT), reply(GOOD)])

    result = service.generate_structured("I", TenderExtractionLLM)

    assert result.data.identity.title == "T"
    assert len(sleeps) == 1 and 40 < sleeps[0] < 41


@pytest.mark.parametrize(
    "message",
    [
        "429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel",
        "400 INVALID_ARGUMENT",
    ],
)
def test_daily_quota_and_other_errors_are_not_retried(monkeypatch, message):
    sleeps = []
    monkeypatch.setattr(L.time, "sleep", sleeps.append)
    service = make_service([Exception(message)])

    with pytest.raises(LLMError):
        service.generate_structured("I", TenderExtractionLLM)

    assert len(service.client.models.calls) == 1 and not sleeps


def test_gives_up_after_three_rate_limit_retries(monkeypatch):
    monkeypatch.setattr(L.time, "sleep", lambda seconds: None)
    service = make_service([Exception(RATE_LIMIT)] * 4)

    with pytest.raises(LLMError, match="429"):
        service.generate_structured("I", TenderExtractionLLM)

    assert len(service.client.models.calls) == 4


# ---------------------------------------------------------------- ExtractionService


class FakeLLM:
    model = "fake-model"

    def __init__(self, data, fail=False):
        self.data, self.fail = data, fail
        self.uploads, self.deleted, self.parts = [], [], None

    def upload_file(self, path):
        self.uploads.append(path)
        return NS(name="files/1")

    def delete_file(self, uploaded):
        self.deleted.append(uploaded.name)

    def generate_structured(self, instructions, schema, parts=None, **kwargs):
        self.parts = parts
        if self.fail:
            raise LLMError("boom")
        return L.StructuredResult(
            data=schema.model_validate(self.data),
            usage=L.Usage(1000, 200, 0, 1),
            repairs=0,
        )


DATA = {
    "identity": {"title": "X"},
    "project": {"estimated_contract_value": {"value": 100}},
    "dates": {
        "submission_end": "2026-03-25",
        "technical_opening": "2026-03-20",
        "published": "soon",
    },
    "evaluation": {
        "weights": {"technical_pct": 30, "other": [{"name": "DPR", "weight_pct": 30}]},
        "criteria": [
            {
                "name": "A1",
                "max_points": 5,
                "sub_criteria": [
                    {"name": "a", "max_points": 3},
                    {"name": "b", "max_points": 1},
                ],
            }
        ],
    },
    "key_personnel": {
        "key_person_months_total": 76,
        "roles": [
            {"role": "TL", "person_months": 12},
            {"role": "BR", "person_months": 7},
            {"role": "opt", "person_months": 4, "is_conditional": True},
        ],
    },
    "contract": {
        "payment_stages": [{"stage": "a", "percent": 60}, {"stage": "b", "percent": 30}]
    },
}


def test_pdf_mode_uploads_cleans_up_and_raises_review_flags():
    llm = FakeLLM(DATA)

    result = E.ExtractionService(llm_service=llm).extract(Path("a.pdf"), mode="pdf")

    assert llm.uploads and llm.deleted == ["files/1"] and llm.parts[0].name == "files/1"
    meta = result.extraction.meta
    assert (
        meta.model == "fake-model" and len(meta.prompt_hash) == 12 and meta.needs_review
    )
    flags = " | ".join(meta.review_flags)
    for needle in [
        "money amounts found but identity.currency is not set",
        "not a clean date: 'soon'",
        "before submission_end",
        "weights add up to 60",
        "roles add up to 19, stated total 76",
        "sub-criteria add up to 4, max is 5",
        "payment stages add up to 90%",
    ]:
        assert needle in flags, (needle, flags)


def test_text_mode_reads_pages_and_flags_pages_without_text(monkeypatch):
    pages = [NS(page_number=1, text="hello world " * 10), NS(page_number=2, text="")]
    monkeypatch.setattr(E, "load_pages", lambda path, **kwargs: pages)
    llm = FakeLLM({"identity": {"title": "Y"}})

    result = E.ExtractionService(llm_service=llm).extract(Path("a.pdf"), mode="text")

    assert not llm.uploads and "=== PAGE 2 ===" in llm.parts[0]
    assert any(
        "no text layer" in flag and "[2]" in flag
        for flag in result.extraction.meta.review_flags
    )


def test_docx_always_uses_text_mode_and_failure_still_deletes_the_upload(monkeypatch):
    monkeypatch.setattr(E, "load_pages", lambda path, **kwargs: [])
    assert (
        E.ExtractionService(llm_service=FakeLLM({}))
        .extract(Path("a.docx"), mode="pdf")
        .mode
        == "text"
    )

    llm = FakeLLM({}, fail=True)
    with pytest.raises(LLMError):
        E.ExtractionService(llm_service=llm).extract(Path("a.pdf"), mode="pdf")
    assert llm.deleted == ["files/1"]


# ---------------------------------------------------------------- derived fixes and checks


def test_person_month_total_is_computed_only_when_the_document_prints_none():
    roles = [
        {"role": "a", "person_months": 12},
        {"role": "b", "person_months": 7},
        {"role": "c", "person_months": 4, "is_conditional": True},
        {"role": "d", "person_months": 3, "category": "support"},
    ]
    doc = TenderExtractionLLM.model_validate({"key_personnel": {"roles": roles}})
    E.fill_derived(doc)
    assert doc.key_personnel.key_person_months_total == 19

    doc = TenderExtractionLLM.model_validate(
        {"key_personnel": {"key_person_months_total": 71, "roles": roles}}
    )
    E.fill_derived(doc)
    assert doc.key_personnel.key_person_months_total == 71
    assert any(
        "roles add up to 19, stated total 71" in f for f in E.consistency_flags(doc)
    )


def test_technical_and_financial_weights_filed_under_other_are_moved_up():
    doc = TenderExtractionLLM.model_validate(
        {
            "evaluation": {
                "weights": {
                    "other": [
                        {"name": "Technical Proposal", "weight_pct": 30},
                        {"name": "DPR Rating", "weight_pct": 30},
                        {"name": "PBG Quote", "weight_pct": 40},
                    ]
                }
            }
        }
    )

    notes = E.fill_derived(doc)

    weights = doc.evaluation.weights
    assert weights.technical_pct == 30 and weights.financial_pct is None
    assert [item.name for item in weights.other] == ["DPR Rating", "PBG Quote"]
    assert notes == ["weights: moved 'Technical Proposal' from other to technical_pct"]
    assert not any("weights add up" in f for f in E.consistency_flags(doc))


def test_weights_already_in_place_are_left_alone_and_double_counting_is_flagged():
    doc = TenderExtractionLLM.model_validate(
        {
            "evaluation": {
                "weights": {
                    "technical_pct": 80,
                    "financial_pct": 20,
                    "other": [
                        {"name": "Technical sub-score A", "weight_pct": 50},
                        {"name": "Technical sub-score B", "weight_pct": 30},
                    ],
                }
            }
        }
    )

    assert E.fill_derived(doc) == []
    assert len(doc.evaluation.weights.other) == 2
    assert any("weights add up to 180" in f for f in E.consistency_flags(doc))


def test_payment_stages_sum_check():
    def flags(stages):
        doc = TenderExtractionLLM.model_validate(
            {"contract": {"payment_stages": stages}}
        )
        return [f for f in E.consistency_flags(doc) if "payment" in f]

    assert not flags([{"stage": "a", "percent": 60}, {"stage": "b", "percent": 40}])
    assert flags([{"stage": "a", "percent": 60}, {"stage": "b", "percent": 30}])
    assert not flags([{"stage": "a", "percent": 60}, {"stage": "b"}])


def test_currency_flag_only_when_money_is_present():
    with_money = {"project": {"estimated_contract_value": {"value": 325000}}}
    assert any(
        "currency" in f
        for f in E.consistency_flags(TenderExtractionLLM.model_validate(with_money))
    )

    with_money["identity"] = {"currency": "USD"}
    assert not any(
        "currency" in f
        for f in E.consistency_flags(TenderExtractionLLM.model_validate(with_money))
    )
    assert not any(
        "currency" in f
        for f in E.consistency_flags(TenderExtractionLLM.model_validate({}))
    )


def test_fingerprint_follows_schema_model_mode_and_prompt(monkeypatch):
    service = E.ExtractionService(llm_service=FakeLLM({}))

    first = service.fingerprint(Path("a.pdf"))

    assert first.startswith("1.2|fake-model|") and first.endswith("|pdf")
    assert service.fingerprint(Path("a.docx")).endswith("|text")
    assert service.fingerprint(Path("a.pdf"), mode="text").endswith("|text")

    monkeypatch.setattr(E, "load_prompt", lambda name: "a different prompt")
    assert service.fingerprint(Path("a.pdf")) != first


# ---------------------------------------------------------------- text mode reads tables and cleans the text


def page(number, *lines):
    return NS(page_number=number, text="\n".join(lines))


def test_text_mode_reads_tables_and_sends_cleaned_text(monkeypatch):
    seen = {}
    body = lambda n: [
        f"Clause {n}.{i}: the consultant shall do task number {n * 10 + i}."
        for i in range(1, 9)
    ]
    pages = [page(i, *body(i), f"Page {i} of 6") for i in range(1, 7)]

    def fake_extract_pages(path, **kwargs):
        seen.update(kwargs)
        return pages

    monkeypatch.setattr(E, "load_pages", fake_extract_pages)
    llm = FakeLLM({"identity": {"title": "Y"}})

    E.ExtractionService(llm_service=llm).extract(Path("a.pdf"), mode="text")

    assert seen["tables"] is True
    sent = llm.parts[0]
    assert "Clause 3.4" in sent and "Page 3 of 6" not in sent  # page stamps are gone


def test_pages_emptied_by_cleaning_are_not_reported_as_scanned(monkeypatch):
    pages = [
        page(
            i,
            "Clause text that is long enough to be a real page " + str(i),
            f"Page {i} of 6",
        )
        for i in range(1, 7)
    ]
    monkeypatch.setattr(E, "load_pages", lambda path, **kwargs: pages)

    result = E.ExtractionService(llm_service=FakeLLM({})).extract(
        Path("a.pdf"), mode="text"
    )

    assert not any(
        "no text layer" in flag for flag in result.extraction.meta.review_flags
    )


# ---------------------------------------------------------------- RFP / NIT numbers are checked against the document


def run_identifiers(
    monkeypatch, model_identity, *page_texts, mode="text", read_fails=False
):
    pages = [page(i + 1, text) for i, text in enumerate(page_texts)]

    def fake_extract_pages(path, **kwargs):
        if read_fails and kwargs.get("max_pages"):
            raise RuntimeError("cannot read")
        return pages

    monkeypatch.setattr(E, "load_pages", fake_extract_pages)
    llm = FakeLLM({"identity": model_identity})
    result = E.ExtractionService(llm_service=llm).extract(Path("a.pdf"), mode=mode)
    return result.extraction.identity, result.extraction.meta.review_flags


def test_a_number_the_model_missed_is_filled_from_the_label(monkeypatch):
    identity, flags = run_identifiers(
        monkeypatch, {"title": "T"}, "ITI Clusters\nRFP Ref. No.: 03/2026"
    )

    assert (
        identity.rfp_number == "03/2026" and flags == []
    )  # an automatic fix is logged, not flagged


def test_the_unlabelled_letter_header_number_is_found(monkeypatch):
    page_text = "Letter of Invitation (LOI)\nNHIDCL/DPR/Siliguri-Guwahati/2025-26/ Dated :\nDear Sir,"

    identity, _ = run_identifiers(
        monkeypatch, {"rfp_number": "NHIDCL/DPR/Siliguri-Guwahati/2025-26/"}, page_text
    )

    assert identity.rfp_number == "NHIDCL/DPR/Siliguri-Guwahati/2025-26"


def test_a_title_around_the_number_is_reduced_to_the_number(monkeypatch):
    page_text = "[TABLE]\nRFP Number | Software Development Services, IIITB/EHRC/2022/IT- 01, 25-Oct-2022\n[/TABLE]"
    model = "Software Development Services, IIITB/EHRC/2022/IT-01, 25-Oct-2022"

    identity, flags = run_identifiers(monkeypatch, {"rfp_number": model}, page_text)

    assert identity.rfp_number == "IIITB/EHRC/2022/IT-01" and flags == []


def test_a_made_up_number_is_replaced_by_the_labelled_one(monkeypatch):
    identity, _ = run_identifiers(
        monkeypatch, {"rfp_number": "04/2027"}, "RFP Ref. No.: 03/2026"
    )

    assert identity.rfp_number == "03/2026"


def test_a_different_grounded_number_is_kept_and_flagged(monkeypatch):
    text = "RFP Ref. No.: 03/2026\nPackage code NHAI/PKG/12/2026 is attached"

    identity, flags = run_identifiers(
        monkeypatch, {"rfp_number": "NHAI/PKG/12/2026"}, text
    )

    assert identity.rfp_number == "NHAI/PKG/12/2026"
    assert any("differs" in flag for flag in flags)


def test_a_number_that_is_nowhere_in_the_text_is_flagged(monkeypatch):
    identity, flags = run_identifiers(
        monkeypatch, {"rfp_number": "XYZ/9/2030"}, "no numbers here"
    )

    assert identity.rfp_number == "XYZ/9/2030"
    assert any("was not found" in flag for flag in flags)


def test_the_rfp_number_is_never_kept_as_a_nit_number(monkeypatch):
    model = {
        "rfp_number": "RFP/Bhopal-Kota/2025-26/efile-299153",
        "nit_number": "RFP/ Bhopal- Kota/2025- 26/efile-299153",
    }

    identity, _ = run_identifiers(
        monkeypatch, model, "RFP Document No.: RFP/ Bhopal-Kota/2025-26/efile-299153"
    )

    assert (
        identity.rfp_number == "RFP/Bhopal-Kota/2025-26/efile-299153"
        and identity.nit_number is None
    )


def test_references_are_tidied_even_when_there_is_nothing_to_compare(monkeypatch):
    identity, flags = run_identifiers(
        monkeypatch, {"rfp_number": "RFP/ Bhopal- Kota/2025-26/"}, "", mode="pdf"
    )

    assert (
        identity.rfp_number == "RFP/Bhopal-Kota/2025-26" and flags == []
    )  # no text, so no 'not found' flag


def test_pdf_mode_reads_only_the_first_pages_for_the_check(monkeypatch):
    seen = {}

    def fake_extract_pages(path, **kwargs):
        seen.update(kwargs)
        return [page(1, "RFP Ref. No.: 03/2026")]

    monkeypatch.setattr(E, "load_pages", fake_extract_pages)
    llm = FakeLLM({"identity": {"title": "T"}})

    result = E.ExtractionService(llm_service=llm).extract(Path("a.pdf"), mode="pdf")

    assert seen["max_pages"] == 40 and llm.uploads  # Gemini still gets the whole PDF
    assert result.extraction.identity.rfp_number == "03/2026"


def test_pdf_mode_survives_a_failed_local_read(monkeypatch):
    identity, flags = run_identifiers(
        monkeypatch, {"rfp_number": "A/1/2025"}, "x", mode="pdf", read_fails=True
    )

    assert identity.rfp_number == "A/1/2025" and flags == []


def test_fingerprint_changes_with_the_pipeline_version(monkeypatch):
    service = E.ExtractionService(llm_service=FakeLLM({}))
    first = service.fingerprint(Path("a.pdf"))

    monkeypatch.setattr(E, "PIPELINE_VERSION", "99")

    assert service.fingerprint(
        Path("a.pdf")
    ) != first and "|p99|" in service.fingerprint(Path("a.pdf"))


# ---------------------------------------------------------------- unit fix: "5.00 Crores" copied as 5.0


def test_a_bare_number_copied_from_a_crore_amount_gets_its_unit():
    doc = TenderExtractionLLM.model_validate(
        {
            "qualification": {
                "financial": [
                    {
                        "kind": "turnover",
                        "label": "turnover",
                        "basis": "absolute_amount",
                        "value": 5.0,
                        "raw_text": "Rs.5.00 Crores",
                    },
                    {
                        "kind": "net_worth",
                        "label": "net worth",
                        "basis": "absolute_amount",
                        "value": 75.0,
                        "raw_text": "Rs 75 lakh",
                    },
                ]
            }
        }
    )

    notes = E.fill_derived(doc)

    assert [t.value for t in doc.qualification.financial] == [50_000_000, 7_500_000]
    assert len(notes) == 2 and "Rs.5.00 Crores" in notes[0]


def test_amounts_that_are_already_right_or_have_no_unit_are_left_alone():
    doc = TenderExtractionLLM.model_validate(
        {
            "qualification": {
                "financial": [
                    {
                        "kind": "turnover",
                        "label": "t",
                        "basis": "absolute_amount",
                        "value": 50_000_000,
                        "raw_text": "Rs.5.00 Crores",
                    },
                    {
                        "kind": "staff_strength",
                        "label": "staff",
                        "basis": "absolute_amount",
                        "value": 500,
                        "raw_text": "500 Cr employees",
                    },
                    {
                        "kind": "turnover",
                        "label": "pct",
                        "basis": "pct_of_estimated_cost",
                        "value": 3,
                        "raw_text": "3 crore",
                    },
                ]
            },
            "project": {
                "estimated_contract_value": {
                    "value": 116400510,
                    "raw_text": "Rs 11,64,00,510",
                }
            },
        }
    )

    assert E.fill_derived(doc) == []
    assert doc.qualification.financial[0].value == 50_000_000
    assert (
        doc.qualification.financial[1].value == 500
        and doc.qualification.financial[2].value == 3
    )
    assert doc.project.estimated_contract_value.value == 116400510


def test_project_amounts_get_the_same_unit_fix():
    doc = TenderExtractionLLM.model_validate(
        {"project": {"project_cost": {"value": 14.86, "raw_text": "₹ 14.86 Cr"}}}
    )

    E.fill_derived(doc)

    assert doc.project.project_cost.value == pytest.approx(148_600_000)


# ---------------------------------------------------------------- rag mode


def _big_pages(count=60):
    filler = lambda n: " ".join(f"zorblat{n}x{k} quillfeather{k}" for k in range(120))
    special = {7: "Bid Security: earnest money deposit Rs 200000 as bank guarantee."}
    return [page(i, special.get(i, filler(i))) for i in range(1, count + 1)]


def test_rag_mode_sends_selected_excerpts_and_records_the_pages(monkeypatch):
    monkeypatch.setattr(E, "load_pages", lambda path, **kwargs: _big_pages())
    llm = FakeLLM({"identity": {"title": "Z"}})
    service = E.ExtractionService(llm_service=llm)
    monkeypatch.setattr(service.settings, "extract_rag_budget_chars", 12_000)

    result = service.extract(Path("a.pdf"), mode="rag")

    sent = llm.parts[0]
    assert result.mode == "rag" and not llm.uploads  # the PDF itself is not uploaded
    assert sent.startswith("SELECTED EXCERPTS") and "=== PAGE 7 ===" in sent
    assert len(sent) < 20_000
    assert 7 in result.extraction.meta.pages_used["rag_evidence"]


def test_rag_mode_on_a_small_document_sends_everything_like_text_mode(monkeypatch):
    monkeypatch.setattr(E, "load_pages", lambda path, **kwargs: _big_pages(3))
    llm = FakeLLM({"identity": {"title": "Z"}})

    result = E.ExtractionService(llm_service=llm).extract(Path("a.pdf"), mode="rag")

    assert "SELECTED EXCERPTS" not in llm.parts[0] and "=== PAGE 3 ===" in llm.parts[0]
    assert result.extraction.meta.pages_used == {}


def test_docx_stays_in_text_mode_even_when_rag_is_asked(monkeypatch):
    monkeypatch.setattr(E, "load_pages", lambda path, **kwargs: [])

    result = E.ExtractionService(llm_service=FakeLLM({})).extract(
        Path("a.docx"), mode="rag"
    )

    assert result.mode == "text"


def test_rag_fingerprint_follows_budget_and_chunk_size(monkeypatch):
    service = E.ExtractionService(llm_service=FakeLLM({}))
    first = service.fingerprint(Path("a.pdf"), mode="rag")

    assert "|rag-s" in first
    monkeypatch.setattr(service.settings, "extract_rag_budget_chars", 99_000)
    assert service.fingerprint(Path("a.pdf"), mode="rag") != first
    assert service.fingerprint(Path("a.pdf"), mode="text").endswith("|text")
