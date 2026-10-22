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
    monkeypatch.setattr(E, "extract_pages", lambda path: pages)
    llm = FakeLLM({"identity": {"title": "Y"}})

    result = E.ExtractionService(llm_service=llm).extract(Path("a.pdf"), mode="text")

    assert not llm.uploads and "=== PAGE 2 ===" in llm.parts[0]
    assert any(
        "no text layer" in flag and "[2]" in flag
        for flag in result.extraction.meta.review_flags
    )


def test_docx_always_uses_text_mode_and_failure_still_deletes_the_upload(monkeypatch):
    monkeypatch.setattr(E, "extract_pages", lambda path: [])
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
