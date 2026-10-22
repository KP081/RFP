"""API-level tests for upload -> extraction -> status -> JSON. Offline: fake extractor, no Gemini.

FastAPI's TestClient runs background tasks before it returns, so after a request the
extraction has already finished (or failed).
"""

from types import SimpleNamespace as NS

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.api.deps import get_rfp_service
from app.api.routes.rfp import router
from app.core.exceptions import AppError, LLMError
from app.repositories.json_repository import JsonRFPRepository
from app.schemas.rfp import ExtractionStatus
from app.schemas.extraction import TenderExtraction
from app.services.rfp_service import RFPService

API = "/api/v1/rfps"


class FakeExtractor:
    def __init__(self):
        self.calls = 0
        self.fail = False
        self.fingerprint_value = "1.2|fake-model|abc123|pdf"

    def fingerprint(self, path, mode=None):
        return self.fingerprint_value

    def extract(self, path, mode=None):
        self.calls += 1
        if self.fail:
            raise LLMError("429 quota exceeded")

        extraction = TenderExtraction.model_validate(
            {
                "identity": {"title": f"Fake tender {self.calls}", "currency": "INR"},
                "meta": {"needs_review": True, "review_flags": ["x"]},
            }
        )
        return NS(
            extraction=extraction,
            usage=NS(prompt_tokens=1000, output_tokens=200, thinking_tokens=0),
            repairs=0,
            mode="pdf",
            seconds=1.5,
        )


class FakeIngestion:
    def ingest(self, file_path, rfp_id, filename):
        return {"pages": 3, "chunks": 5}


class FakeQdrant:
    def delete_by_rfp(self, rfp_id):
        pass


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # uploads/ and data/ are relative paths

    extractor = FakeExtractor()
    service = RFPService(
        repository=JsonRFPRepository(tmp_path / "rfps.json"),
        ingestion_service=FakeIngestion(),
        qdrant_service=FakeQdrant(),
        extraction_service_factory=lambda: extractor,
    )

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_rfp_service] = lambda: service

    @app.exception_handler(AppError)
    async def handle(request, exc):
        return JSONResponse(
            status_code=exc.status_code, content={"detail": exc.message}
        )

    return NS(client=TestClient(app), service=service, extractor=extractor)


def upload(env, text="one"):
    content = b"%PDF-1.4 " + text.encode()
    response = env.client.post(
        API, files={"file": (f"{text}.pdf", content, "application/pdf")}
    )
    assert response.status_code == 202, response.text
    return response.json()


def test_upload_runs_extraction_and_ingestion_in_the_background(env):
    created = upload(env)
    assert created["extraction_status"] == "running"  # what the client sees right away

    info = env.client.get(f"{API}/{created['rfp_id']}").json()
    assert info["extraction_status"] == "done" and info["status"] == "ready"
    assert info["extraction_needs_review"] is True and info["extracted_at"]

    result = env.client.get(f"{API}/{created['rfp_id']}/extraction")
    assert result.status_code == 200
    body = result.json()
    assert (
        body["identity"]["title"] == "Fake tender 1"
        and body["identity"]["currency"] == "INR"
    )
    assert body["meta"]["review_flags"] == ["x"]


def test_usage_is_stored_on_the_record_for_cost_tracking(env):
    rfp_id = upload(env)["rfp_id"]

    record = env.service.get(rfp_id)

    assert record.extraction_prompt_tokens == 1000
    assert record.extraction_output_tokens == 200
    assert record.extraction_seconds == 1.5
    assert record.extraction_key


def test_extract_again_uses_the_stored_result_unless_forced(env):
    rfp_id = upload(env)["rfp_id"]
    assert env.extractor.calls == 1

    again = env.client.post(f"{API}/{rfp_id}/extract")
    assert again.status_code == 202 and again.json()["extraction_status"] == "done"
    assert env.extractor.calls == 1  # cache hit: no LLM call

    forced = env.client.post(f"{API}/{rfp_id}/extract", params={"force": True})
    assert forced.status_code == 202
    assert env.extractor.calls == 2
    assert (
        env.client.get(f"{API}/{rfp_id}/extraction").json()["identity"]["title"]
        == "Fake tender 2"
    )


def test_a_new_prompt_or_model_invalidates_the_cache(env):
    rfp_id = upload(env)["rfp_id"]

    env.extractor.fingerprint_value = "1.2|fake-model|NEWPROMPT|pdf"
    env.client.post(f"{API}/{rfp_id}/extract")

    assert env.extractor.calls == 2


def test_failed_extraction_is_recorded_and_can_be_retried(env):
    env.extractor.fail = True
    rfp_id = upload(env)["rfp_id"]

    info = env.client.get(f"{API}/{rfp_id}").json()
    assert info["extraction_status"] == "failed" and "429" in info["extraction_error"]
    assert info["status"] == "ready"  # ingestion (Q&A) is not affected

    blocked = env.client.get(f"{API}/{rfp_id}/extraction")
    assert blocked.status_code == 409 and "failed" in blocked.json()["detail"]

    env.extractor.fail = False
    assert env.client.post(f"{API}/{rfp_id}/extract").status_code == 202
    assert env.client.get(f"{API}/{rfp_id}").json()["extraction_status"] == "done"


def test_extract_while_running_is_rejected(env):
    rfp_id = upload(env)["rfp_id"]
    env.service.repository.update(rfp_id, extraction_status=ExtractionStatus.RUNNING)

    response = env.client.post(f"{API}/{rfp_id}/extract")

    assert (
        response.status_code == 409 and "already running" in response.json()["detail"]
    )


def test_extraction_endpoint_errors(env):
    assert env.client.get(f"{API}/nope/extraction").status_code == 404

    rfp_id = upload(env)["rfp_id"]
    env.service.repository.update(
        rfp_id, extraction_status=ExtractionStatus.NOT_STARTED
    )
    assert env.client.get(f"{API}/{rfp_id}/extraction").status_code == 409


def test_restart_marks_running_extractions_as_failed(env):
    rfp_id = upload(env)["rfp_id"]
    env.service.repository.update(rfp_id, extraction_status=ExtractionStatus.RUNNING)

    env.service.recover_interrupted()

    record = env.service.get(rfp_id)
    assert (
        record.extraction_status == "failed" and "restarted" in record.extraction_error
    )


def test_delete_removes_the_stored_extraction(env, tmp_path):
    rfp_id = upload(env)["rfp_id"]
    stored = tmp_path / "data" / "rfp_extractions" / f"{rfp_id}.json"
    assert stored.exists()

    assert env.client.delete(f"{API}/{rfp_id}").status_code == 204

    assert not stored.exists()


def test_old_records_without_extraction_fields_still_load(env):
    # a record written before the extraction feature existed
    path = env.service.repository.path
    path.write_text(
        '{"x": {"rfp_id": "x", "filename": "a.pdf", "stored_path": "uploads/x.pdf", "size_bytes": 1,'
        ' "status": "ready", "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z"}}',
        encoding="utf-8",
    )

    info = env.client.get(f"{API}/x").json()

    assert info["extraction_status"] == "not_started" and info["status"] == "ready"
