import json
from io import BytesIO

import httpx
from PIL import Image

from app.core.runtime import get_artifact_publisher, get_job_store, get_minio_storage_service
from app.jobs.tasks import process_minio_document
from app.main import app
from app.storage.minio_service import MinioObjectData


def image_bytes(size=(1200, 1800)):
    image = Image.new("RGB", size, "white")
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def api_client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")


def mock_minio(monkeypatch):
    storage = get_minio_storage_service()

    def fetch_url(url: str):
        ref = storage.parse_image_url(url)
        data = image_bytes()
        return MinioObjectData(
            ref=ref,
            data=data,
            filename=ref.object_key.rsplit("/", 1)[-1],
            content_type="image/png",
            etag="etag-test",
            size=len(data),
            last_modified=None,
        )

    monkeypatch.setattr(storage, "fetch_url", fetch_url)
    return storage


def minio_payload(document_id="123"):
    return {
        "document_id": document_id,
        "document_metadata": {"source": "minio"},
        "pages": [
            {
                "image_url": f"http://minio.test:9000/media/documents/{document_id}/images/page-001.jpg",
                "page_number": 1,
                "page_id": f"{document_id}:p1",
            }
        ],
    }


async def test_health():
    async with api_client() as client:
        response = await client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["storage"]["minio"]["bucket"] == "media"


async def test_each_engineering_module_endpoint_accepts_minio_url(monkeypatch):
    mock_minio(monkeypatch)
    async with api_client() as client:
        for endpoint in ["ocr", "figure-table", "stamp-signature"]:
            response = await client.post(
                f"/api/v1/{endpoint}",
                json={
                    "document_id": "doc_contract",
                    "page_number": 1,
                    "image_url": "http://minio.test:9000/media/documents/doc_contract/images/page.png",
                },
            )
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["document_id"] == "doc_contract"
            assert body["status"]["state"] == "success"


async def test_document_extract_merges_uploaded_pages():
    files = [
        ("images", ("p1.png", image_bytes(), "image/png")),
        ("images", ("p2.png", image_bytes((1000, 1400)), "image/png")),
    ]
    pages = [
        {"page_id": "doc_42:scan_001", "page_number": 1},
        {"page_id": "doc_42:scan_002", "page_number": 2},
    ]
    async with api_client() as client:
        response = await client.post(
            "/api/v1/extract",
            files=files,
            data={
                "document_id": "doc_42",
                "document_metadata_json": json.dumps({"source": "pdf_scan"}),
                "pages_metadata_json": json.dumps(pages),
            },
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["document_id"] == "doc_42"
    assert body["page_count"] == 2
    assert body["processing"]["state"] == "success"
    assert set(body["pages"][0]["modules"]) == {
        "ocr",
        "figure_table",
        "stamp_signature",
    }


async def test_document_extract_can_select_one_module():
    async with api_client() as client:
        response = await client.post(
            "/api/v1/extract",
            files=[
                ("images", ("page-1.png", image_bytes(), "image/png")),
                ("images", ("page-2.png", image_bytes(), "image/png")),
            ],
            data={"document_id": "local_ocr", "module": "ocr"},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["page_count"] == 2
    assert all(set(page["modules"]) == {"ocr"} for page in body["pages"])
    assert {obj["type"] for obj in body["objects"]} == {"paragraph"}


async def test_local_upload_can_optionally_publish(monkeypatch):
    published = []
    publisher = get_artifact_publisher()
    monkeypatch.setattr(publisher, "publish", lambda run: published.append(run.response.document_id) or [])
    async with api_client() as client:
        response = await client.post(
            "/api/v1/extract",
            files=[("images", ("page.png", image_bytes(), "image/png"))],
            data={"document_id": "local_publish", "persist_outputs": "true"},
        )
    assert response.status_code == 200
    assert published == ["local_publish"]


async def test_minio_inspection_endpoint_runs_all_three_models(monkeypatch):
    mock_minio(monkeypatch)
    async with api_client() as client:
        response = await client.post("/api/v1/extract/minio/inspect", json=minio_payload("doc_minio"))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["processing"]["state"] == "success"
    assert set(body["pages"][0]["modules"]) == {
        "ocr",
        "figure_table",
        "stamp_signature",
    }


async def test_minio_inspection_can_select_one_module(monkeypatch):
    mock_minio(monkeypatch)
    payload = minio_payload("doc_minio_figure")
    payload["pages"].append(
        {
            "image_url": (
                "http://minio.test:9000/media/documents/doc_minio_figure/images/page-002.jpg"
            ),
            "page_number": 2,
            "page_id": "doc_minio_figure:p2",
        }
    )
    async with api_client() as client:
        response = await client.post(
            "/api/v1/extract/minio/inspect?module=figure_table",
            json=payload,
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["page_count"] == 2
    assert all(set(page["modules"]) == {"figure_table"} for page in body["pages"])
    assert {obj["type"] for obj in body["objects"]} == {"figure", "table"}


async def test_selected_inspection_cannot_publish_full_artifact_set(monkeypatch):
    mock_minio(monkeypatch)
    async with api_client() as client:
        response = await client.post(
            "/api/v1/extract/minio/inspect?module=ocr&persist_outputs=true",
            json=minio_payload("inspect_selected_publish"),
        )

    assert response.status_code == 422
    assert "all modules" in response.json()["detail"]


async def test_minio_inspection_can_publish_in_same_inference_pass(monkeypatch):
    mock_minio(monkeypatch)
    published = []
    publisher = get_artifact_publisher()
    monkeypatch.setattr(publisher, "publish", lambda run: published.append(run.response.document_id) or [])
    async with api_client() as client:
        response = await client.post(
            "/api/v1/extract/minio/inspect?persist_outputs=true",
            json=minio_payload("inspect_publish"),
        )
    assert response.status_code == 200
    assert published == ["inspect_publish"]


async def test_product_minio_endpoint_returns_202_and_job_id(monkeypatch):
    queued = []

    def fake_apply_async(*, args, task_id, queue):
        queued.append({"args": args, "task_id": task_id, "queue": queue})

    monkeypatch.setattr(process_minio_document, "apply_async", fake_apply_async)
    async with api_client() as client:
        response = await client.post("/api/v1/extract/minio", json=minio_payload("41"))

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["document_id"] == "41"
    assert body["status"] == "queued"
    assert body["job_id"]
    assert queued[0]["task_id"] == body["job_id"]
    assert get_job_store().get(body["job_id"])["status"] == "queued"


def test_celery_product_processing_completes_with_explicit_blocking_executor(monkeypatch):
    mock_minio(monkeypatch)
    callbacks = []
    publisher = get_artifact_publisher()
    monkeypatch.setattr(
        publisher,
        "publish",
        lambda run: [f"documents/{run.response.document_id}/layout.json"],
    )
    monkeypatch.setattr(
        "app.jobs.tasks._deliver_terminal_callback",
        lambda job_id, document_id, payload: callbacks.append(payload),
    )
    job_id = "explicit-executor-job"
    get_job_store().create(job_id, "celery_runtime")

    result = process_minio_document.run(minio_payload("celery_runtime"), job_id)

    assert result["status"] == "completed"
    assert result["written"] == ["documents/celery_runtime/layout.json"]
    assert get_job_store().get(job_id)["status"] == "completed"
    assert "result" in callbacks[0]
    assert "outputs" not in callbacks[0]


async def test_job_status_endpoint_reports_state(monkeypatch):
    monkeypatch.setattr(process_minio_document, "apply_async", lambda **kwargs: None)
    async with api_client() as client:
        accepted = await client.post("/api/v1/extract/minio", json=minio_payload("42"))
        job_id = accepted.json()["job_id"]
        get_job_store().update(job_id, status="processing")
        status_response = await client.get(f"/api/v1/jobs/{job_id}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "processing"


async def test_product_endpoint_rejects_cross_document_object(monkeypatch):
    monkeypatch.setattr(process_minio_document, "apply_async", lambda **kwargs: None)
    payload = minio_payload("41")
    payload["pages"][0]["image_url"] = "http://minio.test:9000/media/documents/99/images/page-001.jpg"
    async with api_client() as client:
        response = await client.post("/api/v1/extract/minio", json=payload)
    assert response.status_code == 422
    assert "documents/41/images/" in response.json()["detail"]


async def test_product_endpoint_returns_503_when_queue_fails(monkeypatch):
    def fail_queue(**kwargs):
        raise RuntimeError("broker unavailable")

    monkeypatch.setattr(process_minio_document, "apply_async", fail_queue)
    async with api_client() as client:
        response = await client.post("/api/v1/extract/minio", json=minio_payload("queue_fail"))
    assert response.status_code == 503
    assert "broker unavailable" in response.json()["detail"]


async def test_invalid_page_rejects_document_before_module_execution():
    async with api_client() as client:
        response = await client.post(
            "/api/v1/extract",
            files=[
                ("images", ("valid.png", image_bytes(), "image/png")),
                ("images", ("broken.png", b"not an image", "image/png")),
            ],
            data={"document_id": "doc_invalid"},
        )
    assert response.status_code == 422


async def test_job_cancel_marks_failed_and_revokes_task(monkeypatch):
    from app.api.v1.endpoints import jobs as jobs_endpoint

    revoked = []
    callbacks = []
    job_id = "cancel-job"
    get_job_store().create(job_id, "cancel-doc")
    get_job_store().update(job_id, status="processing")

    monkeypatch.setattr(
        jobs_endpoint.celery_app.control,
        "revoke",
        lambda task_id, terminate, signal: revoked.append((task_id, terminate, signal)),
    )
    monkeypatch.setattr(
        jobs_endpoint,
        "_deliver_terminal_callback",
        lambda jid, document_id, payload: callbacks.append(payload),
    )

    async with api_client() as client:
        response = await client.delete(f"/api/v1/jobs/{job_id}")

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["error"]["code"] == "CANCELLED"
    assert revoked == [(job_id, True, "SIGTERM")]
    assert callbacks[0]["error"]["code"] == "CANCELLED"
