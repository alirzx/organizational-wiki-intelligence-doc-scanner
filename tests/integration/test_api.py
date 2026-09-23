import json
from io import BytesIO

import httpx
from PIL import Image

from app.core.runtime import get_artifact_publisher, get_minio_storage_service
from app.main import app
from app.storage.minio_service import MinioObjectData, MinioStorageError


def image_bytes(size=(1200, 1800)):
    image = Image.new("RGB", size, "white")
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def api_client():
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    )


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
        endpoints = ["ocr", "figure-table", "stamp-signature"]
        for endpoint in endpoints:
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
            assert body["page_id"] == "doc_contract:p1"
            assert body["status"]["state"] == "success"
            assert body["image"]["source"]["type"] == "minio"
            assert body["image"]["source"]["object_key"] == "documents/doc_contract/images/page.png"
            assert isinstance(body["objects"], list)


async def test_document_extract_merges_multiple_uploaded_pages_and_keeps_page_provenance():
    files = [
        ("images", ("p1.png", image_bytes(), "image/png")),
        ("images", ("p2.png", image_bytes((1000, 1400)), "image/png")),
    ]
    pages = [
        {"page_id": "doc_42:scan_001", "page_number": 1, "metadata": {"scanner": "A"}},
        {"page_id": "doc_42:scan_002", "page_number": 2, "metadata": {"scanner": "A"}},
    ]
    async with api_client() as client:
        response = await client.post(
            "/api/v1/extract",
            files=files,
            data={
                "document_id": "doc_42",
                "document_metadata_json": json.dumps({"source": "pdf_scan", "source_file_id": "file_900"}),
                "pages_metadata_json": json.dumps(pages),
            },
        )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["document_id"] == "doc_42"
    assert body["page_count"] == 2
    assert len(body["pages"]) == 2
    assert body["pages"][0]["image"]["source"]["type"] == "upload"
    assert len(body["objects"]) == sum(len(page["objects"]) for page in body["pages"])
    assert {obj["page_number"] for obj in body["objects"]} == {1, 2}
    assert {obj["page_id"] for obj in body["objects"]} == {"doc_42:scan_001", "doc_42:scan_002"}
    assert body["object_counts"] == {
        "paragraph": 2,
        "table": 2,
        "figure": 2,
        "stamp": 2,
        "signature": 2,
    }
    assert body["processing"]["state"] == "success"


async def test_local_upload_can_optionally_publish_without_losing_detailed_response(monkeypatch):
    published = []
    publisher = get_artifact_publisher()
    monkeypatch.setattr(publisher, "publish", lambda run: published.append(run.response.document_id) or [])

    async with api_client() as client:
        response = await client.post(
            "/api/v1/extract",
            files=[("images", ("page.png", image_bytes(), "image/png"))],
            data={"document_id": "local_publish", "persist_outputs": "true"},
        )

    assert response.status_code == 200, response.text
    assert response.json()["processing"]["state"] == "success"
    assert published == ["local_publish"]


async def test_minio_inspection_endpoint_runs_all_three_models(monkeypatch):
    mock_minio(monkeypatch)
    async with api_client() as client:
        response = await client.post(
            "/api/v1/extract/minio/inspect",
            json={
                "document_id": "doc_minio",
                "document_metadata": {"source": "minio"},
                "pages": [
                    {
                        "image_url": "http://minio.test:9000/media/documents/doc_minio/images/p1.png",
                        "page_number": 1,
                    },
                    {
                        "image_url": "http://minio.test:9000/media/documents/doc_minio/images/p2.png",
                        "page_number": 2,
                    },
                ],
            },
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["page_count"] == 2
    assert body["processing"]["state"] == "success"
    assert all(page["image"]["source"]["type"] == "minio" for page in body["pages"])
    assert body["object_counts"] == {
        "paragraph": 2,
        "table": 2,
        "figure": 2,
        "stamp": 2,
        "signature": 2,
    }


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

    assert response.status_code == 200, response.text
    assert response.json()["processing"]["state"] == "success"
    assert published == ["inspect_publish"]


async def test_product_minio_endpoint_persists_then_returns_small_status(monkeypatch):
    mock_minio(monkeypatch)
    published = []
    publisher = get_artifact_publisher()
    monkeypatch.setattr(publisher, "publish", lambda run: published.append(run.response.document_id) or [])

    async with api_client() as client:
        response = await client.post("/api/v1/extract/minio", json=minio_payload())

    assert response.status_code == 200, response.text
    assert response.json() == {"document_id": "123", "status": "success"}
    assert published == ["123"]


async def test_product_minio_endpoint_returns_502_when_artifact_write_fails(monkeypatch):
    mock_minio(monkeypatch)
    publisher = get_artifact_publisher()

    def fail_publish(run):
        raise MinioStorageError("write denied")

    monkeypatch.setattr(publisher, "publish", fail_publish)

    async with api_client() as client:
        response = await client.post("/api/v1/extract/minio", json=minio_payload("write_fail"))

    assert response.status_code == 502
    assert response.json() == {
        "document_id": "write_fail",
        "status": "failed",
        "error": "write denied",
    }


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
    detail = response.json()["detail"]
    assert detail["document_id"] == "doc_invalid"
    assert detail["page_number"] == 2
    assert "corrupted or unsupported image" in detail["error"]
