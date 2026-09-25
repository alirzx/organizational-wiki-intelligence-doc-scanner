import json
from io import BytesIO

import pytest
from PIL import Image

from app.artifacts.publisher import ArtifactPublisher
from app.core.config import Settings
from app.orchestration.extractor import ExtractionOrchestrator
from app.preprocessing.pipeline import prepare_page
from app.schemas.status import ModuleName


class RecordingStorage:
    def __init__(self):
        self.deleted = []
        self.text = {}
        self.bytes = {}

    def delete_prefix(self, prefix):
        self.deleted.append(prefix)
        return 0

    def put_text(self, object_key, text, *, content_type="text/plain; charset=utf-8"):
        self.text[object_key] = (text, content_type)
        return object_key

    def put_bytes(self, object_key, data, *, content_type):
        self.bytes[object_key] = (data, content_type)
        return object_key


def _settings():
    return Settings(
        ocr_backend="mock",
        figure_table_backend="mock",
        stamp_signature_backend="mock",
        job_store_backend="memory",
    )


def _page(settings: Settings):
    image = Image.new("RGB", (800, 1000), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return prepare_page(
        data=buffer.getvalue(),
        filename="page-001.png",
        mime_type="image/png",
        document_id="123",
        page_id="123:p1",
        page_number=1,
        page_metadata={},
        settings=settings,
    )


async def _run():
    settings = _settings()
    return await ExtractionOrchestrator(settings).extract_document_run(
        document_id="123",
        pages=[_page(settings)],
        request_id="req-artifacts",
        document_metadata={"source": "minio"},
    )


@pytest.mark.asyncio
async def test_artifact_publisher_writes_expected_per_page_layout():
    run = await _run()
    storage = RecordingStorage()
    written = ArtifactPublisher(storage).publish(run)

    assert storage.deleted == [
        "documents/123/OCR/",
        "documents/123/Figure-Table/",
        "documents/123/Stamp-Signature/",
    ]
    assert "documents/123/OCR/page-001.json.txt" in storage.text
    assert "documents/123/OCR/page-001-text.txt" in storage.text
    assert "documents/123/OCR.txt" in storage.text
    assert "documents/123/layout.json" in storage.text

    layout = json.loads(storage.text["documents/123/layout.json"][0])
    assert layout["schema_version"] == "wiki-hami.layout.v2"
    assert layout["document_id"] == "123"
    assert layout["page_count"] == 1
    assert layout["object_count"] == 5
    assert layout["object_counts"] == {
        "paragraph": 1,
        "table": 1,
        "figure": 1,
        "stamp": 1,
        "signature": 1,
    }

    page = layout["pages"][0]
    assert page["page_number"] == 1
    assert page["width"] == 800
    assert page["height"] == 1000
    assert page["coordinate_space"] == "exif_corrected_source_pixels"
    assert page["object_count"] == 5
    assert page["object_counts"] == layout["object_counts"]
    assert page["reading_order_method"] == "bbox_top_to_bottom_then_left_to_right"

    objects = page["objects"]
    assert [obj["reading_order"] for obj in objects] == list(range(1, 6))
    assert {obj["type"] for obj in objects} == {
        "paragraph",
        "table",
        "figure",
        "stamp",
        "signature",
    }
    for obj in objects:
        assert set(obj["bbox"]) == {"x1", "y1", "x2", "y2"}
        assert "polygon" in obj
        assert "confidence" in obj
        assert "metadata" in obj
        assert "provenance" in obj
        assert "module" in obj
        assert "artifacts" in obj
        assert obj["artifacts"]["module_result"].startswith("documents/123/")

    paragraph = next(obj for obj in objects if obj["type"] == "paragraph")
    assert paragraph["text"] is not None
    assert paragraph["raw_text"] is not None
    assert paragraph["artifacts"]["plain_text"] == "documents/123/OCR/page-001-text.txt"
    assert paragraph["artifacts"]["crop"] is None

    table = next(obj for obj in objects if obj["type"] == "table")
    figure = next(obj for obj in objects if obj["type"] == "figure")
    stamp = next(obj for obj in objects if obj["type"] == "stamp")
    signature = next(obj for obj in objects if obj["type"] == "signature")

    assert table["artifacts"]["crop"] == "documents/123/Figure-Table/page-001-table-001.png"
    assert figure["artifacts"]["crop"] == "documents/123/Figure-Table/page-001-figure-001.png"
    assert stamp["artifacts"]["crop"] == "documents/123/Stamp-Signature/page-001-stamp-001.png"
    assert signature["artifacts"]["crop"] == "documents/123/Stamp-Signature/page-001-signature-001.png"

    assert "MOCK_OCR_TEXT" in storage.text["documents/123/OCR/page-001-text.txt"][0]
    assert '"module": "ocr"' in storage.text["documents/123/OCR/page-001.json.txt"][0]
    assert "documents/123/Figure-Table/page-001.json.txt" in storage.text
    assert "documents/123/Stamp-Signature/page-001.json.txt" in storage.text

    crop_keys = set(storage.bytes)
    assert "documents/123/Figure-Table/page-001-table-001.png" in crop_keys
    assert "documents/123/Figure-Table/page-001-figure-001.png" in crop_keys
    assert "documents/123/Stamp-Signature/page-001-stamp-001.png" in crop_keys
    assert "documents/123/Stamp-Signature/page-001-signature-001.png" in crop_keys
    assert not any("/OCR/" in key for key in crop_keys)
    assert all(data.startswith(b"\x89PNG") for data, _ in storage.bytes.values())
    assert set(written) == set(storage.text) | set(storage.bytes)


@pytest.mark.asyncio
async def test_visual_modules_still_write_raw_page_json_when_no_objects_are_detected():
    run = await _run()
    page_run = run.pages[0]
    page_run.modules[ModuleName.FIGURE_TABLE].objects = []
    page_run.modules[ModuleName.STAMP_SIGNATURE].objects = []

    storage = RecordingStorage()
    ArtifactPublisher(storage).publish(run)

    assert "documents/123/Figure-Table/page-001.json.txt" in storage.text
    assert "documents/123/Stamp-Signature/page-001.json.txt" in storage.text
    assert not any("/Figure-Table/" in key for key in storage.bytes)
    assert not any("/Stamp-Signature/" in key for key in storage.bytes)
    assert not any("/OCR/" in key for key in storage.bytes)

    layout = json.loads(storage.text["documents/123/layout.json"][0])
    assert layout["object_count"] == 1
    assert layout["object_counts"] == {
        "paragraph": 1,
        "table": 0,
        "figure": 0,
        "stamp": 0,
        "signature": 0,
    }
    assert [obj["type"] for obj in layout["pages"][0]["objects"]] == ["paragraph"]
