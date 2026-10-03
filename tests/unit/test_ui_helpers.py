import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
from streamlit.testing.v1 import AppTest

from ui.artifacts import build_run_zip, canonical_json_bytes
from ui.module_runs import build_module_document_result, build_module_request, success_message
from ui.visualizer import CLASS_COLORS, annotate_page


def _objects():
    types = ["paragraph", "table", "figure", "stamp", "signature"]
    return [
        {
            "type": object_type,
            "confidence": 0.9,
            "bbox": {"x1": 10 + index * 30, "y1": 20, "x2": 35 + index * 30, "y2": 70},
            "polygon": None,
        }
        for index, object_type in enumerate(types)
    ]


def test_visualizer_has_stable_mapping_and_renders_all_module_classes():
    assert set(CLASS_COLORS) == {"paragraph", "table", "figure", "stamp", "signature"}
    source = Image.new("RGB", (200, 100), "white")
    annotated = annotate_page(source, _objects())
    assert annotated.size == source.size
    assert annotated.tobytes() != source.tobytes()
    for index, object_type in enumerate(CLASS_COLORS):
        assert annotated.getpixel((10 + index * 30, 69)) == tuple(bytes.fromhex(CLASS_COLORS[object_type].removeprefix("#")))


def test_json_and_multi_page_zip_exports_preserve_complete_result():
    result = {
        "schema_version": "wiki-hami.extraction.v1",
        "document_id": "DOC/100",
        "page_count": 2,
        "pages": [
            {"page_id": "DOC/100:p1", "page_number": 1, "image": {"filename": "اول.png"}},
            {"page_id": "DOC/100:p2", "page_number": 2, "image": {"filename": "second.png"}},
        ],
        "objects": _objects(),
        "object_counts": {name: 1 for name in CLASS_COLORS},
        "processing": {"state": "success", "duration_ms": 1, "warnings": []},
    }
    images = [(page, Image.new("RGB", (20, 20), "white")) for page in result["pages"]]
    assert json.loads(canonical_json_bytes(result)) == result
    with ZipFile(BytesIO(build_run_zip(result, images))) as archive:
        names = set(archive.namelist())
        assert names == {
            "DOC_100/extraction.json",
            "DOC_100/manifest.json",
            "DOC_100/pages/page_0001_annotated.png",
            "DOC_100/pages/page_0002_annotated.png",
        }
        assert json.loads(archive.read("DOC_100/extraction.json")) == result
        manifest = json.loads(archive.read("DOC_100/manifest.json"))
        assert [page["page_id"] for page in manifest["pages"]] == ["DOC/100:p1", "DOC/100:p2"]


def _module_page(page_number: int, *, state: str = "success"):
    page_id = f"doc:p{page_number}"
    return {
        "schema_version": "wiki-hami.extraction.v1",
        "request_id": "request-1",
        "document_id": "doc",
        "page_id": page_id,
        "page_number": page_number,
        "module": "ocr",
        "image": {
            "filename": f"page-{page_number}.png",
            "source_width": 100,
            "source_height": 100,
            "processed_width": 100,
            "processed_height": 100,
        },
        "transform": {"scale_x": 1, "scale_y": 1},
        "objects": [
            {
                "page_number": page_number,
                "type": "paragraph",
                "bbox": {"x1": 1, "y1": 2, "x2": 10, "y2": 12},
            }
        ],
        "status": {
            "module": "ocr",
            "state": state,
            "duration_ms": page_number * 10,
            "backend": "mock",
            "warnings": [] if state == "success" else ["page_failed"],
            "error": None,
        },
    }


def test_module_responses_become_multi_page_document_shaped_ui_result():
    result = build_module_document_result(
        [_module_page(2), _module_page(1)],
        document_metadata={"source": "inspection"},
        page_metadata={"doc:p1": {"name": "one"}, "doc:p2": {"name": "two"}},
    )

    assert [page["page_number"] for page in result["pages"]] == [1, 2]
    assert all(set(page["modules"]) == {"ocr"} for page in result["pages"])
    assert result["pages"][0]["page_metadata"] == {"name": "one"}
    assert result["page_count"] == 2
    assert result["object_counts"] == {
        "paragraph": 2,
        "table": 0,
        "figure": 0,
        "stamp": 0,
        "signature": 0,
    }
    assert result["processing"] == {
        "state": "success",
        "duration_ms": 30.0,
        "warnings": [],
    }


def test_module_ui_result_reports_partial_pages_without_claiming_all_modules_ran():
    result = build_module_document_result(
        [_module_page(1), _module_page(2, state="failed")],
        document_metadata={},
    )

    assert result["processing"]["state"] == "partial_success"
    assert success_message(["ocr"], persisted=False) == "OCR succeeded on all pages."
    assert success_message(["ocr"], persisted=False) != "All modules succeeded."


def test_minio_page_descriptor_maps_to_existing_module_request_contract():
    request = build_module_request(
        "doc",
        {
            "image_url": "http://minio/media/documents/doc/images/page.png",
            "page_number": 2,
            "page_id": "doc:p2",
            "metadata": {"minio_object_key": "documents/doc/images/page.png"},
        },
    )

    assert request == {
        "document_id": "doc",
        "image_url": "http://minio/media/documents/doc/images/page.png",
        "page_number": 2,
        "page_id": "doc:p2",
        "page_metadata": {"minio_object_key": "documents/doc/images/page.png"},
    }


def test_streamlit_app_initial_view_renders_without_exceptions():
    app_path = Path(__file__).resolve().parents[2] / "ui/streamlit_app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=10)
    assert not app.exception
    assert app.title[0].value == "Document Extraction Inspector"
    assert len(app.get("file_uploader")) == 1
    assert [button.label for button in app.button[:4]] == [
        "Run All Modules",
        "Run OCR Only",
        "Run Figure/Table Only",
        "Run Stamp/Signature Only",
    ]

    minio_app = app.radio[0].set_value("MinIO").run(timeout=10)
    assert not minio_app.exception
    labels = [button.label for button in minio_app.button]
    assert all(
        label in labels
        for label in [
            "Run All Modules",
            "Run OCR Only",
            "Run Figure/Table Only",
            "Run Stamp/Signature Only",
            "Submit production async job",
        ]
    )
