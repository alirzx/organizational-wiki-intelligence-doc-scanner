import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
from streamlit.testing.v1 import AppTest

from ui.artifacts import build_run_zip, canonical_json_bytes
from ui.module_runs import success_message
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


def test_single_module_success_does_not_claim_all_modules_ran():
    assert success_message(["ocr"], persisted=False) == "OCR succeeded on all pages."
    assert success_message(["ocr"], persisted=False) != "All modules succeeded."


def test_streamlit_app_initial_view_renders_without_exceptions():
    app_path = Path(__file__).resolve().parents[2] / "ui/streamlit_app.py"
    source = app_path.read_text(encoding="utf-8")
    assert "asyncio.run" not in source
    assert "get_ocr_service" not in source
    assert "get_figure_table_service" not in source
    assert "get_stamp_signature_service" not in source
    assert 'api_post("/extract"' in source
    assert '"/extract/minio/inspect"' in source

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
