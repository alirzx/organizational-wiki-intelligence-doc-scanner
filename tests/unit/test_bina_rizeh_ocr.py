import sys
import types
from io import BytesIO

import pytest
from PIL import Image

from app.core.config import Settings
from app.modules.ocr.adapter import lines_to_detected_objects
from app.modules.ocr.backend import MockOCRBackend, create_ocr_backend
from app.modules.ocr.bina_rizeh_backend import (
    BINA_DETECTOR_ID,
    BinaRizehOCRBackend,
    parse_bina_page_output,
)
from app.modules.ocr.paddle_backend import PaddleOCRBackend
from app.modules.ocr.service import OCRService
from app.preprocessing.pipeline import prepare_page


def _prepared_page():
    image = Image.new("RGB", (1000, 1000), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return prepare_page(
        data=buffer.getvalue(),
        filename="page.png",
        mime_type="image/png",
        document_id="doc1",
        page_id="doc1:p1",
        page_number=1,
        page_metadata={},
        settings=Settings(_env_file=None, preprocess_max_long_edge=500),
    )


def test_ocr_backend_factory_selects_only_supported_backends():
    assert isinstance(create_ocr_backend(Settings(_env_file=None, ocr_backend="mock")), MockOCRBackend)
    assert isinstance(
        create_ocr_backend(Settings(_env_file=None, ocr_backend="paddle")), PaddleOCRBackend
    )
    assert isinstance(
        create_ocr_backend(Settings(_env_file=None, ocr_backend="bina_rizeh")),
        BinaRizehOCRBackend,
    )


def test_bina_backend_can_be_selected_from_environment(monkeypatch):
    monkeypatch.setenv("WIKI_HAMI_OCR_BACKEND", "BINA_RIZEH")
    settings = Settings(_env_file=None)

    assert settings.ocr_backend == "bina_rizeh"
    assert settings.ocr_bina_model_id == "Reza2kn/Bina-0.2-Rizeh"
    assert settings.ocr_bina_revision == "4e8cf8806c08442276dcb5ed4a112329945a9bbe"


def test_ocr_backend_configuration_rejects_unknown_backend():
    with pytest.raises(ValueError, match="ocr_backend must be one of"):
        Settings(_env_file=None, ocr_backend="arbitrary_module")


def test_bina_official_page_output_preserves_logical_and_raw_line_text():
    lines = parse_bina_page_output(
        {
            "input_path": "page.png",
            "page_index": 0,
            "text": "سلام",
            "lines": [
                {
                    "text": "سلام",
                    "score": 0.91,
                    "raw_visual_text": "ملاس",
                    "box": [12.0, 20.0, 140.0, 48.0],
                    "row": 0,
                }
            ],
        }
    )

    assert len(lines) == 1
    assert lines[0].text == "سلام"
    assert lines[0].raw_text == "ملاس"
    assert lines[0].confidence == 0.91
    assert lines[0].bbox.model_dump() == {"x1": 12.0, "y1": 20.0, "x2": 140.0, "y2": 48.0}
    assert lines[0].polygon is None


def test_bina_empty_and_malformed_page_output_are_handled_safely():
    assert parse_bina_page_output({"lines": []}) == []
    with pytest.raises(RuntimeError, match="raw_visual_text"):
        parse_bina_page_output(
            {"lines": [{"text": "سلام", "score": 0.8, "box": [1, 2, 3, 4]}]}
        )
    with pytest.raises(RuntimeError, match="positive area"):
        parse_bina_page_output(
            {
                "lines": [
                    {
                        "text": "سلام",
                        "raw_visual_text": "ملاس",
                        "score": 0.8,
                        "box": [3, 2, 1, 4],
                    }
                ]
            }
        )


def _install_fake_bina_dependencies(monkeypatch, tmp_path, captured):
    model_root = tmp_path / "bina"
    (model_root / "inference").mkdir(parents=True)
    (model_root / "detector").mkdir()

    hub = types.ModuleType("huggingface_hub")

    def snapshot_download(**kwargs):
        captured.setdefault("downloads", []).append(kwargs)
        return str(model_root)

    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    paddleocr = types.ModuleType("paddleocr")

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            captured.setdefault("initializations", []).append(kwargs)

        def predict(self, image):
            captured["input_size"] = tuple(image.shape[:2])
            return [
                {
                    "res": {
                        "rec_texts": ["مالس"],
                        "rec_scores": [0.93],
                        "rec_boxes": [[25, 40, 225, 80]],
                    }
                }
            ]

    paddleocr.PaddleOCR = FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", paddleocr)


def test_bina_loads_once_uses_official_page_pipeline_and_reuses_model(monkeypatch, tmp_path):
    captured: dict = {}
    _install_fake_bina_dependencies(monkeypatch, tmp_path, captured)
    backend = BinaRizehOCRBackend(Settings(_env_file=None, ocr_backend="bina_rizeh"))
    image = Image.new("RGB", (320, 180), "white")

    first = backend.predict(image)
    second = backend.predict(image)

    assert [line.text for line in first] == ["سلام"]
    assert [line.raw_text for line in first] == ["مالس"]
    assert [line.text for line in second] == ["سلام"]
    assert len(captured["downloads"]) == 1
    assert captured["downloads"][0]["allow_patterns"] == ["inference/*", "detector/*"]
    assert captured["initializations"] == [
        {
            "text_detection_model_dir": str(tmp_path / "bina" / "detector"),
            "text_recognition_model_dir": str(tmp_path / "bina" / "inference"),
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "text_rec_score_thresh": 0.0,
            "device": "cpu",
        }
    ]
    assert captured["input_size"] == (180, 320)


def test_bina_lines_use_existing_adapter_grouping_source_geometry_and_provenance():
    page = _prepared_page()
    backend = BinaRizehOCRBackend(Settings(_env_file=None, ocr_backend="bina_rizeh"))
    lines = parse_bina_page_output(
        {
            "lines": [
                {
                    "text": "سلام",
                    "raw_visual_text": "ملاس",
                    "score": 0.9,
                    "box": [50, 100, 450, 140],
                    "row": 0,
                }
            ]
        }
    )

    objects = lines_to_detected_objects(
        lines,
        page=page,
        settings=backend.settings,
        backend_metadata=backend.metadata,
    )

    assert len(objects) == 1
    obj = objects[0]
    assert obj.text == "سلام"
    assert obj.raw_text == "ملاس"
    assert obj.bbox.model_dump() == {"x1": 100.0, "y1": 200.0, "x2": 900.0, "y2": 280.0}
    assert obj.polygon is not None
    assert obj.provenance.backend == "bina_rizeh"
    assert obj.provenance.model_id == "Reza2kn/Bina-0.2-Rizeh"
    assert obj.provenance.model_version == "4e8cf8806c08442276dcb5ed4a112329945a9bbe"
    assert obj.metadata["text_detection_model"] == BINA_DETECTOR_ID
    assert obj.metadata["text_recognition_model"] == "Reza2kn/Bina-0.2-Rizeh"


@pytest.mark.asyncio
async def test_bina_service_status_uses_selected_backend_metadata(monkeypatch):
    service = OCRService(Settings(_env_file=None, ocr_backend="bina_rizeh"))
    monkeypatch.setattr(
        service._backend,
        "predict",
        lambda image: parse_bina_page_output(
            {
                "lines": [
                    {
                        "text": "سلام",
                        "raw_visual_text": "مالس",
                        "score": 0.9,
                        "box": [10, 20, 100, 50],
                    }
                ]
            }
        ),
    )

    async def predict_inline(func, *args):
        return func(*args)

    monkeypatch.setattr("app.modules.ocr.service.asyncio.to_thread", predict_inline)
    response = await service.run(_prepared_page(), "request-1")

    assert response.status.backend == "bina_rizeh"
    assert response.status.model_id == "Reza2kn/Bina-0.2-Rizeh"
    assert response.objects[0].provenance.backend == "bina_rizeh"
