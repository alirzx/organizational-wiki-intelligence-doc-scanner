import sys
import types
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from app.core.config import Settings
from app.modules.ocr.adapter import lines_to_detected_objects
from app.modules.ocr.backend import MockOCRBackend, create_ocr_backend
from app.modules.ocr.bina_rizeh_backend import (
    BINA_RUNTIME_FILES,
    DETECTOR_RUNTIME_FILES,
    BinaRizehOCRBackend,
    logical_persian_text,
    parse_bina_detection_output,
    parse_bina_recognition_output,
)
from app.modules.ocr.paddle_backend import PaddleOCRBackend
from app.modules.ocr.service import OCRService
from app.preprocessing.pipeline import prepare_page


def _settings(**overrides):
    return Settings(
        _env_file=None,
        ocr_backend="bina_rizeh",
        ocr_device="cpu",
        **overrides,
    )


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


def _write_runtime_files(root, relative_names):
    for relative_name in relative_names:
        path = root / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        content = b"weights" * 200 if path.suffix == ".pdiparams" else b"runtime"
        path.write_bytes(content)


def _install_fake_dependencies(monkeypatch, tmp_path, captured, *, polygons=None, recognitions=None):
    recognizer_root = tmp_path / "recognizer"
    detector_root = tmp_path / "detector"
    _write_runtime_files(recognizer_root, BINA_RUNTIME_FILES)
    _write_runtime_files(detector_root, DETECTOR_RUNTIME_FILES)

    hub = types.ModuleType("huggingface_hub")

    def snapshot_download(**kwargs):
        captured.setdefault("downloads", []).append(kwargs)
        if kwargs["repo_id"] == _settings().ocr_bina_model_id:
            return str(recognizer_root)
        return str(detector_root)

    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    paddle = types.ModuleType("paddle")
    paddle.is_compiled_with_cuda = lambda: False
    paddle.device = types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda: 0))
    monkeypatch.setitem(sys.modules, "paddle", paddle)

    paddleocr = types.ModuleType("paddleocr")

    class FakeTextDetection:
        def __init__(self, **kwargs):
            captured["detector_init"] = kwargs

        def predict(self, *, input, batch_size):
            captured.setdefault("detector_calls", []).append((input.copy(), batch_size))
            return [{"res": {"dt_polys": polygons if polygons is not None else []}}]

    class FakeTextRecognition:
        def __init__(self, **kwargs):
            captured["recognizer_init"] = kwargs

        def predict(self, *, input, batch_size):
            captured.setdefault("recognizer_calls", []).append((input, batch_size))
            return [
                {"res": {"rec_text": text, "rec_score": score}}
                for text, score in (recognitions or [])
            ]

    paddleocr.TextDetection = FakeTextDetection
    paddleocr.TextRecognition = FakeTextRecognition
    monkeypatch.setitem(sys.modules, "paddleocr", paddleocr)

    return recognizer_root, detector_root


def test_ocr_backend_factory_selects_only_supported_backends():
    assert isinstance(create_ocr_backend(Settings(_env_file=None, ocr_backend="mock")), MockOCRBackend)
    assert isinstance(
        create_ocr_backend(Settings(_env_file=None, ocr_backend="paddle", ocr_device="cpu")),
        PaddleOCRBackend,
    )
    assert isinstance(create_ocr_backend(_settings()), BinaRizehOCRBackend)


def test_bina_backend_can_be_selected_from_environment(monkeypatch):
    monkeypatch.setenv("WIKI_HAMI_OCR_BACKEND", "BINA_RIZEH")
    settings = Settings(_env_file=None)

    assert settings.ocr_backend == "bina_rizeh"
    assert settings.ocr_bina_model_id == "Reza2kn/Bina-0.2-RizehPizeh"
    assert settings.ocr_bina_revision == "993527413ff74ef6d446df91c715a4e0825abe5b"
    assert settings.ocr_detection_model_id == "PaddlePaddle/PP-OCRv6_medium_det"
    assert settings.ocr_detection_model_revision == "8e0f56fb2ef86b461d99cfc7ac5c137738985f61"


def test_ocr_backend_configuration_rejects_unknown_backend():
    with pytest.raises(ValueError, match="ocr_backend must be one of"):
        Settings(_env_file=None, ocr_backend="arbitrary_module")


def test_official_bina_line_output_preserves_logical_and_raw_text():
    assert logical_persian_text("مالسOpenAI123") == "OpenAI123سلام"
    raw_text, score = parse_bina_recognition_output(
        {"rec_text": "مالس", "rec_score": 0.91}, index=0
    )
    assert raw_text == "مالس"
    assert score == 0.91


def test_bina_empty_and_malformed_outputs_are_handled_safely():
    assert parse_bina_detection_output(
        {"dt_polys": []}, image_width=100, image_height=100
    ) == []
    with pytest.raises(RuntimeError, match="dt_polys"):
        parse_bina_detection_output({}, image_width=100, image_height=100)
    with pytest.raises(RuntimeError, match="four finite points"):
        parse_bina_detection_output(
            {"dt_polys": [[[1, 2], [3, 4], [5, 6]]]},
            image_width=100,
            image_height=100,
        )
    with pytest.raises(RuntimeError, match="rec_score"):
        parse_bina_recognition_output({"rec_text": "مالس"}, index=0)


def test_bina_loads_once_downloads_only_runtime_files_and_reuses_models(
    monkeypatch, tmp_path
):
    captured: dict = {}
    polygons = [
        [[25, 40], [225, 40], [225, 80], [25, 80]],
        [[260, 20], [285, 20], [285, 120], [260, 120]],
    ]
    _install_fake_dependencies(
        monkeypatch,
        tmp_path,
        captured,
        polygons=polygons,
        recognitions=[("مود", 0.88), ("مالس", 0.93)],
    )
    backend = BinaRizehOCRBackend(_settings())
    image = Image.new("RGB", (320, 180), "white")

    first = backend.predict(image)
    second = backend.predict(image)

    assert [line.text for line in first] == ["دوم", "سلام"]
    assert [line.raw_text for line in first] == ["مود", "مالس"]
    assert [line.text for line in second] == ["دوم", "سلام"]
    assert len(captured["downloads"]) == 2
    assert captured["downloads"][0]["allow_patterns"] == list(BINA_RUNTIME_FILES)
    assert captured["downloads"][1]["allow_patterns"] == list(DETECTOR_RUNTIME_FILES)
    assert captured["detector_init"] == {
        "model_dir": str(tmp_path / "detector"),
        "device": "cpu",
        "enable_mkldnn": False,
    }
    assert captured["recognizer_init"] == {
        "model_dir": str(tmp_path / "recognizer" / "inference"),
        "device": "cpu",
        "enable_mkldnn": False,
    }
    assert len(captured["detector_calls"]) == 2
    first_crops, batch_size = captured["recognizer_calls"][0]
    assert batch_size == 1
    assert len(first_crops) == 2
    assert first_crops[0].shape[1] > first_crops[0].shape[0]


def test_bina_empty_detection_skips_recognition(monkeypatch, tmp_path):
    captured: dict = {}
    _install_fake_dependencies(monkeypatch, tmp_path, captured)
    backend = BinaRizehOCRBackend(_settings())

    assert backend.predict(Image.new("RGB", (100, 100), "white")) == []
    assert "recognizer_calls" not in captured


def test_bina_rejects_git_lfs_pointer_artifacts(monkeypatch, tmp_path):
    captured: dict = {}
    recognizer_root, _ = _install_fake_dependencies(monkeypatch, tmp_path, captured)
    (recognizer_root / "inference" / "inference.pdiparams").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
    )
    backend = BinaRizehOCRBackend(_settings())

    with pytest.raises(RuntimeError, match="Git LFS pointer"):
        backend.predict(Image.new("RGB", (100, 100), "white"))


def test_bina_lines_use_existing_adapter_grouping_source_geometry_and_provenance(
    monkeypatch, tmp_path
):
    captured: dict = {}
    _install_fake_dependencies(
        monkeypatch,
        tmp_path,
        captured,
        polygons=[[[50, 100], [450, 100], [450, 140], [50, 140]]],
        recognitions=[("مالس", 0.9)],
    )
    page = _prepared_page()
    backend = BinaRizehOCRBackend(_settings())
    lines = backend.predict(page.processed_image)

    objects = lines_to_detected_objects(
        lines,
        page=page,
        settings=backend.settings,
        backend_metadata=backend.metadata,
    )

    assert len(objects) == 1
    obj = objects[0]
    assert obj.text == "سلام"
    assert obj.raw_text == "مالس"
    assert obj.bbox.model_dump() == {"x1": 100.0, "y1": 200.0, "x2": 900.0, "y2": 280.0}
    assert obj.polygon is not None
    assert obj.provenance.backend == "bina_rizeh"
    assert obj.provenance.model_id == "Reza2kn/Bina-0.2-RizehPizeh"
    assert obj.provenance.model_version == "993527413ff74ef6d446df91c715a4e0825abe5b"
    assert obj.metadata["text_detection_model"] == "PaddlePaddle/PP-OCRv6_medium_det"
    assert obj.metadata["text_recognition_model"] == "Reza2kn/Bina-0.2-RizehPizeh"


@pytest.mark.asyncio
async def test_bina_service_status_uses_selected_backend_metadata(monkeypatch, tmp_path):
    captured: dict = {}
    _install_fake_dependencies(
        monkeypatch,
        tmp_path,
        captured,
        polygons=[[[10, 20], [100, 20], [100, 50], [10, 50]]],
        recognitions=[("مالس", 0.9)],
    )
    service = OCRService(_settings())

    async def predict_inline(func, *args):
        return func(*args)

    monkeypatch.setattr("app.modules.ocr.service.asyncio.to_thread", predict_inline)
    response = await service.run(_prepared_page(), "request-1")

    assert response.status.backend == "bina_rizeh"
    assert response.status.model_id == "Reza2kn/Bina-0.2-RizehPizeh"
    assert response.objects[0].provenance.backend == "bina_rizeh"
