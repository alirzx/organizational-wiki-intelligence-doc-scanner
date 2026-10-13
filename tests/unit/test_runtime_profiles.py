import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_model_runtime_defaults_to_gpu():
    assert Settings(_env_file=None).model_runtime == "gpu"


def test_cpu_runtime_accepts_active_cpu_devices():
    settings = Settings(
        _env_file=None,
        model_runtime="cpu",
        text_extraction_mode="ocr",
        ocr_backend="paddle",
        ocr_device="cpu",
        figure_table_backend="pp_doclayout",
        figure_table_device="cpu",
        stamp_signature_backend="rfdetr",
        stamp_signature_device="cpu",
    )
    assert settings.model_runtime == "cpu"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ocr_device", "gpu:0"),
        ("figure_table_device", "gpu:0"),
        ("stamp_signature_device", "cuda:0"),
    ],
)
def test_cpu_runtime_rejects_active_gpu_device(field, value):
    kwargs = {
        "model_runtime": "cpu",
        "text_extraction_mode": "ocr",
        "ocr_backend": "paddle",
        "ocr_device": "cpu",
        "figure_table_backend": "pp_doclayout",
        "figure_table_device": "cpu",
        "stamp_signature_backend": "rfdetr",
        "stamp_signature_device": "cpu",
    }
    kwargs[field] = value
    with pytest.raises(ValidationError, match="MODEL_RUNTIME=cpu"):
        Settings(_env_file=None, **kwargs)


def test_cpu_vlm_runtime_rejects_gpu_paddle_device_used_by_hybrid_verifier():
    with pytest.raises(ValidationError, match="WIKI_HAMI_OCR_DEVICE"):
        Settings(
            _env_file=None,
            model_runtime="cpu",
            text_extraction_mode="vlm",
            ocr_device="gpu:0",
            figure_table_backend="mock",
            stamp_signature_backend="mock",
        )
