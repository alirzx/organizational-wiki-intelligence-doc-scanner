import types

import pytest

from app.core.config import Settings
from app.core.devices import require_paddle_device, require_torch_device


def test_model_device_defaults_use_framework_specific_gpu_syntax():
    settings = Settings(_env_file=None)

    assert settings.ocr_device == "gpu:0"
    assert settings.figure_table_device == "gpu:0"
    assert settings.stamp_signature_device == "cuda:0"


def test_all_model_devices_allow_explicit_cpu_overrides():
    settings = Settings(
        _env_file=None,
        ocr_device="cpu",
        figure_table_device="cpu",
        stamp_signature_device="cpu",
    )
    paddle = types.SimpleNamespace()
    torch = types.SimpleNamespace()

    require_paddle_device(settings.ocr_device, paddle)
    require_paddle_device(settings.figure_table_device, paddle)
    require_torch_device(settings.stamp_signature_device, torch)


def test_paddle_gpu_request_rejects_cpu_only_package():
    paddle = types.SimpleNamespace(
        is_compiled_with_cuda=lambda: False,
        device=types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda: 0)),
    )

    with pytest.raises(RuntimeError, match="CPU-only"):
        require_paddle_device("gpu:0", paddle)


def test_torch_cuda_request_rejects_unavailable_runtime():
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    )

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        require_torch_device("cuda:0", torch)


def test_device_index_must_be_visible():
    paddle = types.SimpleNamespace(
        is_compiled_with_cuda=lambda: True,
        device=types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda: 1)),
    )
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
    )

    with pytest.raises(RuntimeError, match="only 1 CUDA device"):
        require_paddle_device("gpu:1", paddle)
    with pytest.raises(RuntimeError, match="only 1 CUDA device"):
        require_torch_device("cuda:1", torch)
