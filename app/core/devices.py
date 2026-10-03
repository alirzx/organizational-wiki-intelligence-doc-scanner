"""Explicit accelerator validation for optional model runtimes."""

from __future__ import annotations

from typing import Any


def _device_index(device: str, *, prefix: str) -> int | None:
    normalized = device.strip().lower()
    if normalized == "cpu":
        return None
    if normalized == prefix:
        return 0
    expected = f"{prefix}:"
    if not normalized.startswith(expected):
        raise RuntimeError(
            f"Unsupported {prefix} device {device!r}; use 'cpu' or '{prefix}:<index>'"
        )
    try:
        index = int(normalized[len(expected) :])
    except ValueError as exc:
        raise RuntimeError(
            f"Unsupported {prefix} device {device!r}; use 'cpu' or '{prefix}:<index>'"
        ) from exc
    if index < 0:
        raise RuntimeError(f"{prefix} device index must be non-negative")
    return index


def require_paddle_device(device: str, paddle: Any) -> None:
    """Reject unavailable Paddle GPU requests instead of falling back to CPU."""
    index = _device_index(device, prefix="gpu")
    if index is None:
        return
    if not paddle.is_compiled_with_cuda():
        raise RuntimeError(
            f"Paddle device {device!r} was requested, but the installed Paddle package is CPU-only. "
            "Install a CUDA-compatible paddlepaddle-gpu build or configure 'cpu'."
        )
    count = int(paddle.device.cuda.device_count())
    if index >= count:
        raise RuntimeError(
            f"Paddle device {device!r} was requested, but only {count} CUDA device(s) are visible"
        )


def require_torch_device(device: str, torch: Any) -> None:
    """Reject unavailable PyTorch CUDA requests instead of falling back to CPU."""
    index = _device_index(device, prefix="cuda")
    if index is None:
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"PyTorch device {device!r} was requested, but CUDA is unavailable. "
            "Install a CUDA-compatible torch build, expose the GPU, or configure 'cpu'."
        )
    count = int(torch.cuda.device_count())
    if index >= count:
        raise RuntimeError(
            f"PyTorch device {device!r} was requested, but only {count} CUDA device(s) are visible"
        )
