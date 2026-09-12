"""Fail-fast floating-point diagnostics for the H3 execution pipeline.

The validator is deliberately dependency-light at import time.  PyTorch is
loaded only when a value is checked so the rest of the package remains usable
for planning and workflow generation without a CUDA runtime.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def _tensor_path(path: str, suffix: str) -> str:
    return suffix if not path else f"{path}{suffix}"


def _iter_tensors(value: Any, path: str, seen: set[int]) -> Iterator[tuple[str, Any]]:
    """Yield tensors from ordinary containers and H3 NestedTensor values."""

    try:
        import torch  # type: ignore
    except ImportError:
        return

    if isinstance(value, torch.Tensor):
        if bool(getattr(value, "is_nested", False)):
            try:
                streams = tuple(value.unbind())
            except Exception:
                # A NestedTensor must expose unbind for H3.  Yielding it lets
                # the caller report any supported tensor fields without
                # converting or copying the value here.
                yield path or "tensor", value
                return
            for index, stream in enumerate(streams):
                yield from _iter_tensors(stream, f"{path}[{index}]", seen)
            return
        yield path or "tensor", value
        return

    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return

    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_tensors(item, _tensor_path(path, f"[{key!r}]"), seen)
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _iter_tensors(item, _tensor_path(path, f"[{index}]"), seen)


def _complex_inf_masks(tensor: Any) -> tuple[Any, Any]:
    """Return element-level positive/negative infinity masks for complex data."""

    import torch  # type: ignore

    real = tensor.real
    imag = tensor.imag
    positive = (torch.isinf(real) & (real > 0)) | (torch.isinf(imag) & (imag > 0))
    negative = (torch.isinf(real) & (real < 0)) | (torch.isinf(imag) & (imag < 0))
    return positive, negative


def _synchronize_if_needed(tensor: Any) -> None:
    import torch  # type: ignore

    if bool(getattr(tensor, "is_cuda", False)):
        torch.cuda.synchronize(tensor.device)


def _tensor_counts(tensor: Any) -> tuple[int, int, int]:
    import torch  # type: ignore

    if tensor.numel() == 0:
        return 0, 0, 0
    _synchronize_if_needed(tensor)
    with torch.no_grad():
        finite_mask = torch.isfinite(tensor)
        if bool(finite_mask.all()):
            return 0, 0, 0
        nan_mask = torch.isnan(tensor)
        if bool(tensor.is_complex()):
            positive_inf_mask, negative_inf_mask = _complex_inf_masks(tensor)
        else:
            positive_inf_mask = torch.isposinf(tensor)
            negative_inf_mask = torch.isneginf(tensor)
        return (
            int(nan_mask.sum().item()),
            int(positive_inf_mask.sum().item()),
            int(negative_inf_mask.sum().item()),
        )


def validate_finite(value: Any, stage_name: str, *, tensor_name: str = "root") -> Any:
    """Raise ``FloatingPointError`` on the first non-finite tensor.

    Integer, boolean, string, metadata, and empty tensors are intentionally
    ignored.  The original value is returned unchanged for convenient use at
    existing boundaries.
    """

    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("H3 floating-point diagnostics require PyTorch") from exc

    for path, tensor in _iter_tensors(value, tensor_name, set()):
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.numel() == 0 or not (tensor.is_floating_point() or tensor.is_complex()):
            continue
        nan_count, positive_inf_count, negative_inf_count = _tensor_counts(tensor)
        if nan_count or positive_inf_count or negative_inf_count:
            raise FloatingPointError(
                f"FloatingPointError: stage {stage_name!r} encountered non-finite values\n"
                f"(nan={nan_count}, +inf={positive_inf_count}, "
                f"-inf={negative_inf_count}, tensor={path}, "
                f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
                f"device={tensor.device})"
            )
    return value


__all__ = ["validate_finite"]
