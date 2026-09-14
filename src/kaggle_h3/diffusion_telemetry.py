"""Opt-in numerical telemetry for the H3 diffusion sampler.

The sampler remains unchanged when telemetry is disabled.  When enabled, each
record deliberately synchronizes the tensor's CUDA device before collecting
reductions; this makes the reported values trustworthy but also means that a
telemetry run is not a production-timing run.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import uuid
from typing import Any


def diffusion_telemetry_enabled() -> bool:
    """Return whether per-step diffusion statistics are enabled."""

    value = os.environ.get("KAGGLE_H3_DIFFUSION_TELEMETRY", "0").strip().lower()
    return value in {"1", "true", "yes", "on", "enabled", "full", "steps"}


def _iter_tensors(value: Any, path: str, seen: set[int]) -> Iterator[tuple[str, Any]]:
    """Yield ordinary and H3 NestedTensor leaves without copying them."""

    try:
        import torch  # type: ignore
    except ImportError:
        return

    if isinstance(value, torch.Tensor):
        if bool(getattr(value, "is_nested", False)):
            try:
                streams = tuple(value.unbind())
            except Exception:
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
            yield from _iter_tensors(item, f"{path}[{key!r}]", seen)
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _iter_tensors(item, f"{path}[{index}]", seen)


def _scalar(value: Any) -> float | int | None:
    try:
        item = value.item()
    except AttributeError:
        item = value
    try:
        return float(item)
    except (TypeError, ValueError):
        return None


def _tensor_statistics(tensor: Any) -> dict[str, Any]:
    """Collect raw floating-point statistics from one tensor."""

    import torch  # type: ignore

    if tensor.numel() == 0:
        return {
            "present": True,
            "shape": list(tensor.shape),
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "finite_count": 0,
            "numel": 0,
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
        }
    if not (tensor.is_floating_point() or tensor.is_complex()):
        return {}

    if bool(getattr(tensor, "is_cuda", False)):
        torch.cuda.synchronize(tensor.device)
    with torch.no_grad():
        finite_count = int(torch.isfinite(tensor).sum().item())
        # H3 activations are real. For completeness, complex telemetry reports
        # magnitude statistics because min/max are undefined for complex data.
        values = tensor.abs() if tensor.is_complex() else tensor
        minimum = _scalar(values.min())
        maximum = _scalar(values.max())
        mean = _scalar(values.mean())
        std = _scalar(values.float().std(unbiased=False))
    return {
        "present": True,
        "shape": list(tensor.shape),
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "std": std,
        "finite_count": finite_count,
        "numel": int(tensor.numel()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def _tensor_checksum(tensor: Any) -> str:
    """Return a SHA-256 checksum of a tensor's raw values.

    The tensor is never modified.  A contiguous view is used only for the
    byte serialization, and CUDA values are copied to CPU solely for hashing.
    This is intentionally part of opt-in diagnostics because the copy is
    expensive and synchronizes the producing CUDA stream.
    """

    import torch  # type: ignore

    if bool(getattr(tensor, "is_cuda", False)):
        torch.cuda.synchronize(tensor.device)
    with torch.no_grad():
        raw = tensor.detach().contiguous()
        if raw.device.type != "cpu":
            raw = raw.cpu()
        # Viewing as bytes works for float, bfloat16, float8, complex, integer,
        # and boolean tensors without casting their values or changing dtype.
        raw_bytes = raw.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw_bytes).hexdigest()


class H3DiffusionTelemetry:
    """Emit one structured record for each sampler boundary tensor."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        output_path: str | Path | None = None,
    ) -> None:
        self.enabled = diffusion_telemetry_enabled() if enabled is None else bool(enabled)
        configured_path = output_path or os.environ.get(
            "KAGGLE_H3_DIFFUSION_TELEMETRY_PATH", ""
        ).strip()
        self.output_path = Path(configured_path).expanduser() if configured_path else None
        self.generation_id = uuid.uuid4().hex[:12]
        self.records: list[dict[str, Any]] = []
        self._lock = threading.RLock()

    def _write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True, allow_nan=True)
        print(f"[Kaggle H3][diffusion] {line}", flush=True)
        if self.output_path is None:
            return
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception as exc:
            print(
                f"[Kaggle H3][diffusion] WARNING: could not write telemetry to "
                f"{self.output_path}: {exc}",
                flush=True,
            )

    def record_boundary(
        self,
        *,
        step_index: int,
        sigma: Any,
        timestep: Any = None,
        boundary: str,
        value: Any,
        tensor_name: str,
        alias_of: str | None = None,
    ) -> None:
        """Record all numerical tensor leaves at one named sampler boundary."""

        self._record_value(
            step_index=int(step_index),
            stage=f"diffusion_step_{int(step_index)}_{boundary}",
            sigma=sigma,
            timestep=sigma if timestep is None else timestep,
            value=value,
            tensor_name=tensor_name,
            alias_of=alias_of,
        )

    def record_phase(
        self,
        *,
        stage: str,
        value: Any,
        tensor_name: str,
    ) -> None:
        """Record a non-step boundary such as a VAE input or output."""

        self._record_value(
            step_index=None,
            stage=str(stage),
            sigma=None,
            timestep=None,
            value=value,
            tensor_name=tensor_name,
        )

    def _record_value(
        self,
        *,
        step_index: int | None,
        stage: str,
        sigma: Any,
        timestep: Any,
        value: Any,
        tensor_name: str,
        alias_of: str | None = None,
    ) -> None:
        """Serialize tensor statistics and checksums for one boundary."""

        if not self.enabled:
            return
        sigma_value = _scalar(sigma)
        timestep_value = _scalar(timestep)
        found = False
        with self._lock:
            for path, tensor in _iter_tensors(value, tensor_name, set()):
                found = True
                try:
                    statistics = _tensor_statistics(tensor)
                except Exception as exc:
                    statistics = {
                        "present": True,
                        "statistics_error": str(exc),
                        "dtype": str(getattr(tensor, "dtype", None)),
                        "device": str(getattr(tensor, "device", None)),
                    }
                if not statistics:
                    continue
                try:
                    checksum = _tensor_checksum(tensor)
                    checksum_error = None
                except Exception as exc:
                    checksum = None
                    checksum_error = str(exc)
                record = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "generation_id": self.generation_id,
                    "step_index": step_index,
                    "stage": stage,
                    "sigma": sigma_value,
                    "timestep": timestep_value,
                    "tensor": path,
                    "checksum_algorithm": "sha256_raw_contiguous",
                    "checksum": checksum,
                    **statistics,
                }
                if checksum_error is not None:
                    record["checksum_error"] = checksum_error
                if alias_of is not None:
                    record["alias_of"] = alias_of
                self.records.append(record)
                self._write(record)
            if not found:
                record = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "generation_id": self.generation_id,
                    "step_index": step_index,
                    "stage": stage,
                    "sigma": sigma_value,
                    "timestep": timestep_value,
                    "tensor": tensor_name,
                    "checksum_algorithm": "sha256_raw_contiguous",
                    "checksum": None,
                    "present": False,
                    "min": None,
                    "max": None,
                    "mean": None,
                    "std": None,
                    "finite_count": None,
                    "numel": None,
                    "dtype": None,
                    "device": None,
                }
                if alias_of is not None:
                    record["alias_of"] = alias_of
                self.records.append(record)
                self._write(record)

    def summary(self) -> dict[str, Any]:
        """Return the records for runtime manifests and workflow inspection."""

        return {
            "enabled": self.enabled,
            "generation_id": self.generation_id,
            "record_count": len(self.records),
            "output_path": str(self.output_path) if self.output_path else None,
            "records": list(self.records),
        }


__all__ = ["H3DiffusionTelemetry", "diffusion_telemetry_enabled"]
