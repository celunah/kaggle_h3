"""Phase-aware execution helpers for native ComfyUI H3 workflows.

The regular ComfyUI model loader has one global load device.  That is useful
for the explicitly documented one-GPU fallback, but it does not implement the
two-phase layout used by this project.  This module is deliberately small and
lazy: it only imports Torch, Accelerate, and ComfyUI when a real generation is
about to start.

The phase contract is:

* the Qwen language layers are dispatched contiguously across every requested
  GPU during conditioning, then return to CPU before the transformer phase;
* the H3 transformer is dispatched as intact blocks across every requested
  GPU, with CPU and optionally /kaggle/tmp as overflow tiers;
* the transformer is released after denoising;
* audio VAE decode targets GPU0 and video VAE decode targets GPU1.

The native path fails loudly when two GPUs are available but this dispatch
cannot be verified. Set ``KAGGLE_H3_PHASE_SHARDING=off`` only when the
explicit one-GPU fallback is intentional.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import gc
import importlib.util
import inspect
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import weakref
from statistics import median
from typing import Any


try:
    from .fp_diagnostics import finite_diagnostics_enabled, validate_finite
except ImportError:
    try:
        from kaggle_h3_fp_diagnostics import (  # type: ignore
            finite_diagnostics_enabled,
            validate_finite,
        )
    except ImportError:
        _diagnostics_path = Path(__file__).with_name("kaggle_h3_fp_diagnostics.py")
        _diagnostics_spec = importlib.util.spec_from_file_location(
            "kaggle_h3_fp_diagnostics", _diagnostics_path
        )
        if (
            _diagnostics_spec is None
            or _diagnostics_spec.loader is None
            or not _diagnostics_path.is_file()
        ):
            raise ImportError(f"Cannot load H3 floating-point diagnostics: {_diagnostics_path}")
        _diagnostics_module = importlib.util.module_from_spec(_diagnostics_spec)
        sys.modules[_diagnostics_spec.name] = _diagnostics_module
        _diagnostics_spec.loader.exec_module(_diagnostics_module)
        finite_diagnostics_enabled = _diagnostics_module.finite_diagnostics_enabled
        validate_finite = _diagnostics_module.validate_finite


class H3PhaseError(RuntimeError):
    """Raised when the requested H3 execution phase cannot be verified."""


H3_SYNCHRONIZATION_MODES = ("full", "safe", "fast")


def normalize_h3_synchronization_mode(mode: Any) -> str:
    """Validate the user-selectable synchronization policy."""

    normalized = str(mode or "full").strip().lower()
    if normalized not in H3_SYNCHRONIZATION_MODES:
        choices = ", ".join(H3_SYNCHRONIZATION_MODES)
        raise H3PhaseError(
            f"Unsupported H3 synchronization mode {mode!r}; select one of: {choices}."
        )
    return normalized


def h3_synchronization_plan(mode: Any = "full") -> dict[str, Any]:
    """Describe the barriers enabled by one synchronization policy."""

    normalized = normalize_h3_synchronization_mode(mode)
    if normalized == "full":
        return {
            "mode": normalized,
            "sampler_boundary": "all_requested_gpus_after_each_model_call",
            "block_outputs": "every_dispatchable_block",
            "activation_transfers": "source_and_destination_device_barriers",
            "quantized_weight_copies": "synchronous_with_source_stream_barriers",
            "phase_cleanup": "all_requested_gpus",
        }
    if normalized == "safe":
        return {
            "mode": normalized,
            "sampler_boundary": "primary_gpu_after_each_model_call",
            "block_outputs": "every_dispatchable_block",
            "activation_transfers": "source_and_destination_device_barriers",
            "quantized_weight_copies": "synchronous_with_source_stream_barriers",
            "phase_cleanup": "all_requested_gpus",
        }
    return {
        "mode": normalized,
        "sampler_boundary": "explicit_block_and_final_handoff_barriers",
        "block_outputs": "device_transitions_and_final_layer_only",
        "activation_transfers": "cross_device_source_and_destination_barriers",
        "quantized_weight_copies": "synchronous_with_source_stream_barriers",
        "phase_cleanup": "all_requested_gpus",
    }


@dataclass
class H3TransformerPhase:
    """State owned by one denoising phase."""

    model_patcher: Any = None
    transformer: Any = None
    original_transformer: Any = None
    transformer_path: str = ""
    device_ids: tuple[int, ...] = ()
    report: dict[str, Any] = field(default_factory=dict)
    active: bool = False
    released: bool = False
    monitor: Any = None
    activity_handles: list[Any] = field(default_factory=list)
    dispatch_handles: list[Any] = field(default_factory=list)
    activity: dict[str, Any] = field(default_factory=dict)
    synchronization_mode: str = "full"
    synchronization_mode_ref: dict[str, str] | None = None


@dataclass
class H3TextEncoderPhase:
    """State owned by the temporary two-GPU Qwen conditioning phase."""

    clip: Any = None
    patcher: Any = None
    encoder: Any = None
    device_ids: tuple[int, ...] = ()
    report: dict[str, Any] = field(default_factory=dict)
    dispatch_handles: list[Any] = field(default_factory=list)
    active: bool = False
    released: bool = False


_ACTIVE_TEXT_ENCODER_PHASES: list[H3TextEncoderPhase] = []


@dataclass(frozen=True)
class _H3VAEPhaseRegistration:
    """Weak registration for a VAE that may need interruption cleanup."""

    reference: Any
    role: str
    device_id: int


_ACTIVE_VAE_PHASES: dict[int, _H3VAEPhaseRegistration] = {}
_VAE_PHASE_LOCK = threading.RLock()


def synchronize_h3_devices(device_ids: Any, *, reason: str = "") -> list[str]:
    """Synchronize all requested CUDA devices at an H3 phase boundary."""

    try:
        import torch  # type: ignore
    except Exception:
        return []
    if isinstance(device_ids, (str, bytes, int)):
        device_ids = (device_ids,)
    synchronized: list[str] = []
    for device_id in device_ids or ():
        try:
            device = torch.device(device_id)
        except (TypeError, RuntimeError, ValueError):
            continue
        if device.type != "cuda":
            continue
        torch.cuda.synchronize(device)
        synchronized.append(str(device))
    return synchronized


@contextlib.contextmanager
def h3_rmsnorm_dtype_alignment():
    """Align H3 RMSNorm weights to FP32 activations for the sampler scope.

    ComfyUI's sampler carries H3 residual activations in FP32 while some
    checkpoint normalization weights remain BF16.  PyTorch then skips its
    fused RMSNorm implementation.  The H3 sampler is the only caller of this
    context, so the temporary ``torch.rms_norm`` wrapper cannot affect other
    workflows after the sampler returns.  Only the small normalization weight
    is converted; quantized linear weights and activations are untouched.

    Set ``KAGGLE_H3_RMSNORM_DTYPE_ALIGNMENT=0`` to disable this compatibility
    path for an A/B benchmark.
    """

    report: dict[str, Any] = {
        "enabled": False,
        "mode": "disabled",
        "aligned_calls": 0,
    }
    try:
        import torch  # type: ignore
    except Exception:
        yield report
        return

    configured = os.environ.get("KAGGLE_H3_RMSNORM_DTYPE_ALIGNMENT", "1").strip().lower()
    if configured in {"0", "false", "no", "off", "disabled"}:
        yield report
        return

    original = getattr(torch, "rms_norm", None)
    if not callable(original):
        report["mode"] = "unavailable"
        yield report
        return

    def aligned_rms_norm(input_tensor, normalized_shape, weight=None, eps=1e-5, *args, **kwargs):
        if (
            torch.is_tensor(input_tensor)
            and input_tensor.is_floating_point()
            and torch.is_tensor(weight)
            and weight.is_floating_point()
            and weight.dtype != input_tensor.dtype
        ):
            weight = weight.to(device=input_tensor.device, dtype=input_tensor.dtype)
            report["aligned_calls"] += 1
        return original(input_tensor, normalized_shape, weight, eps, *args, **kwargs)

    setattr(torch, "rms_norm", aligned_rms_norm)
    report["enabled"] = True
    report["mode"] = "weight_to_activation_dtype"
    try:
        yield report
    finally:
        if getattr(torch, "rms_norm", None) is aligned_rms_norm:
            setattr(torch, "rms_norm", original)


def register_vae_phase(vae: Any, *, device_id: int, role: str) -> None:
    """Track a VAE without retaining it beyond the Comfy workflow owner."""

    try:
        reference = weakref.ref(vae)
    except TypeError:
        # ComfyUI's VAE wrapper is weak-referenceable.  Do not add a strong
        # fallback here: a cleanup registry must never become a model leak.
        print(
            f"[Kaggle H3] Could not register non-weak-referenceable {role} VAE; "
            "interruption cleanup will rely on the node finally block.",
            flush=True,
        )
        return
    with _VAE_PHASE_LOCK:
        _ACTIVE_VAE_PHASES[id(vae)] = _H3VAEPhaseRegistration(
            reference=reference,
            role=str(role),
            device_id=int(device_id),
        )


def unregister_vae_phase(vae: Any) -> None:
    with _VAE_PHASE_LOCK:
        _ACTIVE_VAE_PHASES.pop(id(vae), None)


def _registered_vae_phases() -> list[tuple[Any, str]]:
    active: list[tuple[Any, str]] = []
    dead: list[int] = []
    with _VAE_PHASE_LOCK:
        for key, registration in _ACTIVE_VAE_PHASES.items():
            vae = registration.reference()
            if vae is None:
                dead.append(key)
            else:
                active.append((vae, registration.role))
        for key in dead:
            _ACTIVE_VAE_PHASES.pop(key, None)
    return active


def _synchronize_h3_devices_for_cleanup(
    device_ids: Any, report: dict[str, Any], *, reason: str
) -> None:
    """Attempt cleanup synchronization without blocking resource release."""

    try:
        synchronize_h3_devices(device_ids, reason=reason)
    except Exception as exc:
        report.setdefault("warnings", []).append(
            f"CUDA synchronization failed during cleanup: {exc}"
        )


class _GpuMemoryMonitor:
    """Low-overhead memory sampler for the lifetime of one H3 phase."""

    def __init__(self, device_ids: tuple[int, ...], interval_seconds: float = 1.0):
        self.device_ids = tuple(device_ids)
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _sample(device_ids: tuple[int, ...]) -> dict[str, Any]:
        try:
            import torch  # type: ignore

            values: dict[str, Any] = {"timestamp": time.time(), "gpus": []}
            for device_id in device_ids:
                free, total = torch.cuda.mem_get_info(device_id)
                values["gpus"].append(
                    {
                        "index": device_id,
                        "free_bytes": int(free),
                        "total_bytes": int(total),
                        "used_bytes": int(total - free),
                        "allocated_bytes": int(torch.cuda.memory_allocated(device_id)),
                        "reserved_bytes": int(torch.cuda.memory_reserved(device_id)),
                    }
                )
            try:
                import psutil  # type: ignore

                values["cpu_available_bytes"] = int(psutil.virtual_memory().available)
            except Exception:
                values["cpu_available_bytes"] = None
            smi = shutil.which("nvidia-smi")
            if smi:
                try:
                    query = subprocess.run(
                        [
                            smi,
                            "--query-gpu=index,utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )
                    utilization: dict[str, dict[str, int]] = {}
                    for line in query.stdout.splitlines():
                        fields = [field.strip() for field in line.split(",")]
                        if len(fields) != 3:
                            continue
                        try:
                            utilization[fields[0]] = {
                                "utilization_percent": int(float(fields[1])),
                                "used_mib": int(float(fields[2])),
                            }
                        except ValueError:
                            continue
                    values["nvidia_smi"] = utilization
                except Exception as exc:
                    values["nvidia_smi_error"] = str(exc)
            return values
        except Exception as exc:
            return {"timestamp": time.time(), "error": str(exc), "gpus": []}

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self._sample(self.device_ids))
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="h3-gpu-phase-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        if not self.samples:
            self.samples.append(self._sample(self.device_ids))
        per_gpu: list[dict[str, Any]] = []
        for device_id in self.device_ids:
            entries = [
                item
                for sample in self.samples
                for item in sample.get("gpus", [])
                if item.get("index") == device_id
            ]
            if not entries:
                per_gpu.append({"index": device_id, "samples": 0})
                continue
            peak = max(entries, key=lambda item: item.get("used_bytes", 0))
            smi_values = [
                sample.get("nvidia_smi", {}).get(str(device_id), {}).get("utilization_percent")
                for sample in self.samples
                if isinstance(sample.get("nvidia_smi", {}).get(str(device_id), {}).get("utilization_percent"), int)
            ]
            per_gpu.append(
                {
                    "index": device_id,
                    "samples": len(entries),
                    "peak_used_gib": round(peak.get("used_bytes", 0) / 2**30, 3),
                    "peak_allocated_gib": round(peak.get("allocated_bytes", 0) / 2**30, 3),
                    "peak_reserved_gib": round(peak.get("reserved_bytes", 0) / 2**30, 3),
                    "minimum_free_gib": round(min(item.get("free_bytes", 0) for item in entries) / 2**30, 3),
                    "peak_utilization_percent": max(smi_values) if smi_values else None,
                    "observed_active_in_same_generation": bool(
                        smi_values and max(smi_values) > 0
                    ),
                    "safety_limit_gib": 14.0,
                    "safety_headroom_pass": peak.get("used_bytes", 0) <= int(14.0 * 2**30),
                }
            )
        cpu_values = [
            item.get("cpu_available_bytes")
            for item in self.samples
            if isinstance(item.get("cpu_available_bytes"), int)
        ]
        return {
            "sample_count": len(self.samples),
            "peak_by_gpu": per_gpu,
            "minimum_cpu_available_gib": round(min(cpu_values) / 2**30, 3) if cpu_values else None,
            "cpu_headroom_limit_gib": 24.0,
            "cpu_headroom_pass": bool(cpu_values and min(cpu_values) >= int(24.0 * 2**30)),
        }


def phase_policy() -> str:
    value = os.environ.get("KAGGLE_H3_PHASE_SHARDING", "auto").strip().lower()
    if value not in {"auto", "required", "off"}:
        raise H3PhaseError(
            "KAGGLE_H3_PHASE_SHARDING must be auto, required, or off; "
            f"got {value!r}"
        )
    return value


def phase_offload_dir() -> Path:
    configured = os.environ.get("KAGGLE_H3_PHASE_OFFLOAD_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    kaggle_tmp = Path("/kaggle/tmp")
    if kaggle_tmp.is_dir():
        return kaggle_tmp / "minimax-h3-layer-offload" / "comfyui"
    return Path.cwd() / "kaggle_h3_layer_offload"


def phase_device_ids() -> tuple[int, ...]:
    """Return local CUDA indices after CUDA_VISIBLE_DEVICES remapping."""

    configured = os.environ.get("KAGGLE_H3_GPU_IDS")
    if configured:
        try:
            ids = tuple(int(item.strip()) for item in configured.split(",") if item.strip())
        except ValueError as exc:
            raise H3PhaseError(f"Invalid KAGGLE_H3_GPU_IDS={configured!r}") from exc
        if ids:
            return ids
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return ()
        return tuple(range(min(int(torch.cuda.device_count()), 2)))
    except Exception:
        return ()


def phase_state_for_model(model: Any) -> H3TransformerPhase | None:
    """Return the active phase state attached to a patcher or its real model."""

    for candidate in (model, getattr(model, "model", None)):
        state = getattr(candidate, "_kaggle_h3_phase_state", None)
        if isinstance(state, H3TransformerPhase) and not state.released:
            return state
    return None


def set_h3_synchronization_mode(state: H3TransformerPhase | None, mode: Any) -> str:
    """Change the active transformer router policy without redispatching it."""

    normalized = normalize_h3_synchronization_mode(mode)
    if state is None:
        return normalized
    state.synchronization_mode = normalized
    if isinstance(state.synchronization_mode_ref, dict):
        state.synchronization_mode_ref["value"] = normalized
    transformer_ref = getattr(
        state.transformer, "_kaggle_h3_synchronization_mode_ref", None
    )
    if isinstance(transformer_ref, dict):
        transformer_ref["value"] = normalized
    state.report["synchronization"] = h3_synchronization_plan(normalized)
    if state.activity:
        state.activity["synchronization_mode"] = normalized
    return normalized


def _text_encoder_root(clip: Any) -> Any:
    """Find the model root that contains Qwen's language-layer sequence."""

    candidates = [
        getattr(clip, "cond_stage_model", None),
        getattr(clip, "model", None),
        clip,
    ]
    for candidate in candidates:
        if candidate is None or not callable(getattr(candidate, "named_modules", None)):
            continue
        try:
            try:
                from .layer_sharding import find_dispatchable_text_encoder_layers
            except ImportError:
                from kaggle_h3_layer_sharding import (  # type: ignore
                    find_dispatchable_text_encoder_layers,
                )

            find_dispatchable_text_encoder_layers(candidate)
            return candidate
        except Exception:
            continue
    raise H3PhaseError(
        "The loaded H3 text encoder does not expose a dispatchable Qwen language-layer sequence"
    )


def prepare_text_encoder_for_h3_phase(
    clip: Any,
    *,
    device_ids: tuple[int, ...] = (0, 1),
    offload_dir: Path | None = None,
) -> H3TextEncoderPhase:
    """Dispatch Qwen language layers across both GPUs until conditioning ends."""

    existing = getattr(clip, "_kaggle_h3_text_encoder_phase", None)
    if isinstance(existing, H3TextEncoderPhase) and not existing.released:
        return existing
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise H3PhaseError("H3 text-encoder sharding requires PyTorch") from exc
    ids = tuple(int(item) for item in device_ids)
    if len(ids) < 2 or ids[0] == ids[1]:
        raise H3PhaseError(
            f"H3 text-encoder sharding requires two distinct GPU ids; got {list(ids)}"
        )
    if not torch.cuda.is_available() or int(torch.cuda.device_count()) < max(ids) + 1:
        raise H3PhaseError(
            f"H3 text-encoder sharding requested {list(ids)}, but visible CUDA devices are insufficient"
        )
    patcher = clip if hasattr(clip, "load_device") else getattr(clip, "patcher", None)
    if patcher is None:
        raise H3PhaseError("The H3 text encoder does not expose a Comfy model patcher")
    encoder = _text_encoder_root(clip)
    try:
        from .layer_sharding import dispatch_h3_text_encoder

        _dispatched, report, handles = dispatch_h3_text_encoder(
            encoder,
            device_ids=ids,
            gpu_limit_gib=13.0,
            cpu_headroom_gib=26.0,
            offload_dir=offload_dir,
        )
    except ImportError:
        from kaggle_h3_layer_sharding import dispatch_h3_text_encoder  # type: ignore

        _dispatched, report, handles = dispatch_h3_text_encoder(
            encoder,
            device_ids=ids,
            gpu_limit_gib=13.0,
            cpu_headroom_gib=26.0,
            offload_dir=offload_dir,
        )
    primary = torch.device(f"cuda:{ids[0]}")
    _retarget_patcher_load_device(patcher, primary)
    patcher.offload_device = torch.device("cpu")
    state = H3TextEncoderPhase(
        clip=clip,
        patcher=patcher,
        encoder=encoder,
        device_ids=ids,
        report=report,
        dispatch_handles=list(handles),
        active=True,
    )
    setattr(clip, "_kaggle_h3_text_encoder_phase", state)
    try:
        setattr(patcher, "_kaggle_h3_text_encoder_phase", state)
    except Exception:
        pass
    _ACTIVE_TEXT_ENCODER_PHASES.append(state)
    execution_devices = sorted(
        {
            str(item.get("execution_device"))
            for item in report.get("layers", [])
            if str(item.get("execution_device", "")).startswith("cuda:")
        }
    )
    print(
        "[Kaggle H3] Text encoder dispatched across "
        f"{execution_devices}; CPU third tier=True; "
        f"language_layers={report['group']['layer_count']}; "
        f"boundary={report.get('activation_boundary')}. ",
        flush=True,
    )
    return state


def release_text_encoder_phase(state: H3TextEncoderPhase | None) -> dict[str, Any]:
    """Remove temporary Qwen routing and return the encoder to CPU."""

    if state is None or state.released:
        return {"status": "already_released" if state else "not_started"}
    state.released = True
    _synchronize_h3_devices_for_cleanup(
        state.device_ids,
        state.report,
        reason="text encoder phase release",
    )
    for handle in state.dispatch_handles:
        remove = getattr(handle, "remove", None)
        if callable(remove):
            try:
                remove()
            except Exception:
                pass
    state.dispatch_handles.clear()
    moved_to_cpu = 0
    try:
        import torch  # type: ignore

        for item in state.report.get("layers", []):
            if not str(item.get("device", "")).startswith("cuda:"):
                continue
            try:
                try:
                    from .layer_sharding import _module_at_path
                except ImportError:
                    from kaggle_h3_layer_sharding import _module_at_path  # type: ignore

                _module_at_path(state.encoder, str(item["path"])).to(torch.device("cpu"))
                moved_to_cpu += 1
            except Exception:
                continue
        unpatch = getattr(state.patcher, "unpatch_model", None)
        if callable(unpatch):
            try:
                unpatch(torch.device("cpu"), unpatch_weights=True)
            except Exception:
                pass
    except Exception:
        pass
    _release_comfy_cache()
    gc.collect()
    try:
        import torch  # type: ignore

        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        setattr(state.clip, "_kaggle_h3_text_encoder_phase", None)
        setattr(state.patcher, "_kaggle_h3_text_encoder_phase", None)
    except Exception:
        pass
    state.active = False
    report = {
        "status": "released",
        "hooks_removed": True,
        "layers_returned_to_cpu": moved_to_cpu,
        "cpu_third_tier": True,
    }
    state.report["release"] = report
    if state in _ACTIVE_TEXT_ENCODER_PHASES:
        _ACTIVE_TEXT_ENCODER_PHASES.remove(state)
    print(
        "[Kaggle H3] Text encoder phase released before transformer dispatch; "
        f"layers_returned_to_cpu={moved_to_cpu}.",
        flush=True,
    )
    return report


def release_all_text_encoder_phases() -> list[dict[str, Any]]:
    """Release every active text phase without masking a generation error."""

    reports: list[dict[str, Any]] = []
    for state in list(_ACTIVE_TEXT_ENCODER_PHASES):
        try:
            reports.append(release_text_encoder_phase(state))
        except Exception as exc:
            reports.append({"status": "cleanup_failed", "error": str(exc)})
    return reports


def _attach_phase_state(model: Any, state: H3TransformerPhase | None) -> None:
    """Keep phase state with the shared Comfy model so ModelPatcher.clone preserves it."""

    for candidate in (model, getattr(model, "model", None)):
        if candidate is not None:
            try:
                setattr(candidate, "_kaggle_h3_phase_state", state)
            except Exception:
                pass


def _retarget_patcher_load_device(patcher: Any, device: Any) -> None:
    """Retarget a Comfy patcher without breaking DynamicVRAM bookkeeping.

    ComfyUI 0.34's dynamic patcher stores per-load-device state in
    ``model.dynamic_pins``. Changing ``load_device`` directly to CPU creates a
    device key that was never registered, so prompt tracking fails before the
    workflow can reach the phase barrier. The model can remain CPU-resident;
    the patcher's logical load target must simply remain a registered CUDA
    device until the explicit dispatch takes ownership.
    """

    register = getattr(patcher, "register_load_device", None)
    if callable(register):
        try:
            register(device)
        except Exception as exc:
            raise H3PhaseError(
                f"ComfyUI could not register H3 load device {device}: {exc}"
            ) from exc
    patcher.load_device = device
    real_model = getattr(patcher, "model", None)
    dynamic_pins = getattr(real_model, "dynamic_pins", None)
    if isinstance(dynamic_pins, dict) and device not in dynamic_pins:
        raise H3PhaseError(
            f"ComfyUI dynamic patcher has no bookkeeping entry for load device {device}"
        )


def prepare_model_for_h3_phase(
    model: Any, *, device_ids: tuple[int, ...] = (0, 1)
) -> dict[str, Any]:
    """Make a Comfy model loader CPU-owned until the explicit H3 phase begins.

    This is intentionally a placement change, not a second model copy.  The
    phase barrier later changes the patcher's primary execution device after
    conditioning and installs the verified block map.  It uses ComfyUI's
    native quantized router for INT8 H3 and Accelerate only for ordinary
    modules.
    """

    try:
        import torch  # type: ignore
    except Exception as exc:
        raise H3PhaseError("H3 loader placement requires PyTorch") from exc
    if not torch.cuda.is_available():
        raise H3PhaseError("H3 sharded loader requires CUDA")
    visible = int(torch.cuda.device_count())
    ids = tuple(int(item) for item in device_ids)
    if len(ids) < 2 or ids[0] == ids[1]:
        raise H3PhaseError(
            f"H3 sharded loader requires two distinct GPU ids; got {list(ids)}"
        )
    if any(item < 0 or item >= visible for item in ids):
        raise H3PhaseError(
            f"H3 sharded loader requested {list(ids)}, but only {visible} CUDA devices are visible"
        )

    patcher = model if hasattr(model, "load_device") else getattr(model, "patcher", None)
    if patcher is None:
        raise H3PhaseError("H3 sharded loader did not receive a Comfy ModelPatcher")
    cpu = torch.device("cpu")
    phase_device = torch.device(f"cuda:{ids[0]}")
    # Keep the real model on CPU, but do not set ModelPatcher.load_device to
    # CPU: DynamicVRAM's prompt tracker indexes dynamic_pins by that value.
    _retarget_patcher_load_device(patcher, phase_device)
    patcher.offload_device = cpu
    real_model = getattr(patcher, "model", None)
    if real_model is not None and hasattr(real_model, "device"):
        real_model.device = cpu
    placement = {
        "loader": "KaggleH3ShardedDiffusionLoader",
        "status": "loaded_cpu_until_conditioning_barrier",
        "device_ids": list(ids),
        "model_residency": "cpu",
        "patcher_load_device": str(phase_device),
        "transformer": "automatic_contiguous_intact_blocks",
        "cpu": "third_offload_tier",
        "disk": str(phase_offload_dir()),
    }
    options = getattr(patcher, "model_options", None)
    if isinstance(options, dict):
        options["kaggle_h3_phase_loader"] = placement
    print(
        "[Kaggle H3] Sharded diffusion loader kept model weights on CPU; "
        f"patcher load target={phase_device}; phase targets cuda:{ids[0]} "
        f"and cuda:{ids[1]} after conditioning.",
        flush=True,
    )
    return placement


def build_phase_runtime_config(
    *, device_ids: tuple[int, ...] = (0, 1), offload_dir: Path | None = None
) -> dict[str, Any]:
    """Return the serializable phase plan consumed by nodes and manifests."""

    ids = tuple(int(item) for item in device_ids)
    offload = str((offload_dir or phase_offload_dir()).resolve())
    return {
        "strategy": "phase_aware_h3",
        "text_encoder": "automatic_contiguous_language_layers_cuda_0_cuda_1_then_cpu",
        "transformer": {
            "dispatch": "automatic_contiguous_intact_blocks",
            "device_ids": list(ids),
            "gpu_budget_gib": 13.0,
            "cpu_headroom_gib": 26.0,
            "offload_dir": offload,
        },
        "audio_vae": f"cuda:{ids[0]}" if ids else "cuda:0",
        "video_vae": f"cuda:{ids[1]}" if len(ids) > 1 else f"cuda:{ids[0]}" if ids else "cuda:0",
        "cpu": "third_offload_tier",
        "disk": offload,
        "release_transformer_before_decode": True,
        "synchronization": h3_synchronization_plan("full"),
    }


def _find_transformer(model: Any) -> tuple[Any, str]:
    """Find the H3 module without assuming one ComfyUI wrapper shape."""

    try:
        from .layer_sharding import find_dispatchable_layers
    except ImportError:
        from kaggle_h3_layer_sharding import find_dispatchable_layers  # type: ignore

    candidates: list[tuple[Any, str]] = []
    patcher_model = getattr(model, "model", None)
    if patcher_model is not None:
        candidates.append((patcher_model, "model"))
        diffusion_model = getattr(patcher_model, "diffusion_model", None)
        if diffusion_model is not None:
            candidates.append((diffusion_model, "model.diffusion_model"))
    candidates.append((model, "root"))
    for candidate, path in candidates:
        try:
            find_dispatchable_layers(candidate)
        except Exception:
            continue
        return candidate, path
    raise H3PhaseError(
        "The loaded ComfyUI H3 model does not expose an indexed transformer "
        "block sequence; refusing to run a one-GPU disguise."
    )


def _call_zero_argument(owner: Any, name: str) -> bool:
    function = getattr(owner, name, None)
    if not callable(function):
        return False
    try:
        signature = inspect.signature(function)
        required = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in {parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD}
        ]
        if required:
            return False
    except (TypeError, ValueError):
        pass
    try:
        function()
        return True
    except Exception:
        return False


def _release_comfy_cache() -> None:
    """Ask ComfyUI to release models before the explicit map is applied."""

    try:
        import comfy.model_management as model_management  # type: ignore
    except Exception:
        return
    for name in ("unload_model_clones", "cleanup_models_gc", "soft_empty_cache"):
        _call_zero_argument(model_management, name)


def release_h3_runtime_resources() -> dict[str, Any]:
    """Release temporary H3 allocations without masking the original error."""

    text_reports = release_all_text_encoder_phases()
    vae_reports = release_all_vae_phases()
    _release_comfy_cache()
    gc.collect()
    cache_cleared = False
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                ipc_collect()
            cache_cleared = True
    except Exception:
        pass
    cleanup_errors = [
        str(item.get("error"))
        for item in (*text_reports, *vae_reports)
        if item.get("status") in {"cleanup_failed", "released_with_errors"}
        and item.get("error")
    ]
    for item in vae_reports:
        cleanup_errors.extend(str(error) for error in item.get("errors", []))
    report = {
        "status": "released" if not cleanup_errors else "released_with_errors",
        "text_encoder": text_reports,
        "vae": vae_reports,
        "errors": cleanup_errors,
        "comfy_cache_cleared": True,
        "torch_cuda_cache_cleared": cache_cleared,
    }
    print(
        "[Kaggle H3] Runtime cleanup completed after generation; "
        f"text_phases={len(text_reports)}, vae_phases={len(vae_reports)}, "
        f"errors={len(cleanup_errors)}, "
        f"torch_cuda_cache_cleared={cache_cleared}.",
        flush=True,
    )
    return report


def release_vae_phase(vae: Any, *, role: str) -> dict[str, Any]:
    """Return a decoded H3 VAE to CPU and clear temporary CUDA allocations."""

    unpatched = False
    errors: list[str] = []
    patcher = getattr(vae, "patcher", None)
    try:
        import torch  # type: ignore

        phase_device = getattr(vae, "device", None)
        if phase_device is None and patcher is not None:
            phase_device = getattr(patcher, "load_device", None)
        synchronize_h3_devices((phase_device,), reason=f"{role} VAE phase release")
        unpatch = getattr(patcher, "unpatch_model", None)
        if callable(unpatch):
            unpatch(torch.device("cpu"), unpatch_weights=True)
            unpatched = True
    except Exception as exc:
        errors.append(str(exc))
    _release_comfy_cache()
    gc.collect()
    try:
        import torch  # type: ignore

        torch.cuda.empty_cache()
    except Exception as exc:
        errors.append(f"CUDA cache cleanup failed: {exc}")
    unregister_vae_phase(vae)
    report = {
        "status": "released" if not errors else "released_with_errors",
        "role": role,
        "patcher_unpatched_to_cpu": unpatched,
        "errors": errors,
    }
    print(
        f"[Kaggle H3] {role} VAE phase released after decode; "
        f"patcher_cpu_release={unpatched}, errors={len(errors)}.",
        flush=True,
    )
    for error in errors:
        print(f"[Kaggle H3] WARNING: {role} VAE cleanup: {error}", flush=True)
    return report


def release_all_vae_phases() -> list[dict[str, Any]]:
    """Release every VAE registered by the current Comfy workflow.

    The weak registry lets an interrupted sampler clean VAEs whose decode
    nodes were never reached without keeping those VAEs alive indefinitely.
    """

    reports: list[dict[str, Any]] = []
    for vae, role in _registered_vae_phases():
        reports.append(release_vae_phase(vae, role=role))
    return reports


def _release_non_transformer_comfy_models(active_model: Any) -> list[dict[str, Any]]:
    """Evict conditioning/auxiliary models while preserving the transformer."""

    try:
        import comfy.model_management as model_management  # type: ignore
    except Exception:
        return []
    unload = getattr(model_management, "unload_model_and_clones", None)
    loaded_models = getattr(model_management, "current_loaded_models", None)
    if not callable(unload) or not isinstance(loaded_models, list):
        return []
    active_real_model = getattr(active_model, "model", None)
    released: list[dict[str, Any]] = []
    for loaded in list(loaded_models):
        candidate = getattr(loaded, "model", None)
        if candidate is None or candidate is active_model:
            continue
        if active_real_model is not None and getattr(candidate, "model", None) is active_real_model:
            continue
        label = candidate.__class__.__name__
        if getattr(candidate, "is_clip", False):
            role = "text_encoder"
        else:
            role = "conditioning_or_auxiliary"
        try:
            unload(candidate, unload_additional_models=False, all_devices=True)
            released.append({"role": role, "class": label, "status": "evicted"})
        except Exception as exc:
            released.append(
                {"role": role, "class": label, "status": "eviction_failed", "error": str(exc)}
            )
    if released:
        print(
            "[Kaggle H3] Evicted non-transformer ComfyUI models before transformer "
            f"dispatch: {released}.",
            flush=True,
        )
    return released


def _same_comfy_model(left: Any, right: Any) -> bool:
    """Compare patchers by identity without requiring a specific Comfy class."""

    if left is right:
        return True
    left_real = getattr(left, "model", None)
    right_real = getattr(right, "model", None)
    return left_real is not None and left_real is right_real


def _attach_dispatched_transformer(
    model: Any,
    transformer: Any,
    dispatched: Any,
    transformer_path: str,
) -> dict[str, Any]:
    """Make the Accelerate-dispatched object the module ComfyUI will execute.

    ``dispatch_model`` may return the same module after installing hooks, or a
    replacement wrapper.  Keeping only the return value in phase state is not
    enough: ComfyUI still follows the module reference held by
    ``ModelPatcher.model``.  Replace that exact reference when Accelerate
    returns a distinct object and fail if the expected path changed underneath
    us instead of silently sampling the original GPU0-managed model.
    """

    if dispatched is transformer:
        return {
            "status": "in_place_hooks",
            "path": transformer_path,
            "same_object": True,
        }
    if transformer_path == "root":
        raise H3PhaseError(
            "Accelerate returned a replacement for the root H3 model, but ComfyUI "
            "has no safe parent slot for reattachment"
        )
    parent_path, separator, attribute = transformer_path.rpartition(".")
    if not separator:
        parent = model
    else:
        parent = model
        for part in parent_path.split("."):
            parent = getattr(parent, part)
    current = getattr(parent, attribute, None)
    if current is not transformer:
        raise H3PhaseError(
            "The ComfyUI model changed while H3 dispatch was running; refusing "
            f"to attach a replacement at {transformer_path!r}"
        )
    setattr(parent, attribute, dispatched)
    if getattr(parent, attribute, None) is not dispatched:
        raise H3PhaseError(
            f"ComfyUI rejected the dispatched H3 transformer at {transformer_path!r}"
        )
    return {
        "status": "reattached",
        "path": transformer_path,
        "same_object": False,
    }


def _prepare_sampling_preserving_phase(
    original_prepare_sampling: Any,
    phase_model: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Run Comfy's preparation while excluding an already-dispatched H3 model."""

    try:
        import comfy.model_management as model_management  # type: ignore
    except Exception:
        return original_prepare_sampling(*args, **kwargs)
    original_load_models_gpu = getattr(model_management, "load_models_gpu", None)
    if not callable(original_load_models_gpu):
        return original_prepare_sampling(*args, **kwargs)

    def load_models_gpu_without_dispatched_model(models, *load_args, **load_kwargs):
        requested = list(models)
        kept = [candidate for candidate in requested if not _same_comfy_model(candidate, phase_model)]
        if len(kept) != len(requested):
            print(
                "[Kaggle H3] Preserving the pre-dispatched transformer; "
                "ComfyUI GPU0-only reload was suppressed for this sampling pass.",
                flush=True,
            )
        if kept:
            return original_load_models_gpu(kept, *load_args, **load_kwargs)
        return None

    model_management.load_models_gpu = load_models_gpu_without_dispatched_model
    try:
        return original_prepare_sampling(*args, **kwargs)
    finally:
        model_management.load_models_gpu = original_load_models_gpu


def _measure_gpu_transfer(device_ids: tuple[int, ...]) -> dict[str, Any]:
    """Measure the activation-boundary copy against a CPU-staged copy."""

    result: dict[str, Any] = {
        "status": "not_measured",
        "device_ids": list(device_ids),
        "payload_megabytes": 4,
        "directions": [],
    }
    if len(device_ids) < 2:
        result["reason"] = "two CUDA devices are required"
        return result
    try:
        import torch  # type: ignore

        elements = 4 * 2**20 // 2
        for source_id, target_id in ((device_ids[0], device_ids[1]), (device_ids[1], device_ids[0])):
            source = torch.randn(elements, device=f"cuda:{source_id}", dtype=torch.float16)
            target = torch.empty_like(source, device=f"cuda:{target_id}")
            direct: list[float] = []
            staged: list[float] = []
            for _ in range(2):
                start = time.perf_counter()
                target.copy_(source)
                torch.cuda.synchronize(target_id)
                direct.append((time.perf_counter() - start) * 1000)
            for _ in range(2):
                start = time.perf_counter()
                copied = source.to("cpu").to(f"cuda:{target_id}")
                torch.cuda.synchronize(target_id)
                staged.append((time.perf_counter() - start) * 1000)
                del copied
            direct_median = round(median(direct), 3)
            staged_median = round(median(staged), 3)
            result["directions"].append(
                {
                    "source": source_id,
                    "target": target_id,
                    "direct_median_ms": direct_median,
                    "cpu_staged_median_ms": staged_median,
                    "direct_to_staged_ratio": round(direct_median / max(staged_median, 0.001), 3),
                    "acceptable": bool(direct_median <= 100.0 and direct_median / max(staged_median, 0.001) <= 1.5),
                }
            )
            del source, target
        torch.cuda.empty_cache()
        result["status"] = "measured"
        result["acceptable"] = all(item["acceptable"] for item in result["directions"])
    except Exception as exc:
        result["status"] = "measurement_failed"
        result["error"] = str(exc)
    return result


def _install_transformer_activity_hooks(state: H3TransformerPhase) -> None:
    """Record calls and optionally validate dispatched H3 block boundaries."""

    diagnostics_enabled = finite_diagnostics_enabled()

    state.activity = {
        "planned_gpu_calls": {str(device_id): 0 for device_id in state.device_ids},
        "input_gpu_ids": [],
        "block_calls": {},
        "diagnostics_enabled": diagnostics_enabled,
        "diagnostic_checks": {},
        "validated_blocks": {"input": [], "output": []},
    }
    planned_layers = (state.report.get("planned") or {}).get("layers") or []
    group_path = str((state.report.get("planned") or {}).get("group", {}).get("container_path", ""))
    observed_execution_map = (
        (state.report.get("observed") or {}).get("reported_execution_map") or {}
    )

    def execution_target(layer: dict[str, Any]) -> str:
        path = str(layer.get("path", ""))
        return str(
            observed_execution_map.get(
                path,
                layer.get("execution_device", layer.get("device", "")),
            )
        )

    block_layers: list[tuple[int, dict[str, Any]]] = []
    for layer in planned_layers:
        path = str(layer.get("path", ""))
        if not group_path or not path.startswith(f"{group_path}."):
            continue
        try:
            index = int(path.rsplit(".", 1)[-1])
        except ValueError:
            continue
        block_layers.append((index, layer))
    block_layers.sort(key=lambda item: item[0])
    primary_gpu = f"cuda:{state.device_ids[0]}" if state.device_ids else "cuda:0"
    secondary_gpu = f"cuda:{state.device_ids[1]}" if len(state.device_ids) > 1 else None
    primary_blocks = [
        (index, layer)
        for index, layer in block_layers
        if execution_target(layer) == primary_gpu
    ]
    secondary_blocks = [
        (index, layer)
        for index, layer in block_layers
        if secondary_gpu is not None
        and execution_target(layer) == secondary_gpu
    ]
    first_primary_path = primary_blocks[0][1].get("path") if primary_blocks else None
    first_secondary_path = secondary_blocks[0][1].get("path") if secondary_blocks else None

    def mark_block_check(kind: str, path: str) -> None:
        checked = state.activity["validated_blocks"][kind]
        if path not in checked:
            checked.append(path)

    def validate_phase_input(stage: str, value: Any, path: str) -> None:
        if not diagnostics_enabled:
            return
        validate_finite(
            {"args": value[0], "kwargs": value[1]},
            stage,
            tensor_name=f"{path}.input",
        )
        state.activity["diagnostic_checks"][stage] = True
        mark_block_check("input", path)

    for layer in planned_layers:
        # The quantized Comfy router records the GPU where a CPU/disk-tier
        # block actually executes separately from its parameter residency.
        target = execution_target(layer)
        if not target.startswith("cuda:"):
            continue
        path = str(layer.get("path", ""))
        module = state.transformer
        try:
            for part in path.split("."):
                module = getattr(module, part)
        except Exception:
            continue

        register = getattr(module, "register_forward_pre_hook", None)
        if not callable(register):
            continue

        def observe_call(_module: Any, args: tuple[Any, ...], _kwargs: dict[str, Any] | None = None, *, _path=path, _target=target):
            state.activity["planned_gpu_calls"][_target] = state.activity["planned_gpu_calls"].get(_target, 0) + 1
            state.activity["block_calls"][_path] = state.activity["block_calls"].get(_path, 0) + 1
            for value in args:
                device = getattr(value, "device", None)
                label = str(device)
                if label.startswith("cuda:") and label not in state.activity["input_gpu_ids"]:
                    state.activity["input_gpu_ids"].append(label)
            kwargs = _kwargs or {}
            if _path == first_primary_path and "sampler_gpu0" not in state.activity["diagnostic_checks"]:
                validate_phase_input("sampler_gpu0", (args, kwargs), _path)
            elif _target == primary_gpu:
                # Validate every subsequent block input on GPU0, not just the
                # segment entry. This catches a non-finite activation before it
                # can be consumed by the next intact block.
                validate_phase_input("sampler_gpu0", (args, kwargs), _path)
            if _path == first_secondary_path:
                # The first GPU1 block is the exact destination boundary of
                # the preceding GPU0 segment. Keep the low-level names first
                # so a failing transition reports its precise location.
                validate_phase_input("gpu0_to_gpu1", (args, kwargs), _path)
                validate_phase_input("sampler_gpu1_in", (args, kwargs), _path)
                if "sampler_gpu1" not in state.activity["diagnostic_checks"]:
                    validate_phase_input("sampler_gpu1", (args, kwargs), _path)
            elif _target == secondary_gpu:
                # Validate every subsequent block input on GPU1. The first
                # block already receives the low-level handoff checks above.
                validate_phase_input("sampler_gpu1", (args, kwargs), _path)

        def observe_output(
            _module: Any,
            _args: tuple[Any, ...],
            _kwargs: dict[str, Any] | None,
            output: Any,
            *,
            _path=path,
            _target=target,
        ) -> Any:
            if not diagnostics_enabled:
                return output
            stage = (
                "sampler_gpu0_out"
                if _target == primary_gpu
                else "sampler_gpu1_out"
            )
            # Attach this to every dispatched block. The path in the required
            # exception format identifies the first block that actually
            # produced non-finite values, rather than only the last block in a
            # GPU segment.
            validate_finite(output, stage, tensor_name=f"{_path}.output")
            state.activity["diagnostic_checks"][stage] = True
            mark_block_check("output", _path)
            return output

        try:
            state.activity_handles.append(register(observe_call, with_kwargs=True))
        except TypeError:
            try:
                state.activity_handles.append(register(lambda module, args, _observe=observe_call: _observe(module, args),))
            except Exception:
                continue
        except Exception:
            continue

        if not diagnostics_enabled:
            continue

        register_output = getattr(module, "register_forward_hook", None)
        if not callable(register_output):
            continue
        try:
            state.activity_handles.append(
                register_output(observe_output, with_kwargs=True)
            )
        except TypeError:
            try:
                state.activity_handles.append(
                    register_output(
                        lambda current, call_args, output, _observe=observe_output: _observe(
                            current, call_args, {}, output
                        )
                    )
                )
            except Exception:
                continue
        except Exception:
            continue


def _remove_transformer_activity_hooks(state: H3TransformerPhase) -> None:
    for handle in state.activity_handles:
        remove = getattr(handle, "remove", None)
        if callable(remove):
            try:
                remove()
            except Exception:
                pass
    state.activity_handles.clear()


def _remove_transformer_dispatch_hooks(state: H3TransformerPhase) -> None:
    """Remove the private Comfy quantized execution hooks installed for a phase."""

    for handle in state.dispatch_handles:
        remove = getattr(handle, "remove", None)
        if callable(remove):
            try:
                remove()
            except Exception:
                pass
    state.dispatch_handles.clear()


def _runtime_synchronization_mode(runtime_config: dict[str, Any] | None) -> str:
    if not isinstance(runtime_config, dict):
        return "full"
    execution = runtime_config.get("execution")
    if not isinstance(execution, dict):
        return "full"
    synchronization = execution.get("synchronization")
    if not isinstance(synchronization, dict):
        return "full"
    return normalize_h3_synchronization_mode(synchronization.get("mode", "full"))


def begin_transformer_phase(
    model: Any,
    runtime_config: dict[str, Any] | None = None,
    *,
    device_ids: tuple[int, ...] | None = None,
    offload_dir: Path | None = None,
    synchronization_mode: str | None = None,
) -> H3TransformerPhase:
    """Dispatch H3 blocks across two GPUs and verify the resulting map."""

    requested_synchronization_mode = normalize_h3_synchronization_mode(
        synchronization_mode
        if synchronization_mode is not None
        else _runtime_synchronization_mode(runtime_config)
    )
    existing = phase_state_for_model(model)
    if existing is not None and existing.active:
        print(
            "[Kaggle H3] Reusing the already verified transformer phase; "
            "the sampler will not reload it through GPU0-only Comfy management.",
            flush=True,
        )
        if isinstance(runtime_config, dict):
            execution = dict(runtime_config.get("execution") or {})
            execution["transformer_phase"] = existing.report
            runtime_config["execution"] = execution
        set_h3_synchronization_mode(existing, requested_synchronization_mode)
        return existing

    policy = phase_policy()
    ids = tuple(device_ids or phase_device_ids())
    if policy == "off":
        print(
            "[Kaggle H3] Transformer phase disabled explicitly; using the "
            "one-GPU ComfyUI fallback.",
            flush=True,
        )
        return H3TransformerPhase(
            device_ids=ids,
            report={"status": "explicit_fallback_only", "policy": policy},
            synchronization_mode=requested_synchronization_mode,
        )
    if len(ids) < 2:
        if policy == "required":
            raise H3PhaseError(
                "H3 phase sharding is required but fewer than two CUDA devices are visible."
            )
        print(
            "[Kaggle H3] Fewer than two CUDA devices are visible; using the "
            "one-GPU fallback.",
            flush=True,
        )
        return H3TransformerPhase(
            device_ids=ids,
            report={"status": "explicit_fallback_only", "policy": policy},
            synchronization_mode=requested_synchronization_mode,
        )

    transformer, transformer_path = _find_transformer(model)
    _release_comfy_cache()
    pre_dispatch_eviction = _release_non_transformer_comfy_models(model)
    try:
        try:
            from .layer_sharding import dispatch_h3_transformer
        except ImportError:
            from kaggle_h3_layer_sharding import dispatch_h3_transformer  # type: ignore

        dispatched, report = dispatch_h3_transformer(
            transformer,
            device_ids=ids,
            offload_dir=offload_dir or phase_offload_dir(),
            gpu_limit_gib=13.0,
            cpu_headroom_gib=26.0,
            preferred_layout="h3_two_t4_preferred",
            synchronization_mode=requested_synchronization_mode,
        )
    except Exception as exc:
        raise H3PhaseError(
            "The two-GPU H3 transformer phase could not be dispatched and verified. "
            "No one-GPU downgrade was performed. Set "
            "KAGGLE_H3_PHASE_SHARDING=off only for the explicit fallback. "
            f"Cause: {exc}"
        ) from exc

    observed = report.get("observed") or {}
    observed_ids = tuple(
        int(item)
        for item in observed.get(
            "observed_execution_gpu_ids", observed.get("observed_gpu_ids", ())
        )
    )
    if not set(ids).issubset(observed_ids):
        raise H3PhaseError(
            "The H3 transformer dispatch report did not prove that every requested "
            f"GPU participated: requested={list(ids)}, observed={list(observed_ids)}"
        )
    report = dict(report)
    report["synchronization"] = h3_synchronization_plan(requested_synchronization_mode)
    report["comfy_attachment"] = _attach_dispatched_transformer(
        model,
        transformer,
        dispatched,
        transformer_path,
    )
    reported_map = observed.get("reported_module_map") or {}
    reported_execution_map = observed.get("reported_execution_map") or reported_map
    reported_devices = sorted(
        {
            str(value)
            for value in reported_execution_map.values()
            if str(value).startswith("cuda:")
        }
    )
    print(
        "[Kaggle H3] Comfy execution attachment: "
        f"status={report['comfy_attachment']['status']}; "
        f"actual transformer execution devices={reported_devices}; "
        f"parameter residency map entries={len(reported_map)}.",
        flush=True,
    )
    report["phase"] = "transformer_denoising"
    report["transformer_path"] = transformer_path
    report["same_generation_multi_gpu"] = True
    report["pre_dispatch_eviction"] = pre_dispatch_eviction
    report["transfer_benchmark"] = _measure_gpu_transfer(ids)
    print(
        "[Kaggle H3] Transformer phase dispatched across "
        f"cuda:{ids[0]} and cuda:{ids[1]}; observed block devices="
        f"{list(observed_ids)}; CPU third tier=True; disk="
        f"{bool(report.get('disk_offload_used'))}.",
        flush=True,
    )
    transfer = report["transfer_benchmark"]
    print(
        "[Kaggle H3] Activation-boundary transfer benchmark: "
        f"status={transfer.get('status')}, acceptable={transfer.get('acceptable')}, "
        f"directions={transfer.get('directions', [])}.",
        flush=True,
    )
    if isinstance(runtime_config, dict):
        execution = dict(runtime_config.get("execution") or {})
        execution["transformer_phase"] = report
        execution["actual_device_map"] = report.get("observed", {}).get(
            "reported_execution_map"
        ) or report.get("observed", {}).get("reported_module_map")
        runtime_config["execution"] = execution
    monitor = _GpuMemoryMonitor(ids)
    monitor.start()
    state = H3TransformerPhase(
        model_patcher=model,
        transformer=dispatched,
        original_transformer=transformer,
        transformer_path=transformer_path,
        device_ids=ids,
        report=report,
        active=True,
        monitor=monitor,
        synchronization_mode=requested_synchronization_mode,
        synchronization_mode_ref=getattr(
            dispatched, "_kaggle_h3_synchronization_mode_ref", None
        ),
    )
    state.dispatch_handles = list(
        getattr(dispatched, "_kaggle_h3_dispatch_handles", ())
    )
    try:
        import torch  # type: ignore

        if hasattr(model, "load_device"):
            _retarget_patcher_load_device(model, torch.device(f"cuda:{ids[0]}"))
        if hasattr(model, "offload_device"):
            model.offload_device = torch.device("cpu")
    except Exception:
        pass
    _attach_phase_state(model, state)
    _install_transformer_activity_hooks(state)
    return state


def _remove_accelerate_hooks(module: Any) -> bool:
    try:
        from accelerate.hooks import remove_hook_from_module  # type: ignore
    except Exception:
        return False
    try:
        remove_hook_from_module(module, recurse=True)
        return True
    except TypeError:
        try:
            remove_hook_from_module(module)
            return True
        except Exception:
            return False
    except Exception:
        return False


def release_transformer_phase(state: H3TransformerPhase | None) -> dict[str, Any]:
    """Release the dispatched transformer before either VAE is decoded."""

    if state is None or state.released:
        return {"status": "already_released" if state else "not_started"}
    state.released = True
    if not state.active:
        state.report["release"] = {"status": "fallback_no_dispatch"}
        return state.report["release"]

    _synchronize_h3_devices_for_cleanup(
        state.device_ids,
        state.report,
        reason="transformer phase release",
    )
    memory_report = state.monitor.stop() if state.monitor is not None else None
    _remove_transformer_activity_hooks(state)
    _remove_transformer_dispatch_hooks(state)
    hooks_removed = _remove_accelerate_hooks(state.transformer)
    patcher = state.model_patcher
    restored_original = False
    if (
        state.original_transformer is not None
        and state.transformer is not state.original_transformer
        and state.transformer_path
    ):
        parent_path, separator, attribute = state.transformer_path.rpartition(".")
        if separator:
            parent = patcher
            for part in parent_path.split("."):
                parent = getattr(parent, part)
        else:
            parent = patcher
        if getattr(parent, attribute, None) is state.transformer:
            setattr(parent, attribute, state.original_transformer)
            restored_original = getattr(parent, attribute, None) is state.original_transformer
    unpatched = False
    # At this point no downstream node needs the model. Restoring the original
    # patcher to its CPU/offload tier releases GPU pages without creating a
    # second checkpoint copy. Some dynamic Comfy builds do not expose this
    # operation, so cache cleanup remains the final safety net.
    unpatch_model = getattr(patcher, "unpatch_model", None)
    if callable(unpatch_model):
        try:
            import torch  # type: ignore

            unpatch_model(torch.device("cpu"), unpatch_weights=True)
            unpatched = True
        except Exception as exc:
            state.report.setdefault("warnings", []).append(
                f"Comfy model patcher CPU release was unavailable: {exc}"
            )
    _release_comfy_cache()
    gc.collect()
    state.report["release"] = {
        "status": "released" if (hooks_removed or unpatched) else "cache_cleanup_only",
        "accelerate_hooks_removed": hooks_removed,
        "comfy_original_transformer_restored": restored_original,
        "comfy_patcher_unpatched_to_cpu": unpatched,
        "gpu_to_gpu_transfer_measurement": "recorded by outer telemetry when available",
    }
    if memory_report is not None:
        state.report["memory_monitor"] = memory_report
        print(
            "[Kaggle H3] Transformer-phase peak memory: "
            + ", ".join(
                f"GPU{item['index']}={item.get('peak_used_gib', 'unknown')} GiB"
                for item in memory_report.get("peak_by_gpu", [])
            )
            + f"; minimum CPU available={memory_report.get('minimum_cpu_available_gib')} GiB.",
            flush=True,
        )
    activity = dict(state.activity)
    planned_gpu_calls = dict(activity.get("planned_gpu_calls") or {})
    used_gpu_ids = {
        int(label.split(":", 1)[1])
        for label, calls in planned_gpu_calls.items()
        if int(calls) > 0 and ":" in label and label.split(":", 1)[1].isdigit()
    }
    state.report["generation_activity"] = {
        "planned_gpu_calls": planned_gpu_calls,
        "input_gpu_ids": list(activity.get("input_gpu_ids") or []),
        "block_calls": dict(activity.get("block_calls") or {}),
        "validated_blocks": {
            kind: list(paths)
            for kind, paths in (activity.get("validated_blocks") or {}).items()
        },
        "observed_gpu_ids": sorted(used_gpu_ids),
        "same_generation_all_requested_gpus": set(state.device_ids).issubset(used_gpu_ids),
    }
    state.report["release"]["generation_activity"] = state.report["generation_activity"]
    validated_block_counts = ", ".join(
        f"{kind}={len(paths)}"
        for kind, paths in state.report["generation_activity"]["validated_blocks"].items()
    )
    print(
        "[Kaggle H3] Transformer generation activity: "
        f"planned_gpu_calls={planned_gpu_calls}, observed_input_devices="
        f"{state.report['generation_activity']['input_gpu_ids']}, "
        f"validated_blocks={validated_block_counts}.",
        flush=True,
    )
    print(
        "[Kaggle H3] Transformer phase released before VAE decode; "
        f"hooks_removed={hooks_removed}, patcher_cpu_release={unpatched}.",
        flush=True,
    )
    _attach_phase_state(patcher, None)
    return state.report["release"]


def configure_vae_phase(
    vae: Any,
    *,
    device_id: int,
    role: str,
    runtime_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Set one ComfyUI VAE wrapper's load and execution device explicitly."""

    try:
        import torch  # type: ignore
    except Exception as exc:
        raise H3PhaseError("VAE phase placement requires PyTorch") from exc
    if not torch.cuda.is_available():
        raise H3PhaseError("VAE phase placement requires CUDA")
    count = int(torch.cuda.device_count())
    if int(device_id) < 0 or int(device_id) >= count:
        raise H3PhaseError(
            f"Cannot place {role} VAE on cuda:{device_id}; only {count} CUDA devices are visible."
        )
    device = torch.device(f"cuda:{int(device_id)}")
    previous: dict[str, Any] = {}
    patcher = getattr(vae, "patcher", None)
    for owner, name in ((vae, "device"), (patcher, "load_device"), (patcher, "offload_device")):
        if owner is not None and hasattr(owner, name):
            previous[name] = str(getattr(owner, name))
    if hasattr(vae, "device"):
        vae.device = device
    if patcher is not None:
        if hasattr(patcher, "load_device"):
            _retarget_patcher_load_device(patcher, device)
        # Keep host RAM as the third tier. This is intentionally not set to
        # the same GPU, which would turn a low-memory decode into a hard OOM.
        if hasattr(patcher, "offload_device"):
            patcher.offload_device = torch.device("cpu")
    execution = (runtime_config or {}).get("execution") if isinstance(runtime_config, dict) else None
    observed = {
        "role": role,
        "device": str(device),
        "offload_device": str(getattr(patcher, "offload_device", "unknown")),
        "previous": previous,
        "transformer_released_before_decode": bool(
            (execution or {}).get("transformer_phase", {}).get("release")
        ),
    }
    register_vae_phase(vae, device_id=int(device_id), role=str(role))
    print(
        f"[Kaggle H3] {role} VAE phase target={observed['device']}; "
        f"CPU offload={observed['offload_device']}.",
        flush=True,
    )
    return observed
