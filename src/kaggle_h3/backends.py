"""Backend capability checks and workers for the H3 execution paths.

The important distinction in this module is between *seeing* two CUDA
devices and a backend actually using both devices for one request.  Every
backend result carries a planned device map, a capacity decision, and the
telemetry/transfer fields needed to verify that distinction after a real
generation.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import requests

from .bootstrap import H3_MODEL_REPOSITORY, H3_MODEL_REVISION, detect_hardware


GPU_SAFETY_LIMIT_GIB = 14.0
CPU_HEADROOM_MIN_GIB = 24.0
CPU_HEADROOM_TARGET_GIB = 26.0


def _gpu_capacity(hardware: dict[str, Any]) -> tuple[float | None, float | None]:
    devices = hardware.get("cuda_devices", [])
    capacities = [
        float(device["total_gib"])
        for device in devices
        if isinstance(device.get("total_gib"), (int, float))
    ]
    capacities.extend(
        float(device["total_mib"]) / 1024.0
        for device in devices
        if not isinstance(device.get("total_gib"), (int, float))
        and isinstance(device.get("total_mib"), (int, float))
    )
    min_capacity = min(capacities) if capacities else None
    cpu_total = hardware.get("cpu_memory", {}).get("total_bytes")
    cpu_total_gib = float(cpu_total) / 2**30 if cpu_total else None
    return min_capacity, cpu_total_gib


def _dual_gpu_base(hardware: dict[str, Any]) -> dict[str, Any]:
    devices = hardware.get("cuda_devices", [])
    return {
        "detected_cuda_devices": devices,
        "device_ids": [device.get("index") for device in devices],
        "same_generation_multi_gpu": False,
        "planned_same_generation_multi_gpu": False,
        "observed_same_generation_multi_gpu": False,
        "generation_attempted": False,
        "observed_device_map": None,
        "cpu_ram_third_tier": True,
        "safety_limits": {
            "max_gpu_used_gib": GPU_SAFETY_LIMIT_GIB,
            "min_cpu_headroom_gib": CPU_HEADROOM_MIN_GIB,
            "target_cpu_headroom_gib": CPU_HEADROOM_TARGET_GIB,
        },
    }


def _capacity_decision(
    hardware: dict[str, Any], *, min_gpu_gib: float, min_cpu_gib: float
) -> dict[str, Any]:
    actual_gpu, actual_cpu = _gpu_capacity(hardware)
    two_devices = int(hardware.get("cuda_device_count", 0)) >= 2
    enough_gpu = actual_gpu is not None and actual_gpu >= min_gpu_gib
    enough_cpu = actual_cpu is not None and actual_cpu >= min_cpu_gib
    return {
        "required_min_gpu_capacity_gib": min_gpu_gib,
        "required_min_cpu_total_gib": min_cpu_gib,
        "observed_min_gpu_capacity_gib": round(actual_gpu, 2)
        if actual_gpu is not None
        else None,
        "observed_cpu_total_gib": round(actual_cpu, 2) if actual_cpu is not None else None,
        "two_devices_visible": two_devices,
        "capacity_eligible": bool(two_devices and enough_gpu and enough_cpu),
        "reason": (
            "capacity requirements met"
            if two_devices and enough_gpu and enough_cpu
            else "hardware is below the backend's documented/observed capacity envelope"
        ),
    }


def _runtime_capacity_decision(
    hardware: dict[str, Any], *, min_free_gpu_gib: float, min_cpu_available_gib: float
) -> dict[str, Any]:
    """Check only the live safety envelope needed for an experimental attempt.

    Unlike the documented-envelope checks above, this intentionally does not
    reject a machine merely because an upstream example used larger cards. A
    real layer-dispatch attempt may proceed when the live T4/RAM headroom is
    safe; the generation telemetry remains the authority on whether it works.
    """

    devices = hardware.get("cuda_devices", [])
    free_values = [
        float(device["free_gib"])
        for device in devices
        if isinstance(device.get("free_gib"), (int, float))
    ]
    cpu_available = hardware.get("cpu_memory", {}).get("available_gib")
    enough_gpu = len(free_values) >= 2 and min(free_values) >= min_free_gpu_gib
    enough_cpu = isinstance(cpu_available, (int, float)) and float(cpu_available) >= min_cpu_available_gib
    return {
        "required_min_free_gpu_gib": min_free_gpu_gib,
        "required_min_cpu_available_gib": min_cpu_available_gib,
        "observed_min_free_gpu_gib": round(min(free_values), 2) if free_values else None,
        "observed_cpu_available_gib": round(float(cpu_available), 2)
        if isinstance(cpu_available, (int, float))
        else None,
        "two_devices_visible": len(free_values) >= 2,
        "capacity_eligible": bool(enough_gpu and enough_cpu),
        "reason": (
            "live GPU/RAM safety envelope met; attempting automatic layer dispatch"
            if enough_gpu and enough_cpu
            else "live GPU/RAM safety envelope is too small for a safe layer-dispatch attempt"
        ),
    }


def evaluate_backends(
    hardware: dict[str, Any] | None = None, *, comfy_root: Path | None = None
) -> dict[str, Any]:
    """Evaluate all requested multi-GPU routes before allowing a generation.

    The requirements are intentionally conservative.  They are not a claim
    that a particular Kaggle image can never be optimized; they describe when
    this package is willing to label a path eligible without silently risking
    the two T4s or the host RAM budget.
    """

    hardware = hardware or detect_hardware()
    devices = hardware.get("cuda_devices", [])
    ids = [device.get("index") for device in devices]
    dual_map = {
        "transformer": "tensor_parallel(cuda:0,cuda:1)",
        "text_encoder": "encoder_parallel(auto)",
        "video_vae": "layerwise_offload_or_cuda:0",
        "audio_vae": "layerwise_offload_or_cuda:0",
        "cpu": "third_offload_tier_when_resident_layers_are evicted",
    }
    explicit_split_map = {
        "text_encoder": "cuda:1",
        "transformer": "cuda:0",
        "video_vae": "cuda:0",
        "audio_vae": "cuda:0",
        "cpu": "third_offload_tier_when_group_offload_is_enabled",
    }
    attempts: list[dict[str, Any]] = []

    sglang = _dual_gpu_base(hardware)
    sglang.update(
        {
            "backend": "sglang",
            "display_name": "SGLang H3 tensor/encoder parallel",
            "available": bool(shutil.which("sglang") or _module_available("sglang")),
            "same_generation_multi_gpu": True,
            "planned_same_generation_multi_gpu": True,
            "actual_device_map": dual_map,
            "planned_launch": {
                "num_gpus": 2,
                "tp_size": 2,
                "ulysses_degree": 1,
                "encoder_parallel": "auto",
                "layerwise_offload_components": ["dit", "text_encoder", "vae"],
            },
            "requirements_evidence": {
                "min_gpu_capacity_gib": 32.0,
                "min_cpu_total_gib": 384.0,
                "source": "SGLang's official two-RTX-5090 H3 memory-mode recipe",
            },
            "limitations": [
                "The official two-card recipe is documented for 2x RTX 5090 32 GiB and a much larger host, not 2x T4.",
                "This package will not call the path verified on T4 without a real generation telemetry record.",
            ],
        }
    )
    sglang.update(_capacity_decision(hardware, min_gpu_gib=32.0, min_cpu_gib=384.0))
    sglang["status"] = (
        "eligible"
        if sglang["available"] and sglang["capacity_eligible"]
        else "capacity_or_installation_blocked"
    )
    attempts.append(sglang)

    balanced = _dual_gpu_base(hardware)
    balanced.update(
        {
            "backend": "diffusers_balanced",
            "display_name": "Diffusers/Accelerate balanced device map probe",
            "available": _module_available("diffusers") and _module_available("accelerate"),
            "same_generation_multi_gpu": True,
            "planned_same_generation_multi_gpu": True,
            "actual_device_map": {
                "strategy": "balanced",
                "inspection": "pipeline.hf_device_map when the loader accepts device_map=balanced",
                "cpu": "enable_group_offload/offload_state_dict",
            },
            "limitations": [
                "Balanced placement is a generic Accelerate strategy; H3's modular pipeline does not guarantee that it shards every component correctly.",
                "A successful load must be followed by a real two-GPU generation and map inspection.",
            ],
        }
    )
    balanced.update(_capacity_decision(hardware, min_gpu_gib=48.0, min_cpu_gib=75.0))
    balanced["status"] = (
        "eligible"
        if balanced["available"] and balanced["capacity_eligible"]
        else "capacity_or_installation_blocked"
    )
    attempts.append(balanced)

    layer_sharded = _dual_gpu_base(hardware)
    layer_sharded.update(
        {
            "backend": "diffusers_layer_sharded",
            "display_name": "Diffusers automatic H3 transformer block dispatch",
            "available": _module_available("diffusers") and _module_available("accelerate"),
            "same_generation_multi_gpu": True,
            "planned_same_generation_multi_gpu": True,
            "actual_device_map": {
                "transformer": "preferred H3 map: blocks[0:23] -> cuda:0; blocks[23:50] -> cuda:1",
                "input_islands": "projections and token_refiner -> cuda:0",
                "output_islands": "final/output layers -> cuda:1",
                "dispatch": "accelerate.dispatch_model",
                "cpu": "remaining safe CPU budget after 24 GiB headroom",
                "disk": "/kaggle/tmp H3 layer offload only when GPU/CPU budgets require it",
            },
            "dispatch_strategy": {
                "layer_discovery": "largest indexed blocks/transformer_blocks/layers sequence",
                "granularity": "whole residual transformer blocks",
                "placement": "preferred 23/27 contiguous groups across two T4s, with automatic balanced fallback for other block counts",
                "no_split_module_classes": "discovered block class",
                "activation_boundary": "one hidden-state transfer after block 22 before block 23 per denoising pass",
                "verification": "actual parameter devices plus same-job telemetry",
            },
            "requirements_evidence": {
                "min_free_gpu_gib": 12.0,
                "min_cpu_available_gib": 24.0,
                "source": "local safety budget; backend is an empirical experiment rather than an upstream capacity claim",
            },
            "limitations": [
                "The transformer must be loadable by the installed Diffusers H3 class before dispatch can be applied.",
                "Disk overflow is safe but can be extremely slow; the manifest records it instead of hiding it.",
                "Adapter stacks remain ComfyUI-only until a compatible Diffusers adapter contract is implemented.",
            ],
        }
    )
    layer_sharded.update(_runtime_capacity_decision(hardware, min_free_gpu_gib=12.0, min_cpu_available_gib=24.0))
    layer_sharded["status"] = (
        "eligible"
        if layer_sharded["available"] and layer_sharded["capacity_eligible"]
        else "capacity_or_installation_blocked"
    )
    attempts.append(layer_sharded)

    split = _dual_gpu_base(hardware)
    split.update(
        {
            "backend": "diffusers_component_split",
            "display_name": "Diffusers explicit H3 component split",
            "available": _module_available("diffusers") and _module_available("accelerate"),
            "same_generation_multi_gpu": True,
            "planned_same_generation_multi_gpu": True,
            "actual_device_map": explicit_split_map,
            "limitations": [
                "This is explicit component placement, not automatic VRAM pooling.",
                "The official H3 example uses two 48 GiB cards with int8 or two 80 GiB cards with BF16; it is not a T4 validation.",
                "Cross-device conditioner/transformer transfers must be measured for each real run.",
            ],
        }
    )
    split.update(_capacity_decision(hardware, min_gpu_gib=48.0, min_cpu_gib=75.0))
    split["status"] = (
        "eligible"
        if split["available"] and split["capacity_eligible"]
        else "capacity_or_installation_blocked"
    )
    attempts.append(split)

    comfy_sp = _dual_gpu_base(hardware)
    comfy_sp.update(
        {
            "backend": "comfyui_sp",
            "display_name": "ComfyUI H3 sequence-parallel custom node",
            "available": _comfy_sp_node_present(comfy_root),
            "same_generation_multi_gpu": True,
            "planned_same_generation_multi_gpu": True,
            "actual_device_map": {
                "MiniMaxH3SPUNETLoader": "world_size=2; sequence/head sharding across cuda:0,cuda:1",
                "MiniMaxH3SPVAEDecode": "optional; keep on one GPU/CPU because VAE costs additional VRAM",
                "transformer_weights": "replicated on each GPU (not weight-sharded)",
                "cpu": "ComfyUI lowvram/CPU offload third tier",
            },
            "custom_node": "buqi-code/buqi-minimax-h3-multigpu",
            "requirements_evidence": {
                "min_gpu_capacity_gib": 24.0,
                "min_cpu_total_gib": 24.0,
                "source": "community H3 sequence-parallel node README; measured on non-T4 hardware",
            },
            "limitations": [
                "The node replicates the DiT, so fp8 is reported around 21 GiB per card; 15 GiB T4s do not meet that envelope.",
                "It is an optional custom node and is not installed by default.",
                "The VAE extension is not recommended on 24 GiB cards; CPU/VAE offload remains available.",
            ],
        }
    )
    comfy_sp.update(_capacity_decision(hardware, min_gpu_gib=24.0, min_cpu_gib=24.0))
    comfy_sp["status"] = (
        "eligible"
        if comfy_sp["available"] and comfy_sp["capacity_eligible"]
        else "custom_node_or_capacity_blocked"
    )
    attempts.append(comfy_sp)

    native = _dual_gpu_base(hardware)
    native.update(
        {
            "backend": "comfyui_native_fallback",
            "display_name": "Native ComfyUI H3 with one-GPU CPU offload",
            "available": bool(devices),
            "same_generation_multi_gpu": False,
            "actual_device_map": {
                "transformer": f"cuda:{ids[0]}" if ids else None,
                "text_encoder": "CPU/offload-managed",
                "video_vae": "CPU (--cpu-vae)",
                "audio_vae": "CPU (--cpu-vae)",
                "secondary_gpu": "unused",
            },
            "status": "explicit_fallback_only" if devices else "unavailable",
            "limitations": [
                "Native ComfyUI does not automatically pool separate GPUs for H3.",
                "This path is never selected as primary architecture unless allow_single_gpu_fallback=True.",
            ],
        }
    )
    attempts.append(native)
    return {
        "hardware": hardware,
        "device_ids": ids,
        "safety_limits": {
            "max_gpu_used_gib": GPU_SAFETY_LIMIT_GIB,
            "min_cpu_headroom_gib": CPU_HEADROOM_MIN_GIB,
            "target_cpu_headroom_gib": CPU_HEADROOM_TARGET_GIB,
        },
        "attempts": attempts,
    }


def _comfy_sp_node_present(comfy_root: Path | None) -> bool:
    if comfy_root is None:
        return False
    custom_nodes = comfy_root / "custom_nodes"
    if not custom_nodes.is_dir():
        return False
    for source in custom_nodes.rglob("*.py"):
        try:
            if "MiniMaxH3SPUNETLoader" in source.read_text(encoding="utf-8", errors="ignore"):
                return True
        except OSError:
            continue
    return False


def _module_available(name: str) -> bool:
    try:
        __import__(name)
    except Exception:
        return False
    return True


def select_backend(
    evaluation: dict[str, Any],
    *,
    requested: str = "auto",
    allow_single_gpu_fallback: bool = False,
) -> dict[str, Any]:
    """Select a backend without silently downgrading the architecture."""

    attempts = evaluation.get("attempts", [])
    by_name = {item.get("backend"): item for item in attempts}
    if requested != "auto":
        selected = by_name.get(requested)
        if selected is None:
            raise ValueError(f"Unknown H3 backend: {requested}")
        if selected.get("status") == "eligible":
            return {"status": "selected", "selected": selected, "fallback_used": False}
        if requested == "comfyui_native_fallback" and allow_single_gpu_fallback:
            return {
                "status": "selected_explicit_fallback",
                "selected": selected,
                "fallback_used": True,
            }
        return {
            "status": "blocked",
            "selected": None,
            "requested": requested,
            "reason": selected.get("status"),
            "candidate": selected,
            "fallback_available": bool(by_name.get("comfyui_native_fallback", {}).get("available")),
        }

    for name in (
        "sglang",
        "diffusers_layer_sharded",
        "diffusers_component_split",
        "diffusers_balanced",
        "comfyui_sp",
    ):
        candidate = by_name.get(name)
        if candidate and candidate.get("status") == "eligible":
            return {"status": "selected", "selected": candidate, "fallback_used": False}
    fallback = by_name.get("comfyui_native_fallback")
    if allow_single_gpu_fallback and fallback and fallback.get("available"):
        return {
            "status": "selected_explicit_fallback",
            "selected": fallback,
            "fallback_used": True,
        }
    return {
        "status": "no_verified_dual_gpu_backend",
        "selected": None,
        "fallback_used": False,
        "reason": "No multi-GPU backend met the installed/capacity checks; one-GPU fallback is disabled.",
        "fallback_available": bool(fallback and fallback.get("available")),
    }


def build_sglang_command(
    *,
    model_path: str = H3_MODEL_REPOSITORY,
    variant: str = "fl2va",
    port: int = 30000,
    resident_layers: int = 20,
    component_weight_paths: dict[str, str] | None = None,
) -> list[str]:
    """Build the documented two-GPU SGLang H3 memory-mode launch command."""

    variant = "fl2va" if variant.lower() == "t2va" else variant.lower()
    command = [
        shutil.which("sglang") or "sglang",
        "serve",
        "--model-path",
        model_path,
        "--model-variant",
        variant,
        "--num-gpus",
        "2",
        "--tp-size",
        "2",
        "--ulysses-degree",
        "1",
        "--encoder-parallel",
        "auto",
        "--performance-mode",
        "memory",
        "--layerwise-offload-components",
        "dit,text_encoder,vae",
        "--dit-offload-prefetch-size",
        "1",
        "--dit-layerwise-resident-layers",
        str(resident_layers),
        "--enable-torch-compile",
        "false",
        "--port",
        str(port),
    ]
    for component, path in (component_weight_paths or {}).items():
        command.extend([f"--component-weights-paths.{component}", path])
    return command


class SGLangClient:
    """Minimal OpenAI-compatible H3 video client for a local SGLang worker."""

    def __init__(self, base_url: str = "http://127.0.0.1:30000", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def submit(self, payload: dict[str, Any]) -> str:
        response = self.session.post(
            f"{self.base_url}/v1/videos", json=payload, timeout=self.timeout
        )
        if not response.ok:
            raise RuntimeError(f"SGLang submit failed ({response.status_code}): {response.text[-2000:]}")
        body = response.json()
        request_id = body.get("id") or body.get("request_id")
        if not request_id:
            raise RuntimeError(f"SGLang response did not contain a video id: {body}")
        return str(request_id)

    def status(self, request_id: str) -> dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/v1/videos/{request_id}", timeout=self.timeout
        )
        if not response.ok:
            raise RuntimeError(f"SGLang status failed ({response.status_code}): {response.text[-2000:]}")
        return response.json()

    def wait(self, request_id: str, *, timeout: float = 3600.0, poll_interval: float = 3.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            body = self.status(request_id)
            state = str(body.get("status", body.get("state", ""))).lower()
            if state in {"completed", "succeeded", "success"} or body.get("completed"):
                return body
            if state in {"failed", "error", "cancelled"}:
                raise RuntimeError(f"SGLang H3 request failed: {json.dumps(body)[:4000]}")
            time.sleep(poll_interval)
        raise TimeoutError(f"Timed out waiting for SGLang request {request_id}")

    def download(self, request_id: str, destination: Path) -> Path:
        response = self.session.get(
            f"{self.base_url}/v1/videos/{request_id}/content", timeout=self.timeout
        )
        if not response.ok:
            raise RuntimeError(f"SGLang download failed ({response.status_code}): {response.text[-2000:]}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)
        return destination


def measure_gpu_transfer(
    device_ids: Iterable[int] = (0, 1),
    *,
    megabytes: int = 16,
    repeats: int = 3,
) -> dict[str, Any]:
    """Measure direct and CPU-staged CUDA copies in both directions."""

    ids = list(device_ids)
    result: dict[str, Any] = {
        "device_ids": ids,
        "payload_megabytes": megabytes,
        "repeats": repeats,
        "status": "not_measured",
        "directions": [],
    }
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available() or len(ids) < 2:
            result["reason"] = "two CUDA devices are required"
            return result
        if any(index >= torch.cuda.device_count() for index in ids[:2]):
            result["reason"] = "requested CUDA device id is not visible"
            return result
        elements = max(1, megabytes * 2**20 // 4)
        for source_id, target_id in ((ids[0], ids[1]), (ids[1], ids[0])):
            with torch.cuda.device(source_id):
                source = torch.randn(elements, device=f"cuda:{source_id}")
            with torch.cuda.device(target_id):
                target = torch.empty_like(source, device=f"cuda:{target_id}")
            # Warm up allocation and the selected CUDA copy path.
            target.copy_(source, non_blocking=False)
            torch.cuda.synchronize(source_id)
            torch.cuda.synchronize(target_id)
            direct_ms: list[float] = []
            staged_ms: list[float] = []
            for _ in range(repeats):
                start = time.perf_counter()
                target.copy_(source, non_blocking=False)
                torch.cuda.synchronize(target_id)
                direct_ms.append((time.perf_counter() - start) * 1000)
            for _ in range(repeats):
                start = time.perf_counter()
                staged = source.to("cpu").to(f"cuda:{target_id}")
                torch.cuda.synchronize(target_id)
                staged_ms.append((time.perf_counter() - start) * 1000)
                del staged
            direction = {
                "source": source_id,
                "target": target_id,
                "direct_median_ms": round(median(direct_ms), 3),
                "cpu_staged_median_ms": round(median(staged_ms), 3),
                "direct_samples_ms": [round(value, 3) for value in direct_ms],
                "cpu_staged_samples_ms": [round(value, 3) for value in staged_ms],
            }
            direction["direct_to_staged_ratio"] = round(
                direction["direct_median_ms"] / max(direction["cpu_staged_median_ms"], 0.001),
                3,
            )
            direction["acceptable"] = bool(
                direction["direct_median_ms"] <= 100.0
                and direction["direct_to_staged_ratio"] <= 1.5
            )
            result["directions"].append(direction)
            del source, target
            torch.cuda.empty_cache()
        result["status"] = "measured"
        result["acceptable"] = all(item["acceptable"] for item in result["directions"])
        return result
    except Exception as exc:  # pragma: no cover - hardware-specific
        result["status"] = "measurement_failed"
        result["error"] = str(exc)
        return result


def start_sglang(
    command: list[str],
    *,
    log_path: Path,
    dry_run: bool = False,
) -> Any:
    """Start a private local SGLang worker, or return its dry-run plan."""

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    env["PYTHONUNBUFFERED"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return {
            "command": command,
            "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
            "log_path": str(log_path),
            "status": "would_start",
        }
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=log_path.parent,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return process
