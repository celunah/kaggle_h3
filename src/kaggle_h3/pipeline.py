"""End-to-end orchestration for ComfyUI and explicitly sharded H3 workers."""

from __future__ import annotations

import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backends import (
    SGLangClient,
    build_sglang_command,
    evaluate_backends,
    measure_gpu_transfer,
    select_backend,
    start_sglang,
)
from .bootstrap import (
    H3_MODEL_REVISION,
    download_selected_models,
    ensure_comfyui,
    install_comfyui_dependencies,
    install_h3_adapter_node,
    model_file_status,
    plan_execution_devices,
    start_comfyui,
)
from .comfy_api import (
    ComfyClient,
    collect_video_outputs,
    stage_input_files,
    wait_for_health,
)
from .telemetry import TelemetryRecorder, sample_memory
from .validation import extract_keyframes, validate_video, write_json
from .workflow import (
    H3Request,
    build_workflow,
    request_as_dict,
    select_production_mode,
    validate_production_workflow_shape,
    workflow_sha256,
)


def _utc_id(seed: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}_seed{seed}"


def _result_paths(project_root: Path, run_id: str) -> tuple[Path, Path]:
    result_dir = project_root / "results"
    return result_dir / f"{run_id}.json", result_dir / run_id


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    write_json(path, manifest)


def _backend_attempt_summary(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "backend": item.get("backend"),
            "status": item.get("status"),
            "available": item.get("available"),
            "same_generation_multi_gpu": item.get("same_generation_multi_gpu"),
            "planned_same_generation_multi_gpu": item.get("planned_same_generation_multi_gpu"),
            "observed_same_generation_multi_gpu": item.get("observed_same_generation_multi_gpu"),
            "generation_attempted": item.get("generation_attempted"),
            "planned_device_map": item.get("actual_device_map"),
            "actual_device_map": item.get("observed_device_map"),
            "observed_device_map": item.get("observed_device_map"),
            "capacity_eligible": item.get("capacity_eligible"),
            "limitations": item.get("limitations", []),
        }
        for item in evaluation.get("attempts", [])
    ]


def _sglang_payload(request: H3Request, mode: str, media_dir: Path | None = None) -> dict[str, Any]:
    from .workflow import dimensions, frame_length, build_prompt

    width, height = dimensions(request)
    frame_info = frame_length(request)
    payload: dict[str, Any] = {
        "model": "MiniMaxAI/MiniMax-H3",
        "task": mode.lower(),
        "prompt": build_prompt(request, mode),
        "seconds": frame_info["actual_seconds"],
        "target": {
            "short_edge": min(width, height),
            "aspect_ratio": request.aspect_ratio,
            "duration_seconds": frame_info["actual_seconds"],
        },
        "conditions": [],
        "num_outputs_per_prompt": 1,
        "num_inference_steps": (
            int(request.turbo_steps) if request.turbo_mode else request.steps or 50
        ),
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
        "seed": int(request.seed),
    }
    conditions: list[dict[str, Any]] = []
    for index, asset in enumerate(request.assets(), start=1):
        path = Path(asset.path)
        if media_dir is not None:
            destination = media_dir / f"{index:02d}_{path.name}"
            if path.resolve() != destination.resolve():
                shutil.copy2(path, destination)
            path = destination
        conditions.append(
            {
                "type": asset.media_type,
                "role": "reference",
                "h3_role": asset.role,
                "uri": path.resolve().as_uri(),
            }
        )
    if mode.upper() == "FL2VA":
        for index, asset in enumerate(request.endpoint_assets(), start=len(request.assets()) + 1):
            path = Path(asset.path)
            if media_dir is not None:
                destination = media_dir / f"{index:02d}_{path.name}"
                if path.resolve() != destination.resolve():
                    shutil.copy2(path, destination)
                path = destination
            conditions.append(
                {
                    "type": "image",
                    "role": "keyframe",
                    "frame_index": 0 if asset.role == "first_frame" else -1,
                    "uri": path.resolve().as_uri(),
                }
            )
    if conditions:
        payload["conditions"] = conditions
    return payload


def _adapter_request_summary(request: H3Request) -> dict[str, Any]:
    return {
        "turbo_mode": bool(request.turbo_mode),
        "turbo_steps": int(request.turbo_steps),
        "mute_generated_audio": bool(request.mute_generated_audio),
        "conditioning_mode": request.conditioning_mode,
        "slots": [
            {"slot": 1, "adapter_id": request.adapter_1, "strength": request.adapter_1_strength},
            {"slot": 2, "adapter_id": request.adapter_2, "strength": request.adapter_2_strength},
            {"slot": 3, "adapter_id": request.adapter_3, "strength": request.adapter_3_strength},
        ],
        "implementation": "KaggleH3AdapterStack + KaggleH3TurboSampler for ComfyUI",
    }


def _request_uses_adapter_stack(request: H3Request) -> bool:
    return bool(
        request.turbo_mode
        or any(
            adapter != "None"
            for adapter in (request.adapter_1, request.adapter_2, request.adapter_3)
        )
    )


def _initial_manifest(
    *,
    request: H3Request,
    mode: str,
    run_id: str,
    evaluation: dict[str, Any],
    selection: dict[str, Any],
    workflow: dict[str, Any],
    workflow_path: Path,
) -> dict[str, Any]:
    selected = selection.get("selected") or {}
    references = []
    for asset in request.assets() + request.endpoint_assets():
        references.append(
            {
                "role": asset.role,
                "media_type": asset.media_type,
                "path": str(asset.path),
            }
        )
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "request": request_as_dict(request),
        "adapter_stack": _adapter_request_summary(request),
        "mode": mode,
        "references": references,
        "h3": {
            "repository": "MiniMaxAI/MiniMax-H3",
            "comfy_model_repository": "WarmBloodAban/Minimax-h3_Singularity",
            "selection_revision": H3_MODEL_REVISION,
        },
        "backend_selection": {
            "status": selection.get("status"),
            "backend": selected.get("backend"),
            "fallback_used": selection.get("fallback_used", False),
            "reason": selection.get("reason"),
        },
        "backend_evaluation": _backend_attempt_summary(evaluation),
        "planned_device_map": selected.get("actual_device_map"),
        "device_map": selected.get("observed_device_map"),
        "quantization_profile": {
            "diffusion": "WarmBloodAban MiniMax H3 Singularity Ref2VA pruned INT8",
            "text_encoder": "Comfy-Org Qwen3-VL NVFP4-AWQ selected by default",
            "video_vae": "FP16",
            "audio_vae": "FP32",
            "compatibility": "must be confirmed on the target Kaggle T4 image; no BF16 full checkpoint is the default",
        },
        "cpu_ram": {
            "third_offload_tier_configured": bool(selected.get("cpu_ram_third_tier", True)),
            "direct_tensor_attribution": "not available from nvidia-smi/psutil alone",
        },
        "hardware": evaluation.get("hardware"),
        "safety_limits": evaluation.get("safety_limits"),
        "workflow": {
            "path": str(workflow_path),
            "sha256": workflow_sha256(workflow),
            "shape": validate_production_workflow_shape(workflow),
            "api": workflow,
        },
        "timing": {},
        "transfer_benchmark": None,
        "telemetry": None,
        "attempts": [],
        "outputs": [],
        "validation": None,
        "notes": [
            "A backend is considered verified for same-generation multi-GPU use only when telemetry shows both GPU indices active during this run.",
            "CPU RAM remains an explicit third offload tier; safety headroom is not optional.",
        ],
    }


def run_generation(
    request: H3Request,
    *,
    project_root: Path,
    comfy_root: Path | None = None,
    backend: str = "auto",
    allow_single_gpu_fallback: bool = False,
    comfy_url: str = "http://127.0.0.1:8188",
    sglang_url: str = "http://127.0.0.1:30000",
    start_local_workers: bool = True,
    download_models: bool = False,
    dry_run: bool = False,
    max_attempts: int = 1,
    telemetry_interval_seconds: float = 1.0,
) -> dict[str, Any]:
    """Run one request or produce a complete, honest dry-run manifest."""

    project_root = project_root.resolve()
    comfy_root = (comfy_root or project_root / "ComfyUI").resolve()
    mode = select_production_mode(request)
    run_id = _utc_id(request.seed)
    manifest_path, run_dir = _result_paths(project_root, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluation = evaluate_backends(comfy_root=comfy_root)
    selection = select_backend(
        evaluation,
        requested=backend,
        allow_single_gpu_fallback=allow_single_gpu_fallback,
    )
    selected = selection.get("selected") or {}
    selected_backend = selected.get("backend")
    execution_plan = plan_execution_devices(
        evaluation["hardware"],
        backend_supports_sharding=bool(selected.get("same_generation_multi_gpu")),
    )
    transfer = (
        measure_gpu_transfer(evaluation.get("device_ids", []))
        if not dry_run
        else {"status": "skipped_dry_run", "reason": "requires a CUDA runtime"}
    )

    workflow = build_workflow(
        request,
        mode=mode,
        run_id=run_id,
        # Production always uses the phase-aware native loader so the pinned
        # Singularity checkpoint and matching adapter contract are enforced.
        loader="native",
        device_ids=tuple(evaluation.get("device_ids", [0, 1])[:2] or [0, 1]),
    )
    workflow_path = run_dir / "workflow_api.json"
    write_json(workflow_path, workflow)
    manifest = _initial_manifest(
        request=request,
        mode=mode,
        run_id=run_id,
        evaluation=evaluation,
        selection=selection,
        workflow=workflow,
        workflow_path=workflow_path,
    )
    manifest["execution_plan"] = execution_plan
    manifest["transfer_benchmark"] = transfer
    if dry_run:
        manifest["status"] = "dry_run_only"
        manifest["attempts"].append(
            {
                "attempt": 1,
                "backend": selected_backend,
                "status": "not_executed",
                "reason": "dry_run=True; no GPU job was launched",
            }
        )
        if selection.get("status") == "no_verified_dual_gpu_backend":
            manifest["notes"].append(
                "No dual-GPU backend is verified by local capability checks; this is expected until Kaggle hardware validation."
            )
        _write_manifest(manifest_path, manifest)
        return manifest

    if not selected_backend:
        manifest["status"] = "blocked_before_generation"
        manifest["attempts"].append(
            {
                "attempt": 1,
                "backend": backend,
                "status": "blocked",
                "reason": selection.get("reason"),
            }
        )
        _write_manifest(manifest_path, manifest)
        return manifest

    if _request_uses_adapter_stack(request) and selected_backend not in {
        "comfyui_native_fallback",
        "comfyui_sp",
    }:
        reason = (
            f"The current {selected_backend} worker has no H3 ComfyUI LoRA/runtime-config adapter path. "
            "Refusing to silently run without the selected adapter stack; use a ComfyUI backend or add an explicit backend adapter."
        )
        manifest["status"] = "blocked_before_generation"
        manifest["attempts"].append(
            {"attempt": 1, "backend": selected_backend, "status": "blocked", "reason": reason}
        )
        manifest["notes"].append(reason)
        _write_manifest(manifest_path, manifest)
        return manifest

    if selected_backend in {"comfyui_native_fallback", "comfyui_sp"}:
        # Establish the checkout before model download; otherwise the created
        # models/ directory would make ensure_comfyui correctly reject the path
        # as a non-empty unrelated directory.
        manifest["comfyui_bootstrap"] = ensure_comfyui(comfy_root)
    if download_models and selected_backend in {"comfyui_native_fallback", "comfyui_sp"}:
        manifest["model_files"] = download_selected_models(comfy_root, mode)
    elif selected_backend in {"comfyui_native_fallback", "comfyui_sp"}:
        # The production loader owns the mutually exclusive Singularity
        # diffusion checkpoint and downloads it just in time. Only the shared
        # text/VAE assets are required before ComfyUI starts.
        manifest["model_files"] = model_file_status(
            comfy_root, mode, include_diffusion=False
        )
    else:
        manifest["model_files"] = {
            "status": "managed_by_selected_backend",
            "repository": "MiniMaxAI/MiniMax-H3",
            "mode": mode,
            "download_requested": download_models,
            "note": "No ComfyUI quantization was downloaded for this non-Comfy worker.",
        }

    for attempt_number in range(1, max(1, max_attempts) + 1):
        attempt_seed = int(request.seed) + attempt_number - 1
        if attempt_seed != request.seed:
            request.seed = attempt_seed
            workflow = build_workflow(
                request,
                mode=mode,
                run_id=run_id,
                loader="native",
            )
            write_json(workflow_path, workflow)
            manifest["workflow"] = {
                "path": str(workflow_path),
                "sha256": workflow_sha256(workflow),
                "shape": validate_production_workflow_shape(workflow),
                "api": workflow,
            }
        attempt: dict[str, Any] = {
            "attempt": attempt_number,
            "backend": selected_backend,
            "seed": attempt_seed,
            "status": "running",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest["attempts"].append(attempt)
        _write_manifest(manifest_path, manifest)
        telemetry_path = run_dir / f"telemetry_attempt_{attempt_number:02d}.jsonl"
        recorder = TelemetryRecorder(telemetry_path, interval_seconds=telemetry_interval_seconds)
        baseline_memory = sample_memory()
        worker_process: Any = None
        comfy_process: Any = None
        started = time.perf_counter()
        try:
            recorder.start()
            if selected_backend in {"comfyui_native_fallback", "comfyui_sp"}:
                if not manifest["model_files"].get("all_present"):
                    raise RuntimeError(
                        "Required ComfyUI model files are missing. Set download_models=True or mount the exact files before quick generation."
                    )
                attempt["comfyui"] = ensure_comfyui(comfy_root)
                attempt["h3_adapter_node"] = install_h3_adapter_node(
                    comfy_root, project_root
                )
                if start_local_workers:
                    attempt["comfyui_dependencies"] = install_comfyui_dependencies(comfy_root)
                else:
                    attempt["comfyui_dependencies"] = {"status": "external_worker_expected"}
                if start_local_workers:
                    comfy_process = start_comfyui(
                        comfy_root,
                        execution_plan=execution_plan,
                        log_dir=run_dir,
                        # The native backend is explicitly the one-GPU
                        # fallback; retain its CPU-VAE behavior. The direct
                        # phase-aware notebook launch leaves this disabled so
                        # its decode nodes can target GPU0/GPU1.
                        cpu_vae=True,
                    )
                client = ComfyClient(comfy_url)
                wait_for_health(client)
                required = [
                    "UNETLoader",
                    "CLIPLoader",
                    "VAELoader",
                    "MiniMaxH3ImageToVideo" if mode != "Ref2VA" else "KaggleH3Ref2VAConditioning",
                    "MiniMaxH3SigmaShift",
                    "RandomNoise",
                    "KSamplerSelect",
                    "BasicScheduler",
                    "BasicGuider",
                    "SamplerCustomAdvanced",
                    "VAEDecode",
                    "VAEDecodeAudio",
                    "CreateVideo",
                    "SaveVideo",
                    "KaggleH3AdapterStack",
                    "KaggleH3TurboSampler",
                    "KaggleH3VAEDecode",
                    "KaggleH3AudioVAEDecode",
                ]
                if request.assets():
                    required.extend(["LoadImage", "LoadVideo", "GetVideoComponents", "LoadAudio"])
                if selected_backend == "comfyui_sp":
                    required.extend(["MiniMaxH3SPUNETLoader", "MiniMaxH3SPVAEDecode"])
                node_check = client.check_required_nodes(required)
                attempt["node_check"] = node_check
                if not node_check["all_present"]:
                    raise RuntimeError(f"ComfyUI is missing required nodes: {node_check['missing']}")
                staged = stage_input_files(request.all_paths(), comfy_root, run_id)
                workflow = build_workflow(
                    request,
                    mode=mode,
                    input_files=staged,
                    run_id=run_id,
                    loader="native",
                    device_ids=tuple(
                        execution_plan.get("model_device_ids")
                        or execution_plan.get("visible_device_ids")
                        or (0,)
                    ),
                )
                write_json(workflow_path, workflow)
                prompt_id = client.queue_prompt(workflow)
                attempt["prompt_id"] = prompt_id
                history = client.wait_for_completion(prompt_id)
                outputs = collect_video_outputs(
                    client,
                    history,
                    comfy_root=comfy_root,
                    destination_dir=project_root / "results",
                    run_id=run_id,
                )
            elif selected_backend in {"sglang"}:
                if start_local_workers:
                    command = build_sglang_command(
                        variant="fl2va" if mode == "T2VA" else mode.lower(), port=30000
                    )
                    worker_process = start_sglang(command, log_path=run_dir / "sglang.log")
                client = SGLangClient(sglang_url)
                payload = _sglang_payload(request, mode, run_dir / "media")
                request_id = client.submit(payload)
                attempt["request_id"] = request_id
                client.wait(request_id)
                outputs = [client.download(request_id, project_root / "results" / f"{run_id}_01.mp4")]
            elif selected_backend == "diffusers_layer_sharded":
                from .diffusers_worker import DiffusersH3Worker

                # Kaggle's persistent working volume is only about 20 GiB,
                # while /kaggle/tmp is the large scratch tier intended for
                # Accelerate's layer offload files. Fall back to the run dir
                # on non-Kaggle machines where that mount does not exist.
                layer_offload_root = (
                    Path("/kaggle/tmp")
                    if Path("/kaggle/tmp").is_dir()
                    else run_dir
                )
                worker = DiffusersH3Worker(
                    workflow=mode.lower(),
                    layer_sharded=True,
                    device_ids=tuple(evaluation.get("device_ids", [])),
                    offload_dir=layer_offload_root / "minimax-h3-layer-offload" / run_id,
                )
                worker.load()
                attempt["device_map"] = worker.actual_device_map
                attempt["layer_dispatch"] = worker.dispatch_report
                output_path = project_root / "results" / f"{run_id}_01.mp4"
                outputs = [worker.generate(request, output_path)]
            elif selected_backend == "diffusers_component_split":
                from .diffusers_worker import DiffusersH3Worker

                worker = DiffusersH3Worker(workflow=mode.lower())
                attempt["device_map"] = worker.actual_device_map
                output_path = project_root / "results" / f"{run_id}_01.mp4"
                outputs = [worker.generate(request, output_path)]
            elif selected_backend == "diffusers_balanced":
                raise RuntimeError(
                    "Diffusers balanced placement is a capability probe only; use the explicit component-split worker after inspecting its map."
                )
            else:
                raise RuntimeError(f"Backend {selected_backend!r} has no generation adapter yet")
            attempt["outputs"] = [str(path) for path in outputs]
            attempt["status"] = "completed"
            if attempt.get("device_map"):
                manifest["device_map"] = attempt["device_map"]
            attempt["validation"] = validate_video(outputs[0], request)
            keyframes = extract_keyframes(outputs[0], run_dir / "keyframes")
            attempt["keyframes"] = [str(path) for path in keyframes]
            manifest["outputs"] = [str(path) for path in outputs]
            manifest["validation"] = attempt["validation"]
            manifest["status"] = "completed"
            break
        except Exception as exc:  # preserve failed attempts in the manifest
            attempt["status"] = "failed"
            attempt["error"] = str(exc)
            attempt["traceback_tail"] = traceback.format_exc()[-4000:]
            manifest["status"] = "failed" if attempt_number >= max_attempts else "retrying"
        finally:
            attempt["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            attempt["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
            manifest["timing"][f"attempt_{attempt_number}"] = attempt["elapsed_seconds"]
            manifest["telemetry"] = recorder.stop()
            baseline_by_gpu = {
                str(item.get("index")): item.get("used_mib")
                for item in baseline_memory.get("gpus", [])
            }
            manifest["telemetry"]["baseline"] = baseline_memory
            for item in manifest["telemetry"].get("peak_by_gpu", []):
                baseline_used = baseline_by_gpu.get(str(item.get("index")))
                item["baseline_used_mib"] = baseline_used
                item["delta_used_mib"] = (
                    round(item.get("peak_used_mib", 0) - baseline_used, 3)
                    if isinstance(baseline_used, (int, float))
                    else None
                )
                item["same_job_activity"] = bool(
                    isinstance(item.get("delta_used_mib"), (int, float))
                    and item["delta_used_mib"] >= 256
                )
            manifest["cpu_ram"]["minimum_observed_available_gib"] = manifest["telemetry"].get(
                "minimum_cpu_available_gib"
            )
            manifest["safety_check"] = {
                "gpu_headroom_pass": all(
                    item.get("safety_headroom_pass", False)
                    for item in manifest["telemetry"].get("peak_by_gpu", [])
                ),
                "cpu_headroom_pass": manifest["telemetry"].get("cpu_headroom_pass"),
                "max_gpu_used_gib": max(
                    [item.get("peak_used_gib", 0.0) for item in manifest["telemetry"].get("peak_by_gpu", [])],
                    default=0.0,
                ),
                "minimum_cpu_available_gib": manifest["telemetry"].get(
                    "minimum_cpu_available_gib"
                ),
            }
            active_gpu_count = sum(
                1
                for item in manifest["telemetry"].get("peak_by_gpu", [])
                if item.get("same_job_activity") is True
            )
            manifest["verification"] = {
                "both_gpus_used_for_same_generation": bool(
                    selected.get("same_generation_multi_gpu") is True
                    and active_gpu_count >= 2
                ),
                "active_gpu_count_by_memory_delta": active_gpu_count,
                "method": "baseline-to-peak nvidia-smi memory delta >= 256 MiB on each GPU plus a backend that declares same-generation multi-GPU placement",
                "status": "verified"
                if selected.get("same_generation_multi_gpu") is True and active_gpu_count >= 2
                else "not_verified",
            }
            for candidate in manifest.get("backend_evaluation", []):
                if candidate.get("backend") == selected_backend:
                    candidate["generation_attempted"] = True
                    candidate["observed_same_generation_multi_gpu"] = manifest["verification"][
                        "both_gpus_used_for_same_generation"
                    ]
                    candidate["observed_device_map"] = manifest.get("device_map")
                    candidate["actual_device_map"] = manifest.get("device_map")
            if worker_process is not None and hasattr(worker_process, "poll"):
                worker_process.terminate()
                try:
                    worker_process.wait(timeout=10)
                except Exception:
                    worker_process.kill()
            if comfy_process is not None:
                comfy_process.stop()
            _write_manifest(manifest_path, manifest)
    return manifest


def run_quick_validation(
    *,
    project_root: Path,
    comfy_root: Path | None = None,
    allow_single_gpu_fallback: bool = False,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Run the required small precondition before medium/full generations."""

    request = H3Request(
        prompt="A simple H3 test shot: a calm close-up with stable subject identity and gentle ambient audio.",
        character_references=[project_root / "smoke_assets" / "CHARACTER_REFERENCE.png"],
        scene_references=[project_root / "smoke_assets" / "SCENE_REFERENCE.png"],
        duration_seconds=5.0,
        quality_mode="quick",
        seed=7,
    )
    result = run_generation(
        request,
        project_root=project_root,
        comfy_root=comfy_root,
        allow_single_gpu_fallback=allow_single_gpu_fallback,
        dry_run=dry_run,
        max_attempts=1,
    )
    result["quick_validation"] = True
    return result


def benchmark_backends(
    request: H3Request,
    *,
    project_root: Path,
    comfy_root: Path | None = None,
    backends: tuple[str, ...] = (
        "sglang",
        "diffusers_component_split",
        "diffusers_balanced",
        "comfyui_sp",
        "comfyui_native_fallback",
    ),
    allow_single_gpu_fallback: bool = True,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Attempt each requested backend sequentially and compare same-request timing.

    Dry-run is the safe local default.  Set ``dry_run=False`` only in Kaggle
    after models and the selected worker dependencies are installed.
    """

    rows: list[dict[str, Any]] = []
    for backend in backends:
        result = run_generation(
            request,
            project_root=project_root,
            comfy_root=comfy_root,
            backend=backend,
            allow_single_gpu_fallback=allow_single_gpu_fallback,
            dry_run=dry_run,
            max_attempts=1,
        )
        rows.append(
            {
                "backend": backend,
                "status": result.get("status"),
                "fallback_used": result.get("backend_selection", {}).get("fallback_used"),
                "timing": result.get("timing"),
                "device_map": result.get("device_map"),
                "telemetry": result.get("telemetry"),
                "transfer_benchmark": result.get("transfer_benchmark"),
                "outputs": result.get("outputs"),
            }
        )
    result = {"status": "dry_run_only" if dry_run else "completed", "backends": rows}
    write_json(project_root / "results" / "backend_benchmark.json", result)
    return result
