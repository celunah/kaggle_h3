"""Kaggle/runtime bootstrap and hardware inspection helpers.

The module deliberately does not download model weights or launch a public
server unless the caller asks it to.  The default H3 profile is the small
Comfy-Org repackaged profile documented in README.md.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model_manager import H3_DIFFUSION_MODEL_SPECS


COMFYUI_REPOSITORY = "https://github.com/Comfy-Org/ComfyUI.git"
COMFYUI_REF = os.environ.get("KAGGLE_H3_COMFYUI_REF", "v0.34.0")
H3_MODEL_REPOSITORY = "Comfy-Org/MiniMax-H3"
H3_MODEL_REVISION = os.environ.get(
    "KAGGLE_H3_MODEL_REVISION",
    "a98869194787969724c7425d95d0ed73ce9202af",
)

COMMON_MODEL_FILES: dict[str, tuple[str, str]] = {
    "clip": (
        "text_encoders",
        "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    ),
    "video_vae": ("vae", "minimax_h3_video_vae_fp16.safetensors"),
    "audio_vae": ("vae", "minimax_h3_audio_vae_fp32.safetensors"),
}
MODE_MODEL_FILES: dict[str, tuple[str, str]] = {
    variant: ("diffusion_models", spec.filename)
    for variant, spec in H3_DIFFUSION_MODEL_SPECS.items()
}
MODE_MODEL_FILES["T2VA"] = MODE_MODEL_FILES["FL2VA"]


@dataclass(frozen=True)
class ComfyProcess:
    """Handle for a privately bound ComfyUI subprocess."""

    process: subprocess.Popen[str]
    log_path: Path
    command: tuple[str, ...] = ()
    cuda_visible_devices: str | None = None

    def stop(self, timeout: float = 10.0) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


def _safe_json(value: Any) -> Any:
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def detect_hardware() -> dict[str, Any]:
    """Return current CUDA and host-memory facts without assuming two GPUs."""

    result: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_devices": [],
        "cpu_memory": {},
        "warnings": [],
    }

    try:
        import psutil  # type: ignore

        memory = psutil.virtual_memory()
        result["cpu_memory"] = {
            "total_bytes": int(memory.total),
            "available_bytes": int(memory.available),
            "used_bytes": int(memory.used),
            "available_gib": round(memory.available / 2**30, 2),
        }
    except Exception as exc:  # pragma: no cover - depends on runtime package
        result["warnings"].append(f"psutil unavailable: {exc}")

    try:
        import torch  # type: ignore

        result["torch_version"] = torch.__version__
        result["cuda_version"] = getattr(torch.version, "cuda", None)
        result["cuda_available"] = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if result["cuda_available"] else 0
        result["cuda_device_count"] = count
        for index in range(count):
            name = str(torch.cuda.get_device_name(index))
            try:
                free, total = torch.cuda.mem_get_info(index)
                free_bytes = int(free)
                total_bytes = int(total)
            except Exception as exc:  # pragma: no cover - driver dependent
                free_bytes = None
                total_bytes = None
                result["warnings"].append(
                    f"Could not query CUDA memory for device {index}: {exc}"
                )
            result["cuda_devices"].append(
                {
                    "index": index,
                    "name": name,
                    "is_t4": "t4" in name.lower(),
                    "free_bytes": free_bytes,
                    "total_bytes": total_bytes,
                    "free_gib": round(free_bytes / 2**30, 2)
                    if free_bytes is not None
                    else None,
                    "total_gib": round(total_bytes / 2**30, 2)
                    if total_bytes is not None
                    else None,
                }
            )
    except Exception as exc:  # pragma: no cover - depends on runtime package
        result["warnings"].append(f"PyTorch CUDA inspection unavailable: {exc}")

    if result["cuda_device_count"] == 0:
        smi = shutil.which("nvidia-smi")
        if smi:
            try:
                completed = subprocess.run(
                    [
                        smi,
                        "--query-gpu=index,name,memory.total,memory.free",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                for line in completed.stdout.splitlines():
                    fields = [field.strip() for field in line.split(",")]
                    if len(fields) != 4:
                        continue
                    result["cuda_devices"].append(
                        {
                            "index": int(fields[0]),
                            "name": fields[1],
                            "is_t4": "t4" in fields[1].lower(),
                            "total_mib": int(fields[2]),
                            "free_mib": int(fields[3]),
                        }
                    )
                result["cuda_device_count"] = len(result["cuda_devices"])
                result["cuda_available"] = bool(result["cuda_devices"])
            except Exception as exc:  # pragma: no cover - driver dependent
                result["warnings"].append(f"nvidia-smi inspection failed: {exc}")

    if result["cuda_device_count"] == 0:
        result["warnings"].append("No CUDA device was detected.")
    elif result["cuda_device_count"] < 2:
        result["warnings"].append(
            "Fewer than two CUDA devices are visible; continuing with one-GPU mode."
        )
    elif not all(device.get("is_t4", False) for device in result["cuda_devices"]):
        result["warnings"].append(
            "The visible GPUs are not all NVIDIA T4 devices; review VRAM before rendering."
        )
    return result


def plan_execution_devices(
    hardware: dict[str, Any], *, backend_supports_sharding: bool = False
) -> dict[str, Any]:
    """Choose placement without treating separate GPUs as one VRAM pool."""

    devices = list(hardware.get("cuda_devices", []))
    if not devices:
        return {
            "backend_supports_sharding": backend_supports_sharding,
            "mode": "cpu_only_unavailable_for_h3",
            "primary_device": None,
            "visible_device_ids": [],
            "model_device_ids": [],
            "notes": ["H3 generation requires a CUDA-capable ComfyUI runtime."],
        }

    if backend_supports_sharding and len(devices) > 1:
        return {
            "backend_supports_sharding": True,
            "mode": "explicit_backend_sharding",
            "primary_device": devices[0]["index"],
            "visible_device_ids": [device["index"] for device in devices],
            "model_device_ids": [device["index"] for device in devices],
            "notes": [
                "The selected backend owns explicit placement; this is not an automatic VRAM pool."
            ],
        }

    primary = int(devices[0]["index"])
    return {
        "backend_supports_sharding": False,
        "mode": "single_gpu_cpu_offload",
        "primary_device": primary,
        "visible_device_ids": [primary],
        "model_device_ids": [primary],
        "secondary_devices": [int(device["index"]) for device in devices[1:]],
        "notes": [
            "Native ComfyUI H3 has no automatic multi-GPU model pool; use one primary GPU with CPU offload.",
            "The secondary GPU is detected and reported but is not claimed for H3 model execution.",
        ],
    }


def required_model_files(
    mode: str, *, include_diffusion: bool = True
) -> dict[str, tuple[str, str]]:
    """Return common files and optionally the diffusion file for ``mode``."""

    normalized = mode.strip().upper()
    canonical = {
        "REF2VA": "Ref2VA",
        "FL2VA": "FL2VA",
        "T2VA": "T2VA",
    }.get(normalized)
    if canonical is None:
        raise ValueError(f"Unsupported H3 mode: {mode}")
    files = dict(COMMON_MODEL_FILES)
    if include_diffusion:
        files["diffusion_model"] = MODE_MODEL_FILES[canonical]
    return files


def model_file_status(
    comfy_root: Path,
    mode: str,
    *,
    models_root: Path | None = None,
    include_diffusion: bool = True,
) -> dict[str, Any]:
    models_root = models_root or (comfy_root / "models")
    status: dict[str, Any] = {
        "repository": H3_MODEL_REPOSITORY,
        "revision": H3_MODEL_REVISION,
        "mode": mode,
        "files": {},
        "all_present": True,
    }
    for role, (directory, filename) in required_model_files(
        mode, include_diffusion=include_diffusion
    ).items():
        path = models_root / directory / filename
        present = path.is_file()
        status["files"][role] = {
            "directory": directory,
            "filename": filename,
            "path": str(path),
            "present": present,
            "size_bytes": path.stat().st_size if present else None,
        }
        status["all_present"] = status["all_present"] and present
    return status


def install_python_dependencies(project_root: Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Install the small notebook-side dependency set into this Python."""

    requirements = project_root / "requirements-kaggle.txt"
    if not requirements.is_file():
        raise FileNotFoundError(requirements)
    command = [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)]
    result: dict[str, Any] = {"command": command, "requirements": str(requirements)}
    if dry_run:
        result["status"] = "skipped_dry_run"
        return result
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    result["status"] = "installed"
    result["stdout_tail"] = completed.stdout[-2000:]
    result["stderr_tail"] = completed.stderr[-2000:]
    return result


def install_comfyui_dependencies(
    comfy_root: Path, *, dry_run: bool = False
) -> dict[str, Any]:
    """Install the pinned checkout's Python requirements into this interpreter."""

    requirements = comfy_root / "requirements.txt"
    if not requirements.is_file():
        raise FileNotFoundError(requirements)
    command = [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)]
    result: dict[str, Any] = {"command": command, "requirements": str(requirements)}
    if dry_run:
        result["status"] = "skipped_dry_run"
        return result
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    result["status"] = "installed"
    result["stdout_tail"] = completed.stdout[-2000:]
    result["stderr_tail"] = completed.stderr[-2000:]
    return result


def install_h3_adapter_node(
    comfy_root: Path, project_root: Path, *, dry_run: bool = False
) -> dict[str, Any]:
    """Install the local H3 adapter node and its dependency-light core.

    The core is copied beside the custom node because a separately launched
    ComfyUI process does not necessarily inherit the notebook's ``src`` path.
    The files are ordinary Python/JSON sources; no model weights are copied and
    the operation never edits a checkpoint.
    """

    source_node = project_root / "custom_nodes" / "kaggle_h3_adapters.py"
    source_core = project_root / "src" / "kaggle_h3" / "adapters.py"
    source_catalog = project_root / "custom_nodes" / "kaggle_h3_adapter_catalog.json"
    source_phase_runtime = project_root / "src" / "kaggle_h3" / "phase_runtime.py"
    source_layer_sharding = project_root / "src" / "kaggle_h3" / "layer_sharding.py"
    source_ref2va = project_root / "src" / "kaggle_h3" / "ref2va.py"
    source_fp_diagnostics = project_root / "src" / "kaggle_h3" / "fp_diagnostics.py"
    source_model_manager = project_root / "src" / "kaggle_h3" / "model_manager.py"
    sources = (
        source_node,
        source_core,
        source_catalog,
        source_phase_runtime,
        source_layer_sharding,
        source_ref2va,
        source_fp_diagnostics,
        source_model_manager,
    )
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing H3 adapter node source files: " + ", ".join(missing))
    target_dir = comfy_root / "custom_nodes"
    target_files = (
        target_dir / "kaggle_h3_adapters.py",
        target_dir / "kaggle_h3_adapter_core.py",
        target_dir / "kaggle_h3_adapter_catalog.json",
        target_dir / "kaggle_h3_phase_runtime.py",
        target_dir / "kaggle_h3_layer_sharding.py",
        target_dir / "kaggle_h3_ref2va.py",
        target_dir / "kaggle_h3_fp_diagnostics.py",
        target_dir / "kaggle_h3_model_manager.py",
    )
    legacy_target_files = (
        target_dir / "celune_h3_adapters.py",
        target_dir / "celune_h3_adapter_core.py",
        target_dir / "celune_h3_adapter_catalog.json",
        target_dir / "celune_h3_phase_runtime.py",
        target_dir / "celune_h3_layer_sharding.py",
        target_dir / "celune_h3_ref2va.py",
    )
    result: dict[str, Any] = {
        "status": "would_install" if dry_run else "installed",
        "sources": [str(path) for path in sources],
        "targets": [str(path) for path in target_files],
        "removed_legacy_targets": [str(path) for path in legacy_target_files if path.is_file()],
        "node_ids": [
            "KaggleH3SmokeReference",
            "KaggleH3Ref2VAConditioning",
            "KaggleH3ShardedDiffusionLoader",
            "KaggleH3TextEncoderLoader",
            "KaggleH3VAELoader",
            "KaggleH3PhaseDispatch",
            "KaggleH3AdapterStack",
            "KaggleH3TurboSampler",
            "KaggleH3VAEDecode",
            "KaggleH3AudioVAEDecode",
        ],
    }
    if dry_run:
        return result
    target_dir.mkdir(parents=True, exist_ok=True)
    for legacy_target in legacy_target_files:
        if legacy_target.is_file():
            legacy_target.unlink()
    for source, target in zip(sources, target_files):
        shutil.copy2(source, target)
    return result


def stage_smoke_assets(
    comfy_root: Path, project_root: Path, *, dry_run: bool = False
) -> dict[str, Any]:
    """Copy the checked-in smoke references into ComfyUI's input directory."""

    asset_dir = project_root / "smoke_assets"
    asset_names = ("CHARACTER_REFERENCE.png", "SCENE_REFERENCE.png")
    sources = [asset_dir / name for name in asset_names]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing smoke workflow assets: " + ", ".join(missing))
    input_dir = comfy_root / "input"
    targets = [input_dir / name for name in asset_names]
    result: dict[str, Any] = {
        "status": "would_stage" if dry_run else "staged",
        "sources": [str(path) for path in sources],
        "targets": [str(path) for path in targets],
    }
    if dry_run:
        return result
    input_dir.mkdir(parents=True, exist_ok=True)
    for source, target in zip(sources, targets):
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    return result


def ensure_comfyui(
    comfy_root: Path, *, ref: str = COMFYUI_REF, dry_run: bool = False
) -> dict[str, Any]:
    """Clone the pinned ComfyUI release only when the checkout is absent."""

    comfy_root = comfy_root.resolve()
    result: dict[str, Any] = {
        "path": str(comfy_root),
        "repository": COMFYUI_REPOSITORY,
        "requested_ref": ref,
    }
    if (comfy_root / "main.py").is_file():
        result["status"] = "present"
        try:
            result["resolved_revision"] = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=comfy_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except Exception:
            result["resolved_revision"] = None
        return result
    if dry_run:
        result["status"] = "would_clone"
        return result
    if comfy_root.exists() and any(comfy_root.iterdir()):
        raise RuntimeError(
            f"Refusing to clone into non-empty path that is not ComfyUI: {comfy_root}"
        )
    comfy_root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", ref, COMFYUI_REPOSITORY, str(comfy_root)],
        check=True,
    )
    result["status"] = "cloned"
    result["resolved_revision"] = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=comfy_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return result


def ensure_github_checkout(
    repository: str,
    destination: Path,
    *,
    ref: str = "main",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Clone or fast-forward a public GitHub project into writable storage.

    Kaggle datasets are read-only and large model files should not be committed
    to GitHub.  This helper is therefore intended for the small package and
    notebook source only; H3 weights continue to download into the separately
    configured ``/kaggle/tmp`` model store.  Existing uncommitted checkout
    changes are never overwritten: the detached checkout update fails with an
    actionable Git error instead.
    """

    repository = str(repository).strip()
    ref = str(ref).strip() or "main"
    if not repository:
        raise ValueError("A GitHub repository URL is required")
    destination = Path(destination).expanduser().resolve()
    result: dict[str, Any] = {
        "path": str(destination),
        "repository": repository,
        "requested_ref": ref,
    }
    git_dir = destination / ".git"

    def git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=destination,
            check=check,
            capture_output=True,
            text=True,
        )

    def resolved_revision() -> str | None:
        try:
            return git("rev-parse", "HEAD").stdout.strip()
        except Exception:
            return None

    if not git_dir.is_dir():
        if destination.exists() and any(destination.iterdir()):
            raise RuntimeError(
                "Refusing to clone the GitHub package into a non-empty path "
                f"without a .git checkout: {destination}. Choose a fresh Kaggle working path."
            )
        if dry_run:
            result["status"] = "would_clone"
            return result
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = ["git", "clone", "--depth", "1"]
        if not (len(ref) == 40 and all(char in "0123456789abcdefABCDEF" for char in ref)):
            command.extend(["--branch", ref])
        command.extend([repository, str(destination)])
        try:
            subprocess.run(command, check=True)
            if len(ref) == 40 and all(char in "0123456789abcdefABCDEF" for char in ref):
                subprocess.run(
                    ["git", "-C", str(destination), "fetch", "--depth", "1", "origin", ref],
                    check=True,
                )
                subprocess.run(
                    ["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"],
                    check=True,
                )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Could not clone GitHub package {repository!r} at {ref!r}: {exc}"
            ) from exc
        result["status"] = "cloned"
        result["resolved_revision"] = resolved_revision()
        return result

    remote = git("config", "--get", "remote.origin.url").stdout.strip()
    def normalize(value: str) -> str:
        return str(value).strip().removesuffix(".git").rstrip("/")
    if remote and normalize(remote) != normalize(repository):
        raise RuntimeError(
            f"Existing checkout {destination} points to {remote!r}, not {repository!r}"
        )
    if dry_run:
        result["status"] = "would_update"
        result["resolved_revision"] = resolved_revision()
        return result
    try:
        git("fetch", "--depth", "1", "origin", ref)
        git("checkout", "--detach", "FETCH_HEAD")
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(
            f"Could not update GitHub package {destination} to {ref!r}: {detail}"
        ) from exc
    result["status"] = "updated"
    result["resolved_revision"] = resolved_revision()
    return result


def download_selected_models(
    comfy_root: Path,
    mode: str,
    *,
    models_root: Path | None = None,
    include_diffusion: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Download shared files; defer diffusion weights to the ComfyUI loader.

    ``include_diffusion`` is retained for non-Comfy callers that explicitly
    need the old eager behavior.  Kaggle startup and the normal ComfyUI path
    leave it false so only the selected workflow downloads a diffusion model.
    """

    models_root = models_root or (comfy_root / "models")
    status = model_file_status(
        comfy_root,
        mode,
        models_root=models_root,
        include_diffusion=include_diffusion,
    )
    if not include_diffusion:
        status["diffusion_model"] = {
            "status": "deferred_to_comfy_loader",
            "note": "KaggleH3ShardedDiffusionLoader downloads the selected variant at execution time.",
        }
    if status["all_present"]:
        status["download_status"] = "already_present"
        return status
    if dry_run:
        status["download_status"] = "skipped_dry_run"
        return status
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required only when downloading missing model files."
        ) from exc

    downloaded: list[str] = []
    for role, (directory, filename) in required_model_files(
        mode, include_diffusion=include_diffusion
    ).items():
        destination = models_root / directory
        destination.mkdir(parents=True, exist_ok=True)
        if (destination / filename).is_file():
            continue
        hf_hub_download(
            repo_id=H3_MODEL_REPOSITORY,
            filename=f"{directory}/{filename}",
            revision=H3_MODEL_REVISION,
            local_dir=str(models_root),
            token=os.environ.get("HF_TOKEN"),
        )
        downloaded.append(f"{directory}/{filename}")
    status = model_file_status(
        comfy_root,
        mode,
        models_root=models_root,
        include_diffusion=include_diffusion,
    )
    if not include_diffusion:
        status["diffusion_model"] = {
            "status": "deferred_to_comfy_loader",
            "note": "KaggleH3ShardedDiffusionLoader downloads the selected variant at execution time.",
        }
    status["download_status"] = "downloaded"
    status["downloaded"] = downloaded
    return status


def configure_comfyui_model_paths(
    comfy_root: Path, models_root: Path, *, dry_run: bool = False
) -> dict[str, Any]:
    """Point ComfyUI at a model tree outside its writable working directory."""

    models_root = models_root.resolve()
    if models_root.name == "models":
        base_path = models_root.parent
        relative_models = "models/"
    else:
        base_path = models_root
        relative_models = "."
    config_path = comfy_root / "extra_model_paths.yaml"
    config = (
        "# Generated by kaggle_h3; keep H3 weights outside /kaggle/working.\n"
        "h3_external:\n"
        f"  base_path: {base_path.as_posix()}\n"
        f"  diffusion_models: {relative_models}diffusion_models/\n"
        f"  text_encoders: {relative_models}text_encoders/\n"
        f"  vae: {relative_models}vae/\n"
    )
    result: dict[str, Any] = {
        "status": "would_write" if dry_run else "written",
        "config_path": str(config_path),
        "base_path": str(base_path),
        "models_root": str(models_root),
    }
    if dry_run:
        result["config"] = config
        return result
    comfy_root.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config, encoding="utf-8")
    return result


def comfy_launch_command(
    comfy_root: Path,
    port: int,
    *,
    listen_host: str = "127.0.0.1",
    visible_device_ids: list[int] | None = None,
    enable_cors_header: str | None = None,
    cpu_vae: bool = False,
    disable_cuda_malloc: bool = True,
) -> list[str]:
    """Build a ComfyUI command with an explicit VAE phase policy.

    ``--lowvram`` remains enabled. ``--cpu-vae`` is opt-in for the explicit
    one-GPU fallback; phase-aware H3 needs GPU VAE targets after denoising.
    ``--gpu-only`` is never included because it defeats CPU/RAM offload.
    ``--disable-cuda-malloc`` is enabled for H3 by default because the
    quantized phase router uses explicit synchronous device boundaries and
    must not combine them with ComfyUI's cudaMallocAsync allocator.
    """

    command = [
        sys.executable,
        str(comfy_root / "main.py"),
        "--listen",
        listen_host,
        "--port",
        str(port),
        "--lowvram",
        "--disable-auto-launch",
    ]
    if disable_cuda_malloc:
        command.insert(command.index("--disable-auto-launch"), "--disable-cuda-malloc")
    if cpu_vae:
        command.insert(command.index("--disable-auto-launch"), "--cpu-vae")
    if enable_cors_header is not None:
        command.extend(["--enable-cors-header", enable_cors_header])
    if visible_device_ids and len(visible_device_ids) > 1:
        # CUDA_VISIBLE_DEVICES remaps these to local indices 0,1 inside ComfyUI.
        command[4:4] = ["--cuda-device", ",".join(str(index) for index in range(len(visible_device_ids)))]
    return command


def _runtime_visible_cuda_device_ids() -> list[int]:
    """Return local CUDA indices visible to this Python process.

    This is only a safety net for callers that omit an execution plan. It
    cannot recover a GPU hidden by the parent process's CUDA visibility mask;
    the notebook preflight therefore remains responsible for discovering the
    physical devices before launching ComfyUI.
    """

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return list(range(int(torch.cuda.device_count())))
    except Exception:
        pass
    return []


def start_comfyui(
    comfy_root: Path,
    *,
    port: int = 8188,
    listen_host: str = "127.0.0.1",
    execution_plan: dict[str, Any] | None = None,
    log_dir: Path | None = None,
    enable_cors_header: str | None = None,
    cpu_vae: bool = False,
    disable_cuda_malloc: bool = True,
    dry_run: bool = False,
) -> ComfyProcess | dict[str, Any]:
    """Start ComfyUI, optionally exposing it beyond loopback."""

    plan = execution_plan or {}
    if "visible_device_ids" in plan:
        visible = [int(item) for item in plan.get("visible_device_ids", [])]
    else:
        # Do not silently fall back to a single card just because a caller
        # omitted the plan. ComfyUI receives every CUDA device visible here.
        visible = _runtime_visible_cuda_device_ids()
    command = comfy_launch_command(
        comfy_root,
        port,
        listen_host=listen_host,
        visible_device_ids=visible,
        enable_cors_header=enable_cors_header,
        cpu_vae=cpu_vae,
        disable_cuda_malloc=disable_cuda_malloc,
    )
    log_dir = (log_dir or (comfy_root / ".." / "kaggle_h3_runs")).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "comfyui.log"
    env = os.environ.copy()
    if visible:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(item) for item in visible)
    env["PYTHONUNBUFFERED"] = "1"
    print("[Kaggle H3] ComfyUI CUDA_VISIBLE_DEVICES:", env.get("CUDA_VISIBLE_DEVICES"))
    print("[Kaggle H3] ComfyUI launch command:", shlex.join(command))
    if dry_run:
        return {
            "command": command,
            "log_path": str(log_path),
            "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
            "visible_device_ids": visible,
            "status": "would_start",
        }
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=comfy_root,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    time.sleep(1.0)
    if process.poll() is not None:
        raise RuntimeError(
            f"ComfyUI exited immediately with code {process.returncode}; see {log_path}"
        )
    return ComfyProcess(
        process=process,
        log_path=log_path,
        command=tuple(command),
        cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES"),
    )


def wait_for_comfyui_device_visibility(
    comfy_process: ComfyProcess,
    *,
    expected_count: int,
    base_url: str = "http://127.0.0.1:8188",
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Wait for ``/system_stats`` and require the requested device count.

    ComfyUI's primary ``Device: cuda:0`` log line is not sufficient evidence
    of visibility: the API's ``devices`` array is the authoritative process
    view. Failing here prevents a later H3 node from starting a misleading
    one-GPU run.
    """

    import urllib.request

    deadline = time.monotonic() + timeout
    last_payload: dict[str, Any] | None = None
    last_error: str | None = None
    while time.monotonic() < deadline:
        if comfy_process.process.poll() is not None:
            raise RuntimeError(
                f"ComfyUI exited with code {comfy_process.process.returncode}; "
                f"see {comfy_process.log_path}"
            )
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + "/system_stats", timeout=3) as response:
                payload = json.load(response)
            if isinstance(payload, dict):
                last_payload = payload
            devices = payload.get("devices", []) if isinstance(payload, dict) else []
            if isinstance(devices, list) and len(devices) >= expected_count:
                return {
                    "status": "ready",
                    "expected_count": expected_count,
                    "actual_count": len(devices),
                    "devices": devices,
                    "system_stats": payload,
                }
            last_error = f"ComfyUI reports {len(devices) if isinstance(devices, list) else 0} device(s)"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(2.0)

    actual_devices = last_payload.get("devices", []) if last_payload else []
    log_tail = comfy_process.log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
    raise RuntimeError(
        "ComfyUI did not expose the required CUDA devices. "
        f"Expected {expected_count}, observed "
        f"{len(actual_devices) if isinstance(actual_devices, list) else 0}; "
        f"last check: {last_error}.\n"
        f"Launch CUDA_VISIBLE_DEVICES={comfy_process.cuda_visible_devices!r}\n"
        f"Launch command: {shlex.join(comfy_process.command)}\n"
        f"Log: {comfy_process.log_path}\n{log_tail}"
    )
