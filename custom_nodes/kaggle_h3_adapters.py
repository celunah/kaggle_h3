"""ComfyUI nodes for a metadata-validated MiniMax H3 adapter stack.

The Kaggle bootstrap installs this node and its catalog into
``ComfyUI/custom_nodes``. Dependency-light helpers are installed into the
private ``kaggle_h3_support`` package beside the ComfyUI application so
ComfyUI does not mistake them for separate custom nodes.
"""

from __future__ import annotations

from contextvars import ContextVar
import json
import math
import os
from pathlib import Path
import importlib.util
import re
import sys
from typing import Any


_H3_NODE_PATH = Path(__file__).resolve()
for _import_root in (_H3_NODE_PATH.parents[1], _H3_NODE_PATH.parents[2]):
    if _import_root.is_dir() and str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))


_H3_SAMPLER_SYNCHRONIZATION_MODE: ContextVar[str] = ContextVar(
    "kaggle_h3_sampler_synchronization_mode", default="full"
)
_H3_DIFFUSION_TELEMETRY: ContextVar[Any] = ContextVar(
    "kaggle_h3_diffusion_telemetry", default=None
)

H3_EULER_SAMPLER_NAMES = ("euler", "euler_dualclock")


try:
    from kaggle_h3.adapters import (
        AdapterCache,
        H3AdapterError,
        apply_adapter_stack,
        build_runtime_config,
        detect_conditioning_mode,
        load_catalog,
        AUTO_SINGULARITY_ADAPTER,
        MAX_TURBO_STEPS,
        MIN_TURBO_STEPS,
        resolve_singularity_turbo_adapter,
        selector_options,
    )
except ImportError:
    try:
        from kaggle_h3_support.adapters import (  # type: ignore
            AdapterCache,
            H3AdapterError,
            apply_adapter_stack,
            build_runtime_config,
            detect_conditioning_mode,
            load_catalog,
            AUTO_SINGULARITY_ADAPTER,
            MAX_TURBO_STEPS,
            MIN_TURBO_STEPS,
            resolve_singularity_turbo_adapter,
            selector_options,
        )
    except ImportError:
        try:
            # This supports loading the node directly from the delivered
            # package before the bootstrap has copied its core beside ComfyUI.
            node_path = Path(__file__).resolve()
            for import_root in (
                node_path.parents[1] / "src",
                node_path.parents[2] / "src",
                node_path.parent,
            ):
                if import_root.is_dir() and str(import_root) not in sys.path:
                    sys.path.insert(0, str(import_root))
            from kaggle_h3.adapters import (  # type: ignore
                AdapterCache,
                H3AdapterError,
                apply_adapter_stack,
                build_runtime_config,
                detect_conditioning_mode,
                load_catalog,
                AUTO_SINGULARITY_ADAPTER,
                MAX_TURBO_STEPS,
                MIN_TURBO_STEPS,
                resolve_singularity_turbo_adapter,
                selector_options,
            )
        except ImportError:  # installed standalone beside the ComfyUI custom node
            # ComfyUI loads this file with an importlib file spec. Import the
            # sibling helper by path for the legacy standalone bundle.
            core_path = Path(__file__).resolve().with_name("kaggle_h3_adapter_core.py")
            core_spec = importlib.util.spec_from_file_location(
                "kaggle_h3_adapter_core", core_path
            )
            if core_spec is None or core_spec.loader is None or not core_path.is_file():
                raise ImportError(f"Cannot load standalone H3 adapter core: {core_path}")
            core_module = importlib.util.module_from_spec(core_spec)
            sys.modules[core_spec.name] = core_module
            core_spec.loader.exec_module(core_module)
            AdapterCache = core_module.AdapterCache
            H3AdapterError = core_module.H3AdapterError
            apply_adapter_stack = core_module.apply_adapter_stack
            build_runtime_config = core_module.build_runtime_config
            detect_conditioning_mode = core_module.detect_conditioning_mode
            load_catalog = core_module.load_catalog
            AUTO_SINGULARITY_ADAPTER = core_module.AUTO_SINGULARITY_ADAPTER
            MAX_TURBO_STEPS = core_module.MAX_TURBO_STEPS
            MIN_TURBO_STEPS = core_module.MIN_TURBO_STEPS
            resolve_singularity_turbo_adapter = core_module.resolve_singularity_turbo_adapter
            selector_options = core_module.selector_options


try:
    from kaggle_h3.ref2va import (
        H3_MAX_SECONDS,
        H3_MIN_FRAMES,
        REF2VA_SIZE_PRESETS,
        resolve_ref2va_dimensions,
        resolve_ref2va_length,
    )
except ImportError:
    try:
        from kaggle_h3_support.ref2va import (  # type: ignore
            H3_MAX_SECONDS,
            H3_MIN_FRAMES,
            REF2VA_SIZE_PRESETS,
            resolve_ref2va_dimensions,
            resolve_ref2va_length,
        )
    except ImportError:
        try:
            from kaggle_h3_ref2va import (  # type: ignore
                H3_MAX_SECONDS,
                H3_MIN_FRAMES,
                REF2VA_SIZE_PRESETS,
                resolve_ref2va_dimensions,
                resolve_ref2va_length,
            )
        except ImportError:
            ref2va_path = Path(__file__).with_name("kaggle_h3_ref2va.py")
            ref2va_spec = importlib.util.spec_from_file_location(
                "kaggle_h3_ref2va", ref2va_path
            )
            if ref2va_spec is None or ref2va_spec.loader is None or not ref2va_path.is_file():
                raise ImportError(f"Cannot load standalone H3 Ref2VA helper: {ref2va_path}")
            ref2va_module = importlib.util.module_from_spec(ref2va_spec)
            sys.modules[ref2va_spec.name] = ref2va_module
            ref2va_spec.loader.exec_module(ref2va_module)
            H3_MAX_SECONDS = ref2va_module.H3_MAX_SECONDS
            H3_MIN_FRAMES = ref2va_module.H3_MIN_FRAMES
            REF2VA_SIZE_PRESETS = ref2va_module.REF2VA_SIZE_PRESETS
            resolve_ref2va_dimensions = ref2va_module.resolve_ref2va_dimensions
            resolve_ref2va_length = ref2va_module.resolve_ref2va_length


try:
    from kaggle_h3.model_manager import (
        H3DiffusionModelError,
        H3DiffusionModelManager,
        h3_diffusion_directory_from_comfy,
    )
except ImportError:
    try:
        from kaggle_h3_support.model_manager import (  # type: ignore
            H3DiffusionModelError,
            H3DiffusionModelManager,
            h3_diffusion_directory_from_comfy,
        )
    except ImportError:
        try:
            from kaggle_h3_model_manager import (  # type: ignore
                H3DiffusionModelError,
                H3DiffusionModelManager,
                h3_diffusion_directory_from_comfy,
            )
        except ImportError:
            model_manager_path = Path(__file__).with_name("kaggle_h3_model_manager.py")
            model_manager_spec = importlib.util.spec_from_file_location(
                "kaggle_h3_model_manager", model_manager_path
            )
            if (
                model_manager_spec is None
                or model_manager_spec.loader is None
                or not model_manager_path.is_file()
            ):
                raise ImportError(f"Cannot load standalone H3 model manager: {model_manager_path}")
            model_manager_module = importlib.util.module_from_spec(model_manager_spec)
            sys.modules[model_manager_spec.name] = model_manager_module
            model_manager_spec.loader.exec_module(model_manager_module)
            H3DiffusionModelError = model_manager_module.H3DiffusionModelError
            H3DiffusionModelManager = model_manager_module.H3DiffusionModelManager
            h3_diffusion_directory_from_comfy = model_manager_module.h3_diffusion_directory_from_comfy


CATALOG_PATH = Path(__file__).with_name("kaggle_h3_adapter_catalog.json")

try:
    from kaggle_h3.phase_runtime import (
        H3PhaseError,
        begin_transformer_phase,
        configure_vae_phase,
        phase_state_for_model,
        phase_policy,
        phase_device_ids,
        prepare_model_for_h3_phase,
        prepare_text_encoder_for_h3_phase,
        _retarget_patcher_load_device,
        _prepare_sampling_preserving_phase,
        h3_synchronization_plan,
        normalize_h3_synchronization_mode,
        release_all_text_encoder_phases,
        release_h3_runtime_resources,
        release_vae_phase,
        release_transformer_phase,
        h3_rmsnorm_dtype_alignment,
        set_h3_synchronization_mode,
        synchronize_h3_devices,
        validate_finite,
    )
except ImportError:
    try:
        from kaggle_h3_support.phase_runtime import (  # type: ignore
            H3PhaseError,
            begin_transformer_phase,
            configure_vae_phase,
            phase_state_for_model,
            phase_policy,
            phase_device_ids,
            prepare_model_for_h3_phase,
            prepare_text_encoder_for_h3_phase,
            _retarget_patcher_load_device,
            _prepare_sampling_preserving_phase,
            h3_synchronization_plan,
            normalize_h3_synchronization_mode,
            release_all_text_encoder_phases,
            release_h3_runtime_resources,
            release_vae_phase,
            release_transformer_phase,
            h3_rmsnorm_dtype_alignment,
            set_h3_synchronization_mode,
            synchronize_h3_devices,
            validate_finite,
        )
    except ImportError:
        try:
            from kaggle_h3_phase_runtime import (  # type: ignore
                H3PhaseError,
                begin_transformer_phase,
                configure_vae_phase,
                phase_state_for_model,
                phase_policy,
                phase_device_ids,
                prepare_model_for_h3_phase,
                prepare_text_encoder_for_h3_phase,
                _retarget_patcher_load_device,
                _prepare_sampling_preserving_phase,
                h3_synchronization_plan,
                normalize_h3_synchronization_mode,
                release_all_text_encoder_phases,
                release_h3_runtime_resources,
                release_vae_phase,
                release_transformer_phase,
                h3_rmsnorm_dtype_alignment,
                set_h3_synchronization_mode,
                synchronize_h3_devices,
                validate_finite,
            )
        except ImportError:
            # The legacy standalone bundle kept the helper beside the node.
            phase_path = Path(__file__).with_name("kaggle_h3_phase_runtime.py")
            phase_spec = importlib.util.spec_from_file_location(
                "kaggle_h3_phase_runtime", phase_path
            )
            if phase_spec is None or phase_spec.loader is None or not phase_path.is_file():
                raise ImportError(f"Cannot load standalone H3 phase runtime: {phase_path}")
            phase_module = importlib.util.module_from_spec(phase_spec)
            sys.modules[phase_spec.name] = phase_module
            phase_spec.loader.exec_module(phase_module)
            H3PhaseError = phase_module.H3PhaseError
            begin_transformer_phase = phase_module.begin_transformer_phase
            configure_vae_phase = phase_module.configure_vae_phase
            phase_state_for_model = phase_module.phase_state_for_model
            phase_policy = phase_module.phase_policy
            phase_device_ids = phase_module.phase_device_ids
            prepare_model_for_h3_phase = phase_module.prepare_model_for_h3_phase
            prepare_text_encoder_for_h3_phase = phase_module.prepare_text_encoder_for_h3_phase
            _retarget_patcher_load_device = phase_module._retarget_patcher_load_device
            _prepare_sampling_preserving_phase = phase_module._prepare_sampling_preserving_phase
            h3_synchronization_plan = phase_module.h3_synchronization_plan
            normalize_h3_synchronization_mode = phase_module.normalize_h3_synchronization_mode
            release_all_text_encoder_phases = phase_module.release_all_text_encoder_phases
            release_h3_runtime_resources = phase_module.release_h3_runtime_resources
            release_vae_phase = phase_module.release_vae_phase
            release_transformer_phase = phase_module.release_transformer_phase
            h3_rmsnorm_dtype_alignment = phase_module.h3_rmsnorm_dtype_alignment
            set_h3_synchronization_mode = phase_module.set_h3_synchronization_mode
            synchronize_h3_devices = phase_module.synchronize_h3_devices
            validate_finite = phase_module.validate_finite


try:
    from kaggle_h3.diffusion_telemetry import H3DiffusionTelemetry
except ImportError:
    try:
        from kaggle_h3_support.diffusion_telemetry import H3DiffusionTelemetry  # type: ignore
    except ImportError:
        try:
            from kaggle_h3_diffusion_telemetry import H3DiffusionTelemetry  # type: ignore
        except ImportError:
            H3DiffusionTelemetry = None  # type: ignore[assignment,misc]


try:
    from kaggle_h3.sage_attention import (
        H3SageAttentionError,
        h3_sage_attention,
        sage_attention_status,
    )
except ImportError:
    try:
        from kaggle_h3_support.sage_attention import (  # type: ignore
            H3SageAttentionError,
            h3_sage_attention,
            sage_attention_status,
        )
    except ImportError:
        try:
            from kaggle_h3_sage_attention import (  # type: ignore
                H3SageAttentionError,
                h3_sage_attention,
                sage_attention_status,
            )
        except ImportError:
            sage_path = Path(__file__).with_name("kaggle_h3_sage_attention.py")
            sage_spec = importlib.util.spec_from_file_location(
                "kaggle_h3_sage_attention", sage_path
            )
            if sage_spec is None or sage_spec.loader is None or not sage_path.is_file():
                raise ImportError(f"Cannot load standalone H3 SageAttention helper: {sage_path}")
            sage_module = importlib.util.module_from_spec(sage_spec)
            sys.modules[sage_spec.name] = sage_module
            sage_spec.loader.exec_module(sage_module)
            H3SageAttentionError = sage_module.H3SageAttentionError
            h3_sage_attention = sage_module.h3_sage_attention
            sage_attention_status = sage_module.sage_attention_status


# ComfyUI's current MiniMax H3 nodes use the V3 schema.  Keep this import
# optional so the legacy adapter module remains importable in dependency-light
# tests and on older ComfyUI installations.
try:
    from comfy_api.latest import io as _H3IO  # type: ignore
except ImportError:
    _H3IO = None  # type: ignore[assignment,misc]


def _catalog():
    return load_catalog(CATALOG_PATH)


def _folder_options(category: str) -> list[str]:
    """Return a Comfy model-folder selector without importing Comfy in tests."""

    try:
        import folder_paths  # type: ignore

        values = list(folder_paths.get_filename_list(category))
        return values or [""]
    except Exception:
        return [""]


H3_SMOKE_REFERENCE_FILES = {
    "character": "CHARACTER_REFERENCE.png",
    "scene": "SCENE_REFERENCE.png",
}


def _resolve_h3_smoke_reference(reference: str) -> Path:
    """Resolve a checked-in smoke reference without relying on a UI filename list."""

    key = str(reference or "").strip().lower()
    filename = H3_SMOKE_REFERENCE_FILES.get(key)
    if filename is None:
        raise H3PhaseError(
            f"Unknown H3 smoke reference {reference!r}; choose character or scene."
        )

    candidates: list[Path] = []
    configured_asset_dir = os.environ.get("KAGGLE_H3_SMOKE_ASSET_DIR", "").strip()
    if configured_asset_dir:
        candidates.append(Path(configured_asset_dir).expanduser() / filename)
    configured_project_root = os.environ.get("KAGGLE_H3_PROJECT_ROOT", "").strip()
    if configured_project_root:
        candidates.append(Path(configured_project_root).expanduser() / "smoke_assets" / filename)

    node_path = Path(__file__).resolve()
    # This covers the normal checkout layout: <project>/ComfyUI/custom_nodes.
    for parent in node_path.parents:
        candidates.append(parent / "smoke_assets" / filename)

    try:
        import folder_paths  # type: ignore

        input_directory = getattr(folder_paths, "get_input_directory", None)
        if callable(input_directory):
            candidates.append(Path(input_directory()) / filename)
        configured_input_directory = getattr(folder_paths, "input_directory", None)
        if configured_input_directory:
            candidates.append(Path(configured_input_directory) / filename)
    except Exception:
        # The project and environment candidates above are enough for the
        # standalone node; ComfyUI's folder_paths API is optional here.
        pass

    seen: set[Path] = set()
    searched: list[str] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        searched.append(str(resolved))
        if resolved.is_file():
            return resolved
    raise H3PhaseError(
        f"Could not resolve H3 smoke {key} reference {filename!r}. Searched: "
        + ", ".join(searched)
        + ". Run the updated Kaggle startup cell or set KAGGLE_H3_PROJECT_ROOT."
    )


class KaggleH3SmokeReference:
    """Load the repository's fixed smoke image references by semantic role."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference": (
                    list(H3_SMOKE_REFERENCE_FILES),
                    {"default": "character"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_reference"
    CATEGORY = "loaders/MiniMax H3"

    def load_reference(self, reference: str):
        path = _resolve_h3_smoke_reference(reference)
        try:
            import numpy as np  # type: ignore
            import torch  # type: ignore
            from PIL import Image  # type: ignore
        except Exception as exc:
            raise H3PhaseError(
                "Kaggle H3 smoke references require ComfyUI's Pillow, NumPy, and PyTorch."
            ) from exc
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            array = np.array(rgb, dtype=np.float32, copy=True) / 255.0
        tensor = torch.from_numpy(array)[None, ...]
        print(f"[Kaggle H3] Resolved smoke {reference} reference: {path}", flush=True)
        return (tensor,)


def _set_explicit_patcher_devices(owner: Any, *, load_device: Any, offload_device: Any) -> None:
    patcher = owner if hasattr(owner, "load_device") else getattr(owner, "patcher", None)
    if patcher is None:
        raise H3PhaseError(f"{type(owner).__name__} does not expose a Comfy model patcher")
    _retarget_patcher_load_device(patcher, load_device)
    patcher.offload_device = offload_device


def _release_comfy_model_cache_before_h3_switch() -> None:
    """Release a previous H3 model before replacing its on-disk checkpoint."""

    try:
        import comfy.model_management as model_management  # type: ignore

        unload = getattr(model_management, "unload_all_models", None)
        if callable(unload):
            unload()
        empty = getattr(model_management, "soft_empty_cache", None)
        if callable(empty):
            empty()
        print(
            "[Kaggle H3] Released ComfyUI's cached model residency before diffusion switch.",
            flush=True,
        )
    except Exception as exc:
        # The native loader remains the source of truth on older Comfy builds;
        # failing to expose a cache-release helper must not hide a download
        # error or change the selected checkpoint.
        print(f"[Kaggle H3] Cache release before diffusion switch unavailable: {exc}", flush=True)


def _h3_stream_tensor(samples: dict[str, Any], stream: str) -> Any:
    """Return one H3 AV stream without copying the other stream.

    Native H3 sampling produces a NestedTensor pair: video at index 0 and
    audio at index 1. The sampler converts that pair into two ordinary
    ComfyUI LATENT values so each downstream VAE owns only the tensor it will
    consume. Accept the old packed form here as a migration path for hand
    edited workflows, but reject an ambiguous audio input rather than decoding
    the video stream through the audio VAE.
    """

    packed = samples.get("samples") if isinstance(samples, dict) else None
    if packed is None:
        raise RuntimeError(f"H3 {stream} latent is missing its samples tensor")
    if bool(getattr(packed, "is_nested", False)):
        unbind = getattr(packed, "unbind", None)
        if not callable(unbind):
            raise RuntimeError("H3 NestedTensor does not expose unbind()")
        streams = tuple(unbind())
        if len(streams) != 2:
            raise RuntimeError(
                f"H3 AV latent must contain exactly video+audio streams; got {len(streams)}"
            )
        return streams[0 if stream == "video" else 1]
    declared = samples.get("_kaggle_h3_stream")
    if declared == stream:
        return packed
    if stream == "video" and declared is None:
        # This is the legacy video output shape. Audio remains intentionally
        # strict because treating a plain video tensor as audio is destructive.
        return packed
    raise RuntimeError(
        f"H3 {stream} decoder received an ambiguous latent. Reconnect it to the "
        "dedicated output of Kaggle H3 | Turbo Sampler."
    )


def _h3_move_for_consumer(tensor: Any, *, device_id: int, role: str) -> Any:
    """Move one stream directly to its next consumer, retaining CPU as tier 3."""

    import torch

    target = torch.device(f"cuda:{int(device_id)}")
    source = getattr(tensor, "device", None)
    source_label = str(source) if source is not None else "unknown"
    # The sampler or preceding VAE may have produced the tensor on a worker
    # stream. Synchronize the producer and consumer devices before the copy so
    # a direct peer transfer cannot observe a partially written activation.
    synchronize_h3_devices((source, target), reason=f"{role} transfer before copy")
    if source_label == str(target):
        print(
            f"[Kaggle H3] Consumer routing retained {role} latent on {target}.",
            flush=True,
        )
        return tensor
    moved = tensor.to(target, non_blocking=False)
    synchronize_h3_devices((target,), reason=f"{role} transfer after copy")
    print(
        f"[Kaggle H3] Consumer routing moved {role} latent {source_label} -> {target} "
        "directly for its next consumer.",
        flush=True,
    )
    return moved


def _h3_clone_sampler_value(value: Any) -> Any:
    """Detach and clone only values retained by a multistep sampler.

    ComfyUI's ``res_multistep`` keeps the previous denoised model output for
    its second-order history.  H3's quantized execution path can expose a
    view backed by a temporary output/workspace buffer, so retaining that
    view across the next model call is unsafe even when all CUDA launches
    have been synchronized.  Clone the history value without moving it or
    changing its dtype/device; ordinary metadata remains shared.
    """

    try:
        import torch
    except Exception:
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_h3_clone_sampler_value(item) for item in value)
    if isinstance(value, list):
        return [_h3_clone_sampler_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _h3_clone_sampler_value(item) for key, item in value.items()
        }
    return value


def _h3_sampler_boundary_devices(mode: str) -> tuple[int, ...]:
    """Choose devices for the post-model sampler barrier."""

    device_ids = tuple(phase_device_ids())
    if mode == "full":
        return device_ids
    if mode == "safe":
        return device_ids[:1]
    return ()


def _h3_res_multistep(
    model: Any,
    x: Any,
    sigmas: Any,
    extra_args: dict[str, Any] | None = None,
    callback: Any = None,
    disable: Any = None,
    s_noise: float = 1.0,
    noise_sampler: Any = None,
):
    """Run ComfyUI's base ``res_multistep`` with safe H3 history ownership.

    The update equations intentionally mirror ComfyUI v0.34.0's
    ``sample_res_multistep`` entry point (eta=0, non-CFG).  The only H3
    additions are the selected H3 synchronization boundary and an owned copy
    of ``old_denoised``. Turbo remains on ComfyUI's normal Euler path.
    """

    import torch
    from comfy.k_diffusion import sampling as k_diffusion_sampling  # type: ignore

    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    if noise_sampler is None:
        noise_sampler = k_diffusion_sampling.default_noise_sampler(x, seed=seed)
    model_sampling = model.inner_model.model_patcher.get_model_object(
        "model_sampling"
    )
    s_noise = s_noise * getattr(model_sampling, "noise_scale", 1.0)
    s_in = x.new_ones([x.shape[0]])

    def sigma_fn(t):
        return t.neg().exp()

    def t_fn(sigma):
        return sigma.log().neg()

    def phi1_fn(t):
        return torch.expm1(t) / t

    def phi2_fn(t):
        return (phi1_fn(t) - 1.0) / t

    old_sigma_down = None
    old_denoised = None
    telemetry = _H3_DIFFUSION_TELEMETRY.get()

    for index in range(len(sigmas) - 1):
        if telemetry is not None:
            telemetry.record_boundary(
                step_index=index,
                sigma=sigmas[index],
                boundary="input",
                value=x,
                tensor_name="sampler_input",
            )
        denoised = model(x, sigmas[index] * s_in, **extra_args)
        synchronize_h3_devices(
            _h3_sampler_boundary_devices(_H3_SAMPLER_SYNCHRONIZATION_MODE.get()),
            reason=f"res_multistep model output boundary {index}",
        )
        if telemetry is not None:
            telemetry.record_boundary(
                step_index=index,
                sigma=sigmas[index],
                boundary="model_output",
                value=denoised,
                tensor_name="model_output",
            )
            # The ComfyUI sampler API exposes the model's denoised prediction
            # directly. There is no second denoising operation at this layer,
            # so retain an explicit alias rather than inventing a transformation.
            telemetry.record_boundary(
                step_index=index,
                sigma=sigmas[index],
                boundary="denoised",
                value=denoised,
                tensor_name="denoised",
                alias_of="model_output",
            )
            telemetry.record_boundary(
                step_index=index,
                sigma=sigmas[index],
                boundary="history",
                value=old_denoised,
                tensor_name="history",
            )
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": index,
                    "sigma": sigmas[index],
                    "sigma_hat": sigmas[index],
                    "denoised": denoised,
                }
            )

        # eta=0 is the official non-ancestral base-H3 contract. Keep this
        # call for exact parity with ComfyUI's implementation and to make
        # the wrapper safe if Comfy changes the helper's scalar behavior.
        sigma_down, sigma_up = k_diffusion_sampling.get_ancestral_step(
            sigmas[index], sigmas[index + 1], eta=0.0
        )
        if sigma_down == 0 or old_denoised is None:
            d = k_diffusion_sampling.to_d(x, sigmas[index], denoised)
            dt = sigma_down - sigmas[index]
            x = x + d * dt
        else:
            t = t_fn(sigmas[index])
            t_old = t_fn(old_sigma_down)
            t_next = t_fn(sigma_down)
            t_prev = t_fn(sigmas[index - 1])
            h = t_next - t
            c2 = (t_prev - t_old) / h
            phi1_value = phi1_fn(-h)
            phi2_value = phi2_fn(-h)
            b1 = torch.nan_to_num(phi1_value - phi2_value / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2_value / c2, nan=0.0)
            x = sigma_fn(h) * x + h * (b1 * denoised + b2 * old_denoised)

        if sigma_up > 0:
            x = x + noise_sampler(
                sigmas[index], sigmas[index + 1]
            ) * s_noise * sigma_up

        # Do not retain a view/workspace owned by the just-completed H3
        # forward. This is the key difference from ComfyUI's direct sampler
        # function and is intentionally limited to the multistep history.
        old_denoised = _h3_clone_sampler_value(denoised)
        old_sigma_down = sigma_down
        if telemetry is not None:
            telemetry.record_boundary(
                step_index=index,
                sigma=sigmas[index],
                boundary="updated_latent",
                value=x,
                tensor_name="updated_latent",
            )

    return x


def _h3_euler_with_telemetry(
    model: Any,
    x: Any,
    sigmas: Any,
    extra_args: dict[str, Any] | None = None,
    callback: Any = None,
    disable: Any = None,
    s_churn: float = 0.0,
    s_tmin: float = 0.0,
    s_tmax: float = float("inf"),
    s_noise: float = 1.0,
):
    """Mirror ComfyUI Euler while exposing opt-in H3 boundary telemetry."""

    import torch
    from comfy.k_diffusion import sampling as k_diffusion_sampling  # type: ignore

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    telemetry = _H3_DIFFUSION_TELEMETRY.get()
    trange = getattr(k_diffusion_sampling, "trange", None)
    step_iterator = (
        trange(len(sigmas) - 1, disable=disable)
        if callable(trange)
        else range(len(sigmas) - 1)
    )

    for index in step_iterator:
        if telemetry is not None:
            telemetry.record_boundary(
                step_index=index,
                sigma=sigmas[index],
                boundary="input",
                value=x,
                tensor_name="sampler_input",
            )

        if s_churn > 0:
            gamma = (
                min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
                if s_tmin <= sigmas[index] <= s_tmax
                else 0.0
            )
        else:
            gamma = 0.0
        sigma_hat = sigmas[index] * (gamma + 1)

        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat**2 - sigmas[index] ** 2) ** 0.5

        denoised = model(x, sigma_hat * s_in, **extra_args)
        if telemetry is not None:
            telemetry.record_boundary(
                step_index=index,
                sigma=sigmas[index],
                timestep=sigma_hat,
                boundary="model_output",
                value=denoised,
                tensor_name="model_output",
            )
            telemetry.record_boundary(
                step_index=index,
                sigma=sigmas[index],
                timestep=sigma_hat,
                boundary="denoised",
                value=denoised,
                tensor_name="denoised",
                alias_of="model_output",
            )
            # Euler has no retained second-order history. Emit the boundary
            # explicitly so Euler and res_multistep traces have the same shape.
            telemetry.record_boundary(
                step_index=index,
                sigma=sigmas[index],
                timestep=sigma_hat,
                boundary="history",
                value=None,
                tensor_name="history",
            )

        d = k_diffusion_sampling.to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": index,
                    "sigma": sigmas[index],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )

        dt = sigmas[index + 1] - sigma_hat
        x = x + d * dt
        if telemetry is not None:
            telemetry.record_boundary(
                step_index=index,
                sigma=sigmas[index],
                timestep=sigmas[index + 1],
                boundary="updated_latent",
                value=x,
                tensor_name="updated_latent",
            )

    return x


def _h3_prepare_sampler(
    sampler_name: str, sampler: Any, *, dual_clock: bool = False
) -> Any:
    """Install H3 sampler wrappers while preserving ComfyUI's equations.

    Base H3 ``res_multistep`` uses the owned-history implementation. Turbo's
    Euler sampler delegates to ComfyUI exactly when telemetry is disabled and
    uses the mirrored implementation only for opt-in telemetry.
    """

    if sampler_name == "euler":
        original_sampler_function = getattr(sampler, "sampler_function", None)
        if not callable(original_sampler_function):
            raise RuntimeError("H3 could not access ComfyUI's Euler sampler function")

        def euler_sampler_with_optional_telemetry(*args: Any, **kwargs: Any):
            if _H3_DIFFUSION_TELEMETRY.get() is None:
                return original_sampler_function(*args, **kwargs)
            return _h3_euler_with_telemetry(*args, **kwargs)

        euler_sampler_with_optional_telemetry.__name__ = (
            "sample_h3_euler_dualclock"
            if dual_clock
            else "sample_h3_euler"
        )
        sampler.sampler_function = euler_sampler_with_optional_telemetry
        if dual_clock:
            print(
                "[Kaggle H3] euler_dualclock selected: using stock Euler with "
                "native ModelSamplingAV video/audio clocks; no second manual "
                "audio update is applied.",
                flush=True,
            )
        else:
            print(
                "[Kaggle H3] Turbo Euler preserves ComfyUI execution unless "
                "diffusion telemetry is enabled.",
                flush=True,
            )
        return sampler

    if sampler_name != "res_multistep":
        return sampler
    if not callable(getattr(sampler, "sampler_function", None)):
        raise RuntimeError(
            "H3 could not access ComfyUI's res_multistep sampler function"
        )
    sampler.sampler_function = _h3_res_multistep
    print(
        "[Kaggle H3] Using base res_multistep with owned history and "
        "model-output CUDA barriers.",
        flush=True,
    )
    return sampler


def _h3_time_shift_sigma(
    sigma: float, from_shift: float, to_shift: float
) -> float:
    """Map one video sigma onto H3's independent audio flow schedule."""

    sigma = float(sigma)
    base_sigma = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base_sigma / (1.0 + (to_shift - 1.0) * base_sigma)


def _h3_actual_clock_schedule(
    sigmas: Any, *, shift_video: float, shift_audio: float
) -> dict[str, Any]:
    """Describe the schedules actually used by native H3 ModelSamplingAV.

    The returned audio values are telemetry only.  The sampler continues to
    advance the packed latent on ComfyUI's video sigma grid; the H3 model and
    ``ModelSamplingAV`` carry the audio stream onto its shifted clock exactly
    once.  This function must never be used to update the latent directly.
    """

    video_sigmas = [float(value) for value in sigmas]
    audio_sigmas = [
        _h3_time_shift_sigma(value, float(shift_video), float(shift_audio))
        for value in video_sigmas
    ]
    return {
        "video": {
            "sigmas": video_sigmas,
            "model_timesteps": [value * 1000.0 for value in video_sigmas],
            "flow_timesteps": [1.0 - value for value in video_sigmas],
        },
        "audio": {
            "sigmas": audio_sigmas,
            "model_timesteps": [value * 1000.0 for value in audio_sigmas],
            "flow_timesteps": [1.0 - value for value in audio_sigmas],
        },
        "shift_video": float(shift_video),
        "shift_audio": float(shift_audio),
        "mapping": "time_shift_sigma(video_sigma, shift_video, shift_audio)",
        "latent_update": "native_model_sampling_av",
        "manual_audio_update": False,
    }


def _h3_finite_audio(audio: dict[str, Any]) -> dict[str, Any]:
    """Make the final audio codec boundary finite and CPU-owned.

    The H3 audio VAE can produce a non-finite sample when an unstable audio
    latent reaches its normalization step. PyAV reports this much later as
    ``avcodec_send_frame`` error 22, which otherwise hides the real boundary.
    Replace only non-finite samples with silence, log the event, clamp the
    codec input to the legal PCM range, and release the GPU waveform.
    """

    import torch

    waveform = audio.get("waveform")
    if waveform is None:
        raise RuntimeError("H3 audio VAE returned no waveform")
    finite = torch.isfinite(waveform)
    nonfinite_count = int((~finite).sum().item())
    if nonfinite_count:
        print(
            "[Kaggle H3] WARNING: audio VAE produced "
            f"{nonfinite_count} non-finite samples; replacing them with silence "
            "so SaveVideo/AAC can complete.",
            flush=True,
        )
        waveform = torch.nan_to_num(
            waveform,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    peak = float(waveform.detach().abs().amax().item()) if waveform.numel() else 0.0
    if peak > 1.0:
        print(
            f"[Kaggle H3] Clamping audio codec input peak {peak:.4g} to [-1, 1].",
            flush=True,
        )
    waveform = waveform.clamp(-1.0, 1.0).to("cpu")
    return {"waveform": waveform, "sample_rate": audio["sample_rate"]}


def _native_ref2va_conditioning():
    """Load ComfyUI's reference conditioner only when the node executes."""

    try:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Kaggle H3 Ref2VA Conditioning requires ComfyUI's native "
            "comfy_extras.nodes_minimax_h3 module. Update ComfyUI to a build "
            "that includes MiniMax H3 support."
        ) from exc
    return MiniMaxH3ReferenceToVideo


def _native_fl2va_conditioning():
    """Load ComfyUI's native FL2VA/T2VA conditioner only when needed."""

    try:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Kaggle H3 Conditioning requires ComfyUI's native "
            "comfy_extras.nodes_minimax_h3 module. Update ComfyUI to a build "
            "that includes MiniMax H3 support."
        ) from exc
    return MiniMaxH3ImageToVideo


def _native_h3_add_guide():
    """Load the native arbitrary-frame H3 guide helper when required."""

    try:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Ref2VA start/end guides require ComfyUI's native "
            "MiniMaxH3AddGuide node. Update ComfyUI to a build that includes "
            "MiniMax H3 guide support."
        ) from exc
    return MiniMaxH3AddGuide


def _compact_h3_autogrow(value: Any, name: str) -> dict[str, Any]:
    """Normalize a V3 autogrow dictionary and omit disconnected sockets."""

    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(
            f"Kaggle H3 Conditioning expected {name} to be an autogrow mapping; "
            f"got {type(value).__name__}."
        )
    return {str(key): item for key, item in value.items() if item is not None}


_H3_EMPTY_FILE = "None"
# One row can be image, video, or audio. The total is bounded by the native
# H3 limits (9 image + 3 video + 3 audio references).
_H3_INLINE_REFERENCE_SLOTS = 15


def _h3_input_file_options(content_types: list[str]) -> list[str]:
    """Return upload/select options matching ComfyUI's native input loaders."""

    try:
        import folder_paths  # type: ignore

        input_dir = folder_paths.get_input_directory()
        os.makedirs(input_dir, exist_ok=True)
        files = [
            name
            for name in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, name))
        ]
        files = folder_paths.filter_files_content_types(files, content_types)
        return [_H3_EMPTY_FILE, *sorted(files)]
    except Exception:
        # Keep the node importable in dependency-light tests and old ComfyUI
        # installations. The upload control still works once ComfyUI supplies
        # its folder_paths module.
        return [_H3_EMPTY_FILE]


def _h3_selected_file(value: Any) -> str | None:
    if value is None:
        return None
    selected = str(value).strip()
    if not selected or selected == _H3_EMPTY_FILE:
        return None
    return selected


def _h3_slot_index(name: str, fallback: int) -> int:
    match = re.search(r"(\d+)$", str(name))
    return int(match.group(1)) if match else fallback


def _h3_ordered_mapping(value: Any, name: str) -> list[tuple[int, Any]]:
    """Return non-empty autogrow entries in their numeric slot order."""

    compacted = _compact_h3_autogrow(value, name)
    entries = [
        (_h3_slot_index(key, fallback), item)
        for fallback, (key, item) in enumerate(compacted.items())
    ]
    return sorted(entries, key=lambda entry: entry[0])


def _load_h3_inline_file(kind: str, filename: str) -> tuple[Any, Any | None]:
    """Load one inline selector through ComfyUI's native media loaders.

    The second return value is an optional soundtrack extracted from an inline
    video. H3 expects video references and their audio references in matching
    slots, so a video's own soundtrack is paired automatically.
    """

    try:
        if kind == "image":
            from nodes import LoadImage  # type: ignore

            return LoadImage().load_image(filename)[0], None
        if kind == "audio":
            from comfy_extras.nodes_audio import LoadAudio  # type: ignore

            return _node_output_values(LoadAudio.execute(filename))[0], None
        if kind == "video":
            from comfy_extras.nodes_video import GetVideoComponents, LoadVideo  # type: ignore

            video = _node_output_values(LoadVideo.execute(filename))[0]
            components = _node_output_values(GetVideoComponents.execute(video))
            return components[0], components[1]
    except Exception as exc:
        raise RuntimeError(
            f"Kaggle H3 Conditioning could not load inline {kind} file "
            f"{filename!r}: {exc}"
        ) from exc
    raise ValueError(f"Kaggle H3 Conditioning has unsupported inline media type {kind!r}.")


def _resolve_h3_inline_references(
    reference_slots: list[Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the global mixed-media selector into native H3 reference groups."""

    images: list[Any] = []
    videos: list[Any] = []
    video_audios: dict[int, Any] = {}
    audios: list[Any] = []
    cache: dict[tuple[str, str], tuple[Any, Any | None]] = {}

    for slot in reference_slots:
        if not isinstance(slot, dict):
            continue
        kind = str(slot.get("reference", slot.get("type", "none"))).strip().lower()
        if kind not in {"image", "video", "audio"}:
            # DynamicCombo stores its selected key under the dynamic input's
            # id (for example ``reference_0``), not necessarily ``reference``.
            detected_kind = next(
                (
                    str(value).strip().lower()
                    for key, value in slot.items()
                    if key != "file"
                    and str(value).strip().lower() in {"image", "video", "audio", "none"}
                ),
                kind,
            )
            kind = detected_kind
        filename = _h3_selected_file(slot.get("file"))
        if kind in {"", "none"} or filename is None:
            continue
        if kind not in {"image", "video", "audio"}:
            raise ValueError(
                "Kaggle H3 Conditioning reference selector has unsupported "
                f"type {kind!r}; choose image, video, audio, or None."
            )
        cache_key = (kind, filename)
        if cache_key not in cache:
            cache[cache_key] = _load_h3_inline_file(kind, filename)
        media, soundtrack = cache[cache_key]
        if kind == "image":
            images.append(media)
        elif kind == "video":
            video_index = len(videos)
            videos.append(media)
            if soundtrack is not None:
                video_audios[video_index] = soundtrack
        else:
            audios.append(media)

    return (
        {f"ref_image_{index}": value for index, value in enumerate(images)},
        {f"ref_video_{index}": value for index, value in enumerate(videos)},
        {
            f"ref_video_audio_{index}": value
            for index, value in sorted(video_audios.items())
        },
        {f"ref_audio_{index}": value for index, value in enumerate(audios)},
    )


def _merge_h3_reference_sources(
    inline: dict[str, Any],
    sockets: Any,
    name: str,
    prefix: str,
    *,
    offset: int = 0,
) -> dict[str, Any]:
    """Append typed socket refs after inline refs without changing either source."""

    merged = dict(inline)
    for position, (_source_index, item) in enumerate(_h3_ordered_mapping(sockets, name)):
        merged[f"{prefix}_{offset + position}"] = item
    return merged


def _merge_h3_video_audio_sources(
    inline: dict[str, Any],
    sockets: Any,
    *,
    video_offset: int,
    typed_video_indices: list[int],
) -> dict[str, Any]:
    merged = dict(inline)
    typed_video_positions = {
        source_index: position
        for position, source_index in enumerate(typed_video_indices)
    }
    for source_index, item in _h3_ordered_mapping(sockets, "ref_video_audios"):
        if source_index not in typed_video_positions:
            # Preserve the original key so _execute_global_h3_conditioning can
            # issue its normal matching-slot error with the user's slot name.
            merged[f"ref_video_audio_{source_index}"] = item
            continue
        merged[
            f"ref_video_audio_{video_offset + typed_video_positions[source_index]}"
        ] = item
    return merged


def _resolve_h3_inline_frame(
    selected_file: Any,
    connected_frame: Any,
    label: str,
) -> Any:
    filename = _h3_selected_file(selected_file)
    if filename is None:
        return connected_frame
    if connected_frame is not None:
        raise ValueError(
            f"Kaggle H3 Conditioning {label} has both a file selector and a connected "
            "IMAGE socket. Use one input, not both."
        )
    return _load_h3_inline_file("image", filename)[0]


def _h3_optional_override(value: Any) -> int | None:
    """Treat ComfyUI's zero-valued optional widgets as unset."""

    if value is None:
        return None
    parsed = int(value)
    return None if parsed == 0 else parsed


def _execute_global_h3_conditioning(
    *,
    clip: Any,
    vae: Any,
    audio_vae: Any,
    prompt: str,
    mode: str,
    width: int,
    height: int,
    length: int,
    ref_image_size: str,
    start_frame: Any = None,
    end_frame: Any = None,
    ref_images: dict[str, Any] | None = None,
    ref_videos: dict[str, Any] | None = None,
    ref_video_audios: dict[str, Any] | None = None,
    ref_audios: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Execute one global H3 conditioning contract for both model variants."""

    normalized_mode = str(mode).strip()
    if normalized_mode not in {"Ref2VA", "FL2VA"}:
        raise ValueError(
            f"Kaggle H3 Conditioning mode must be Ref2VA or FL2VA; got {mode!r}."
        )
    if not prompt or not str(prompt).strip():
        raise ValueError("Kaggle H3 Conditioning prompt must not be empty")

    images = _compact_h3_autogrow(ref_images, "ref_images")
    videos = _compact_h3_autogrow(ref_videos, "ref_videos")
    video_audios = _compact_h3_autogrow(ref_video_audios, "ref_video_audios")
    audios = _compact_h3_autogrow(ref_audios, "ref_audios")
    if normalized_mode == "FL2VA":
        if images or videos or video_audios or audios:
            raise ValueError(
                "Kaggle H3 Conditioning FL2VA accepts start_frame/end_frame only. "
                "Select Ref2VA for image, video, or audio references."
            )
        native = _native_fl2va_conditioning()
        try:
            result = native.execute(
                clip=clip,
                vae=vae,
                prompt=prompt,
                width=int(width),
                height=int(height),
                length=int(length),
                first_frame=start_frame,
                last_frame=end_frame,
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Kaggle H3 FL2VA conditioning failed: {exc}") from exc
        values = _node_output_values(result)
        if len(values) != 2:
            raise RuntimeError(
                "Kaggle H3 native FL2VA conditioner did not return conditioning and latent"
            )
        return values[0], values[1]

    if video_audios:
        orphan_audio = sorted(
            key
            for key in video_audios
            if f"ref_video_{key.rsplit('_', 1)[-1]}" not in videos
        )
        if orphan_audio:
            raise ValueError(
                "Each ref_video_audio_N must be connected with the matching "
                f"ref_video_N; orphan audio slot(s): {', '.join(orphan_audio)}."
            )
    if (
        not images
        and not videos
        and start_frame is None
        and end_frame is None
    ):
        raise ValueError(
            "Kaggle H3 Conditioning Ref2VA needs at least one image/video "
            "reference or a start/end frame guide; audio alone is insufficient."
        )

    native = _native_ref2va_conditioning()
    try:
        result = native.execute(
            clip=clip,
            vae=vae,
            audio_vae=audio_vae,
            prompt=prompt,
            width=int(width),
            height=int(height),
            length=int(length),
            ref_image_size=ref_image_size,
            ref_images=images,
            ref_videos=videos,
            ref_video_audios=video_audios,
            ref_audios=audios,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Kaggle H3 Ref2VA conditioning failed: {exc}") from exc
    values = _node_output_values(result)
    if len(values) != 2:
        raise RuntimeError(
            "Kaggle H3 native Ref2VA conditioner did not return conditioning and latent"
        )
    conditioning, latent = values

    # Ref2VA's native reference conditioner has no first/last-frame inputs.
    # Use the native guide contract so these controls remain real conditioning
    # rather than a UI-only label.  Guides are appended in temporal order.
    guide = _native_h3_add_guide if (start_frame is not None or end_frame is not None) else None
    if guide is not None:
        guide_node = guide()
        actual_frame_count = int(length)
        if start_frame is not None:
            guided = guide_node.execute(
                positive=conditioning,
                latent=latent,
                frame_idx=0,
                vae=vae,
                image=start_frame,
            )
            conditioning = _node_output_values(guided)[0]
        if end_frame is not None:
            guided = guide_node.execute(
                positive=conditioning,
                latent=latent,
                frame_idx=actual_frame_count - 1,
                vae=vae,
                image=end_frame,
            )
            conditioning = _node_output_values(guided)[0]
    return conditioning, latent


if _H3IO is not None:

    class KaggleH3Conditioning(_H3IO.ComfyNode):  # type: ignore[misc, valid-type]
        """One global H3 conditioner with sockets and inline media selectors."""

        @classmethod
        def define_schema(cls):
            image_files = _h3_input_file_options(["image"])
            video_files = _h3_input_file_options(["video"])
            audio_files = _h3_input_file_options(["audio", "video"])

            def inline_file_option(
                key: str,
                options: list[str],
                upload_type: Any,
                tooltip: str,
            ):
                return _H3IO.DynamicCombo.Option(
                    key,
                    [
                        _H3IO.Combo.Input(
                            "file",
                            options=options,
                            default=_H3_EMPTY_FILE,
                            upload=upload_type,
                            tooltip=tooltip,
                        )
                    ],
                )

            reference_options = [
                _H3IO.DynamicCombo.Option("none", []),
                inline_file_option(
                    "image",
                    image_files,
                    _H3IO.UploadType.image,
                    "Reference image file from ComfyUI input; upload is supported.",
                ),
                inline_file_option(
                    "video",
                    video_files,
                    _H3IO.UploadType.video,
                    "Reference video file from ComfyUI input; upload is supported.",
                ),
                inline_file_option(
                    "audio",
                    audio_files,
                    _H3IO.UploadType.audio,
                    "Reference audio file from ComfyUI input; upload is supported.",
                ),
            ]
            autogrow_image = _H3IO.Autogrow.TemplatePrefix(
                input=_H3IO.Image.Input(
                    "ref_image",
                    tooltip="Reference image; order becomes <Picture i>.",
                ),
                prefix="ref_image_",
                min=0,
                max=9,
            )
            autogrow_video = _H3IO.Autogrow.TemplatePrefix(
                input=_H3IO.Image.Input(
                    "ref_video",
                    tooltip="Reference video frames; order becomes <Video i>.",
                ),
                prefix="ref_video_",
                min=0,
                max=3,
            )
            autogrow_video_audio = _H3IO.Autogrow.TemplatePrefix(
                input=_H3IO.Audio.Input(
                    "ref_video_audio",
                    tooltip="Audio paired with the same-numbered reference video.",
                ),
                prefix="ref_video_audio_",
                min=0,
                max=3,
            )
            autogrow_audio = _H3IO.Autogrow.TemplatePrefix(
                input=_H3IO.Audio.Input(
                    "ref_audio",
                    tooltip="Standalone reference audio.",
                ),
                prefix="ref_audio_",
                min=0,
                max=3,
            )
            return _H3IO.Schema(
                node_id="KaggleH3Conditioning",
                display_name="Kaggle H3 Conditioning",
                category="conditioning/MiniMax H3",
                description=(
                    "Global H3 conditioning: prompt, Ref2VA mode, "
                    "inline upload/select controls, "
                    "and socket-based references."
                ),
                inputs=[
                    _H3IO.Clip.Input("clip"),
                    _H3IO.Vae.Input("vae", optional=True),
                    _H3IO.Vae.Input("audio_vae", optional=True),
                    _H3IO.Image.Input("start_frame", optional=True),
                    _H3IO.Image.Input("end_frame", optional=True),
                    _H3IO.Combo.Input(
                        "start_frame_file",
                        display_name="Start frame (select/upload)",
                        options=image_files,
                        default=_H3_EMPTY_FILE,
                        optional=True,
                        upload=_H3IO.UploadType.image,
                    ),
                    _H3IO.Combo.Input(
                        "end_frame_file",
                        display_name="End frame (select/upload)",
                        options=image_files,
                        default=_H3_EMPTY_FILE,
                        optional=True,
                        upload=_H3IO.UploadType.image,
                    ),
                    *[
                        _H3IO.DynamicCombo.Input(
                            f"reference_{index}",
                            options=reference_options,
                            display_name=f"Reference {index + 1} (type + file)",
                            optional=True,
                            tooltip=(
                                "Choose image, video, audio, or None. "
                                "Rows are passed to H3 in listed order."
                            ),
                        )
                        for index in range(_H3_INLINE_REFERENCE_SLOTS)
                    ],
                    _H3IO.Autogrow.Input(
                        "ref_images", optional=True, template=autogrow_image
                    ),
                    _H3IO.Autogrow.Input(
                        "ref_videos", optional=True, template=autogrow_video
                    ),
                    _H3IO.Autogrow.Input(
                        "ref_video_audios",
                        optional=True,
                        template=autogrow_video_audio,
                    ),
                    _H3IO.Autogrow.Input(
                        "ref_audios", optional=True, template=autogrow_audio
                    ),
                    _H3IO.String.Input(
                        "prompt", multiline=True, dynamic_prompts=True
                    ),
                    _H3IO.Combo.Input(
                        "mode", options=["Ref2VA"], default="Ref2VA"
                    ),
                    _H3IO.Float.Input(
                        "seconds",
                        default=5.0,
                        min=H3_MIN_FRAMES / 24.0,
                        max=H3_MAX_SECONDS,
                        step=0.001,
                    ),
                    _H3IO.Combo.Input(
                        "size_preset",
                        options=list(REF2VA_SIZE_PRESETS),
                        default="360p",
                    ),
                    _H3IO.Combo.Input(
                        "aspect_ratio",
                        options=["16:9", "4:3"],
                        default="16:9",
                    ),
                    _H3IO.Combo.Input(
                        "ref_image_size", options=["match", "max"], default="match"
                    ),
                    # Advanced exact geometry overrides keep API-generated
                    # graphs compatible with older clients; normal interactive
                    # use should prefer seconds + presets.
                    # ComfyUI may submit an optional integer widget as 0 even
                    # when it is not configured. Zero is the unset sentinel;
                    # nonzero values are validated below.
                    _H3IO.Int.Input(
                        "width", optional=True, default=0, min=0, max=8192, step=32
                    ),
                    _H3IO.Int.Input(
                        "height", optional=True, default=0, min=0, max=8192, step=32
                    ),
                    _H3IO.Int.Input(
                        "length", optional=True, default=0, min=0, max=3600, step=1
                    ),
                ],
                outputs=[
                    _H3IO.Conditioning.Output(display_name="positive"),
                    _H3IO.Latent.Output(),
                ],
            )

        @classmethod
        def execute(
            cls,
            clip,
            prompt,
            mode,
            seconds,
            size_preset,
            aspect_ratio,
            ref_image_size="match",
            vae=None,
            audio_vae=None,
            start_frame=None,
            end_frame=None,
            start_frame_file=None,
            end_frame_file=None,
            reference_0=None,
            reference_1=None,
            reference_2=None,
            reference_3=None,
            reference_4=None,
            reference_5=None,
            reference_6=None,
            reference_7=None,
            reference_8=None,
            reference_9=None,
            reference_10=None,
            reference_11=None,
            reference_12=None,
            reference_13=None,
            reference_14=None,
            ref_images=None,
            ref_videos=None,
            ref_video_audios=None,
            ref_audios=None,
            width=None,
            height=None,
            length=None,
        ):
            width = _h3_optional_override(width)
            height = _h3_optional_override(height)
            length = _h3_optional_override(length)
            start_frame = _resolve_h3_inline_frame(
                start_frame_file, start_frame, "start_frame"
            )
            end_frame = _resolve_h3_inline_frame(
                end_frame_file, end_frame, "end_frame"
            )
            inline_images, inline_videos, inline_video_audios, inline_audios = (
                _resolve_h3_inline_references(
                    [
                        reference_0,
                        reference_1,
                        reference_2,
                        reference_3,
                        reference_4,
                        reference_5,
                        reference_6,
                        reference_7,
                        reference_8,
                        reference_9,
                        reference_10,
                        reference_11,
                        reference_12,
                        reference_13,
                        reference_14,
                    ]
                )
            )
            typed_video_items = _h3_ordered_mapping(ref_videos, "ref_videos")
            ref_images = _merge_h3_reference_sources(
                inline_images,
                ref_images,
                "ref_images",
                "ref_image",
                offset=len(inline_images),
            )
            ref_videos = _merge_h3_reference_sources(
                inline_videos,
                ref_videos,
                "ref_videos",
                "ref_video",
                offset=len(inline_videos),
            )
            ref_audios = _merge_h3_reference_sources(
                inline_audios,
                ref_audios,
                "ref_audios",
                "ref_audio",
                offset=len(inline_audios),
            )
            ref_video_audios = _merge_h3_video_audio_sources(
                inline_video_audios,
                ref_video_audios,
                video_offset=len(inline_videos),
                typed_video_indices=[index for index, _item in typed_video_items],
            )
            preset_width, preset_height = resolve_ref2va_dimensions(
                size_preset, aspect_ratio
            )
            if (width is None) != (height is None):
                raise ValueError(
                    "Kaggle H3 Conditioning width and height overrides must be supplied together."
                )
            resolved_width = int(preset_width if width is None else width)
            resolved_height = int(preset_height if height is None else height)
            if resolved_width % 32 or resolved_height % 32:
                raise ValueError(
                    "Kaggle H3 Conditioning width and height must be multiples of 32."
                )
            duration = resolve_ref2va_length(seconds)
            resolved_length = int(duration["actual_frames"] if length is None else length)
            if resolved_length < H3_MIN_FRAMES or (resolved_length - 5) % 17:
                raise ValueError(
                    "Kaggle H3 Conditioning length must use H3's 17*k+5 frame grid."
                )
            print(
                "[Kaggle H3] Global conditioning: "
                f"mode={mode}, canvas={resolved_width}x{resolved_height}, "
                f"requested_seconds={float(seconds):g}, "
                f"frames={resolved_length}, "
                f"actual_seconds={resolved_length / 24.0:.4f}, "
                f"images={len(_compact_h3_autogrow(ref_images, 'ref_images'))}, "
                f"videos={len(_compact_h3_autogrow(ref_videos, 'ref_videos'))}, "
                f"audio={len(_compact_h3_autogrow(ref_audios, 'ref_audios'))}.",
                flush=True,
            )
            conditioning, latent = _execute_global_h3_conditioning(
                clip=clip,
                vae=vae,
                audio_vae=audio_vae,
                prompt=prompt,
                mode=mode,
                width=resolved_width,
                height=resolved_height,
                length=resolved_length,
                ref_image_size=ref_image_size,
                start_frame=start_frame,
                end_frame=end_frame,
                ref_images=ref_images,
                ref_videos=ref_videos,
                ref_video_audios=ref_video_audios,
                ref_audios=ref_audios,
            )
            return _H3IO.NodeOutput(conditioning, latent)

else:
    KaggleH3Conditioning = None  # type: ignore[assignment,misc]


def _node_output_values(result: Any) -> tuple[Any, ...]:
    """Convert a native V3 NodeOutput to a V1 custom-node return tuple."""

    values = getattr(result, "result", None)
    if values is not None:
        return tuple(values)
    if isinstance(result, tuple):
        return result
    raise RuntimeError(
        "ComfyUI's MiniMax H3 Ref2VA conditioner returned an unsupported result "
        f"type: {type(result).__name__}"
    )


class KaggleH3Ref2VAConditioning:
    """Ref2VA conditioner with seconds, aligned size presets, and ordered refs.

    V1 custom nodes cannot expose native ``io.Autogrow`` sockets, so this node
    provides bounded optional slots.  Users can connect any number of slots up
    to H3's native limits; empty slots are omitted before calling the native
    conditioner.  The native implementation remains responsible for actual
    reference resizing, tokenization, and VAE reference encoding.
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional: dict[str, Any] = {}
        for index in range(9):
            optional[f"ref_image_{index}"] = ("IMAGE",)
        for index in range(3):
            optional[f"ref_video_{index}"] = ("IMAGE",)
            optional[f"ref_video_audio_{index}"] = ("AUDIO",)
            optional[f"ref_audio_{index}"] = ("AUDIO",)
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "seconds": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": H3_MIN_FRAMES / 24.0,
                        "max": H3_MAX_SECONDS,
                        "step": 0.001,
                    },
                ),
                "size_preset": (
                    list(REF2VA_SIZE_PRESETS),
                    {"default": "480p"},
                ),
                "aspect_ratio": (["16:9", "4:3"], {"default": "16:9"}),
                "ref_image_size": (["match", "max"], {"default": "match"}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "condition"
    CATEGORY = "conditioning/MiniMax H3"

    def condition(
        self,
        clip: Any,
        vae: Any,
        audio_vae: Any,
        prompt: str,
        seconds: float,
        size_preset: str,
        aspect_ratio: str,
        ref_image_size: str = "match",
        **references: Any,
    ):
        width, height = resolve_ref2va_dimensions(size_preset, aspect_ratio)
        length = resolve_ref2va_length(seconds)
        ref_images = {
            f"ref_image_{index}": references.get(f"ref_image_{index}")
            for index in range(9)
            if references.get(f"ref_image_{index}") is not None
        }
        ref_videos = {
            f"ref_video_{index}": references.get(f"ref_video_{index}")
            for index in range(3)
            if references.get(f"ref_video_{index}") is not None
        }
        ref_video_audios = {
            f"ref_video_audio_{index}": references.get(f"ref_video_audio_{index}")
            for index in range(3)
            if references.get(f"ref_video_audio_{index}") is not None
        }
        ref_audios = {
            f"ref_audio_{index}": references.get(f"ref_audio_{index}")
            for index in range(3)
            if references.get(f"ref_audio_{index}") is not None
        }
        if not ref_images and not ref_videos:
            raise ValueError(
                "Kaggle H3 Ref2VA Conditioning needs at least one image or video "
                "reference; audio alone cannot define Ref2VA subject conditioning."
            )
        orphan_video_audio = sorted(
            index
            for index in range(3)
            if f"ref_video_audio_{index}" in ref_video_audios
            and f"ref_video_{index}" not in ref_videos
        )
        if orphan_video_audio:
            indexes = ", ".join(str(index) for index in orphan_video_audio)
            raise ValueError(
                "Each ref_video_audio_N must be connected with the matching "
                f"ref_video_N; orphan audio slot(s): {indexes}."
            )
        if not prompt or not str(prompt).strip():
            raise ValueError("Kaggle H3 Ref2VA Conditioning prompt must not be empty")

        print(
            "[Kaggle H3] Ref2VA conditioning: "
            f"canvas={width}x{height}, requested_seconds={float(seconds):g}, "
            f"frames={length['actual_frames']}, actual_seconds={length['actual_seconds']}, "
            f"images={len(ref_images)}, videos={len(ref_videos)}, "
            f"video_audio={len(ref_video_audios)}, audio={len(ref_audios)}.",
            flush=True,
        )
        native = _native_ref2va_conditioning()
        try:
            result = native.execute(
                clip=clip,
                vae=vae,
                audio_vae=audio_vae,
                prompt=prompt,
                width=width,
                height=height,
                length=int(length["actual_frames"]),
                ref_image_size=ref_image_size,
                ref_images=ref_images,
                ref_videos=ref_videos,
                ref_video_audios=ref_video_audios,
                ref_audios=ref_audios,
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Kaggle H3 Ref2VA conditioning failed: {exc}") from exc
        return _node_output_values(result)


class KaggleH3ShardedDiffusionLoader:
    """Select, download, and phase-load exactly one H3 diffusion variant."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_variant": (["Ref2VA"], {"default": "Ref2VA"}),
                "model_profile": (
                    ["singularity"],
                    {
                        "default": "singularity",
                        "tooltip": "Pinned MiniMax H3 Singularity Ref2VA checkpoint.",
                    },
                ),
                "precision": (
                    ["int8_convrot"],
                    {
                        "default": "int8_convrot",
                        "tooltip": "Pinned Singularity INT8 ConvRot checkpoint for the production T4 path.",
                    },
                ),
                "weight_dtype": (["default"], {"default": "default"}),
                "gpu_0": ("INT", {"default": 0, "min": 0, "max": 7}),
                "gpu_1": ("INT", {"default": 1, "min": 0, "max": 7}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "loaders/MiniMax H3"

    def load_model(
        self,
        model_variant: str,
        model_profile: str = "auto",
        precision: str = "int8_convrot",
        weight_dtype: str = "default",
        gpu_0: int = 0,
        gpu_1: int = 1,
    ):
        if str(model_variant) != "Ref2VA":
            raise H3PhaseError("The production H3 loader supports Ref2VA only.")
        if str(model_profile) != "singularity":
            raise H3PhaseError("The production H3 loader requires the Singularity profile.")
        if str(precision) != "int8_convrot":
            raise H3PhaseError("The production H3 loader requires the pinned INT8 ConvRot checkpoint.")
        if str(weight_dtype) != "default":
            raise H3PhaseError("The production H3 loader exposes only the validated default weight dtype.")
        try:
            import torch  # type: ignore
            import nodes  # type: ignore
        except Exception as exc:
            raise H3PhaseError("The H3 sharded loader must run inside ComfyUI with PyTorch") from exc
        ids = (int(gpu_0), int(gpu_1))
        try:
            _release_comfy_model_cache_before_h3_switch()
            diffusion_dir = h3_diffusion_directory_from_comfy()
            switch = H3DiffusionModelManager(diffusion_dir).switch(
                str(model_variant),
                precision=str(precision),
                profile=str(model_profile),
            )
        except H3DiffusionModelError as exc:
            raise H3PhaseError(f"Kaggle H3 diffusion auto-loader failed: {exc}") from exc
        try:
            # Resolve the file through ComfyUI's native loader after the
            # just-in-time switch. The native loader still owns model parsing,
            # dtype selection, patcher construction, and offloading behavior.
            loaded = nodes.UNETLoader().load_unet(switch.spec.filename, str(weight_dtype))
        except Exception as exc:
            raise H3PhaseError(
                "Kaggle H3 downloaded the selected diffusion checkpoint but "
                f"ComfyUI could not load {switch.spec.filename!r}: {exc}"
            ) from exc
        model = loaded[0]
        for owner in (model, getattr(model, "model", None)):
            if owner is None:
                continue
            try:
                setattr(owner, "h3_model_variant", switch.spec.variant.lower())
                setattr(owner, "h3_diffusion_precision", switch.spec.precision)
                setattr(owner, "h3_diffusion_profile", switch.spec.profile)
                setattr(owner, "kaggle_h3_diffusion_path", str(switch.path))
                setattr(owner, "kaggle_h3_diffusion_revision", switch.spec.revision)
            except Exception:
                pass
            options = getattr(owner, "model_options", None)
            if isinstance(options, dict):
                options["h3_model_variant"] = switch.spec.variant.lower()
                options["h3_diffusion_precision"] = switch.spec.precision
                options["h3_diffusion_profile"] = switch.spec.profile
                options["kaggle_h3_diffusion_path"] = str(switch.path)
        if phase_policy() == "off":
            print(
                "[Kaggle H3] H3 diffusion auto-loader selected "
                f"{switch.spec.variant} ({switch.spec.precision}; {switch.precision_reason}); "
                "native ComfyUI loader remains in control "
                "for the explicit one-GPU fallback.",
                flush=True,
            )
            return (model,)
        if not torch.cuda.is_available() or int(torch.cuda.device_count()) < 2:
            raise H3PhaseError(
                "Kaggle H3 Sharded Diffusion Loader requires two visible CUDA devices; "
                "use the standard UNETLoader only for the explicit one-GPU fallback."
            )
        prepare_model_for_h3_phase(model, device_ids=ids)
        print(
            "[Kaggle H3] H3 diffusion auto-loader selected: "
            f"variant={switch.spec.variant}, precision={switch.spec.precision}, "
            f"profile={switch.spec.profile}, "
            f"precision_reason={switch.precision_reason}, checkpoint={switch.path}, "
            f"downloaded={switch.downloaded}, removed={len(switch.removed)}, "
            f"loader_device=cpu, phase_gpus={list(ids)}.",
            flush=True,
        )
        return (model,)


class KaggleH3TextEncoderLoader:
    """Load and automatically shard the H3 Qwen language encoder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (_folder_options("text_encoders"),),
                "type": (["minimax"], {"default": "minimax"}),
                "device_id": ("INT", {"default": 0, "min": 0, "max": 7}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("CLIP",)
    FUNCTION = "load_encoder"
    CATEGORY = "loaders/MiniMax H3"

    def load_encoder(self, clip_name: str, type: str = "minimax", device_id: int = 0):
        try:
            import torch  # type: ignore
            import nodes  # type: ignore
        except Exception as exc:
            raise H3PhaseError("The H3 text encoder loader must run inside ComfyUI with PyTorch") from exc
        if not torch.cuda.is_available() or int(device_id) >= int(torch.cuda.device_count()):
            raise H3PhaseError(f"Cannot place the H3 text encoder on cuda:{device_id}")
        clip = nodes.CLIPLoader().load_clip(str(clip_name), str(type), "default")[0]
        if phase_policy() != "off":
            ids = tuple(phase_device_ids())
            if len(ids) < 2:
                raise H3PhaseError(
                    "H3 text-encoder sharding requires two visible GPUs; set "
                    "KAGGLE_H3_PHASE_SHARDING=off only for the explicit fallback."
                )
            if int(device_id) != ids[0]:
                raise H3PhaseError(
                    "The automatic H3 text-encoder sharding strategy is anchored on "
                    f"cuda:{ids[0]}; got cuda:{int(device_id)}."
                )
            state = prepare_text_encoder_for_h3_phase(
                clip,
                device_ids=ids,
                offload_dir=Path("/kaggle/tmp/minimax-h3-layer-offload/text_encoder"),
            )
            setattr(clip, "_kaggle_h3_conditioning_device", "cuda:0,cuda:1")
            setattr(clip, "_kaggle_h3_text_encoder_report", state.report)
        else:
            device = torch.device(f"cuda:{int(device_id)}")
            _set_explicit_patcher_devices(
                clip,
                load_device=device,
                offload_device=torch.device("cpu"),
            )
            setattr(clip, "_kaggle_h3_conditioning_device", str(device))
            print(
                f"[Kaggle H3] Text encoder explicit fallback target={device}; offload_device=cpu.",
                flush=True,
            )
        return (clip,)


class KaggleH3VAELoader:
    """Load one H3 VAE with an explicit phase target and CPU offload tier."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (_folder_options("vae"),),
                "role": (["video", "audio"], {"default": "video"}),
                "device_id": ("INT", {"default": 1, "min": 0, "max": 7}),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "loaders/MiniMax H3"

    def load_vae(self, vae_name: str, role: str = "video", device_id: int = 1):
        try:
            import nodes  # type: ignore
        except Exception as exc:
            raise H3PhaseError("The H3 VAE loader must run inside ComfyUI") from exc
        vae = nodes.VAELoader().load_vae(str(vae_name))[0]
        configure_vae_phase(vae, device_id=int(device_id), role=str(role))
        return (vae,)


class KaggleH3PhaseDispatch:
    """Dispatch H3 after all text/reference conditioning has completed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "latent_image": ("LATENT",),
            },
            "optional": {"runtime_config": ("H3_RUNTIME_CONFIG",)},
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "LATENT", "H3_RUNTIME_CONFIG")
    RETURN_NAMES = ("MODEL", "conditioning", "latent_image", "H3_RUNTIME_CONFIG")
    FUNCTION = "dispatch"
    CATEGORY = "model/dispatch/MiniMax H3"

    def dispatch(
        self,
        model: Any,
        conditioning: Any,
        latent_image: dict[str, Any],
        runtime_config: dict[str, Any] | None = None,
    ):
        validate_finite(conditioning, "conditioning_in", tensor_name="conditioning")
        state = phase_state_for_model(model)
        text_release: list[dict[str, Any]] = []
        try:
            if state is None:
                state = begin_transformer_phase(model, runtime_config)
            else:
                print(
                    "[Kaggle H3] Phase dispatch barrier found an existing verified transformer map.",
                    flush=True,
                )
        finally:
            # Conditioning has completed at this node. Release the temporary
            # Qwen layer router even when transformer dispatch itself fails.
            text_release = release_all_text_encoder_phases()
        if isinstance(runtime_config, dict):
            execution = dict(runtime_config.get("execution") or {})
            execution["dispatch_barrier"] = "conditioning_complete"
            execution["text_encoder_phase_release"] = text_release
            runtime_config["execution"] = execution
        return model, conditioning, latent_image, runtime_config


def _is_h3_video_vae(vae: Any) -> bool:
    """Identify only the MiniMax H3 video VAE, never the audio VAE."""

    model = getattr(vae, "first_stage_model", None)
    if model is None:
        return False
    class_names = {
        cls.__name__
        for cls in type(model).__mro__
    }
    if "MiniMaxH3VideoVAE" in class_names:
        return True
    # Some ComfyUI builds wrap the concrete VAE class. Keep the fallback
    # narrow: H3's temporal decoder has the distinctive x_embedder input
    # projection that appears in the decode traceback.
    decoder = getattr(model, "decoder", None)
    return callable(getattr(model, "decode_temporal", None)) and getattr(
        decoder, "x_embedder", None
    ) is not None


def _parameter_dtype(module: Any) -> Any:
    """Return a module's concrete weight dtype when it is observable."""

    weight = getattr(module, "weight", None)
    dtype = getattr(weight, "dtype", None)
    if dtype is not None:
        return dtype
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return None
    try:
        for parameter in parameters():
            dtype = getattr(parameter, "dtype", None)
            if dtype is not None:
                return dtype
    except Exception:
        # ComfyUI may expose lazy parameters while a model is offloaded.
        pass
    return None


def _h3_decode_input_dtype(vae: Any, fallback: Any) -> Any:
    """Find the dtype required by H3's first video-decoder projection.

    The ComfyUI VAE wrapper can report FP32 while the loaded H3 decoder has
    mixed FP16/FP32 weights. The input to ``decoder.x_embedder`` is the
    decisive boundary: PyTorch linear layers reject a float32 input with the
    FP16 weight observed in ComfyUI 0.34.0.
    """

    model = getattr(vae, "first_stage_model", None)
    if model is None:
        return fallback
    decoder = getattr(model, "decoder", None)
    x_embedder = getattr(decoder, "x_embedder", None)
    dtype = _parameter_dtype(x_embedder)
    if dtype is not None:
        return dtype
    dtype = _parameter_dtype(decoder)
    if dtype is not None:
        return dtype
    # Retain the old broad fallback for unusual wrapped implementations.
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            for parameter in parameters():
                dtype = getattr(parameter, "dtype", None)
                if dtype is not None:
                    return dtype
        except Exception:
            pass
    return fallback


def _h3_vae_dtype(vae: Any, fallback: Any) -> Any:
    """Read the actual dtype required by the H3 video decoder input."""

    if _is_h3_video_vae(vae):
        return _h3_decode_input_dtype(vae, fallback)
    return fallback


def _enable_h3_decode_weight_casting(vae: Any) -> int:
    """Enable ComfyUI's offload-aware per-layer casts for H3 video decode.

    ``--cpu-vae`` makes ComfyUI's VAE wrapper feed FP32 tensors to the VAE,
    while the FP16 H3 checkpoint can retain FP16 weights in decoder layers.
    ComfyUI's ``manual_cast`` operation handles this boundary without
    permanently converting the checkpoint or retaining a second VAE copy.
    """

    if not _is_h3_video_vae(vae):
        return 0
    model = getattr(vae, "first_stage_model", None)
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return 0
    changed = 0
    try:
        for module in modules():
            if not hasattr(module, "comfy_cast_weights"):
                continue
            if not getattr(module, "comfy_cast_weights"):
                module.comfy_cast_weights = True
                changed += 1
    except Exception:
        # A partially lazy/offloaded model may not expose every module. The
        # modules already updated remain safe, and the normal decode path can
        # still report any unresolved incompatibility.
        pass
    if changed:
        print(
            "[Kaggle H3] Enabled ComfyUI offload-aware mixed-dtype casting "
            f"for {changed} video VAE layers.",
            flush=True,
        )
    return changed


def _coerce_h3_video_samples(vae: Any, samples: dict[str, Any]) -> dict[str, Any]:
    """Match video latents to the loaded FP16/FP32 VAE without moving it.

    ComfyUI's CPU-VAE path can leave the H3 FP16 decoder weights in place while
    the sampler returns a float32 latent.  The H3 ViT decoder requires matching
    dtypes for its first linear projection.  Copying only the small latent
    dictionary preserves model offloading and avoids a duplicate VAE.
    """

    if not _is_h3_video_vae(vae):
        return samples
    latent = samples.get("samples")
    if latent is None or getattr(latent, "is_nested", False):
        return samples
    import torch

    target_dtype = _h3_vae_dtype(vae, getattr(latent, "dtype", torch.float32))
    if getattr(latent, "dtype", None) == target_dtype:
        return samples
    converted = dict(samples)
    converted["samples"] = latent.to(dtype=target_dtype)
    print(
        "[Kaggle H3] Cast video decode latent "
        f"{latent.dtype} -> {target_dtype} to match the loaded video VAE.",
        flush=True,
    )
    return converted


def _h3_video_images_to_comfy(images: Any) -> Any:
    """Convert H3 VAE video output to ComfyUI's frame-major IMAGE layout.

    Depending on the ComfyUI VAE wrapper/build, the MiniMax H3 decoder is
    observable either as ``[B, C, T, H, W]`` or as ``[B, T, H, W, C]``.
    ComfyUI image nodes consume a frame batch shaped ``[N, H, W, C]``.  Detect
    the channel axis before flattening so neither valid representation is
    reinterpreted as a different frame geometry.
    """

    if len(images.shape) != 5:
        return images
    if int(images.shape[-1]) == 3:
        # The current ComfyUI 0.34 H3 VAE wrapper returns this layout already.
        return images.reshape(-1, images.shape[2], images.shape[3], images.shape[4])
    if int(images.shape[1]) == 3:
        # Retain compatibility with the native channel-first decoder layout.
        images = images.permute(0, 2, 3, 4, 1).contiguous()
        return images.reshape(-1, images.shape[2], images.shape[3], images.shape[4])
    if int(images.shape[-1]) != 3 and int(images.shape[1]) != 3:
        raise RuntimeError(
            "MiniMax H3 video VAE returned an unexpected shape: "
            f"{tuple(images.shape)}; expected [batch, time, height, width, 3] "
            "or [batch, 3, time, height, width]."
        )


def _h3_phase_telemetry(runtime_config: dict[str, Any] | None) -> Any:
    """Create opt-in boundary telemetry for a VAE phase.

    The sampler and VAE nodes are separate ComfyUI executions, so the VAE
    cannot use the sampler's ContextVar directly. Reuse its generation id and
    output path from the serializable runtime config when available, allowing
    sampler and decode records to be compared in one trace.
    """

    if H3DiffusionTelemetry is None:
        return None
    existing: dict[str, Any] = {}
    if isinstance(runtime_config, dict):
        execution = runtime_config.get("execution")
        if isinstance(execution, dict):
            candidate = execution.get("diffusion_telemetry")
            if isinstance(candidate, dict):
                existing = candidate
    output_path = existing.get("output_path")
    telemetry = H3DiffusionTelemetry(output_path=output_path or None)
    if not telemetry.enabled:
        return None
    generation_id = existing.get("generation_id")
    if generation_id:
        telemetry.generation_id = str(generation_id)
    return telemetry


def _h3_store_phase_telemetry(
    runtime_config: dict[str, Any] | None,
    telemetry: Any,
) -> None:
    """Attach VAE records to the runtime config without affecting decoding."""

    if telemetry is None or not getattr(telemetry, "records", None):
        return
    if not isinstance(runtime_config, dict):
        return
    execution = dict(runtime_config.get("execution") or {})
    previous = execution.get("vae_telemetry")
    previous_records = (
        list(previous.get("records") or [])
        if isinstance(previous, dict)
        else []
    )
    records = previous_records + list(telemetry.records)
    execution["vae_telemetry"] = {
        "enabled": bool(getattr(telemetry, "enabled", False)),
        "generation_id": getattr(telemetry, "generation_id", None),
        "record_count": len(records),
        "output_path": (
            str(telemetry.output_path) if telemetry.output_path is not None else None
        ),
        "records": records,
    }
    runtime_config["execution"] = execution


class H3VAEDecode:
    """H3 video decode node with explicit latent/VAE dtype reconciliation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "device_id": ("INT", {"default": 1, "min": 0, "max": 7}),
            },
            "optional": {
                "runtime_config": ("H3_RUNTIME_CONFIG",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "model/latent/MiniMax H3"

    def decode(
        self,
        vae: Any,
        samples: dict[str, Any],
        device_id: int = 1,
        runtime_config: dict[str, Any] | None = None,
    ):
        telemetry = _h3_phase_telemetry(runtime_config)
        try:
            configure_vae_phase(
                vae,
                device_id=int(device_id),
                role="video",
                runtime_config=runtime_config,
            )
            _enable_h3_decode_weight_casting(vae)
            video_samples = {"samples": _h3_stream_tensor(samples, "video")}
            # Inspect the stream before Comfy's VAE compatibility cast. The
            # destination check below separately covers the post-transfer
            # tensor immediately before decode.
            validate_finite(
                video_samples["samples"],
                "vae_video_in",
                tensor_name="video_latent.source",
            )
            if telemetry is not None:
                telemetry.record_phase(
                    stage="vae_video_in_source",
                    value=video_samples["samples"],
                    tensor_name="video_latent.source",
                )
            video_samples = _coerce_h3_video_samples(vae, video_samples)
            latent = _h3_move_for_consumer(
                video_samples["samples"], device_id=int(device_id), role="video VAE"
            )
            if telemetry is not None:
                telemetry.record_phase(
                    stage="vae_video_in",
                    value=latent,
                    tensor_name="video_latent",
                )
            validate_finite(latent, "vae_video_in", tensor_name="video_latent")
            validate_finite(latent, "vae_video", tensor_name="video_latent")
            images = vae.decode(latent)
            synchronize_h3_devices(
                (getattr(images, "device", None),),
                reason="video VAE decode completion",
            )
            validate_finite(images, "vae_video_out", tensor_name="video_frames")
            validate_finite(images, "vae_video", tensor_name="video_frames")
            if telemetry is not None:
                telemetry.record_phase(
                    stage="vae_video_out",
                    value=images,
                    tensor_name="video_frames",
                )
            images = _h3_video_images_to_comfy(images)
            # The encoder and CreateVideo do not need GPU-resident frames.
            output = images.to("cpu")
            if telemetry is not None:
                telemetry.record_phase(
                    stage="vae_video_output",
                    value=output,
                    tensor_name="video_frames.cpu",
                )
            return (output,)
        finally:
            _h3_store_phase_telemetry(runtime_config, telemetry)
            release_vae_phase(vae, role="video")


class H3AudioVAEDecode:
    """Decode H3 audio on GPU0 after the transformer phase is released."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "device_id": ("INT", {"default": 0, "min": 0, "max": 7}),
                "mute_generated_audio": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Skip audio VAE decoding and produce a video-only MP4.",
                    },
                ),
            },
            "optional": {
                "runtime_config": ("H3_RUNTIME_CONFIG",),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "decode"
    CATEGORY = "model/latent/MiniMax H3"

    def decode(
        self,
        vae: Any,
        samples: dict[str, Any],
        device_id: int = 0,
        mute_generated_audio: bool = False,
        runtime_config: dict[str, Any] | None = None,
    ):
        if bool(mute_generated_audio):
            print(
                "[Kaggle H3] Generated audio muted; skipping audio VAE decode.",
                flush=True,
            )
            return (None,)
        telemetry = _h3_phase_telemetry(runtime_config)
        try:
            configure_vae_phase(
                vae,
                device_id=int(device_id),
                role="audio",
                runtime_config=runtime_config,
            )
            try:
                from comfy_extras.nodes_audio import vae_decode_audio  # type: ignore
            except ImportError:
                from nodes_audio import vae_decode_audio  # type: ignore
            audio_samples = {
                "samples": _h3_stream_tensor(samples, "audio"),
            }
            if telemetry is not None:
                telemetry.record_phase(
                    stage="vae_audio_in_source",
                    value=audio_samples["samples"],
                    tensor_name="audio_latent.source",
                )
            validate_finite(
                audio_samples["samples"],
                "vae_audio_in",
                tensor_name="audio_latent.source",
            )
            audio_tensor = _h3_move_for_consumer(
                audio_samples["samples"], device_id=int(device_id), role="audio VAE"
            )
            if telemetry is not None:
                telemetry.record_phase(
                    stage="vae_audio_in",
                    value=audio_tensor,
                    tensor_name="audio_latent",
                )
            validate_finite(audio_tensor, "vae_audio_in", tensor_name="audio_latent")
            validate_finite(audio_tensor, "vae_audio", tensor_name="audio_latent")
            audio = vae_decode_audio(vae, {"samples": audio_tensor})
            audio_waveform = audio.get("waveform") if isinstance(audio, dict) else audio
            synchronize_h3_devices(
                (getattr(audio_waveform, "device", None),),
                reason="audio VAE decode completion",
            )
            validate_finite(audio, "vae_audio_out", tensor_name="audio_output")
            validate_finite(audio, "vae_audio", tensor_name="audio_output")
            if telemetry is not None:
                telemetry.record_phase(
                    stage="vae_audio_out",
                    value=audio,
                    tensor_name="audio_output",
                )
            output = _h3_finite_audio(audio)
            if telemetry is not None:
                telemetry.record_phase(
                    stage="vae_audio_output",
                    value=output,
                    tensor_name="audio_output.sanitized",
                )
            return (output,)
        finally:
            _h3_store_phase_telemetry(runtime_config, telemetry)
            release_vae_phase(vae, role="audio")


def _install_phase_dispatch_hook(
    sampler_helpers: Any,
    runtime_config: dict[str, Any] | None,
    phase_holder: dict[str, Any],
    *,
    synchronization_mode: str = "full",
):
    """Install the post-Comfy-load H3 dispatch hook and return the original."""

    original_prepare_sampling = sampler_helpers.prepare_sampling

    def prepare_sampling_then_dispatch(*args, **kwargs):
        """Let ComfyUI finish its managed load, then replace its one-GPU map."""

        phase_model = args[0] if args else kwargs.get("model")
        if phase_model is None:
            raise H3PhaseError(
                "ComfyUI did not expose the model patcher at its sampling load boundary"
            )
        existing = phase_state_for_model(phase_model)
        if existing is not None and existing.active:
            phase_holder["state"] = existing
            set_h3_synchronization_mode(existing, synchronization_mode)
            # A custom phase barrier has already applied the verified H3 map
            # (Comfy-native for quantized H3, Accelerate for ordinary modules).
            # Preserve it while still allowing Comfy to prepare conditioning
            # hooks and any unrelated auxiliary models.
            return _prepare_sampling_preserving_phase(
                original_prepare_sampling,
                phase_model,
                args,
                kwargs,
            )
        result = original_prepare_sampling(*args, **kwargs)
        phase_holder["state"] = begin_transformer_phase(
            phase_model,
            runtime_config,
            synchronization_mode=synchronization_mode,
        )
        return result

    sampler_helpers.prepare_sampling = prepare_sampling_then_dispatch
    return original_prepare_sampling


def _install_standard_h3_decode_compatibility() -> None:
    """Fix existing VAEDecode graphs without requiring a workflow rewrite.

    The shim is narrowly gated to MiniMaxH3VideoVAE and is idempotent. It also
    covers VAEDecodeTiled for users who replace the default decoder later.
    """

    try:
        import nodes
    except Exception:
        return
    for class_name in ("VAEDecode", "VAEDecodeTiled"):
        node_class = getattr(nodes, class_name, None)
        original = getattr(node_class, "decode", None)
        if node_class is None or not callable(original):
            continue
        if getattr(node_class, "_kaggle_h3_dtype_compat", False):
            continue

        def decode_with_h3_dtype(self, vae, samples, *args, _original=original, **kwargs):
            _enable_h3_decode_weight_casting(vae)
            samples = _coerce_h3_video_samples(vae, samples)
            return _original(self, vae, samples, *args, **kwargs)

        node_class.decode = decode_with_h3_dtype
        node_class._kaggle_h3_dtype_compat = True


class H3AdapterStack:
    """Resolve, validate, and apply up to three ordered H3 adapters."""

    @classmethod
    def INPUT_TYPES(cls):
        catalog = _catalog()
        options = [AUTO_SINGULARITY_ADAPTER] + [
            item
            for item in selector_options(catalog)
            if item == "None" or item.startswith("h3_singularity_ref2va_")
        ]
        return {
            "required": {
                "model": ("MODEL",),
                "model_variant": (["ref2va"], {"default": "ref2va", "advanced": True}),
                "turbo_mode": ("BOOLEAN", {"default": True}),
                "turbo_steps": ("INT", {"default": 4, "min": MIN_TURBO_STEPS, "max": MAX_TURBO_STEPS}),
                "adapter_1": (options, {"default": AUTO_SINGULARITY_ADAPTER}),
                "strength_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "adapter_2": (options, {"default": "None"}),
                "strength_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "adapter_3": (options, {"default": "None"}),
                "strength_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "conditioning_mode": (["Ref2VA"], {"default": "Ref2VA"}),
            },
        }

    RETURN_TYPES = ("MODEL", "H3_RUNTIME_CONFIG")
    RETURN_NAMES = ("MODEL", "H3_RUNTIME_CONFIG")
    FUNCTION = "apply"
    CATEGORY = "model/patching/MiniMax H3"

    def apply(
        self,
        model: Any,
        turbo_mode: bool,
        turbo_steps: str | int,
        adapter_1: str,
        strength_1: float,
        adapter_2: str,
        strength_2: float,
        adapter_3: str,
        strength_3: float,
        conditioning_mode: str = "Ref2VA",
        model_variant: str = "ref2va",
    ):
        if str(model_variant).lower() != "ref2va":
            raise H3AdapterError("The production H3 Adapter Stack supports Ref2VA only.")
        if str(conditioning_mode) != "Ref2VA":
            raise H3AdapterError("The production H3 Adapter Stack requires Ref2VA conditioning.")
        catalog = _catalog()
        runtime_config = build_runtime_config(
            model,
            catalog,
            turbo_mode=bool(turbo_mode),
            turbo_steps=turbo_steps,
            adapter_slots=(
                (adapter_1, strength_1),
                (adapter_2, strength_2),
                (adapter_3, strength_3),
            ),
            conditioning_mode=conditioning_mode,
            model_variant=model_variant,
        )
        if runtime_config.get("turbo", {}).get("injection_skipped"):
            print(
                "[Kaggle H3] Turbo already fused into checkpoint; skipping duplicate injection.",
                flush=True,
            )
        try:
            patched, runtime_config = apply_adapter_stack(
                model, runtime_config, AdapterCache(), catalog
            )
        except H3AdapterError:
            raise
        except Exception as exc:
            raise RuntimeError(f"H3 Adapter Stack failed: {exc}") from exc
        return (patched, runtime_config)


class H3TurboSampler:
    """Dedicated H3 sampler that consumes runtime config and enforces NFE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "sampler_name": (
                    list(H3_EULER_SAMPLER_NAMES),
                    {
                        "default": "euler",
                        "tooltip": (
                            "Euler is the default. Euler Dual-clock explicitly selects "
                            "H3's native video/audio ModelSamplingAV protocol."
                        ),
                    },
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "synchronize_mode": (
                    ["full", "safe", "fast"],
                    {"default": "full"},
                ),
            },
            "optional": {
                "runtime_config": ("H3_RUNTIME_CONFIG",),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "LATENT")
    RETURN_NAMES = ("video_latent", "audio_latent", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "sampling/MiniMax H3"

    @staticmethod
    def _resolve_sampling_parameters(
        runtime_config: dict[str, Any] | None,
        sampler_name: str,
        steps: int = 20,
    ):
        """Validate the sampler contract before touching the loaded model."""

        sampler_name = str(sampler_name)
        if runtime_config is None:
            if sampler_name not in H3_EULER_SAMPLER_NAMES:
                raise RuntimeError(
                    "Production H3 sampling requires sampler 'euler' or "
                    "'euler_dualclock'."
                )
            return (
                {
                    "steps": int(steps),
                    "nfe": int(steps),
                    "sampler_name": sampler_name,
                    "base_sampler_name": "euler",
                    "dual_clock": sampler_name == "euler_dualclock",
                    "scheduler": "simple",
                    "shift_video": 12.0,
                    "shift_audio": 3.0,
                    "source": "production_singularity_fallback",
                },
                max(1, int(steps)),
            )
        if not isinstance(runtime_config, dict) or runtime_config.get("schema_version") != "kaggle_h3.runtime.v1":
            raise RuntimeError("H3 Turbo Sampler received an invalid H3_RUNTIME_CONFIG")
        turbo = (runtime_config or {}).get("turbo") or {}
        schedule = dict(runtime_config.get("sampler") or {})
        expected_steps = schedule.get("steps")
        if turbo.get("enabled"):
            if expected_steps is None or not MIN_TURBO_STEPS <= int(expected_steps) <= MAX_TURBO_STEPS:
                raise RuntimeError(f"H3 runtime config contains unsupported Turbo steps: {expected_steps!r}")
            expected_sampler_name = str(schedule.get("sampler_name") or "euler")
            if (
                sampler_name not in H3_EULER_SAMPLER_NAMES
                or expected_sampler_name not in H3_EULER_SAMPLER_NAMES
            ):
                raise RuntimeError(
                    f"H3 Turbo schedule requires sampler Euler or Euler Dual-clock; "
                    f"got expected={expected_sampler_name!r}, selected={sampler_name!r}"
                )
            # Both modes use stock ComfyUI Euler.  The explicit mode changes
            # the protocol selection and telemetry only; audio is not updated
            # a second time outside native ModelSamplingAV.
            schedule["base_sampler_name"] = "euler"
            schedule["sampler_name"] = sampler_name
            schedule["dual_clock"] = sampler_name == "euler_dualclock"
        elif sampler_name != schedule.get("sampler_name", sampler_name):
            raise RuntimeError(
                f"H3 runtime config requires sampler {schedule.get('sampler_name')!r}; got {sampler_name!r}"
            )
        return schedule, (int(steps) if expected_steps is None else int(expected_steps))

    @staticmethod
    def _shift_model(model: Any, schedule: dict[str, Any]):
        """Apply H3's clone-based AV sigma-shift operation exactly once.

        Recent ComfyUI H3 builds may already carry a ``ModelSamplingAV``
        object from a native sigma-shift node.  Replacing an equivalent object
        is harmless mathematically but can duplicate the apparent sampling
        protocol in diagnostics and custom wrappers.  Reuse an equivalent
        native object and only install a replacement when the requested shifts
        differ.
        """

        import comfy.model_sampling  # type: ignore

        if not hasattr(comfy.model_sampling, "ModelSamplingAV"):
            raise RuntimeError(
                "This ComfyUI build does not provide ModelSamplingAV, so the "
                "H3 dual-clock sampler cannot be enabled. Update ComfyUI or "
                "choose a compatible H3 build."
            )

        shifted = model.clone()

        shift_video = float(schedule["shift_video"])
        shift_audio = float(schedule["shift_audio"])
        original = shifted.get_model_object("model_sampling")
        same_native_av = False
        try:
            same_native_av = (
                hasattr(original, "audio_scale")
                and math.isclose(float(getattr(original, "shift")), shift_video)
                and math.isclose(float(getattr(original, "audio_shift")), shift_audio)
            )
        except (AttributeError, TypeError, ValueError):
            same_native_av = False

        class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
            pass

        if not same_native_av:
            model_sampling = ModelSamplingAdvanced(model.model.model_config)
            model_sampling.set_parameters(
                shift=shift_video,
                audio_shift=shift_audio,
            )
            if hasattr(original, "noise_scale"):
                model_sampling.set_noise_scale(original.noise_scale)
            shifted.add_object_patch("model_sampling", model_sampling)
        copied_options = dict(getattr(shifted, "model_options", {}) or {})
        transformer_options = dict(copied_options.get("transformer_options") or {})
        transformer_options["minimax_h3_sigma_shift_video"] = shift_video
        transformer_options["minimax_h3_sigma_shift_audio"] = shift_audio
        copied_options["transformer_options"] = transformer_options
        shifted.model_options = copied_options
        if same_native_av:
            print(
                "[Kaggle H3] Reusing existing native ModelSamplingAV object; "
                "dual-clock shifts are not applied twice.",
                flush=True,
            )
        return shifted

    def sample(
        self,
        model: Any,
        runtime_config: dict[str, Any] | None,
        conditioning: Any,
        latent_image: dict[str, Any],
        noise_seed: int,
        sampler_name: str,
        steps: int = 20,
        synchronize_mode: str = "full",
        sage_attention: bool = False,
    ):
        if bool(sage_attention):
            raise RuntimeError("SageAttention is not part of the production H3 profile yet.")
        import comfy.sample  # type: ignore
        import comfy.samplers  # type: ignore
        import comfy.sampler_helpers  # type: ignore
        import comfy.model_management  # type: ignore
        import comfy.utils  # type: ignore
        import latent_preview  # type: ignore
        from comfy_extras.nodes_custom_sampler import Guider_Basic, Noise_RandomNoise  # type: ignore

        synchronize_mode = normalize_h3_synchronization_mode(synchronize_mode)
        schedule, expected_steps = self._resolve_sampling_parameters(
            runtime_config, sampler_name, steps
        )
        turbo = (runtime_config or {}).get("turbo") or {}

        if isinstance(runtime_config, dict):
            execution = dict(runtime_config.get("execution") or {})
            execution["synchronization"] = h3_synchronization_plan(synchronize_mode)
            runtime_config["execution"] = execution

        # Keep the sampler safe for hand-edited graphs that bypass the phase
        # dispatch node. The normal Ref2VA graph validates this same boundary
        # at KaggleH3PhaseDispatch first.
        validate_finite(conditioning, "conditioning_in", tensor_name="conditioning")
        resolved_mode = (runtime_config or {}).get("model", {}).get("conditioning_mode", "auto")
        if resolved_mode == "auto":
            resolved_mode = detect_conditioning_mode(conditioning)
        candidates = (runtime_config or {}).get("model", {}).get("conditioning_mode_candidates", [])
        if candidates and resolved_mode not in candidates:
            raise RuntimeError(
                f"H3 conditioning mode {resolved_mode!r} is incompatible with the adapter stack; "
                f"expected one of {candidates}"
            )

        sampling_model = self._shift_model(model, schedule)
        phase_state = None
        phase_holder: dict[str, Any] = {}
        # ComfyUI 0.34 calls this module-level helper from CFGGuider.outer_sample.
        # Installing a narrowly-scoped wrapper is necessary: dispatching before
        # guider.sample() would be undone by ComfyUI's own load_models_gpu call.
        original_prepare_sampling = _install_phase_dispatch_hook(
            comfy.sampler_helpers,
            runtime_config,
            phase_holder,
            synchronization_mode=synchronize_mode,
        )
        try:
            sigmas = comfy.samplers.calculate_sigmas(
                sampling_model.get_model_object("model_sampling"),
                str(schedule.get("scheduler", "simple")),
                int(expected_steps),
            ).cpu()
            sampler_backend_name = (
                "euler" if sampler_name == "euler_dualclock" else sampler_name
            )
            actual_clock_schedule = _h3_actual_clock_schedule(
                sigmas,
                shift_video=float(schedule.get("shift_video", 12.0)),
                shift_audio=float(schedule.get("shift_audio", 3.0)),
            )
            sampler_selection = {
                "requested": sampler_name,
                "backend": sampler_backend_name,
                "dual_clock": sampler_name == "euler_dualclock",
                "protocol": "native_model_sampling_av",
                "manual_audio_update": False,
            }
            print(
                "[Kaggle H3] sampler selection and actual video/audio schedules: "
                + json.dumps(
                    {
                        "selection": sampler_selection,
                        "schedule": actual_clock_schedule,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            if isinstance(runtime_config, dict):
                execution = dict(runtime_config.get("execution") or {})
                execution["sampler_selection"] = sampler_selection
                execution["video_audio_schedule"] = actual_clock_schedule
                runtime_config["execution"] = execution
            sampler = _h3_prepare_sampler(
                sampler_backend_name,
                comfy.samplers.sampler_object(sampler_backend_name),
                dual_clock=bool(schedule.get("dual_clock", False)),
            )
            guider = Guider_Basic(sampling_model)
            guider.set_conds(conditioning)

            latent = latent_image
            latent_image_tensor = latent["samples"]
            latent = latent.copy()
            latent_image_tensor = comfy.sample.fix_empty_latent_channels(
                guider.model_patcher,
                latent_image_tensor,
                latent.get("downscale_ratio_spacial"),
                latent.get("downscale_ratio_temporal"),
            )
            latent["samples"] = latent_image_tensor
            noise_mask = latent.get("noise_mask")
            x0_output: dict[str, Any] = {}
            callback = latent_preview.prepare_callback(
                guider.model_patcher, sigmas.shape[-1] - 1, x0_output
            )
            synchronization_token = _H3_SAMPLER_SYNCHRONIZATION_MODE.set(
                synchronize_mode
            )
            diffusion_telemetry = (
                H3DiffusionTelemetry() if H3DiffusionTelemetry is not None else None
            )
            diffusion_telemetry_token = _H3_DIFFUSION_TELEMETRY.set(
                diffusion_telemetry
            )
            rmsnorm_report: dict[str, Any] = {}
            sage_attention_report: dict[str, Any] = {}
            try:
                with h3_sage_attention(bool(sage_attention)) as sage_attention_report:
                    with h3_rmsnorm_dtype_alignment() as rmsnorm_report:
                        samples = guider.sample(
                            Noise_RandomNoise(int(noise_seed)).generate_noise(latent),
                            latent_image_tensor,
                            sampler,
                            sigmas,
                            denoise_mask=noise_mask,
                            callback=callback,
                            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                            seed=int(noise_seed),
                        )
            finally:
                _H3_DIFFUSION_TELEMETRY.reset(diffusion_telemetry_token)
                _H3_SAMPLER_SYNCHRONIZATION_MODE.reset(synchronization_token)
            if isinstance(runtime_config, dict):
                execution = dict(runtime_config.get("execution") or {})
                execution["rmsnorm_dtype_alignment"] = dict(rmsnorm_report)
                execution["sage_attention"] = dict(sage_attention_report)
                execution["dual_clock_sampling"] = {
                    "requested": bool(schedule.get("dual_clock", False)),
                    "sampler_mode": sampler_name,
                    "backend": "native_model_sampling_av",
                    "video_shift": schedule.get("shift_video"),
                    "audio_shift": schedule.get("shift_audio"),
                    "manual_audio_update": False,
                }
                if diffusion_telemetry is not None and diffusion_telemetry.enabled:
                    execution["diffusion_telemetry"] = diffusion_telemetry.summary()
                runtime_config["execution"] = execution
            synchronize_h3_devices(
                _h3_sampler_boundary_devices(synchronize_mode),
                reason="transformer sampler completion",
            )
            validate_finite(
                samples,
                "sampler_final_out",
                tensor_name="sampler_output",
            )
            validate_finite(
                samples,
                "sampler_final",
                tensor_name="sampler_output",
            )

            # H3 returns a NestedTensor pair. Keep the final output on the
            # phase primary GPU, then expose separate consumer-owned streams:
            # video can cross GPU0 -> GPU1 once, while audio stays on GPU0.
            # Moving the packed result to Comfy's intermediate device here
            # would force both streams through CPU and retain an unnecessary
            # video+audio container until both decoders finish.
            video_tensor = _h3_stream_tensor({"samples": samples}, "video")
            audio_tensor = _h3_stream_tensor({"samples": samples}, "audio")
            phase_ids = tuple(phase_device_ids())
            primary_id = int(phase_ids[0]) if phase_ids else 0
            video_id = int(phase_ids[1]) if len(phase_ids) > 1 else primary_id
            video_tensor = _h3_move_for_consumer(
                video_tensor, device_id=primary_id, role="final video output"
            )
            audio_tensor = _h3_move_for_consumer(
                audio_tensor, device_id=primary_id, role="final audio output"
            )
            video_output = {
                "samples": video_tensor,
                "type": "h3_video",
                "_kaggle_h3_stream": "video",
                "_kaggle_h3_final_device": str(video_tensor.device),
                "_kaggle_h3_next_device": f"cuda:{video_id}",
            }
            audio_output = {
                "samples": audio_tensor,
                "type": "h3_audio",
                "_kaggle_h3_stream": "audio",
                "_kaggle_h3_final_device": str(audio_tensor.device),
                "_kaggle_h3_next_device": f"cuda:{primary_id}",
            }
            if isinstance(runtime_config, dict):
                execution = dict(runtime_config.get("execution") or {})
                execution["consumer_aware_latent_routing"] = {
                    "h3_final_output": f"cuda:{primary_id}",
                    "video": {
                        "source": str(video_tensor.device),
                        "next_consumer": f"cuda:{video_id}",
                        "transfer": "direct_gpu_to_gpu" if video_id != primary_id else "same_device",
                        "cpu_staging": False,
                    },
                    "audio": {
                        "source": str(audio_tensor.device),
                        "next_consumer": f"cuda:{primary_id}",
                        "transfer": "same_device",
                        "cpu_staging": False,
                    },
                    "decoded_outputs": "GPU_to_CPU_before_encode",
                }
                runtime_config["execution"] = execution
            if "x0" in x0_output:
                denoised = latent.copy()
                # Convert this before the transformer is unloaded; this is
                # the only post-sample operation that still needs the model.
                denoised["samples"] = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
            else:
                # Do not retain a second packed AV NestedTensor just for an
                # unconsumed fallback output. The primary video result is the
                # same denoised sample in this branch and shares its storage.
                denoised = video_output
        finally:
            comfy.sampler_helpers.prepare_sampling = original_prepare_sampling
            phase_state = phase_holder.get("state")
            original_exception = sys.exc_info()[0] is not None
            release_report: dict[str, Any]
            cleanup_error: Exception | None = None
            try:
                release_report = release_transformer_phase(phase_state)
            except Exception as exc:
                cleanup_error = exc
                release_report = {"status": "cleanup_failed", "error": str(exc)}
                print(
                    f"[Kaggle H3] Transformer cleanup raised after sampling: {exc}",
                    flush=True,
                )
            try:
                runtime_cleanup_report = release_h3_runtime_resources()
                if runtime_cleanup_report.get("status") == "released_with_errors":
                    cleanup_error = H3PhaseError(
                        "H3 runtime cleanup reported errors: "
                        + "; ".join(runtime_cleanup_report.get("errors", []))
                    )
            except Exception as exc:
                cleanup_error = cleanup_error or exc
                print(
                    f"[Kaggle H3] Final runtime cleanup raised: {exc}",
                    flush=True,
                )
            if cleanup_error is not None and not original_exception:
                raise H3PhaseError(
                    f"H3 runtime cleanup failed after sampling: {cleanup_error}"
                ) from cleanup_error
            if (
                phase_state is not None
                and phase_state.active
                and not (release_report.get("generation_activity") or {}).get(
                    "same_generation_all_requested_gpus", False
                )
            ):
                raise H3PhaseError(
                    "The H3 generation did not execute dispatchable transformer blocks "
                    f"on every requested GPU: {release_report.get('generation_activity')}. "
                    "The result was rejected instead of being labeled sharded."
                )
        print(
            f"[Kaggle H3] {resolved_mode} sampler consumed H3 runtime config: "
            f"turbo={bool(turbo.get('enabled'))}, steps={expected_steps}, "
            f"shift={schedule.get('shift_video')}/{schedule.get('shift_audio')}, "
            f"sampler={sampler_name}, dual_clock={sampler_name == 'euler_dualclock'}, "
            f"sage_attention={bool(sage_attention)}, synchronize={synchronize_mode}",
            flush=True,
        )
        return (video_output, audio_output, denoised)


NODE_CLASS_MAPPINGS = {
    "KaggleH3SmokeReference": KaggleH3SmokeReference,
    "KaggleH3Ref2VAConditioning": KaggleH3Ref2VAConditioning,
    "KaggleH3ShardedDiffusionLoader": KaggleH3ShardedDiffusionLoader,
    "KaggleH3TextEncoderLoader": KaggleH3TextEncoderLoader,
    "KaggleH3VAELoader": KaggleH3VAELoader,
    "KaggleH3PhaseDispatch": KaggleH3PhaseDispatch,
    "KaggleH3AdapterStack": H3AdapterStack,
    "KaggleH3TurboSampler": H3TurboSampler,
    "KaggleH3VAEDecode": H3VAEDecode,
    "KaggleH3AudioVAEDecode": H3AudioVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KaggleH3SmokeReference": "Kaggle H3 | Smoke Reference (auto-resolved)",
    "KaggleH3Ref2VAConditioning": "Kaggle H3 | Ref2VA Conditioning (Seconds + Presets)",
    "KaggleH3ShardedDiffusionLoader": "Kaggle H3 | Singularity Ref2VA INT8 Auto Loader",
    "KaggleH3TextEncoderLoader": "Kaggle H3 | Text Encoder (GPU0 + GPU1)",
    "KaggleH3VAELoader": "Kaggle H3 | VAE Loader",
    "KaggleH3PhaseDispatch": "Kaggle H3 | Transformer Dispatch Barrier",
    "KaggleH3AdapterStack": "Kaggle H3 | Adapter Stack",
    "KaggleH3TurboSampler": "Kaggle H3 | Turbo Sampler",
    "KaggleH3VAEDecode": "Kaggle H3 | Video VAE Decode (GPU1)",
    "KaggleH3AudioVAEDecode": "Kaggle H3 | Audio VAE Decode (GPU0)",
}


_install_standard_h3_decode_compatibility()
