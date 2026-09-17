"""Optional, sampler-scoped SageAttention integration for MiniMax H3.

ComfyUI imports H3's ``optimized_attention`` function into the MiniMax model
module.  Changing only the generic attention module after ComfyUI has booted
therefore does not affect an already-imported H3 model.  This helper patches
that one H3 symbol for the duration of a sampler call and restores it in the
same ``finally`` path as the phase dispatcher.

The dependency remains optional.  Enabling the node option without a working
``sageattention`` installation raises a clear error instead of silently
running a different attention backend.
"""

from __future__ import annotations

from contextlib import contextmanager
import re
from typing import Any, Iterator


class H3SageAttentionError(RuntimeError):
    """Raised when the optional SageAttention backend cannot be enabled."""


def _version_tuple(value: Any) -> tuple[int, ...] | None:
    match = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", str(value or ""))
    if not match:
        return None
    return tuple(int(item or 0) for item in match.groups())


def _cuda_runtime_status() -> dict[str, Any]:
    result: dict[str, Any] = {
        "cuda_available": False,
        "gpu_architectures": [],
        "gpu_names": [],
    }
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return result
        result["cuda_available"] = True
        for index in range(int(torch.cuda.device_count())):
            major, minor = torch.cuda.get_device_capability(index)
            result["gpu_architectures"].append(f"sm{major}{minor}")
            result["gpu_names"].append(str(torch.cuda.get_device_name(index)))
    except Exception as exc:
        result["cuda_error"] = f"{type(exc).__name__}: {exc}"
    return result


def sage_attention_status() -> dict[str, Any]:
    """Report the installed SageAttention and ComfyUI integration surface."""

    status: dict[str, Any] = {
        "available": False,
        "package": None,
        "version": None,
        "comfy_attention_sage": False,
        "supports_h3_packed_containers": False,
        "implementation": "SageAttention v1",
        "triton_version": None,
        "triton_compatible": False,
        "cuda_available": False,
        "gpu_architectures": [],
        "gpu_names": [],
        "runtime_compatible": False,
    }
    status.update(_cuda_runtime_status())
    try:
        import sageattention  # type: ignore

        status["package"] = "sageattention"
        status["version"] = getattr(sageattention, "__version__", None)
        status["available"] = callable(getattr(sageattention, "sageattn", None))
        sage_version = _version_tuple(status["version"])
        status["sage_v1_compatible"] = bool(
            sage_version is not None and sage_version[0] == 1
        )
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        return status

    try:
        import triton  # type: ignore

        status["triton_version"] = getattr(triton, "__version__", None)
        triton_version = _version_tuple(status["triton_version"])
        status["triton_compatible"] = bool(
            triton_version is not None
            and triton_version[0] == 3
            and triton_version[1] in {1, 2}
        )
    except Exception as exc:
        status["triton_error"] = f"{type(exc).__name__}: {exc}"

    # SageAttention v1's T4 path is the only path enabled by this integration.
    # On a CPU-only local test host we defer this decision until a tensor is
    # observed, so the import and patching contract remains testable.
    architectures = tuple(status.get("gpu_architectures") or ())
    status["t4_architecture_compatible"] = bool(
        not architectures or all(architecture == "sm75" for architecture in architectures)
    )
    status["runtime_compatible"] = bool(
        status.get("sage_v1_compatible")
        and status.get("triton_compatible")
        and status.get("t4_architecture_compatible")
    )

    try:
        import comfy.ldm.modules.attention as attention  # type: ignore

        status["comfy_attention_sage"] = callable(
            getattr(attention, "attention_sage", None)
        )
        status["supports_h3_packed_containers"] = bool(
            status["comfy_attention_sage"]
            and hasattr(attention, "AttentionTensorContainer")
        )
    except Exception as exc:
        status["comfy_error"] = f"{type(exc).__name__}: {exc}"
    return status


def install_h3_sage_attention() -> tuple[dict[str, Any], Any]:
    """Patch H3 transformer attention and return ``(status, restore)``.

    Only ``comfy.ldm.minimax.model`` is patched.  The H3 VAE and all unrelated
    ComfyUI models retain their existing attention implementation.  The
    returned restore callback is safe to call once and is also safe to call
    repeatedly during exception cleanup.
    """

    status = sage_attention_status()
    if not status.get("available"):
        detail = status.get("error") or "sageattention.sageattn is unavailable"
        raise H3SageAttentionError(
            "SageAttention was requested for H3, but the optional backend is not "
            f"available: {detail}. Install a T4-compatible SageAttention build "
            "and Triton before enabling the sampler toggle."
        )
    if status.get("cuda_available") and not status.get("runtime_compatible"):
        raise H3SageAttentionError(
            "SageAttention v1 is installed, but the current runtime is not "
            "T4-compatible: require sm75 plus Triton 3.1 or 3.2; "
            f"got architectures={status.get('gpu_architectures') or 'unknown'}, "
            f"Triton={status.get('triton_version') or 'unknown'}."
        )
    if not status.get("supports_h3_packed_containers"):
        raise H3SageAttentionError(
            "SageAttention was found, but this ComfyUI build does not expose "
            "attention_sage/AttentionTensorContainer for the packed H3 transformer."
        )

    try:
        import comfy.ldm.minimax.model as minimax_model  # type: ignore
        import comfy.ldm.modules.attention as attention  # type: ignore
    except Exception as exc:
        raise H3SageAttentionError(
            f"Could not import the ComfyUI H3 attention modules: {exc}"
        ) from exc

    replacement = getattr(attention, "attention_sage", None)
    if not callable(replacement):
        raise H3SageAttentionError(
            "ComfyUI's H3 attention module does not provide attention_sage."
        )
    if not hasattr(minimax_model, "optimized_attention"):
        raise H3SageAttentionError(
            "This ComfyUI H3 model does not expose an optimized_attention hook."
        )

    original = minimax_model.optimized_attention
    original_masked = getattr(minimax_model, "optimized_attention_masked", None)

    def _tensor_metadata(value: Any) -> tuple[str, str] | None:
        device = getattr(value, "device", None)
        dtype = getattr(value, "dtype", None)
        if device is None and dtype is None:
            # ComfyUI's packed H3 attention contract uses
            # AttentionTensorContainer(tensor) with a non-consuming peek().
            peek = getattr(value, "peek", None)
            if callable(peek):
                try:
                    return _tensor_metadata(peek())
                except Exception:
                    return None
        if device is None and dtype is None:
            return None
        return str(device or "unknown"), str(dtype or "unknown")

    fallback_reported = False

    def _fallback_reason(reason: str) -> None:
        nonlocal fallback_reported
        status["fallback"] = True
        status["fallback_reason"] = reason
        if fallback_reported:
            return
        fallback_reported = True
        print(
            "[Kaggle H3] SageAttention fallback: "
            f"backend=comfyui_default, reason={reason}, "
            f"gpu_architecture={status.get('gpu_architectures') or 'unknown'}, "
            f"dtype={status.get('dtype') or 'unknown'}.",
            flush=True,
        )

    def h3_attention_with_ownership(*args: Any, **kwargs: Any) -> Any:
        # H3's packed attention call supplies Q/K/V as its first three tensor
        # arguments.  Do not move or cast them here: the active dispatched
        # block owns the device and SageAttention must consume that island.
        metadata = [_tensor_metadata(value) for value in args[:3]]
        metadata = [item for item in metadata if item is not None]
        devices = {item[0] for item in metadata}
        dtypes = {item[1] for item in metadata}
        status["device"] = next(iter(devices), "unknown")
        status["dtype"] = "/".join(sorted(dtypes)) or "unknown"
        if len(devices) > 1:
            _fallback_reason(
                "H3 attention inputs crossed device ownership boundaries: "
                + ",".join(sorted(devices))
            )
            return original(*args, **kwargs)
        if metadata and next(iter(devices)).split(":", 1)[0] != "cuda":
            _fallback_reason("current H3 attention block is not on CUDA")
            return original(*args, **kwargs)
        if status.get("cuda_available") and not status.get("runtime_compatible"):
            _fallback_reason(
                "SageAttention v1 requires Triton 3.1/3.2 and T4-compatible sm75 devices"
            )
            return original(*args, **kwargs)
        try:
            return replacement(*args, **kwargs)
        except Exception as exc:
            _fallback_reason(f"SageAttention kernel failed: {type(exc).__name__}: {exc}")
            return original(*args, **kwargs)

    minimax_model.optimized_attention = h3_attention_with_ownership
    # H3 currently uses the unmasked path, but patching the matching alias
    # keeps a future H3 block from unexpectedly bypassing the selected toggle.
    if original_masked is not None:
        minimax_model.optimized_attention_masked = h3_attention_with_ownership

    restored = False

    def restore() -> None:
        nonlocal restored
        if restored:
            return
        minimax_model.optimized_attention = original
        if original_masked is not None:
            minimax_model.optimized_attention_masked = original_masked
        restored = True

    status.update(
        {
            "enabled": True,
            "backend": "comfyui.attention_sage",
            "patched_module": "comfy.ldm.minimax.model",
            "patched_vae": False,
            "fallback": False,
            "gpu_architecture": ",".join(status.get("gpu_architectures") or ()) or "unknown",
            "dtype": "unknown_until_first_attention",
        }
    )
    print(
        "[Kaggle H3] SageAttention enabled for the H3 transformer only "
        "(VAE attention remains unchanged); "
        f"backend=SageAttention v1, Triton={status.get('triton_version') or 'unknown'}, "
        f"gpu_architecture={status.get('gpu_architectures') or 'unknown'}, "
        "dtype=unknown_until_first_attention, fallback=False.",
        flush=True,
    )
    return status, restore


@contextmanager
def h3_sage_attention(enabled: bool) -> Iterator[dict[str, Any]]:
    """Temporarily enable SageAttention, falling back to normal attention."""

    if not enabled:
        yield {
            "enabled": False,
            "backend": "comfyui_default",
            "fallback": False,
            "gpu_architecture": ",".join(_cuda_runtime_status().get("gpu_architectures") or ())
            or "unknown",
            "dtype": "unchanged",
        }
        return
    try:
        status, restore = install_h3_sage_attention()
    except H3SageAttentionError as exc:
        runtime = _cuda_runtime_status()
        report = {
            "enabled": False,
            "backend": "comfyui_default",
            "requested": True,
            "fallback": True,
            "fallback_reason": str(exc),
            "gpu_architecture": ",".join(runtime.get("gpu_architectures") or ()) or "unknown",
            "dtype": "unchanged",
        }
        print(
            "[Kaggle H3] SageAttention unavailable; using normal attention. "
            f"gpu_architecture={report['gpu_architecture']}, "
            f"dtype={report['dtype']}, reason={exc}",
            flush=True,
        )
        yield report
        return
    try:
        yield status
    finally:
        restore()
        print("[Kaggle H3] SageAttention restored after H3 sampling.", flush=True)
