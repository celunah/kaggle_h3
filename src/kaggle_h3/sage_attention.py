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
from typing import Any, Iterator


class H3SageAttentionError(RuntimeError):
    """Raised when the optional SageAttention backend cannot be enabled."""


def sage_attention_status() -> dict[str, Any]:
    """Report the installed SageAttention and ComfyUI integration surface."""

    status: dict[str, Any] = {
        "available": False,
        "package": None,
        "version": None,
        "comfy_attention_sage": False,
        "supports_h3_packed_containers": False,
    }
    try:
        import sageattention  # type: ignore

        status["package"] = "sageattention"
        status["version"] = getattr(sageattention, "__version__", None)
        status["available"] = callable(getattr(sageattention, "sageattn", None))
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        return status

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
    minimax_model.optimized_attention = replacement
    # H3 currently uses the unmasked path, but patching the matching alias
    # keeps a future H3 block from unexpectedly bypassing the selected toggle.
    if original_masked is not None:
        minimax_model.optimized_attention_masked = replacement

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
        }
    )
    print(
        "[Kaggle H3] SageAttention enabled for the H3 transformer only "
        "(VAE attention remains unchanged).",
        flush=True,
    )
    return status, restore


@contextmanager
def h3_sage_attention(enabled: bool) -> Iterator[dict[str, Any]]:
    """Temporarily enable SageAttention when ``enabled`` is true."""

    if not enabled:
        yield {"enabled": False, "backend": "comfyui_default"}
        return
    status, restore = install_h3_sage_attention()
    try:
        yield status
    finally:
        restore()
        print("[Kaggle H3] SageAttention restored after H3 sampling.", flush=True)
