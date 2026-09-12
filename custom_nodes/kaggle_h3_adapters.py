"""ComfyUI nodes for a metadata-validated MiniMax H3 adapter stack.

The Kaggle bootstrap installs this file together with
``kaggle_h3_adapter_core.py`` and ``kaggle_h3_adapter_catalog.json`` into
ComfyUI/custom_nodes. Keeping the Comfy-specific shim separate from the
dependency-light core makes the resolver and ordering rules unit-testable.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
from typing import Any

try:
    from kaggle_h3.adapters import (
        AdapterCache,
        H3AdapterError,
        apply_adapter_stack,
        build_runtime_config,
        detect_conditioning_mode,
        load_catalog,
        selector_options,
    )
except ImportError:
    try:
        # This supports loading the node directly from the delivered package
        # before the bootstrap has copied its core beside the Comfy checkout.
        node_path = Path(__file__).resolve()
        # ComfyUI loads a custom-node file by path and does not guarantee that
        # the package's sibling ``src`` tree or ``custom_nodes`` directory is
        # importable. Cover both delivered layouts before using the copied
        # dependency-light core as the final fallback.
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
            selector_options,
        )
    except ImportError:  # installed standalone beside the ComfyUI custom node
        # ComfyUI loads this file with an importlib file spec. Import the
        # sibling helper by path so this does not depend on custom_nodes being
        # present on sys.path.
        import importlib.util

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
        release_all_text_encoder_phases,
        release_h3_runtime_resources,
        release_vae_phase,
        release_transformer_phase,
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
            release_all_text_encoder_phases,
            release_h3_runtime_resources,
            release_vae_phase,
            release_transformer_phase,
            validate_finite,
        )
    except ImportError:
        # The bootstrap copies this helper next to the node. The direct
        # package import above remains preferred for tests and notebook-side
        # development; this fallback keeps an installed ComfyUI checkout
        # self-contained.
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
        release_all_text_encoder_phases = phase_module.release_all_text_encoder_phases
        release_h3_runtime_resources = phase_module.release_h3_runtime_resources
        release_vae_phase = phase_module.release_vae_phase
        release_transformer_phase = phase_module.release_transformer_phase
        validate_finite = phase_module.validate_finite


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


def _set_explicit_patcher_devices(owner: Any, *, load_device: Any, offload_device: Any) -> None:
    patcher = owner if hasattr(owner, "load_device") else getattr(owner, "patcher", None)
    if patcher is None:
        raise H3PhaseError(f"{type(owner).__name__} does not expose a Comfy model patcher")
    _retarget_patcher_load_device(patcher, load_device)
    patcher.offload_device = offload_device


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
    if source_label == str(target):
        print(
            f"[Kaggle H3] Consumer routing retained {role} latent on {target}.",
            flush=True,
        )
        return tensor
    moved = tensor.to(target)
    print(
        f"[Kaggle H3] Consumer routing moved {role} latent {source_label} -> {target} "
        "directly for its next consumer.",
        flush=True,
    )
    return moved


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
    """Load H3 without allowing ComfyUI to materialize it on GPU0 first."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (_folder_options("diffusion_models"),),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"], {"default": "default"}),
                "gpu_0": ("INT", {"default": 0, "min": 0, "max": 7}),
                "gpu_1": ("INT", {"default": 1, "min": 0, "max": 7}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "loaders/MiniMax H3"

    def load_model(
        self, unet_name: str, weight_dtype: str = "default", gpu_0: int = 0, gpu_1: int = 1
    ):
        if "minimax_h3" not in str(unet_name).lower():
            raise H3PhaseError(
                "Kaggle H3 Sharded Diffusion Loader received a non-H3 checkpoint: "
                f"{unet_name!r}"
            )
        try:
            import torch  # type: ignore
            import nodes  # type: ignore
        except Exception as exc:
            raise H3PhaseError("The H3 sharded loader must run inside ComfyUI with PyTorch") from exc
        ids = (int(gpu_0), int(gpu_1))
        loaded = nodes.UNETLoader().load_unet(str(unet_name), str(weight_dtype))
        model = loaded[0]
        if phase_policy() == "off":
            print(
                "[Kaggle H3] H3 sharded loader honoring explicit one-GPU fallback; "
                "the standard ComfyUI load device remains in control.",
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
            "[Kaggle H3] H3 sharded loader selected: "
            f"checkpoint={unet_name}, loader_device=cpu, phase_gpus={list(ids)}.",
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
            video_samples = _coerce_h3_video_samples(vae, video_samples)
            latent = _h3_move_for_consumer(
                video_samples["samples"], device_id=int(device_id), role="video VAE"
            )
            validate_finite(latent, "vae_video_in", tensor_name="video_latent")
            validate_finite(latent, "vae_video", tensor_name="video_latent")
            images = vae.decode(latent)
            validate_finite(images, "vae_video_out", tensor_name="video_frames")
            validate_finite(images, "vae_video", tensor_name="video_frames")
            if len(images.shape) == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            # The encoder and CreateVideo do not need GPU-resident frames.
            return (images.to("cpu"),)
        finally:
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
        runtime_config: dict[str, Any] | None = None,
    ):
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
            validate_finite(
                audio_samples["samples"],
                "vae_audio_in",
                tensor_name="audio_latent.source",
            )
            audio_tensor = _h3_move_for_consumer(
                audio_samples["samples"], device_id=int(device_id), role="audio VAE"
            )
            validate_finite(audio_tensor, "vae_audio_in", tensor_name="audio_latent")
            validate_finite(audio_tensor, "vae_audio", tensor_name="audio_latent")
            audio = vae_decode_audio(vae, {"samples": audio_tensor})
            validate_finite(audio, "vae_audio_out", tensor_name="audio_output")
            validate_finite(audio, "vae_audio", tensor_name="audio_output")
            return (_h3_finite_audio(audio),)
        finally:
            release_vae_phase(vae, role="audio")


def _install_phase_dispatch_hook(
    sampler_helpers: Any,
    runtime_config: dict[str, Any] | None,
    phase_holder: dict[str, Any],
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
        phase_holder["state"] = begin_transformer_phase(phase_model, runtime_config)
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
        options = selector_options(_catalog())
        return {
            "required": {
                "model": ("MODEL",),
                "model_variant": (["auto", "fl2va", "ref2va"], {"default": "auto", "advanced": True}),
                "turbo_mode": ("BOOLEAN", {"default": False}),
                "turbo_steps": (["4", "8"], {"default": "4"}),
                "adapter_1": (options, {"default": "None"}),
                "strength_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "adapter_2": (options, {"default": "None"}),
                "strength_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "adapter_3": (options, {"default": "None"}),
                "strength_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                # FL2VA is shared by T2VA and FL2VA. Leave this at auto unless
                # the graph carries explicit mode metadata from its loader.
                "conditioning_mode": (["auto", "T2VA", "FL2VA", "Ref2VA"], {"default": "auto"}),
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
        conditioning_mode: str = "auto",
        model_variant: str = "auto",
    ):
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
                "sampler_name": (["euler", "res_multistep"], {"default": "euler"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
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

        if runtime_config is None:
            return (
                {
                    "steps": None,
                    "nfe": None,
                    "sampler_name": sampler_name,
                    "scheduler": "simple",
                    "shift_video": 12.0,
                    "shift_audio": 3.0,
                    "source": "base_h3_schedule",
                },
                max(1, int(steps)),
            )
        if not isinstance(runtime_config, dict) or runtime_config.get("schema_version") != "kaggle_h3.runtime.v1":
            raise RuntimeError("H3 Turbo Sampler received an invalid H3_RUNTIME_CONFIG")
        turbo = (runtime_config or {}).get("turbo") or {}
        schedule = dict(runtime_config.get("sampler") or {})
        expected_steps = schedule.get("steps")
        if turbo.get("enabled"):
            if expected_steps not in (4, 8):
                raise RuntimeError(f"H3 runtime config contains unsupported Turbo steps: {expected_steps!r}")
            if sampler_name != schedule.get("sampler_name"):
                raise RuntimeError(
                    f"H3 Turbo schedule requires sampler {schedule.get('sampler_name')!r}; got {sampler_name!r}"
                )
        elif expected_steps is not None:
            raise RuntimeError("Non-Turbo H3 runtime config must not contain a Turbo step count")
        elif sampler_name != schedule.get("sampler_name", sampler_name):
            raise RuntimeError(
                f"H3 runtime config requires sampler {schedule.get('sampler_name')!r}; got {sampler_name!r}"
            )
        return schedule, (int(steps) if expected_steps is None else int(expected_steps))

    @staticmethod
    def _shift_model(model: Any, schedule: dict[str, Any]):
        """Apply H3's official clone-based AV sigma-shift operation."""

        import comfy.model_sampling  # type: ignore

        shifted = model.clone()

        class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
            pass

        original = shifted.get_model_object("model_sampling")
        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(
            shift=float(schedule["shift_video"]),
            audio_shift=float(schedule["shift_audio"]),
        )
        if hasattr(original, "noise_scale"):
            model_sampling.set_noise_scale(original.noise_scale)
        shifted.add_object_patch("model_sampling", model_sampling)
        copied_options = dict(getattr(shifted, "model_options", {}) or {})
        transformer_options = dict(copied_options.get("transformer_options") or {})
        transformer_options["minimax_h3_sigma_shift_video"] = float(schedule["shift_video"])
        transformer_options["minimax_h3_sigma_shift_audio"] = float(schedule["shift_audio"])
        copied_options["transformer_options"] = transformer_options
        shifted.model_options = copied_options
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
    ):
        import comfy.sample  # type: ignore
        import comfy.samplers  # type: ignore
        import comfy.sampler_helpers  # type: ignore
        import comfy.model_management  # type: ignore
        import comfy.utils  # type: ignore
        import latent_preview  # type: ignore
        from comfy_extras.nodes_custom_sampler import Guider_Basic, Noise_RandomNoise  # type: ignore

        schedule, expected_steps = self._resolve_sampling_parameters(
            runtime_config, sampler_name, steps
        )
        turbo = (runtime_config or {}).get("turbo") or {}

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
            comfy.sampler_helpers, runtime_config, phase_holder
        )
        try:
            sigmas = comfy.samplers.calculate_sigmas(
                sampling_model.get_model_object("model_sampling"),
                str(schedule.get("scheduler", "simple")),
                int(expected_steps),
            ).cpu()
            sampler = comfy.samplers.sampler_object(sampler_name)
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
                release_h3_runtime_resources()
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
            f"shift={schedule.get('shift_video')}/{schedule.get('shift_audio')}",
            flush=True,
        )
        return (video_output, audio_output, denoised)


NODE_CLASS_MAPPINGS = {
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
    "KaggleH3Ref2VAConditioning": "Kaggle H3 | Ref2VA Conditioning (Seconds + Presets)",
    "KaggleH3ShardedDiffusionLoader": "Kaggle H3 | Diffusion Shard Loader (GPU0 + GPU1)",
    "KaggleH3TextEncoderLoader": "Kaggle H3 | Text Encoder (GPU0 + GPU1)",
    "KaggleH3VAELoader": "Kaggle H3 | VAE Loader",
    "KaggleH3PhaseDispatch": "Kaggle H3 | Transformer Dispatch Barrier",
    "KaggleH3AdapterStack": "Kaggle H3 | Adapter Stack",
    "KaggleH3TurboSampler": "Kaggle H3 | Turbo Sampler",
    "KaggleH3VAEDecode": "Kaggle H3 | Video VAE Decode (GPU1)",
    "KaggleH3AudioVAEDecode": "Kaggle H3 | Audio VAE Decode (GPU0)",
}


_install_standard_h3_decode_compatibility()
