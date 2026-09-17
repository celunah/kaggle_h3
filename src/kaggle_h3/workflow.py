"""Request normalization, Kaggle H3 prompting, and H3 API workflows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .bootstrap import COMMON_MODEL_FILES, MODE_MODEL_FILES
from .adapters import AUTO_SINGULARITY_ADAPTER, MAX_TURBO_STEPS, MIN_TURBO_STEPS
from .ref2va import quality_mode_to_ref2va_preset, resolve_ref2va_dimensions

PRODUCTION_SAMPLER_NAMES = ("euler", "euler_dualclock")

ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "16:9": (16, 9),
    "9:16": (9, 16),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "1:1": (1, 1),
    "21:9": (21, 9),
}

WORKFLOW_REGISTRY: dict[str, dict[str, str]] = {
    "Ref2VA": {
        "class_type": "KaggleH3Conditioning",
        "template": "workflows/kaggle_h3_ref2va.json",
        "model_role": "ref2va",
        "label": "Kaggle H3 | Ref2VA reference-to-video",
    },
}

WORKFLOW_NODE_LABELS: dict[str, str] = {
    "unet": "Kaggle H3 | Singularity Ref2VA Diffusion Auto Loader",
    "clip": "Kaggle H3 | Text Encoder (GPU0)",
    "vae_video": "Kaggle H3 | Video VAE (GPU1)",
    "vae_audio": "Kaggle H3 | Audio VAE (GPU0)",
    "character_reference": "Kaggle H3 | Character Reference",
    "scene_reference": "Kaggle H3 | Scene Reference",
    "first_frame": "Kaggle H3 | First Frame",
    "last_frame": "Kaggle H3 | Last Frame",
    "h3": "Kaggle H3 | Conditioning",
    "h3_adapter": "Kaggle H3 | Adapter Stack",
    "phase": "Kaggle H3 | Transformer Dispatch Barrier",
    "sample": "Kaggle H3 | H3 Sampler",
    "decode": "Kaggle H3 | Video Decode (GPU1)",
    "decode_audio": "Kaggle H3 | Audio Decode (GPU0)",
    "video": "Kaggle H3 | Assemble Video",
    "save": "Kaggle H3 | Save MP4",
}


@dataclass(frozen=True)
class ReferenceAsset:
    path: str | Path
    role: str
    media_type: str | None = None

    def normalized(self) -> "ReferenceAsset":
        path = Path(self.path).expanduser().resolve()
        media_type = self.media_type or _infer_media_type(path)
        if media_type not in {"image", "video", "audio"}:
            raise ValueError(f"Unsupported reference media type: {media_type}")
        return ReferenceAsset(path=path, role=self.role, media_type=media_type)


@dataclass
class H3Request:
    prompt: str
    character_references: list[str | Path] = field(default_factory=list)
    scene_references: list[str | Path] = field(default_factory=list)
    outfit_references: list[str | Path] = field(default_factory=list)
    motion_references: list[str | Path] = field(default_factory=list)
    audio_references: list[str | Path] = field(default_factory=list)
    first_frame: str | Path | None = None
    last_frame: str | Path | None = None
    duration_seconds: float = 5.0
    aspect_ratio: str = "16:9"
    width: int | None = None
    height: int | None = None
    seed: int = 42
    quality_mode: str = "medium"
    steps: int | None = None
    ref_image_size: str = "match"
    turbo_mode: bool = True
    turbo_steps: int = 4
    adapter_1: str = AUTO_SINGULARITY_ADAPTER
    adapter_1_strength: float = 1.0
    adapter_2: str = "None"
    adapter_2_strength: float = 1.0
    adapter_3: str = "None"
    adapter_3_strength: float = 1.0
    conditioning_mode: str = "Ref2VA"
    model_profile: str = "singularity"
    sage_attention: bool = False
    synchronize_mode: str = "full"
    sampler_name: str = "euler"
    mute_generated_audio: bool = False

    def assets(self) -> list[ReferenceAsset]:
        assets: list[ReferenceAsset] = []
        for role, values in (
            ("character", self.character_references),
            ("scene", self.scene_references),
            ("outfit", self.outfit_references),
            ("motion", self.motion_references),
            ("audio", self.audio_references),
        ):
            for value in values:
                assets.append(ReferenceAsset(value, role).normalized())
        return assets

    def endpoint_assets(self) -> list[ReferenceAsset]:
        result: list[ReferenceAsset] = []
        if self.first_frame is not None:
            result.append(ReferenceAsset(self.first_frame, "first_frame", "image").normalized())
        if self.last_frame is not None:
            result.append(ReferenceAsset(self.last_frame, "last_frame", "image").normalized())
        return result

    def all_paths(self) -> list[Path]:
        return [Path(asset.path) for asset in self.assets() + self.endpoint_assets()]


def _infer_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".mp4", ".mov", ".webm", ".mkv", ".avi"}:
        return "video"
    if suffix in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
        return "audio"
    return "image"


def select_mode(request: H3Request) -> str:
    """Choose a mode without silently discarding incompatible inputs."""

    has_refs = bool(request.assets())
    if has_refs and all(asset.media_type == "audio" for asset in request.assets()):
        raise ValueError("Ref2VA needs at least one image or video reference; audio cannot be the only condition.")
    counts = {"image": 0, "video": 0, "audio": 0}
    for asset in request.assets():
        counts[asset.media_type or "image"] += 1
    if counts["image"] > 9 or counts["video"] > 3 or counts["audio"] > 3:
        raise ValueError("H3 supports at most 9 image, 3 video, and 3 standalone audio references.")
    has_endpoints = request.first_frame is not None or request.last_frame is not None
    if has_refs and has_endpoints:
        raise ValueError(
            "Use either Ref2VA references or FL2VA first/last endpoints in one request; "
            "the separate workflows avoid silently dropping conditions."
        )
    if has_endpoints:
        return "FL2VA"
    if has_refs:
        return "Ref2VA"
    return "T2VA"


def select_production_mode(request: H3Request) -> str:
    """Validate the supported production contract and return Ref2VA."""

    mode = select_mode(request)
    if mode != "Ref2VA":
        raise ValueError(
            "The production Kaggle H3 workflow supports Ref2VA references only; "
            "FL2VA/T2VA and first/last-frame endpoints are not available."
        )
    if not request.assets():
        raise ValueError("The production Ref2VA workflow requires at least one image or video reference.")
    if not request.turbo_mode:
        raise ValueError("The production Kaggle H3 workflow requires Turbo mode.")
    if request.conditioning_mode != "Ref2VA":
        raise ValueError("The production Kaggle H3 workflow requires Ref2VA conditioning.")
    if request.model_profile != "singularity":
        raise ValueError("The production Kaggle H3 workflow requires the Singularity model profile.")
    if request.sampler_name not in PRODUCTION_SAMPLER_NAMES:
        raise ValueError(
            "Production H3 supports sampler 'euler' or 'euler_dualclock'; "
            f"got {request.sampler_name!r}."
        )
    if request.width is not None or request.height is not None:
        raise ValueError("Production H3 uses the 240p, 360p, and 480p presets; explicit dimensions are disabled.")
    steps = int(request.steps if request.steps is not None else request.turbo_steps)
    if not MIN_TURBO_STEPS <= steps <= MAX_TURBO_STEPS:
        raise ValueError(
            f"Production H3 steps must be between {MIN_TURBO_STEPS} and {MAX_TURBO_STEPS}; got {steps}."
        )
    if request.quality_mode.lower() not in {"quick", "medium", "full"}:
        raise ValueError("quality_mode must be quick, medium, or full")
    if request.aspect_ratio not in {"16:9", "4:3"}:
        raise ValueError("Production Ref2VA supports only 16:9 and 4:3 aspect ratios.")
    return mode


def _ratio(request: H3Request) -> tuple[int, int]:
    value = request.aspect_ratio.strip().lower()
    if value == "auto":
        return (16, 9)
    if value not in ASPECT_RATIOS:
        raise ValueError(f"Unsupported aspect ratio {request.aspect_ratio!r}")
    return ASPECT_RATIOS[value]


def dimensions(request: H3Request) -> tuple[int, int]:
    if request.width and request.height:
        width, height = request.width, request.height
    else:
        preset = quality_mode_to_ref2va_preset(request.quality_mode)
        width, height = resolve_ref2va_dimensions(preset, request.aspect_ratio)
    if width < 256 or height < 256:
        raise ValueError("H3 canvas dimensions must each be at least 256 pixels")
    return (_round_multiple(width, 32), _round_multiple(height, 32))


def _round_multiple(value: int, multiple: int) -> int:
    return max(multiple, int(round(value / multiple) * multiple))


def frame_length(request: H3Request) -> dict[str, Any]:
    seconds = max(5.0, min(float(request.duration_seconds), 15.0))
    requested_frames = max(5, int(round(seconds * 24)))
    # H3's temporal stride requires 17*k + 5 frames.
    k = max(0, int(round((requested_frames - 5) / 17)))
    candidates = [17 * max(0, k - 1) + 5, 17 * k + 5, 17 * (k + 1) + 5]
    length = min((candidate for candidate in candidates if candidate >= requested_frames), default=17 * k + 5)
    return {
        "requested_seconds": request.duration_seconds,
        "actual_frames": length,
        "actual_seconds": round(length / 24.0, 4),
        "fps": 24,
        "alignment": "17*k+5",
    }


def quality_steps(request: H3Request) -> int:
    steps = int(request.steps if request.steps is not None else request.turbo_steps)
    if not MIN_TURBO_STEPS <= steps <= MAX_TURBO_STEPS:
        raise ValueError(
            f"H3 steps must be between {MIN_TURBO_STEPS} and {MAX_TURBO_STEPS}; got {steps}."
        )
    return steps


def build_prompt(request: H3Request, mode: str) -> str:
    """Build a compact prompt that binds each reference role to H3 labels."""

    assets = request.assets()
    picture_index = 0
    video_index = 0
    audio_index = 0
    role_lines: list[str] = []
    for asset in assets:
        if asset.media_type == "image":
            label = f"<Picture {picture_index + 1}>"
            picture_index += 1
        elif asset.media_type == "video":
            label = f"<Video {video_index + 1}>"
            video_index += 1
        else:
            label = f"<Audio {audio_index + 1}>"
            audio_index += 1
        role_lines.append(f"{asset.role} reference {label}: preserve its identity and role.")

    base = request.prompt.strip()
    if not base:
        raise ValueError("prompt must not be empty")
    sections = [
        f"Mode: {mode}.",
        "Subject identity: the supplied subject reference outranks generic prompt interpretation.",
        f"Action and camera: {base}",
        "Identity and continuity locks: keep the subject's face, hair, body proportions, selected outfit, lighting logic, and spatial relationships stable across the full clip.",
    ]
    if role_lines:
        sections.append(
            "Reference bindings: "
            + " ".join(role_lines)
            + " Outfit references define garment structure and outrank scene references; scene references define environment only."
        )
    if any(
        token in base.lower()
        for token in ("soul spirit", "personified soul", "aura soul")
    ):
        sections.append(
            "Soul-form transition: distinguish Personified Soul, Soul Spirit, and Aura Soul while preserving one identity; show one continuous overlapping transformation with intermediate forms and a material-looking face visible during Soul Spirit; do not cut into disconnected shots unless explicitly requested."
        )
    sections.append(
        "Audio: produce coherent synchronized ambience and effects; preserve reference audio only when an audio reference is supplied."
    )
    return "\n".join(sections)


def _model_name(mode: str) -> str:
    normalized = mode.strip().upper()
    if normalized not in {"REF2VA", "FL2VA", "T2VA"}:
        raise ValueError(f"Unsupported H3 mode: {mode}")
    key = "Ref2VA" if normalized == "REF2VA" else "FL2VA" if normalized == "FL2VA" else "T2VA"
    return MODE_MODEL_FILES[key][1]


def _path_key(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _input_path(input_files: dict[Path, str] | None, path: str | Path) -> str:
    normalized = _path_key(path)
    if input_files:
        for source, target in input_files.items():
            if _path_key(source) == normalized:
                return target
    return normalized.name


def _load_asset_node(
    nodes: dict[str, Any], asset: ReferenceAsset, input_files: dict[Path, str] | None, index: int
) -> tuple[str, int]:
    node_id = f"ref_{index:02d}"
    path = _input_path(input_files, asset.path)
    role_label = asset.role.replace("_", " ").title()
    reference_title = f"Kaggle H3 | {role_label} Reference"
    if asset.media_type == "image":
        nodes[node_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": path},
            "_meta": {"title": reference_title},
        }
        return node_id, 0
    if asset.media_type == "audio":
        nodes[node_id] = {
            "class_type": "LoadAudio",
            "inputs": {"audio": path},
            "_meta": {"title": reference_title},
        }
        return node_id, 0
    nodes[node_id] = {
        "class_type": "LoadVideo",
        "inputs": {"file": path},
        "_meta": {"title": reference_title},
    }
    components_id = f"{node_id}_components"
    nodes[components_id] = {
        "class_type": "GetVideoComponents",
        "inputs": {"video": [node_id, 0]},
        "_meta": {"title": f"Kaggle H3 | {role_label} Video Components"},
    }
    return components_id, 0


def build_workflow(
    request: H3Request,
    *,
    mode: str | None = None,
    input_files: dict[Path, str] | None = None,
    run_id: str = "kaggle_h3",
    loader: str = "native",
    device_ids: tuple[int, int] = (0, 1),
) -> dict[str, Any]:
    """Build ComfyUI's flat API-format graph for one H3 mode."""

    mode = mode or select_production_mode(request)
    canonical_mode = str(mode).strip()
    if canonical_mode != "Ref2VA":
        raise ValueError(
            "Production workflow construction is Ref2VA-only; FL2VA/T2VA are not supported."
        )
    select_production_mode(request)
    dimensions(request)
    nodes: dict[str, Any] = {}
    use_explicit_h3_loaders = loader == "native"
    unet_inputs: dict[str, Any] = {
        "weight_dtype": "default",
    }
    if loader == "comfyui_sp":
        unet_inputs["unet_name"] = _model_name(canonical_mode)
        unet_inputs.update({"world_size": 2, "devices": "auto"})
        unet_class = "MiniMaxH3SPUNETLoader"
    elif use_explicit_h3_loaders:
        # The custom loader owns the mutually exclusive diffusion checkpoint:
        # it downloads the selected variant only when this graph executes.
        unet_inputs["model_variant"] = "Ref2VA"
        unet_inputs["precision"] = "int8_convrot"
        unet_inputs["model_profile"] = "singularity"
        unet_inputs.update({"gpu_0": int(device_ids[0]), "gpu_1": int(device_ids[1])})
        unet_class = "KaggleH3ShardedDiffusionLoader"
    else:
        unet_inputs["unet_name"] = _model_name(canonical_mode)
        unet_class = "UNETLoader"
    nodes["unet"] = {"class_type": unet_class, "inputs": unet_inputs}
    nodes["clip"] = {
        "class_type": "KaggleH3TextEncoderLoader" if use_explicit_h3_loaders else "CLIPLoader",
        "inputs": {
            "clip_name": COMMON_MODEL_FILES["clip"][1],
            "type": "minimax",
        },
    }
    if use_explicit_h3_loaders:
        nodes["clip"]["inputs"]["device_id"] = int(device_ids[0]) if device_ids else 0
    else:
        # Use ComfyUI's normal accelerator path during conditioning. The
        # sampler releases the text encoder before transformer dispatch.
        nodes["clip"]["inputs"]["device"] = "default"
    nodes["vae_video"] = {
        "class_type": "KaggleH3VAELoader" if use_explicit_h3_loaders else "VAELoader",
        "inputs": {"vae_name": COMMON_MODEL_FILES["video_vae"][1]},
    }
    nodes["vae_audio"] = {
        "class_type": "KaggleH3VAELoader" if use_explicit_h3_loaders else "VAELoader",
        "inputs": {"vae_name": COMMON_MODEL_FILES["audio_vae"][1]},
    }
    if use_explicit_h3_loaders:
        nodes["vae_video"]["inputs"].update({"role": "video", "device_id": int(device_ids[1])})
        nodes["vae_audio"]["inputs"].update({"role": "audio", "device_id": int(device_ids[0])})

    h3_inputs: dict[str, Any] = {
        "clip": ["clip", 0],
        "vae": ["vae_video", 0],
        "prompt": build_prompt(request, canonical_mode),
        "mode": "Ref2VA",
        "seconds": float(request.duration_seconds),
        "size_preset": quality_mode_to_ref2va_preset(request.quality_mode),
        "aspect_ratio": (
            request.aspect_ratio
            if request.aspect_ratio in {"16:9", "4:3"}
            else "16:9"
        ),
        "ref_image_size": request.ref_image_size,
    }
    if request.aspect_ratio not in {"16:9", "4:3"}:
        raise ValueError(
            "The production Ref2VA node supports only 16:9 and 4:3; "
            f"got aspect_ratio={request.aspect_ratio!r}."
        )
    h3_inputs.update(
        {
            "audio_vae": ["vae_audio", 0],
            "size_preset": quality_mode_to_ref2va_preset(request.quality_mode),
            "aspect_ratio": request.aspect_ratio,
            "ref_image_size": request.ref_image_size,
        }
    )
    assets = request.assets()
    if len(assets) > 15:
        raise ValueError(
            "Kaggle H3 Conditioning supports at most 15 inline reference rows "
            "(the native H3 limits are 9 images, 3 videos, and 3 audio refs)."
        )
    for reference_index, asset in enumerate(assets):
        if asset.media_type not in {"image", "video", "audio"}:
            raise ValueError(f"Unsupported inline H3 reference type: {asset.media_type}")
        h3_inputs[f"reference_{reference_index}"] = asset.media_type
        h3_inputs[f"reference_{reference_index}.file"] = _input_path(
            input_files, asset.path
        )
    h3_class = "KaggleH3Conditioning"
    nodes["h3"] = {"class_type": h3_class, "inputs": h3_inputs}
    uses_adapter_stack = True
    if uses_adapter_stack:
        nodes["h3_adapter"] = {
            "class_type": "KaggleH3AdapterStack",
            "inputs": {
                "model": ["unet", 0],
                "model_variant": "ref2va",
                "turbo_mode": True,
                "turbo_steps": int(quality_steps(request)),
                "adapter_1": request.adapter_1,
                "strength_1": float(request.adapter_1_strength),
                "adapter_2": request.adapter_2,
                "strength_2": float(request.adapter_2_strength),
                "adapter_3": request.adapter_3,
                "strength_3": float(request.adapter_3_strength),
                "conditioning_mode": "Ref2VA",
            },
        }
        sample_model = ["h3_adapter", 0]
        sample_runtime = ["h3_adapter", 1]
    else:
        sample_model = ["unet", 0]
        sample_runtime = None

    if use_explicit_h3_loaders:
        phase_inputs: dict[str, Any] = {
            "model": sample_model,
            "conditioning": ["h3", 0],
            "latent_image": ["h3", 1],
        }
        if sample_runtime is not None:
            phase_inputs["runtime_config"] = sample_runtime
        nodes["phase"] = {
            "class_type": "KaggleH3PhaseDispatch",
            "inputs": phase_inputs,
        }
        sample_model = ["phase", 0]

    nodes["sample"] = {
        "class_type": "KaggleH3TurboSampler",
        "inputs": {
            "model": sample_model,
            "runtime_config": ["phase", 3] if use_explicit_h3_loaders else sample_runtime,
            "conditioning": ["phase", 1] if use_explicit_h3_loaders else ["h3", 0],
            "latent_image": ["phase", 2] if use_explicit_h3_loaders else ["h3", 1],
            "noise_seed": int(request.seed),
            "sampler_name": request.sampler_name,
            "steps": quality_steps(request),
            "synchronize_mode": request.synchronize_mode,
            "use_sage_attention": bool(request.sage_attention),
        },
    }
    # The stock H3 video VAE is an FP16 checkpoint.  Use the local decoder
    # shim for native ComfyUI graphs so CPU-VAE offload can reconcile the
    # float32 sampler latent with the actual loaded VAE dtype.  The optional
    # sequence-parallel node owns its own decode implementation.
    decode_class = "MiniMaxH3SPVAEDecode" if loader == "comfyui_sp" else "KaggleH3VAEDecode"
    nodes["decode"] = {
        "class_type": decode_class,
        "inputs": {
            "samples": ["sample", 0],
            "vae": ["vae_video", 0],
        },
    }
    if loader != "comfyui_sp":
        nodes["decode"]["inputs"]["device_id"] = (
            int(device_ids[1]) if len(device_ids) > 1
            else int(device_ids[0]) if device_ids
            else 0
        )
    if not request.mute_generated_audio:
        nodes["decode_audio"] = {
            "class_type": "KaggleH3AudioVAEDecode" if loader != "comfyui_sp" else "VAEDecodeAudio",
            # The dedicated sampler output is the audio stream on GPU0. Keeping
            # this separate from the video latent enables a direct GPU0 -> GPU1
            # video handoff without retaining a packed AV NestedTensor.
            "inputs": {"samples": ["sample", 1], "vae": ["vae_audio", 0]},
        }
        if loader != "comfyui_sp":
            nodes["decode_audio"]["inputs"]["device_id"] = int(device_ids[0]) if device_ids else 0
    if uses_adapter_stack and loader != "comfyui_sp":
        nodes["decode"]["inputs"]["runtime_config"] = ["h3_adapter", 1]
        if "decode_audio" in nodes:
            nodes["decode_audio"]["inputs"]["runtime_config"] = ["h3_adapter", 1]
    video_inputs: dict[str, Any] = {
        "images": ["decode", 0],
        "fps": 24.0,
        "bit_depth": 8,
        # ComfyUI 0.34.0 exposes sRGB here; "auto" is not a valid choice
        # in the CreateVideo node shipped by the Kaggle image.
        "color_space": "sRGB",
    }
    if not request.mute_generated_audio:
        video_inputs["audio"] = ["decode_audio", 0]
    nodes["video"] = {
        "class_type": "CreateVideo",
        "inputs": video_inputs,
    }
    nodes["save"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["video", 0],
            "filename_prefix": f"kaggle_h3/{run_id}",
            "format": "mp4",
            "codec": "auto",
        },
    }
    for node_id, node in nodes.items():
        label = node.get("_meta", {}).get("title") or WORKFLOW_NODE_LABELS.get(
            node_id, f"Kaggle H3 | {node_id}"
        )
        if node_id == "h3":
            label = "Kaggle H3 | Global Conditioning (Singularity Ref2VA)"
        elif node_id == "unet" and use_explicit_h3_loaders:
            label = (
                "Kaggle H3 | Singularity Ref2VA INT8 Auto Loader "
                "(replaces inactive checkpoint)"
            )
        elif node_id == "sample":
            sampler_label = (
                "Euler Dual-clock"
                if request.sampler_name == "euler_dualclock"
                else "Euler"
            )
            label = (
                "Kaggle H3 | Singularity Turbo Sampler "
                f"({sampler_label}; {quality_steps(request)} steps)"
            )
        elif node_id == "video" and request.mute_generated_audio:
            label = "Kaggle H3 | Assemble Video (generated audio muted)"
        node["_meta"] = {"title": label}
    return nodes


def validate_workflow_shape(workflow: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict) or not isinstance(node.get("class_type"), str):
            errors.append(f"{node_id}: missing class_type")
        if not isinstance(node.get("inputs"), dict):
            errors.append(f"{node_id}: missing inputs object")
    serialized = json.dumps(workflow)
    if "--gpu-only" in serialized:
        errors.append("workflow contains forbidden --gpu-only execution")
    return {"valid": not errors, "errors": errors, "node_count": len(workflow)}


def validate_production_workflow_shape(workflow: dict[str, Any]) -> dict[str, Any]:
    """Validate the deliberately narrow, user-facing production graph."""

    result = validate_workflow_shape(workflow)
    errors = list(result["errors"])
    serialized = json.dumps(workflow, sort_keys=True).lower()
    forbidden = {
        "fl2va": "FL2VA",
        "t2va": "T2VA",
        "fp8": "FP8",
        "res_multistep": "non-Turbo res_multistep",
    }
    for token, label in forbidden.items():
        if token in serialized:
            errors.append(f"production workflow contains forbidden {label} surface")
    return {**result, "valid": not errors, "errors": errors}


def workflow_sha256(workflow: dict[str, Any]) -> str:
    payload = json.dumps(workflow, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def request_as_dict(request: H3Request) -> dict[str, Any]:
    result = asdict(request)
    for key, value in list(result.items()):
        if isinstance(value, list):
            result[key] = [str(item) for item in value]
        elif isinstance(value, Path):
            result[key] = str(value)
    return result
