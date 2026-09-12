"""MiniMax H3 adapter metadata, resolution, compatibility, and patch planning.

The module is deliberately free of torch and ComfyUI imports at module load time.
It is used by the notebook tests and by the ComfyUI custom node, where the latter
imports the Comfy-only functions lazily after ComfyUI has finished booting.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping


NONE_ADAPTER = "None"
SUPPORTED_TURBO_STEPS = (4, 8)
NORMAL_SHIFT_VIDEO = 12.0
NORMAL_SHIFT_AUDIO = 3.0


class H3AdapterError(RuntimeError):
    """Base error for adapter resolution, compatibility, and patch failures."""


class H3AdapterCompatibilityError(H3AdapterError):
    """Raised before any model patch is applied when metadata does not match."""


class H3AdapterDownloadError(H3AdapterError):
    """Raised when a selected adapter cannot be resolved or downloaded."""


def _normalise_variant(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "")
    if text in {"fl2v", "fl2va", "t2v", "t2va"}:
        return "fl2va"
    if text in {"ref2v", "ref2va", "r2v", "r2va"}:
        return "ref2va"
    return text


def _normalise_mode(value: Any) -> str:
    text = str(value or "").strip().upper().replace("-", "")
    if text in {"FL2V", "FL2VA", "I2V", "I2VA"}:
        return "FL2VA"
    if text in {"T2V", "T2VA"}:
        return "T2VA"
    if text in {"REF2V", "REF2VA", "R2V", "R2VA"}:
        return "Ref2VA"
    return text


def _safe_hf_filename(filename: str) -> str:
    normalized = str(filename).replace("\\", "/").strip("/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe Hugging Face adapter filename: {filename!r}")
    if path.suffix.lower() != ".safetensors":
        raise ValueError(
            f"H3 adapters must use .safetensors; got {filename!r}"
        )
    return normalized


def _tuple_strings(values: Any, normalizer: Callable[[Any], str] | None = None) -> tuple[str, ...]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, Iterable):
        return ()
    result: list[str] = []
    for value in values:
        normalized = normalizer(value) if normalizer else str(value).strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return tuple(result)


@dataclass(frozen=True)
class H3AdapterMetadata:
    """Validated metadata for one incremental H3 adapter file."""

    adapter_id: str
    name: str
    repository: str
    filename: str
    revision: str
    model_variants: tuple[str, ...]
    conditioning_modes: tuple[str, ...]
    recommended_strength: float
    supported_steps: tuple[int, ...]
    fused: bool = False
    license: str = "Unknown; inspect the upstream repository before redistribution."
    attribution: str = ""
    kind: str = "generic"
    schedules: Mapping[int, Mapping[str, Any]] = field(default_factory=dict)
    sha256: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "H3AdapterMetadata":
        adapter_id = str(value.get("adapter_id") or value.get("id") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", adapter_id):
            raise ValueError(f"Invalid H3 adapter_id: {adapter_id!r}")
        name = str(value.get("name") or "").strip()
        repository = str(
            value.get("repository")
            or value.get("huggingface_repository")
            or value.get("hf_repo")
            or ""
        ).strip()
        filename = _safe_hf_filename(
            str(value.get("filename") or value.get("huggingface_filename") or "")
        )
        revision = str(value.get("revision") or "").strip()
        if not name or not repository or not revision:
            raise ValueError(f"Adapter {adapter_id!r} needs name, repository, and revision")
        variants = _tuple_strings(
            value.get("model_variants") or value.get("supported_h3_model_variants"),
            _normalise_variant,
        )
        modes = _tuple_strings(
            value.get("conditioning_modes") or value.get("supported_conditioning_modes"),
            _normalise_mode,
        )
        if not variants or not modes:
            raise ValueError(f"Adapter {adapter_id!r} needs model_variants and conditioning_modes")
        strength = float(value.get("recommended_strength", 1.0))
        if not 0.0 <= strength <= 2.0:
            raise ValueError(f"Adapter {adapter_id!r} recommended_strength must be in [0, 2]")
        steps = tuple(sorted({int(item) for item in (value.get("supported_steps") or ())}))
        if not steps or any(step < 1 or step > 10000 for step in steps):
            raise ValueError(
                f"Adapter {adapter_id!r} supported_steps must contain positive step counts"
            )
        schedules: dict[int, Mapping[str, Any]] = {}
        raw_schedules = value.get("schedules") or value.get("sampler_schedules") or {}
        if not isinstance(raw_schedules, Mapping):
            raise ValueError(f"Adapter {adapter_id!r} schedules must be an object")
        for raw_step, schedule in raw_schedules.items():
            step = int(raw_step)
            if step not in steps or not isinstance(schedule, Mapping):
                raise ValueError(f"Adapter {adapter_id!r} has an invalid schedule for {raw_step!r}")
            schedules[step] = dict(schedule)
        sha256 = value.get("sha256")
        if sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", str(sha256)):
            raise ValueError(f"Adapter {adapter_id!r} sha256 must be 64 hexadecimal characters")
        return cls(
            adapter_id=adapter_id,
            name=name,
            repository=repository,
            filename=filename,
            revision=revision,
            model_variants=variants,
            conditioning_modes=modes,
            recommended_strength=strength,
            supported_steps=steps,
            fused=bool(value.get("fused", False)),
            license=str(value.get("license") or value.get("license_and_attribution") or "Unknown"),
            attribution=str(value.get("attribution") or ""),
            kind=str(value.get("kind") or "generic").strip().lower(),
            schedules=schedules,
            sha256=str(sha256).lower() if sha256 else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "name": self.name,
            "repository": self.repository,
            "filename": self.filename,
            "revision": self.revision,
            "model_variants": list(self.model_variants),
            "conditioning_modes": list(self.conditioning_modes),
            "recommended_strength": self.recommended_strength,
            "supported_steps": list(self.supported_steps),
            "fused": self.fused,
            "license": self.license,
            "attribution": self.attribution,
            "kind": self.kind,
            "schedules": {str(key): dict(value) for key, value in self.schedules.items()},
            **({"sha256": self.sha256} if self.sha256 else {}),
        }

    def schedule_for(self, steps: int) -> dict[str, Any]:
        if steps not in self.supported_steps:
            raise H3AdapterCompatibilityError(
                f"Adapter {self.name!r} does not support {steps}-step inference; "
                f"supported steps: {list(self.supported_steps)}"
            )
        schedule = dict(self.schedules.get(steps, {}))
        schedule.setdefault("steps", steps)
        schedule.setdefault("nfe", steps)
        schedule.setdefault("sampler_name", "euler")
        schedule.setdefault("scheduler", "simple")
        schedule.setdefault("shift_video", NORMAL_SHIFT_VIDEO)
        schedule.setdefault("shift_audio", NORMAL_SHIFT_AUDIO)
        return schedule


@dataclass(frozen=True)
class H3ModelContext:
    is_h3: bool
    model_variant: str
    conditioning_mode: str
    conditioning_mode_candidates: tuple[str, ...]
    fused_turbo: bool
    fused_reason: str | None
    existing_adapter_ids: tuple[str, ...]
    evidence: tuple[str, ...]


def _walk_metadata(value: Any, *, depth: int = 0) -> Iterable[tuple[str, Any]]:
    if depth > 3:
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key), child
            yield from _walk_metadata(child, depth=depth + 1)
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            yield "", child
            yield from _walk_metadata(child, depth=depth + 1)


def _model_evidence(model: Any) -> tuple[list[str], list[tuple[str, Any]]]:
    evidence: list[str] = []
    pairs: list[tuple[str, Any]] = []
    objects = [model]
    for attribute in ("model", "model_config", "metadata", "model_options", "attachments"):
        try:
            child = getattr(model, attribute, None)
        except Exception:
            child = None
        if child is not None:
            objects.append(child)
    for obj in objects:
        if isinstance(obj, Mapping):
            pairs.extend(_walk_metadata(obj))
            evidence.extend(str(key) for key in obj.keys())
        for attribute in (
            "filename",
            "model_filename",
            "name",
            "model_name",
            "model_variant",
            "h3_model_variant",
            "conditioning_mode",
            "h3_conditioning_mode",
            "is_h3",
            "fused_turbo",
            "turbo_fused",
        ):
            try:
                value = getattr(obj, attribute, None)
            except Exception:
                value = None
            if value is not None:
                pairs.append((attribute, value))
        if isinstance(obj, str):
            evidence.append(obj)
        elif obj is not model and not isinstance(obj, (Mapping, list, tuple, set)):
            text = str(obj)
            if text and len(text) < 500:
                evidence.append(text)
    for key, value in pairs:
        if isinstance(value, (str, int, float, bool)):
            evidence.extend([str(key), str(value)])
    return evidence, pairs


def _explicit_bool(pairs: Iterable[tuple[str, Any]], keys: set[str]) -> bool | None:
    for key, value in pairs:
        if key.lower() in keys and isinstance(value, bool):
            return value
    return None


def _existing_adapters(pairs: Iterable[tuple[str, Any]]) -> tuple[str, ...]:
    found: list[str] = []
    for key, value in pairs:
        if "applied_adapter" not in key.lower() and key.lower() not in {"kaggle_h3_applied_adapters", "adapters"}:
            continue
        values = value if isinstance(value, (list, tuple, set)) else [value]
        for item in values:
            if isinstance(item, Mapping):
                item = item.get("adapter_id") or item.get("id")
            if item:
                text = str(item)
                if text not in found:
                    found.append(text)
    return tuple(found)


def detect_h3_model_context(
    model: Any,
    conditioning_mode: str | None = None,
    model_variant: str | None = None,
) -> H3ModelContext:
    """Inspect a Comfy ``MODEL`` without moving or copying its weights."""

    evidence, pairs = _model_evidence(model)
    lower = " ".join(evidence).lower()
    explicit_h3 = _explicit_bool(pairs, {"is_h3", "h3_model"})
    override_variant = _normalise_variant(model_variant) if model_variant and model_variant != "auto" else ""
    is_h3 = bool(explicit_h3) or bool(override_variant) or any(token in lower for token in ("minimax", "mini-max", "h3"))
    explicit_variant = next(
        (_normalise_variant(value) for key, value in pairs if key.lower() in {"model_variant", "h3_model_variant"} and value),
        "",
    )
    if override_variant in {"fl2va", "ref2va"}:
        variant = override_variant
    elif explicit_variant in {"fl2va", "ref2va"}:
        variant = explicit_variant
    elif "ref2va" in lower or "ref2v" in lower:
        variant = "ref2va"
    elif "fl2va" in lower or "fl2v" in lower or "t2va" in lower:
        variant = "fl2va"
    else:
        variant = "unknown"

    explicit_mode = next(
        (_normalise_mode(value) for key, value in pairs if key.lower() in {"conditioning_mode", "h3_conditioning_mode"} and value),
        "",
    )
    if explicit_mode in {"T2VA", "FL2VA", "Ref2VA"}:
        mode = explicit_mode
        candidates = (explicit_mode,)
    elif variant == "ref2va":
        mode = "Ref2VA"
        candidates = ("Ref2VA",)
    elif variant == "fl2va":
        # T2VA and FL2VA intentionally share the FL2VA checkpoint. The
        # sampler resolves the exact mode later from H3 conditioning metadata.
        mode = "auto"
        candidates = ("T2VA", "FL2VA")
    else:
        mode = "unknown"
        candidates = ()

    explicit_fused = _explicit_bool(
        pairs,
        {"fused_turbo", "turbo_fused", "h3_turbo_fused", "turbo_already_fused"},
    )
    fused_reason: str | None = None
    if explicit_fused is True:
        fused_reason = "checkpoint metadata explicitly marks Turbo as fused"
    elif explicit_fused is False:
        fused_reason = "checkpoint metadata explicitly marks Turbo as not fused"
    elif "turbo" in lower and any(token in lower for token in ("fused", "merged", "checkpoint")):
        explicit_fused = True
        fused_reason = "checkpoint identity contains an explicit Turbo/fused marker"
    fused = bool(explicit_fused)
    if override_variant:
        evidence.append(f"explicit node model_variant={override_variant}")
    return H3ModelContext(
        is_h3=is_h3,
        model_variant=variant,
        conditioning_mode=mode,
        conditioning_mode_candidates=candidates,
        fused_turbo=fused,
        fused_reason=fused_reason,
        existing_adapter_ids=_existing_adapters(pairs),
        evidence=tuple(dict.fromkeys(evidence)),
    )


def detect_conditioning_mode(conditioning: Any) -> str:
    """Resolve H3 mode from native conditioning payloads at sampler time."""

    text = json.dumps(conditioning, default=str).lower()
    if "minimax_refs" in text or "ref2va" in text or "picture" in text:
        return "Ref2VA"
    if "minimax_keyframes" in text or "fl2va" in text or "first_frame" in text:
        return "FL2VA"
    return "T2VA"


class AdapterCache:
    """Content-addressed-enough local resolver for safe ComfyUI LoRAs."""

    def __init__(self, root: str | Path | None = None):
        if root is None:
            root = os.environ.get("KAGGLE_H3_LORA_CACHE")
        if root is None:
            try:
                import folder_paths  # type: ignore

                paths = folder_paths.get_folder_paths("loras")
                root = paths[0] if paths else Path.cwd() / "models" / "loras"
            except Exception:
                root = Path.cwd() / "models" / "loras"
        self.root = Path(root).expanduser().resolve()

    def _candidate(self, metadata: H3AdapterMetadata) -> Path:
        # Adapter selectors are registry IDs; the actual file remains a normal
        # ComfyUI loras file so it can also be inspected by built-in loaders.
        return self.root / Path(metadata.filename)

    @staticmethod
    def _sidecar(path: Path) -> Path:
        return path.with_name(path.name + ".h3.json")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _write_sidecar(self, path: Path, metadata: H3AdapterMetadata, *, source: str) -> None:
        self._sidecar(path).write_text(
            json.dumps(
                {
                    "adapter_id": metadata.adapter_id,
                    "repository": metadata.repository,
                    "filename": metadata.filename,
                    "revision": metadata.revision,
                    "sha256": self._sha256(path),
                    "source": source,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def _verify(self, path: Path, metadata: H3AdapterMetadata) -> Path:
        if not path.is_file():
            raise H3AdapterDownloadError(f"Resolved adapter path is not a file: {path}")
        if path.suffix.lower() != ".safetensors":
            raise H3AdapterDownloadError(f"Refusing non-safetensors adapter: {path.name}")
        if metadata.sha256:
            actual = self._sha256(path)
            if actual.lower() != metadata.sha256.lower():
                raise H3AdapterDownloadError(
                    f"SHA-256 mismatch for {metadata.name}: expected {metadata.sha256}, got {actual}"
                )
        return path

    def resolve(self, metadata: H3AdapterMetadata) -> Path:
        """Return a cached file or download exactly the pinned HF revision."""

        candidate = self._candidate(metadata)
        if candidate.is_file():
            print(f"[Kaggle H3] Adapter cached: {metadata.name} -> {candidate}", flush=True)
            verified = self._verify(candidate, metadata)
            sidecar = self._sidecar(verified)
            if sidecar.is_file():
                try:
                    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
                except Exception as exc:
                    raise H3AdapterDownloadError(
                        f"Adapter provenance sidecar is invalid: {sidecar}: {exc}"
                    ) from exc
                expected = {
                    "adapter_id": metadata.adapter_id,
                    "repository": metadata.repository,
                    "filename": metadata.filename,
                    "revision": metadata.revision,
                }
                if any(provenance.get(key) != value for key, value in expected.items()):
                    raise H3AdapterDownloadError(
                        f"Cached adapter provenance does not match the pinned metadata: {sidecar}"
                    )
            else:
                self._write_sidecar(verified, metadata, source="preexisting_local_file")
            return verified
        self.root.mkdir(parents=True, exist_ok=True)
        print(
            f"[Kaggle H3] Downloading adapter {metadata.name} from "
            f"{metadata.repository}/{metadata.filename} @ {metadata.revision}",
            flush=True,
        )
        try:
            from huggingface_hub import hf_hub_download  # type: ignore

            downloaded = hf_hub_download(
                repo_id=metadata.repository,
                filename=metadata.filename,
                revision=metadata.revision,
                local_dir=str(self.root),
                token=os.environ.get("HF_TOKEN"),
            )
        except Exception as exc:
            raise H3AdapterDownloadError(
                f"Could not download H3 adapter {metadata.name!r}; "
                f"repository={metadata.repository!r}, filename={metadata.filename!r}, "
                f"revision={metadata.revision!r}. Check Internet/HF_TOKEN and the file name. {exc}"
            ) from exc
        downloaded_path = Path(downloaded)
        if not downloaded_path.is_absolute():
            downloaded_path = self.root / downloaded_path
        downloaded_path = downloaded_path.resolve()
        # Keep the well-known ComfyUI path even if a hub version returns a
        # temporary cache path. Copying the adapter does not touch the base
        # checkpoint and keeps subsequent Comfy loads deterministic.
        if downloaded_path != candidate.resolve():
            candidate.parent.mkdir(parents=True, exist_ok=True)
            if not candidate.exists():
                shutil.copy2(downloaded_path, candidate)
        verified = self._verify(candidate, metadata)
        self._write_sidecar(verified, metadata, source="huggingface_hub")
        print(f"[Kaggle H3] Adapter ready: {verified}", flush=True)
        return verified


def builtin_catalog() -> dict[str, H3AdapterMetadata]:
    """Official 768p LightX2V ComfyUI Turbo entries, pinned to HF commits."""

    raw = [
        {
            "adapter_id": "h3_fl2va_turbo_4step_v1_0_768p",
            "name": "FL2VA Turbo 4-step v1.0 768p",
            "repository": "lightx2v/Minimax-h3-Turbo",
            "filename": "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
            "revision": "3ec17a324ced54151364f24f8b5fb6bf7e26414f",
            "model_variants": ["fl2va"],
            "conditioning_modes": ["T2VA", "FL2VA"],
            "recommended_strength": 1.0,
            "supported_steps": [4],
            "fused": False,
            "license": "See lightx2v/Minimax-h3-Turbo repository and model card.",
            "attribution": "LightX2V / ModelTC MiniMax-H3-Turbo",
            "kind": "turbo",
            "schedules": {
                "4": {"steps": 4, "nfe": 4, "sampler_name": "euler", "shift_video": 6.0, "shift_audio": 3.0, "resolution": "768p"}
            },
        },
        {
            "adapter_id": "h3_fl2va_turbo_8step_v1_0_768p",
            "name": "FL2VA Turbo 8-step v1.0 768p",
            "repository": "lightx2v/Minimax-h3-Turbo",
            "filename": "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
            "revision": "3ec17a324ced54151364f24f8b5fb6bf7e26414f",
            "model_variants": ["fl2va"],
            "conditioning_modes": ["T2VA", "FL2VA"],
            "recommended_strength": 1.0,
            "supported_steps": [8],
            "fused": False,
            "license": "See lightx2v/Minimax-h3-Turbo repository and model card.",
            "attribution": "LightX2V / ModelTC MiniMax-H3-Turbo",
            "kind": "turbo",
            "schedules": {
                "8": {"steps": 8, "nfe": 8, "sampler_name": "euler", "shift_video": 6.0, "shift_audio": 3.0, "resolution": "768p"}
            },
        },
        {
            "adapter_id": "h3_ref2va_turbo_4step_v0_1",
            "name": "Ref2VA Turbo 4-step v0.1",
            "repository": "lightx2v/Minimax-h3-Turbo",
            "filename": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
            "revision": "3ec17a324ced54151364f24f8b5fb6bf7e26414f",
            "model_variants": ["ref2va"],
            "conditioning_modes": ["Ref2VA"],
            "recommended_strength": 1.0,
            "supported_steps": [4],
            "fused": False,
            "license": "See lightx2v/Minimax-h3-Turbo repository and model card.",
            "attribution": "LightX2V / ModelTC MiniMax-H3-Turbo",
            "kind": "turbo",
            "schedules": {
                "4": {"steps": 4, "nfe": 4, "sampler_name": "euler", "shift_video": 12.0, "shift_audio": 3.0, "resolution": "544p"}
            },
        },
        {
            "adapter_id": "h3_ref2va_turbo_8step_v1_0_768p",
            "name": "Ref2VA Turbo 8-step v1.0 768p",
            "repository": "lightx2v/Minimax-h3-Turbo",
            "filename": "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
            "revision": "3ec17a324ced54151364f24f8b5fb6bf7e26414f",
            "model_variants": ["ref2va"],
            "conditioning_modes": ["Ref2VA"],
            "recommended_strength": 1.0,
            "supported_steps": [8],
            "fused": False,
            "license": "Apache-2.0; see lightx2v/Minimax-h3-Turbo repository and model card.",
            "attribution": "LightX2V / ModelTC MiniMax-H3-Turbo",
            "kind": "turbo",
            "sha256": "6a56f41ab4229c9dd845b9501bbd475ee57e112d846cf2e819d534a1ae928c5a",
            "schedules": {
                "8": {"steps": 8, "nfe": 8, "sampler_name": "euler", "shift_video": 6.0, "shift_audio": 3.0, "resolution": "768p"}
            },
        },
    ]
    return {item.adapter_id: item for item in map(H3AdapterMetadata.from_mapping, raw)}


def load_catalog(path: str | Path | None = None) -> dict[str, H3AdapterMetadata]:
    """Load built-ins, then merge user entries from a JSON catalog if present."""

    result = builtin_catalog()
    if path is None:
        return result
    catalog_path = Path(path)
    if not catalog_path.is_file():
        return result
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        entries = payload.get("adapters", []) if isinstance(payload, Mapping) else payload
        if not isinstance(entries, list):
            raise ValueError("catalog must be a list or an object with an adapters list")
        for entry in entries:
            metadata = H3AdapterMetadata.from_mapping(entry)
            result[metadata.adapter_id] = metadata
    except Exception as exc:
        raise H3AdapterError(f"Invalid H3 adapter catalog {catalog_path}: {exc}") from exc
    return result


def selector_options(catalog: Mapping[str, H3AdapterMetadata]) -> list[str]:
    return [NONE_ADAPTER] + [key for key in sorted(catalog, key=lambda item: catalog[item].name.lower())]


def validate_adapter_compatibility(
    metadata: H3AdapterMetadata,
    context: H3ModelContext,
    *,
    conditioning_mode: str | None,
    turbo_enabled: bool,
    turbo_steps: int,
) -> None:
    if not context.is_h3:
        raise H3AdapterCompatibilityError(
            "H3 Adapter Stack received a model that does not identify as MiniMax H3. "
            "Load a native H3 diffusion model before applying adapters."
        )
    if context.model_variant == "unknown":
        raise H3AdapterCompatibilityError(
            "Could not detect the loaded H3 model variant (expected FL2VA or Ref2VA); "
            "the adapter was not applied. Add explicit h3_model_variant metadata to the loader."
        )
    if context.model_variant not in metadata.model_variants:
        raise H3AdapterCompatibilityError(
            f"Adapter {metadata.name!r} targets {list(metadata.model_variants)}, "
            f"but the loaded checkpoint is {context.model_variant!r}. No download or patch was applied."
        )
    requested_mode = _normalise_mode(conditioning_mode) if conditioning_mode and conditioning_mode != "auto" else "auto"
    candidates = (
        (requested_mode,)
        if requested_mode.lower() != "auto"
        else context.conditioning_mode_candidates
    )
    if not candidates:
        candidates = (context.conditioning_mode,) if context.conditioning_mode != "unknown" else ()
    if not set(candidates).intersection(metadata.conditioning_modes):
        raise H3AdapterCompatibilityError(
            f"Adapter {metadata.name!r} supports conditioning modes {list(metadata.conditioning_modes)}, "
            f"but the detected mode candidates are {list(candidates) or ['unknown']}."
        )
    if metadata.fused:
        raise H3AdapterCompatibilityError(
            f"Adapter {metadata.name!r} is marked fused in metadata and cannot be injected as an incremental LoRA."
        )
    if turbo_enabled and turbo_steps not in metadata.supported_steps:
        raise H3AdapterCompatibilityError(
            f"Adapter {metadata.name!r} does not support Turbo {turbo_steps}-step inference; "
            f"supported steps: {list(metadata.supported_steps)}"
        )


def build_runtime_config(
    model: Any,
    catalog: Mapping[str, H3AdapterMetadata],
    *,
    turbo_mode: bool,
    turbo_steps: int | str,
    adapter_slots: Iterable[tuple[str, float]],
    conditioning_mode: str = "auto",
    model_variant: str = "auto",
) -> dict[str, Any]:
    """Validate all slots and produce the serializable H3 sampler contract."""

    steps = int(turbo_steps)
    if steps not in SUPPORTED_TURBO_STEPS:
        raise H3AdapterCompatibilityError(
            f"Unsupported H3 Turbo step count {steps}; select exactly 4 or 8."
        )
    context = detect_h3_model_context(model, conditioning_mode, model_variant)
    normalized_mode = _normalise_mode(conditioning_mode) if conditioning_mode != "auto" else "auto"
    selected: list[tuple[H3AdapterMetadata, float, int]] = []
    seen: set[str] = set()
    for index, (adapter_id, raw_strength) in enumerate(adapter_slots, start=1):
        adapter_id = str(adapter_id or NONE_ADAPTER)
        strength = float(raw_strength)
        if not 0.0 <= strength <= 2.0:
            raise H3AdapterCompatibilityError(
                f"Adapter {index} strength must be in [0, 2], got {strength}."
            )
        if adapter_id == NONE_ADAPTER or strength == 0.0:
            continue
        metadata = catalog.get(adapter_id)
        if metadata is None:
            raise H3AdapterCompatibilityError(
                f"Unknown H3 adapter selector {adapter_id!r}; add it to the adapter catalog first."
            )
        if adapter_id in seen:
            raise H3AdapterCompatibilityError(
                f"Adapter {metadata.name!r} was selected more than once. "
                "This is rejected to prevent accidental duplicate injection."
            )
        seen.add(adapter_id)
        validate_adapter_compatibility(
            metadata,
            context,
            conditioning_mode=normalized_mode,
            turbo_enabled=bool(turbo_mode),
            turbo_steps=steps,
        )
        selected.append((metadata, strength, index))

    turbo_selected = [item for item in selected if item[0].kind == "turbo" or "turbo" in item[0].adapter_id.lower()]
    if len(turbo_selected) > 1:
        raise H3AdapterCompatibilityError(
            "Select at most one Turbo adapter; multiple Turbo files cannot be composed safely."
        )
    if turbo_selected and not turbo_mode:
        raise H3AdapterCompatibilityError(
            "A Turbo adapter is selected while Turbo mode is disabled. Enable Turbo mode so the sampler schedule changes too."
        )
    if turbo_mode and not turbo_selected and not context.fused_turbo:
        raise H3AdapterCompatibilityError(
            "Turbo mode is enabled but no compatible Turbo adapter is selected and the checkpoint is not marked fused."
        )

    turbo_metadata = turbo_selected[0][0] if turbo_selected else None
    if turbo_mode and turbo_metadata:
        schedule = turbo_metadata.schedule_for(steps)
    elif turbo_mode and context.fused_turbo:
        # A fused checkpoint still needs the fast sampler contract. Its exact
        # schedule is the standard H3 Turbo 768p contract unless loader metadata
        # overrides it; no incremental LoRA is downloaded in this branch.
        schedule = {
            "steps": steps,
            "nfe": steps,
            "sampler_name": "euler",
            "scheduler": "simple",
            "shift_video": 6.0,
            "shift_audio": 3.0,
            "source": "fused_checkpoint_default",
        }
    else:
        schedule = {
            "steps": None,
            "nfe": None,
            "sampler_name": "res_multistep",
            "scheduler": "simple",
            "shift_video": NORMAL_SHIFT_VIDEO,
            "shift_audio": NORMAL_SHIFT_AUDIO,
            "source": "base_h3_schedule",
        }

    applied_or_existing: list[dict[str, Any]] = []
    for metadata, strength, slot in selected:
        status = "planned"
        reason = None
        if metadata.adapter_id in context.existing_adapter_ids:
            status = "skipped_already_applied"
            reason = "the incoming ModelPatcher already records this adapter"
        elif metadata.kind == "turbo" and context.fused_turbo:
            status = "skipped_fused_turbo"
            reason = context.fused_reason or "Turbo is fused into the checkpoint"
        applied_or_existing.append(
            {
                "slot": slot,
                "adapter_id": metadata.adapter_id,
                "name": metadata.name,
                "strength": strength,
                "repository": metadata.repository,
                "filename": metadata.filename,
                "revision": metadata.revision,
                "status": status,
                **({"reason": reason} if reason else {}),
            }
        )

    return {
        "schema_version": "kaggle_h3.runtime.v1",
        "model": {
            "is_h3": context.is_h3,
            "variant": context.model_variant,
            "conditioning_mode": normalized_mode,
            "detected_conditioning_mode": context.conditioning_mode,
            "conditioning_mode_candidates": list(context.conditioning_mode_candidates),
            "fused_turbo": context.fused_turbo,
            "fused_reason": context.fused_reason,
            "evidence": list(context.evidence),
        },
        "turbo": {
            "enabled": bool(turbo_mode),
            "steps": steps if turbo_mode else None,
            "adapter_id": turbo_metadata.adapter_id if turbo_metadata else None,
            "injection_skipped": bool(context.fused_turbo and turbo_mode),
        },
        "sampler": schedule,
        "adapters": applied_or_existing,
        "applied_order": [item["adapter_id"] for item in applied_or_existing if item["status"] == "planned"],
        "offload": {
            "preserve_comfy_model_patcher": True,
            "lora_state_loaded_on_cpu": True,
            "base_checkpoint_modified": False,
            "duplicate_full_gpu_model_copy": False,
        },
        "execution": {
            "strategy": "phase_aware_h3",
            "text_encoder": "automatic_contiguous_language_layers_cuda_0_cuda_1_then_cpu",
            "transformer": {
                "dispatch": "automatic_contiguous_intact_blocks",
                "gpu_budget_gib": 13.0,
                "cpu_headroom_gib": 26.0,
            },
            "audio_vae": "cuda:0",
            "video_vae": "cuda:1",
            "cpu": "third_offload_tier",
            "release_transformer_before_decode": True,
        },
    }


def _patch_keys(model: Any) -> set[Any] | None:
    try:
        return set(model.get_key_patches().keys())
    except Exception:
        return None


def _annotate_model(model: Any, runtime_config: Mapping[str, Any]) -> None:
    options = getattr(model, "model_options", None)
    if isinstance(options, Mapping):
        copied = dict(options)
        transformer = dict(copied.get("transformer_options") or {})
        transformer["kaggle_h3_applied_adapters"] = [
            item["adapter_id"] for item in runtime_config.get("adapters", [])
            if item.get("status") in {"planned", "applied"}
        ]
        transformer["kaggle_h3_runtime_config"] = dict(runtime_config)
        copied["transformer_options"] = transformer
        try:
            model.model_options = copied
        except Exception:
            pass
    setter = getattr(model, "set_attachments", None)
    if callable(setter):
        setter("kaggle_h3_runtime_config", dict(runtime_config))


def apply_adapter_stack(
    model: Any,
    runtime_config: Mapping[str, Any],
    cache: AdapterCache,
    catalog: Mapping[str, H3AdapterMetadata],
) -> tuple[Any, dict[str, Any]]:
    """Apply selected files in order using Comfy's clone/patch API."""

    planned = [item for item in runtime_config.get("adapters", []) if item.get("status") == "planned"]
    if not planned:
        # A no-op/fused stack must not even mutate the incoming ModelPatcher's
        # metadata. The separate runtime-config output is sufficient for the
        # dedicated sampler.
        return model, dict(runtime_config)
    try:
        import comfy.sd  # type: ignore
        import comfy.utils  # type: ignore
    except Exception as exc:
        raise H3AdapterError("ComfyUI is required to apply an H3 adapter stack") from exc

    # Resolve every file before patching any layer. An incompatible or missing
    # later adapter therefore cannot leave a partially patched model behind.
    resolved: list[tuple[dict[str, Any], Path, H3AdapterMetadata]] = []
    for item in planned:
        metadata = catalog[item["adapter_id"]]
        path = cache.resolve(metadata)
        resolved.append((item, path, metadata))

    patched = model
    for item, path, metadata in resolved:
        print(
            f"[Kaggle H3] Applying adapter {item['slot']} in listed order: "
            f"{metadata.name} strength={item['strength']:.3f}",
            flush=True,
        )
        try:
            loaded = comfy.utils.load_torch_file(
                str(path), safe_load=True, return_metadata=True
            )
            if isinstance(loaded, tuple):
                state_dict, lora_metadata = loaded
            else:
                state_dict, lora_metadata = loaded, None
            before = _patch_keys(patched)
            try:
                patched, _ = comfy.sd.load_lora_for_models(
                    patched,
                    None,
                    state_dict,
                    float(item["strength"]),
                    0.0,
                    lora_metadata=lora_metadata,
                )
            except TypeError:
                # Compatibility with an older ComfyUI checkout that lacks the
                # optional metadata keyword, while retaining safe loading.
                patched, _ = comfy.sd.load_lora_for_models(
                    patched, None, state_dict, float(item["strength"]), 0.0
                )
            finally:
                del state_dict
            after = _patch_keys(patched)
            if before is not None and after is not None and not (after - before):
                raise H3AdapterCompatibilityError(
                    f"Adapter {metadata.name!r} loaded but matched no H3 model patches; "
                    "the file is not compatible with this checkpoint."
                )
        except H3AdapterError:
            raise
        except Exception as exc:
            raise H3AdapterError(
                f"Failed applying H3 adapter {metadata.name!r} from {path}: {exc}"
            ) from exc
        item["status"] = "applied"
        item["path"] = str(path)
    _annotate_model(patched, runtime_config)
    runtime = dict(runtime_config)
    runtime["applied_order"] = [item["adapter_id"] for item in runtime.get("adapters", []) if item.get("status") == "applied"]
    return patched, runtime
