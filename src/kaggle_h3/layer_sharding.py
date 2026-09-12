"""Automatic block-level dispatch for the MiniMax H3 transformer.

This module deliberately has no import-time dependency on torch or
Accelerate.  The planner is useful in unit tests and in the preflight report;
the actual dispatch imports optional runtime dependencies only when a worker
is selected.

H3's denoiser is a sequential transformer.  Dispatching complete blocks is
the useful granularity: splitting attention and MLP submodules would create
multiple device transfers inside every residual block and would also make
the block's residual path unsafe to split.  The returned map is compatible
with ``accelerate.dispatch_model`` and leaves a CPU/disk tier for anything
that cannot safely fit in the GPU budgets.  ComfyUI's quantized H3 models use
``comfy_kitchen.QuantizedTensor`` parameters.  Accelerate's generic offload
hook is not compatible with that parameter constructor, so those models use a
Comfy-aware router that preserves the quantized parameter objects and routes
the block activations across the planned devices.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


GIB = 2**30
# This is the H3 phase residency budget, not a claim that the whole process
# may safely consume this much. ComfyUI's quantized casts and denoising
# activations need room above the resident block weights.
DEFAULT_GPU_LIMIT_GIB = 13.0
DEFAULT_GPU_RESERVE_GIB = 0.5
DEFAULT_CPU_HEADROOM_GIB = 26.0
PREFERRED_H3_FIRST_GPU_BLOCKS = 23


class H3LayerShardingError(RuntimeError):
    """Raised when a real layer-dispatch attempt cannot be made safely."""


@dataclass(frozen=True)
class DispatchableLayer:
    """One intact repeated transformer layer and its estimated weight size."""

    index: int
    path: str
    class_name: str
    size_bytes: int


@dataclass(frozen=True)
class DispatchableLayerGroup:
    """The repeated module sequence that can be placed independently."""

    container_path: str
    container_class: str
    layers: tuple[DispatchableLayer, ...]


def parameter_bytes(module: Any) -> int:
    """Estimate unique parameter and buffer storage for ``module``."""

    def storage_bytes(value: Any) -> int:
        # A comfy-kitchen QuantizedTensor reports its logical dtype/numel so
        # PyTorch shape and operator checks remain correct.  That is not its
        # physical footprint, however: budgeting the wrapper as bf16/fp32
        # needlessly sends H3 blocks to the disk tier.  Count the quantized
        # payload and its tensor-valued scale/metadata instead.
        qdata = getattr(value, "_qdata", None)
        if qdata is not None and callable(getattr(qdata, "numel", None)):
            total = int(qdata.numel()) * int(qdata.element_size())
            params = getattr(value, "_params", None)
            fields = getattr(params, "__dataclass_fields__", {})
            for name in fields:
                field_value = getattr(params, name, None)
                if field_value is qdata:
                    continue
                if callable(getattr(field_value, "numel", None)):
                    total += int(field_value.numel()) * int(field_value.element_size())
            return total
        numel = int(getattr(value, "numel", lambda: 0)())
        element_size = getattr(value, "element_size", None)
        if callable(element_size):
            size = int(element_size())
        else:
            dtype = getattr(value, "dtype", None)
            size = int(getattr(dtype, "itemsize", 4))
        return numel * size

    total = 0
    seen: set[int] = set()
    for iterator_name in ("parameters", "buffers"):
        iterator = getattr(module, iterator_name, None)
        if not callable(iterator):
            continue
        try:
            values = iterator(recurse=True)
        except TypeError:
            values = iterator()
        for value in values:
            identity = id(value)
            if identity in seen:
                continue
            seen.add(identity)
            total += storage_bytes(value)
    return total


def _module_at_path(root: Any, path: str) -> Any:
    current = root
    if path:
        for part in path.split("."):
            current = getattr(current, part)
    return current


def _indexed_children(module: Any) -> list[tuple[str, Any]]:
    named_children = getattr(module, "named_children", None)
    if not callable(named_children):
        return []
    children = list(named_children())
    if len(children) < 2:
        return []
    try:
        indexed = [(int(name), name, child) for name, child in children]
    except (TypeError, ValueError):
        return []
    indexed.sort()
    if [item[0] for item in indexed] != list(range(len(indexed))):
        return []
    return [(name, child) for _, name, child in indexed]


def _candidate_score(path: str, count: int) -> tuple[int, int, int, str]:
    lower = path.lower()
    leaf = lower.rsplit(".", 1)[-1]
    preferred = {
        "blocks": 100,
        "transformer_blocks": 96,
        "layers": 90,
        "resblocks": 88,
    }.get(leaf, 0)
    # Prefer the top-level H3 DiT blocks over a small token-refiner block list.
    top_level_bonus = 20 if "." not in path else 0
    return preferred + top_level_bonus, count, -path.count("."), path


def find_dispatchable_layers(model: Any) -> DispatchableLayerGroup:
    """Find the largest indexed module sequence suitable for block dispatch.

    H3 currently exposes its denoiser as ``blocks``.  The discovery is kept
    structural so a compatible future H3 class can use ``transformer_blocks``
    or ``layers`` without hardcoding a model-specific parameter path.
    """

    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        raise H3LayerShardingError("Loaded H3 transformer does not expose named_modules()")

    candidates: list[tuple[tuple[int, int, int, str], str, Any, list[tuple[str, Any]]]] = []
    for path, module in named_modules():
        children = _indexed_children(module)
        if not children:
            continue
        candidates.append((_candidate_score(path, len(children)), path, module, children))
    if not candidates:
        raise H3LayerShardingError(
            "Could not find an indexed H3 transformer block sequence; refusing to split arbitrary submodules."
        )

    _, container_path, container, children = max(candidates, key=lambda item: item[0])
    layers = tuple(
        DispatchableLayer(
            index=index,
            path=f"{container_path}.{name}" if container_path else name,
            class_name=child.__class__.__name__,
            size_bytes=parameter_bytes(child),
        )
        for index, (name, child) in enumerate(children)
    )
    return DispatchableLayerGroup(
        container_path=container_path,
        container_class=container.__class__.__name__,
        layers=layers,
    )


def _text_encoder_candidate_score(path: str, count: int) -> tuple[int, int, int, str]:
    """Rank indexed module sequences for the Qwen text side of H3.

    Qwen3-VL contains both a language transformer and a vision transformer.
    The language path is the useful dispatch boundary for H3 conditioning;
    selecting a visual block list merely because it is longer would leave the
    actual text encoder on ComfyUI's primary GPU.
    """

    lower = path.lower()
    leaf = lower.rsplit(".", 1)[-1]
    if leaf not in {"layers", "blocks", "transformer_blocks", "resblocks"}:
        return (-1, count, -path.count("."), path)
    score = {"layers": 100, "transformer_blocks": 96, "blocks": 88, "resblocks": 84}[leaf]
    if any(token in lower for token in ("language_model", "text_model", "llm", "decoder")):
        score += 220
    if any(token in lower for token in ("vision", "visual", "image_tower")):
        score -= 260
    return score, count, -path.count("."), path


def find_dispatchable_text_encoder_layers(model: Any) -> DispatchableLayerGroup:
    """Find the language-layer sequence inside a Qwen3-VL H3 text encoder."""

    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        raise H3LayerShardingError(
            "Loaded H3 text encoder does not expose named_modules()"
        )
    candidates: list[tuple[tuple[int, int, int, str], str, Any, list[tuple[str, Any]]]] = []
    for path, module in named_modules():
        children = _indexed_children(module)
        if len(children) < 4:
            continue
        score = _text_encoder_candidate_score(path, len(children))
        if score[0] < 0:
            continue
        candidates.append((score, path, module, children))
    if not candidates:
        raise H3LayerShardingError(
            "Could not find a language-layer sequence in the H3 text encoder; "
            "refusing to split arbitrary Qwen/vision submodules."
        )
    _, container_path, container, children = max(candidates, key=lambda item: item[0])
    layers = tuple(
        DispatchableLayer(
            index=index,
            path=f"{container_path}.{name}" if container_path else name,
            class_name=child.__class__.__name__,
            size_bytes=parameter_bytes(child),
        )
        for index, (name, child) in enumerate(children)
    )
    return DispatchableLayerGroup(
        container_path=container_path,
        container_class=container.__class__.__name__,
        layers=layers,
    )


def plan_text_encoder_device_map(
    model: Any,
    *,
    device_ids: Iterable[int],
    gpu_budgets: dict[int, int] | None = None,
    cpu_budget_bytes: int = 0,
    allow_disk: bool = True,
) -> dict[str, Any]:
    """Plan contiguous Qwen language-layer placement across the requested GPUs."""

    ids = [int(value) for value in device_ids]
    if len(ids) < 2:
        raise H3LayerShardingError(
            "Automatic H3 text-encoder sharding needs at least two visible CUDA devices"
        )
    group = find_dispatchable_text_encoder_layers(model)
    if gpu_budgets is None:
        runtime = runtime_memory_budgets(ids)
        gpu_budgets = {device_id: int(runtime[device_id]) for device_id in ids}
        cpu_budget_bytes = int(runtime["cpu"])
    else:
        gpu_budgets = {int(key): int(value) for key, value in gpu_budgets.items()}
    if any(gpu_budgets.get(device_id, 0) <= 0 for device_id in ids):
        raise H3LayerShardingError("The H3 text encoder has no safe GPU dispatch budget")

    split = max(1, min(len(group.layers) - 1, (len(group.layers) + 1) // 2))
    device_map: dict[str, int | str] = {"": "cpu"}
    execution_map: dict[str, str] = {}
    loads = {device_id: 0 for device_id in ids}
    cpu_used = 0
    cpu_layers: list[str] = []
    disk_layers: list[str] = []
    for layer in group.layers:
        preferred = ids[0] if layer.index < split else ids[1]
        if loads[preferred] + layer.size_bytes <= gpu_budgets[preferred]:
            target: int | str = preferred
            loads[preferred] += layer.size_bytes
        elif cpu_used + layer.size_bytes <= cpu_budget_bytes:
            target = "cpu"
            cpu_used += layer.size_bytes
            cpu_layers.append(layer.path)
        elif allow_disk:
            target = "disk"
            disk_layers.append(layer.path)
        else:
            raise H3LayerShardingError(
                f"H3 text-encoder layer {layer.index} ({layer.size_bytes / GIB:.2f} GiB) "
                "does not fit the configured GPU/CPU budgets"
            )
        device_map[layer.path] = target
        execution_map[layer.path] = _device_label(
            preferred if isinstance(target, str) else target
        )

    assigned_gpu_ids = {
        int(device_map[layer.path])
        for layer in group.layers
        if isinstance(device_map.get(layer.path), int)
    }
    if not set(ids).issubset(assigned_gpu_ids):
        raise H3LayerShardingError(
            "The H3 text-encoder map did not place language layers on every requested GPU"
        )
    return {
        "device_map": device_map,
        "execution_device_map": execution_map,
        "group": {
            "container_path": group.container_path,
            "container_class": group.container_class,
            "layer_count": len(group.layers),
        },
        "layers": [
            {
                "index": layer.index,
                "path": layer.path,
                "class_name": layer.class_name,
                "size_bytes": layer.size_bytes,
                "size_gib": round(layer.size_bytes / GIB, 4),
                "device": _device_label(device_map[layer.path]),
                "execution_device": execution_map[layer.path],
            }
            for layer in group.layers
        ],
        "gpu_budgets_gib": {
            str(device_id): round(gpu_budgets[device_id] / GIB, 4) for device_id in ids
        },
        "gpu_load_estimates_gib": {
            str(device_id): round(loads[device_id] / GIB, 4) for device_id in ids
        },
        "cpu_budget_gib": round(cpu_budget_bytes / GIB, 4),
        "cpu_load_estimate_gib": round(cpu_used / GIB, 4),
        "cpu_layers": cpu_layers,
        "disk_layers": disk_layers,
        "uses_all_requested_gpus": True,
        "strategy": "contiguous_qwen_language_layers",
        "activation_boundary": f"after {group.container_path}.{split - 1} before {group.container_path}.{split}",
    }


def runtime_memory_budgets(
    device_ids: Iterable[int],
    *,
    gpu_limit_gib: float = DEFAULT_GPU_LIMIT_GIB,
    gpu_reserve_gib: float = DEFAULT_GPU_RESERVE_GIB,
    cpu_headroom_gib: float = DEFAULT_CPU_HEADROOM_GIB,
) -> dict[str | int, int | float | dict[str, Any]]:
    """Return runtime budgets derived from current free GPU/RAM capacity.

    The GPU budget is capped at the requested safety limit and also leaves a
    small reserve below the current free amount.  CPU budget is the amount
    that can be consumed while retaining the requested available-RAM
    headroom.  A disk tier is still available to Accelerate if RAM is too
    small, but its use is reported as a potentially very slow path.
    """

    ids = [int(value) for value in device_ids]
    if not ids:
        raise H3LayerShardingError("No CUDA devices were supplied for layer dispatch")
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise H3LayerShardingError("Layer dispatch requires PyTorch") from exc
    if not torch.cuda.is_available():
        raise H3LayerShardingError("Layer dispatch requires CUDA")

    budgets: dict[str | int, int | float | dict[str, Any]] = {}
    gpu_limit_bytes = int(float(gpu_limit_gib) * GIB)
    reserve_bytes = int(float(gpu_reserve_gib) * GIB)
    for device_id in ids:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
        budget = max(0, min(gpu_limit_bytes, int(free_bytes) - reserve_bytes))
        budgets[device_id] = budget
        budgets[f"cuda:{device_id}"] = {
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "budget_bytes": budget,
            "budget_gib": round(budget / GIB, 3),
        }

    try:
        import psutil  # type: ignore

        available_bytes = int(psutil.virtual_memory().available)
    except Exception:
        available_bytes = 0
    cpu_budget = max(0, available_bytes - int(float(cpu_headroom_gib) * GIB))
    budgets["cpu"] = cpu_budget
    budgets["cpu_report"] = {
        "available_bytes": available_bytes,
        "headroom_bytes": int(float(cpu_headroom_gib) * GIB),
        "budget_bytes": cpu_budget,
        "budget_gib": round(cpu_budget / GIB, 3),
    }
    return budgets


def _device_label(device: int | str) -> str:
    if isinstance(device, int):
        return f"cuda:{device}"
    return str(device)


def _choose_block_device(
    layer: DispatchableLayer,
    *,
    current_index: int,
    device_ids: list[int],
    remaining_gpu_bytes: dict[int, int],
    assigned_counts: dict[int, int],
    remaining_after: int,
    cpu_remaining: int,
) -> tuple[int | str, int]:
    """Assign one layer while keeping GPU assignments contiguous."""

    index = current_index
    while index < len(device_ids):
        device_id = device_ids[index]
        available = remaining_gpu_bytes[device_id]
        other_gpu_capacity = sum(
            remaining_gpu_bytes[other] for other in device_ids[index + 1 :]
        )
        # Keep at least one layer available for each remaining GPU whenever
        # possible. This prevents the first card from swallowing the entire
        # sequence just because it has a few more free bytes.
        remaining_devices = len(device_ids) - index - 1
        fits = layer.size_bytes <= available
        leaves_room = remaining_after <= other_gpu_capacity + cpu_remaining
        should_advance = (
            index < len(device_ids) - 1
            and assigned_counts[device_id] > 0
            and not leaves_room
        )
        if fits and not should_advance:
            remaining_gpu_bytes[device_id] -= layer.size_bytes
            assigned_counts[device_id] += 1
            return device_id, index
        if fits and remaining_devices == 0:
            remaining_gpu_bytes[device_id] -= layer.size_bytes
            assigned_counts[device_id] += 1
            return device_id, index
        index += 1

    if layer.size_bytes <= cpu_remaining:
        return "cpu", len(device_ids) - 1
    return "disk", len(device_ids) - 1


def _preferred_h3_component_device(name: str, device_ids: list[int]) -> int | None:
    """Return the requested side of the two-T4 H3 layout for a component.

    The official Diffusers names and the ComfyUI names are intentionally both
    covered here.  Unknown components remain subject to the ordinary
    least-loaded placement logic; this is a preference for known H3 islands,
    not a hardcoded map that would break a future H3 revision.
    """

    lower = name.lower().replace("-", "_")
    output_names = {
        "final_layer",
        "norm_out",
        "proj_out",
        "audio_proj_out",
    }
    input_names = {
        "token_refiner",
        "proj_in",
        "audio_proj_in",
        "context_embedder",
        "time_proj",
        "time_embedder",
        "rope",
        "condition_proj",
        "video_patch_proj",
        "audio_patch_proj",
    }
    # ComfyUI's INT8/ConvRot output path ends with a dtype/device conversion
    # inside MiniMax's FinalLayer. Keep that complete path on the primary side
    # so comfy_kitchen never materializes a quantized tensor on one GPU and
    # converts it on the other. The repeated transformer blocks still span
    # both GPUs; only this boundary is intentionally pinned.
    if lower in output_names or lower.endswith(".final_layer"):
        return device_ids[0]
    if lower in input_names:
        return device_ids[0]
    return None


def _plan_preferred_h3_t4_layout(
    model: Any,
    *,
    group: DispatchableLayerGroup,
    device_ids: list[int],
    gpu_budgets: dict[int, int],
    cpu_budget_bytes: int,
    allow_disk: bool,
) -> dict[str, Any]:
    """Plan the proposed H3 two-T4 layout while retaining CPU/disk tiers.

    For the current 50-block H3 transformer this places blocks 0-22 on the
    first GPU and blocks 23-49 on the second GPU. The input projections,
    token refiner, and quantized output/final path follow the first side. The
    output path is deliberately kept intact on one device because ComfyUI's
    quantized tensor conversion is not safe across a live device boundary.
    Other indexed block counts use the same contiguous split at the nearest
    balanced boundary, so this remains structural rather than tied to one
    checkpoint filename.
    """

    if len(device_ids) != 2:
        raise H3LayerShardingError(
            "The preferred H3 T4 layout currently requires exactly two CUDA devices"
        )
    if group.container_path.rsplit(".", 1)[-1].lower() not in {
        "blocks",
        "transformer_blocks",
    }:
        raise H3LayerShardingError(
            "The preferred H3 T4 layout needs a top-level blocks or transformer_blocks sequence"
        )

    device_map: dict[str, int | str] = {"": "cpu"}
    loads = {device_id: 0 for device_id in device_ids}
    cpu_used = 0
    cpu_components: list[str] = []
    disk_components: list[str] = []
    preferred_fallbacks: list[str] = []

    def place_component(name: str, size: int, preferred: int | None) -> None:
        nonlocal cpu_used
        normalized = name.lower().replace("-", "_")
        output_path = normalized in {
            "final_layer",
            "norm_out",
            "proj_out",
            "audio_proj_out",
        }
        candidates = sorted(device_ids, key=lambda device_id: loads[device_id])
        if preferred is not None and output_path:
            # A quantized output module may use .to(dtype) internally before
            # the parent hook runs. It must stay on the primary device or move
            # to CPU/disk; falling back to GPU1 recreates the native abort.
            candidates = [preferred]
        elif preferred is not None:
            candidates = [preferred] + [device_id for device_id in candidates if device_id != preferred]
        for device_id in candidates:
            if loads[device_id] + size <= gpu_budgets[device_id]:
                device_map[name] = device_id
                loads[device_id] += size
                if preferred is not None and device_id != preferred:
                    preferred_fallbacks.append(name)
                return
        if cpu_used + size <= cpu_budget_bytes:
            device_map[name] = "cpu"
            cpu_used += size
            cpu_components.append(name)
            if preferred is not None:
                preferred_fallbacks.append(name)
            return
        if allow_disk:
            device_map[name] = "disk"
            disk_components.append(name)
            if preferred is not None:
                preferred_fallbacks.append(name)
            return
        raise H3LayerShardingError(
            f"H3 component {name!r} ({size / GIB:.2f} GiB) cannot fit the configured budgets"
        )

    named_children = getattr(model, "named_children", None)
    top_children = list(named_children()) if callable(named_children) else []
    group_root = group.container_path.split(".")[0]
    for name, child in top_children:
        if name == group_root:
            continue
        place_component(name, parameter_bytes(child), _preferred_h3_component_device(name, device_ids))

    # Keep the recommended 23/27 split for the current 50-block H3.  Future
    # compatible models get a balanced split without pretending their layer
    # count is still exactly 50.
    split = (
        PREFERRED_H3_FIRST_GPU_BLOCKS
        if len(group.layers) == 50
        else max(1, min(len(group.layers) - 1, (len(group.layers) + 1) // 2))
    )
    cpu_layers: list[str] = []
    disk_layers: list[str] = []
    for layer in group.layers:
        target = device_ids[0] if layer.index < split else device_ids[1]
        if loads[target] + layer.size_bytes <= gpu_budgets[target]:
            device_map[layer.path] = target
            loads[target] += layer.size_bytes
        elif cpu_used + layer.size_bytes <= cpu_budget_bytes:
            device_map[layer.path] = "cpu"
            cpu_used += layer.size_bytes
            cpu_layers.append(layer.path)
        elif allow_disk:
            device_map[layer.path] = "disk"
            disk_layers.append(layer.path)
        else:
            raise H3LayerShardingError(
                f"H3 layer {layer.index} ({layer.size_bytes / GIB:.2f} GiB) does not fit the preferred GPU/CPU budgets"
            )

    assigned_layer_gpu_ids = {
        int(device_map[layer.path])
        for layer in group.layers
        if isinstance(device_map.get(layer.path), int)
    }
    if not set(device_ids).issubset(assigned_layer_gpu_ids):
        raise H3LayerShardingError(
            "The preferred H3 map did not place transformer blocks on both requested GPUs; refusing a one-GPU disguise"
        )

    return {
        "device_map": device_map,
        "group": {
            "container_path": group.container_path,
            "container_class": group.container_class,
            "layer_count": len(group.layers),
        },
        "layers": [
            {
                "index": layer.index,
                "path": layer.path,
                "class_name": layer.class_name,
                "size_bytes": layer.size_bytes,
                "size_gib": round(layer.size_bytes / GIB, 4),
                "device": _device_label(device_map[layer.path]),
            }
            for layer in group.layers
        ],
        "gpu_budgets_gib": {
            str(device_id): round(gpu_budgets[device_id] / GIB, 4) for device_id in device_ids
        },
        "gpu_load_estimates_gib": {
            str(device_id): round((loads[device_id]) / GIB, 4) for device_id in device_ids
        },
        "cpu_budget_gib": round(cpu_budget_bytes / GIB, 4),
        "cpu_load_estimate_gib": round(cpu_used / GIB, 4),
        "cpu_layers": cpu_layers + cpu_components,
        "disk_layers": disk_layers + disk_components,
        "uses_all_requested_gpus": True,
        "strategy": "h3_two_t4_preferred_contiguous_blocks",
        "preferred_layout": {
            "gpu_0": f"{group.container_path}[0:{split}] plus input projections/token_refiner and final/output layers",
            "gpu_1": f"{group.container_path}[{split}:]",
            "activation_boundary": f"after {group.container_path}.{split - 1} before {group.container_path}.{split}",
            "conditioning": "temporary conditioner placement is managed by the worker before denoiser materialization",
            "preferred_component_fallbacks": preferred_fallbacks,
        },
    }


def plan_layer_device_map(
    model: Any,
    *,
    device_ids: Iterable[int],
    gpu_budgets: dict[int, int] | None = None,
    cpu_budget_bytes: int = 0,
    allow_disk: bool = True,
    preferred_layout: str = "auto",
) -> dict[str, Any]:
    """Build a safe hierarchical device map for one loaded H3 transformer.

    Non-block children are assigned to the least-loaded GPU when they fit;
    the repeated transformer blocks are then assigned in contiguous order.
    Unavoidable overflow is sent to CPU and finally disk, never silently
    ignored.  The caller can pass the map directly to Accelerate.
    """

    ids = [int(value) for value in device_ids]
    if len(ids) < 2:
        raise H3LayerShardingError(
            "Automatic H3 layer sharding needs at least two visible CUDA devices"
        )
    group = find_dispatchable_layers(model)
    if gpu_budgets is None:
        runtime = runtime_memory_budgets(ids)
        gpu_budgets = {device_id: int(runtime[device_id]) for device_id in ids}
        cpu_budget_bytes = int(runtime["cpu"])
    else:
        gpu_budgets = {int(key): int(value) for key, value in gpu_budgets.items()}

    for device_id in ids:
        if gpu_budgets.get(device_id, 0) <= 0:
            raise H3LayerShardingError(f"GPU {device_id} has no safe dispatch budget")

    if preferred_layout == "h3_two_t4_preferred":
        return _plan_preferred_h3_t4_layout(
            model,
            group=group,
            device_ids=ids,
            gpu_budgets=gpu_budgets,
            cpu_budget_bytes=cpu_budget_bytes,
            allow_disk=allow_disk,
        )

    device_map: dict[str, int | str] = {"": "cpu"}
    loads = {device_id: 0 for device_id in ids}
    cpu_used = 0
    group_prefix = f"{group.container_path}." if group.container_path else ""

    # Place all non-block top-level components. A hierarchical root entry
    # covers nested parameters that do not have a more specific assignment.
    named_children = getattr(model, "named_children", None)
    top_children = list(named_children()) if callable(named_children) else []
    for name, child in top_children:
        if name == group.container_path or name == group.container_path.split(".")[0]:
            continue
        size = parameter_bytes(child)
        if size == 0:
            device_map[name] = ids[0]
            continue
        preferred = _preferred_h3_component_device(name, ids)
        if preferred is not None and name.lower().replace("-", "_") in {
            "final_layer",
            "norm_out",
            "proj_out",
            "audio_proj_out",
        }:
            # The quantized output path must not fall back to the other GPU:
            # its internal dtype conversion happens before an outer hook can
            # repair the activation device.
            candidates = [preferred]
        else:
            candidates = sorted(ids, key=lambda device_id: loads[device_id])
        placed = False
        for device_id in candidates:
            if loads[device_id] + size <= gpu_budgets[device_id]:
                device_map[name] = device_id
                loads[device_id] += size
                placed = True
                break
        if placed:
            continue
        if cpu_used + size <= cpu_budget_bytes:
            device_map[name] = "cpu"
            cpu_used += size
        elif allow_disk:
            device_map[name] = "disk"
        else:
            raise H3LayerShardingError(
                f"Non-block H3 component {name!r} ({size / GIB:.2f} GiB) cannot fit the configured budgets"
            )

    remaining_gpu = {
        device_id: max(0, gpu_budgets[device_id] - loads[device_id]) for device_id in ids
    }
    counts = {device_id: 0 for device_id in ids}
    remaining_sizes = sum(layer.size_bytes for layer in group.layers)
    current_index = 0
    disk_layers: list[str] = []
    cpu_layers: list[str] = []
    for layer in group.layers:
        remaining_sizes -= layer.size_bytes
        target, target_index = _choose_block_device(
            layer,
            current_index=current_index,
            device_ids=ids,
            remaining_gpu_bytes=remaining_gpu,
            assigned_counts=counts,
            remaining_after=remaining_sizes,
            cpu_remaining=max(0, cpu_budget_bytes - cpu_used),
        )
        if target == "disk" and not allow_disk:
            raise H3LayerShardingError(
                f"H3 layer {layer.index} ({layer.size_bytes / GIB:.2f} GiB) does not fit GPU/CPU budgets"
            )
        device_map[layer.path] = target
        if target == "cpu":
            cpu_used += layer.size_bytes
            cpu_layers.append(layer.path)
        elif target == "disk":
            disk_layers.append(layer.path)
        current_index = max(current_index, target_index)

    assigned_layer_gpu_ids = {
        int(device_map[layer.path])
        for layer in group.layers
        if isinstance(device_map.get(layer.path), int)
    }
    if not set(ids).issubset(assigned_layer_gpu_ids):
        raise H3LayerShardingError(
            "The computed map did not place dispatchable H3 layers on every requested GPU; refusing a one-GPU disguise"
        )

    return {
        "device_map": device_map,
        "group": {
            "container_path": group.container_path,
            "container_class": group.container_class,
            "layer_count": len(group.layers),
        },
        "layers": [
            {
                "index": layer.index,
                "path": layer.path,
                "class_name": layer.class_name,
                "size_bytes": layer.size_bytes,
                "size_gib": round(layer.size_bytes / GIB, 4),
                "device": _device_label(device_map[layer.path]),
            }
            for layer in group.layers
        ],
        "gpu_budgets_gib": {
            str(device_id): round(gpu_budgets[device_id] / GIB, 4) for device_id in ids
        },
        "gpu_load_estimates_gib": {
            str(device_id): round((gpu_budgets[device_id] - remaining_gpu[device_id]) / GIB, 4)
            for device_id in ids
        },
        "cpu_budget_gib": round(cpu_budget_bytes / GIB, 4),
        "cpu_load_estimate_gib": round(cpu_used / GIB, 4),
        "cpu_layers": cpu_layers,
        "disk_layers": disk_layers,
        "uses_all_requested_gpus": True,
        "strategy": "contiguous_intact_transformer_blocks",
    }


def _observed_module_device(module: Any) -> str:
    for iterator_name in ("parameters", "buffers"):
        iterator = getattr(module, iterator_name, None)
        if not callable(iterator):
            continue
        try:
            value = next(iter(iterator(recurse=True)))
        except (StopIteration, TypeError):
            continue
        return str(getattr(value, "device", "unknown"))
    return "parameterless"


def _is_comfy_quantized_value(value: Any) -> bool:
    """Return whether ``value`` looks like a Comfy/comfy-kitchen quantized tensor.

    This intentionally uses structural checks.  Importing ``comfy`` or
    ``comfy_kitchen`` in the package would make the dependency-light planner
    unusable outside a ComfyUI process, and Comfy can wrap the tensor in a
    ``torch.nn.Parameter`` depending on the installed version.
    """

    classes = getattr(value, "__class__", None)
    mro = getattr(classes, "__mro__", ())
    names = {str(getattr(item, "__name__", "")) for item in mro}
    modules = {str(getattr(item, "__module__", "")) for item in mro}
    if "QuantizedTensor" in names or any("comfy_kitchen" in item for item in modules):
        return True
    return any(
        getattr(value, attribute, None) is not None
        for attribute in ("_qdata", "_layout_cls", "_params")
    )


def is_comfy_quantized_model(model: Any) -> bool:
    """Detect ComfyUI's quantized parameter model without importing ComfyUI."""

    for iterator_name in ("parameters", "buffers"):
        iterator = getattr(model, iterator_name, None)
        if not callable(iterator):
            continue
        try:
            values = iterator(recurse=True)
        except TypeError:
            values = iterator()
        for value in values:
            if _is_comfy_quantized_value(value):
                return True
    return False


def _move_tensor_tree(value: Any, device: Any, *, non_blocking: bool = False) -> Any:
    """Move H3 arguments without touching model parameters.

    Synchronous routing is intentional for Comfy's quantized path. An async
    peer copy can race the next quantized weight transfer and surface later as
    ``cudaErrorIllegalAddress`` inside comfy_kitchen.
    """

    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover - only reached in optional runtime code
        return value
    if isinstance(value, torch.Tensor):
        if getattr(value, "device", None) == device:
            return value
        return value.to(device=device, non_blocking=non_blocking)
    if isinstance(value, tuple):
        return tuple(
            _move_tensor_tree(item, device, non_blocking=non_blocking)
            for item in value
        )
    if isinstance(value, list):
        return [
            _move_tensor_tree(item, device, non_blocking=non_blocking)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _move_tensor_tree(item, device, non_blocking=non_blocking)
            for key, item in value.items()
        }
    return value


def _execution_device_for_path(
    path: str,
    target: int | str,
    *,
    group: DispatchableLayerGroup,
    device_ids: list[int],
) -> str:
    """Resolve a GPU for a CPU/disk-resident entry's actual forward pass."""

    if isinstance(target, int):
        return _device_label(target)
    if len(device_ids) < 2:
        return _device_label(device_ids[0])
    if path.startswith(f"{group.container_path}."):
        layer_name = path.rsplit(".", 1)[-1]
        try:
            layer_index = int(layer_name)
        except ValueError:
            layer_index = 0
        split = (
            PREFERRED_H3_FIRST_GPU_BLOCKS
            if len(group.layers) == 50
            else max(1, min(len(group.layers) - 1, (len(group.layers) + 1) // 2))
        )
        return _device_label(device_ids[0] if layer_index < split else device_ids[1])
    preferred = _preferred_h3_component_device(path, device_ids)
    return _device_label(preferred if preferred is not None else device_ids[0])


def _remove_handles(handles: Iterable[Any]) -> None:
    for handle in handles:
        remove = getattr(handle, "remove", None)
        if callable(remove):
            try:
                remove()
            except Exception:
                pass


class _AttributeRestoreHandle:
    """Make a temporary Comfy runtime override participate in cleanup."""

    def __init__(self, owner: Any, name: str, original: Any) -> None:
        self.owner = owner
        self.name = name
        self.original = original
        self.active = True

    def remove(self) -> None:
        if not self.active:
            return
        setattr(self.owner, self.name, self.original)
        self.active = False


def dispatch_h3_text_encoder(
    model: Any,
    *,
    device_ids: Iterable[int],
    gpu_limit_gib: float = DEFAULT_GPU_LIMIT_GIB,
    cpu_headroom_gib: float = DEFAULT_CPU_HEADROOM_GIB,
    offload_dir: Path | None = None,
) -> tuple[Any, dict[str, Any], list[Any]]:
    """Install a contiguous two-GPU router for the Qwen language encoder.

    ComfyUI's CLIP patcher still owns the outer encoder and may load it on its
    registered primary device.  The language-layer hooks below immediately
    move each selected layer to its planned side, route its activation there,
    and send overflow layers back to CPU after each call.  This preserves the
    normal Comfy patcher contract while ensuring the same conditioning pass
    actually executes language layers on both requested GPUs.
    """

    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise H3LayerShardingError(
            "H3 text-encoder dispatch requires PyTorch"
        ) from exc
    ids = [int(value) for value in device_ids]
    if len(ids) < 2:
        raise H3LayerShardingError(
            "H3 text-encoder dispatch requires at least two CUDA devices"
        )
    runtime = runtime_memory_budgets(
        ids,
        gpu_limit_gib=gpu_limit_gib,
        cpu_headroom_gib=cpu_headroom_gib,
    )
    plan = plan_text_encoder_device_map(
        model,
        device_ids=ids,
        gpu_budgets={device_id: int(runtime[device_id]) for device_id in ids},
        cpu_budget_bytes=int(runtime["cpu"]),
        allow_disk=True,
    )
    group_path = str(plan["group"]["container_path"])
    primary = torch.device(_device_label(ids[0]))
    handles: list[Any] = []
    movement: list[dict[str, Any]] = []

    def register_pre_hook(module: Any, callback: Any) -> None:
        register = getattr(module, "register_forward_pre_hook", None)
        if not callable(register):
            raise H3LayerShardingError(
                f"Text-encoder module {module.__class__.__name__} does not support forward pre-hooks"
            )
        try:
            handles.append(register(callback, with_kwargs=True))
        except TypeError:
            handles.append(register(lambda _module, args: callback(_module, args, {})))

    def register_post_hook(module: Any, callback: Any) -> None:
        register = getattr(module, "register_forward_hook", None)
        if not callable(register):
            raise H3LayerShardingError(
                f"Text-encoder module {module.__class__.__name__} does not support forward hooks"
            )
        try:
            handles.append(register(callback, with_kwargs=True))
        except TypeError:
            handles.append(register(lambda _module, args, output: callback(_module, args, {}, output)))

    try:
        for item in plan["layers"]:
            path = str(item["path"])
            module = _module_at_path(model, path)
            target = torch.device(str(item["execution_device"]))
            planned_residency = str(item["device"])
            if planned_residency.startswith("cuda:"):
                try:
                    module.to(target)
                except Exception as exc:
                    raise H3LayerShardingError(
                        f"Could not place text-encoder layer {path!r} on {target}: {exc}"
                    ) from exc

            def route_layer_inputs(
                current: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                _target=target,
                _planned=planned_residency,
                _path=path,
            ) -> tuple[tuple[Any, ...], dict[str, Any]]:
                if _planned == "cpu" or _planned == "disk":
                    try:
                        current.to(_target)
                    except Exception as exc:
                        raise H3LayerShardingError(
                            f"Could not stage text-encoder layer {_path!r} on {_target}: {exc}"
                        ) from exc
                return (
                    tuple(_move_tensor_tree(value, _target) for value in args),
                    _move_tensor_tree(kwargs, _target),
                )

            register_pre_hook(module, route_layer_inputs)
            if planned_residency == "cpu" or planned_residency == "disk":

                def offload_layer(
                    current: Any,
                    _args: tuple[Any, ...],
                    _kwargs: dict[str, Any],
                    output: Any,
                    *,
                    _path=path,
                ) -> Any:
                    try:
                        current.to(torch.device("cpu"))
                    except Exception as exc:
                        raise H3LayerShardingError(
                            f"Could not return overflow text-encoder layer {_path!r} to CPU: {exc}"
                        ) from exc
                    return output

                register_post_hook(module, offload_layer)
            movement.append(
                {
                    "path": path,
                    "planned_residency": planned_residency,
                    "execution_device": str(target),
                }
            )

        # The parent continues into its final norm/projection after the layer
        # sequence. Route those inputs back to the primary side so a layer-1
        # output cannot leak into a cuda:0-only Qwen head.
        parent_path, separator, container_name = group_path.rpartition(".")
        parent = _module_at_path(model, parent_path) if separator else model
        named_children = getattr(parent, "named_children", None)
        if callable(named_children):
            for name, sibling in named_children():
                if name == container_name:
                    continue

                def route_boundary(
                    _module: Any,
                    args: tuple[Any, ...],
                    kwargs: dict[str, Any],
                    *,
                    _target=primary,
                ) -> tuple[tuple[Any, ...], dict[str, Any]]:
                    return (
                        tuple(_move_tensor_tree(value, _target) for value in args),
                        _move_tensor_tree(kwargs, _target),
                    )

                register_pre_hook(sibling, route_boundary)

    except Exception:
        _remove_handles(handles)
        raise

    plan["movement"] = movement
    plan["router"] = {
        "status": "installed",
        "backend": "comfy_native_text_layer_routing",
        "cpu_third_tier": True,
        "offload_dir": str(offload_dir.resolve()) if offload_dir else None,
        "primary_boundary_device": str(primary),
        "hook_count": len(handles),
    }
    return model, plan, handles


def _install_comfy_quantized_router(
    transformer: Any,
    plan: dict[str, Any],
    *,
    device_ids: list[int],
) -> tuple[dict[str, str], dict[str, Any], list[Any]]:
    """Install a Comfy-safe execution router for a quantized H3 transformer.

    Accelerate's ``set_module_tensor_to_device`` reconstructs the parameter
    class with a ``requires_grad`` keyword.  ``comfy_kitchen`` deliberately
    exposes a frozen ``QuantizedTensor`` constructor without that keyword.
    Moving a Comfy quantized module through its own ``Module._apply`` keeps the
    quantized storage and scale tensors coherent, so GPU-budgeted entries are
    moved in place.  Overflow entries stay CPU-owned (or remain eligible for a
    Comfy disk-backed source) and their H3 arguments are routed to the side of
    the transformer where they execute.  This uses CPU RAM as the third tier
    without creating a second full checkpoint copy.
    """

    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise H3LayerShardingError(
            "Comfy quantized H3 dispatch requires PyTorch"
        ) from exc

    group_info = plan.get("group") or {}
    group = DispatchableLayerGroup(
        container_path=str(group_info.get("container_path", "blocks")),
        container_class=str(group_info.get("container_class", "")),
        layers=tuple(
            DispatchableLayer(
                index=int(item["index"]),
                path=str(item["path"]),
                class_name=str(item.get("class_name", "")),
                size_bytes=int(item.get("size_bytes", 0)),
            )
            for item in (plan.get("layers") or [])
        ),
    )
    device_map = dict(plan.get("device_map") or {})
    execution_map: dict[str, str] = {}
    handles: list[Any] = []
    movement: list[dict[str, Any]] = []

    # ComfyUI calculates this helper inside cast_bias_weight() immediately
    # before it copies a quantized weight. Disable non-blocking transfers for
    # the lifetime of this phase so a Comfy offload stream cannot race a
    # cross-device QuantizedTensor copy. The original helper is restored by
    # the same cleanup path as the forward hooks.
    try:
        import comfy.model_management as comfy_model_management  # type: ignore

        non_blocking_helper = getattr(
            comfy_model_management, "device_supports_non_blocking", None
        )
        if callable(non_blocking_helper):
            def h3_quantized_copies_are_synchronous(_device: Any) -> bool:
                return False

            comfy_model_management.device_supports_non_blocking = (
                h3_quantized_copies_are_synchronous
            )
            handles.append(
                _AttributeRestoreHandle(
                    comfy_model_management,
                    "device_supports_non_blocking",
                    non_blocking_helper,
                )
            )
    except Exception:
        # The dependency-light planner and unit tests do not provide ComfyUI.
        pass

    # Route all explicitly mapped components, including each intact block.
    # The root entry is a residency default, not a callable module.
    mapped_paths = [path for path in device_map if path]
    try:
        for path in mapped_paths:
            target = device_map[path]
            execution_device = _execution_device_for_path(
                path, target, group=group, device_ids=device_ids
            )
            execution_map[path] = execution_device
            module = _module_at_path(transformer, path)
            residency_before = _observed_module_device(module)
            residency_after = residency_before
            move_error = None
            if isinstance(target, int):
                try:
                    module.to(torch.device(execution_device))
                    residency_after = _observed_module_device(module)
                except Exception as exc:
                    move_error = str(exc)
                    raise H3LayerShardingError(
                        "ComfyUI could not move a quantized H3 module through its "
                        f"native _apply path at {path!r} to {execution_device}: {exc}"
                    ) from exc
            movement.append(
                {
                    "path": path,
                    "planned_residency": _device_label(target),
                    "execution_device": execution_device,
                    "residency_before": residency_before,
                    "residency_after": residency_after,
                    "move_error": move_error,
                }
            )

            register = getattr(module, "register_forward_pre_hook", None)
            if not callable(register):
                raise H3LayerShardingError(
                    f"Mapped H3 module {path!r} does not support forward pre-hooks"
                )

            def route_inputs(
                _module: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                _target=execution_device,
            ) -> tuple[tuple[Any, ...], dict[str, Any]]:
                target_device = torch.device(_target)
                return (
                    tuple(_move_tensor_tree(item, target_device) for item in args),
                    _move_tensor_tree(kwargs, target_device),
                )

            try:
                handle = register(route_inputs, with_kwargs=True)
            except TypeError:
                # PyTorch versions before the kwargs-aware hook API still
                # cover H3's positional activation/timestep arguments.
                handle = register(
                    lambda _module, args, _target=execution_device: tuple(
                        _move_tensor_tree(item, torch.device(_target)) for item in args
                    )
                )
            handles.append(handle)

            # MiniMaxH3Model applies masks and the audio sigma conversion after
            # _forward returns.  Those tensors are owned by the sampler's
            # primary device, so return the final head output there after a
            # GPU1 final layer instead of leaving a cross-device mismatch for
            # the outer model wrapper.
            leaf = path.rsplit(".", 1)[-1].lower()
            if leaf in {"final_layer", "norm_out", "proj_out", "audio_proj_out"}:
                # FinalLayer contains norm, AdaLN, and two output heads. A
                # parent hook alone is insufficient for Comfy's lazy
                # quantized operations: one child can still materialize on
                # cuda:0 while the activation and sibling child are on
                # cuda:1. Route every final-layer child through the same
                # device island before its forward call.
                named_nested = getattr(module, "named_modules", None)
                if callable(named_nested):
                    for nested_path, nested_module in named_nested():
                        if not nested_path:
                            continue

                        def route_final_child_inputs(
                            _nested: Any,
                            nested_args: tuple[Any, ...],
                            nested_kwargs: dict[str, Any],
                            *,
                            _target=torch.device(execution_device),
                        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
                            return (
                                tuple(_move_tensor_tree(value, _target) for value in nested_args),
                                _move_tensor_tree(nested_kwargs, _target),
                            )

                        try:
                            nested_handle = nested_module.register_forward_pre_hook(
                                route_final_child_inputs,
                                with_kwargs=True,
                            )
                        except TypeError:
                            nested_handle = nested_module.register_forward_pre_hook(
                                lambda _nested, nested_args, _target=torch.device(execution_device): tuple(
                                    _move_tensor_tree(value, _target) for value in nested_args
                                )
                            )
                        handles.append(nested_handle)

                register_forward = getattr(module, "register_forward_hook", None)
                if callable(register_forward):

                    def route_outputs(
                        _module: Any,
                        _args: tuple[Any, ...],
                        _kwargs: dict[str, Any],
                        output: Any,
                        *,
                        _target=torch.device(_device_label(device_ids[0])),
                    ) -> Any:
                        return _move_tensor_tree(output, _target)

                    try:
                        handles.append(register_forward(route_outputs, with_kwargs=True))
                    except TypeError:
                        handles.append(
                            register_forward(
                                lambda _module, _args, output, _target=torch.device(
                                    _device_label(device_ids[0])
                                ): _move_tensor_tree(output, _target)
                            )
                        )
    except Exception:
        _remove_handles(handles)
        raise

    plan["execution_device_map"] = execution_map
    for layer in plan.get("layers", []):
        path = str(layer["path"])
        layer["execution_device"] = execution_map[path]
    plan["comfy_quantized_router"] = {
        "status": "installed",
        "backend": "comfy_native_quantized_apply_plus_activation_routing",
        "mapped_module_count": len(mapped_paths),
        "gpu_resident_module_count": sum(
            1 for item in movement if item["planned_residency"].startswith("cuda:")
        ),
        "cpu_or_disk_resident_module_count": sum(
            1
            for item in movement
            if not item["planned_residency"].startswith("cuda:")
        ),
        "movement": movement,
        "cpu_third_tier": True,
        "disk_storage": "Comfy-managed source/offload when available; no Accelerate disk hooks",
        "outer_model_return_device": _device_label(device_ids[0]),
        "quantized_copies": "synchronous",
        "activation_transfers": "synchronous",
    }
    return execution_map, plan["comfy_quantized_router"], handles


def inspect_dispatched_map(
    model: Any,
    plan: dict[str, Any],
    *,
    execution_device_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Inspect parameter residency and, when present, actual execution routing."""

    device_map = dict(plan.get("device_map") or {})
    observed_layers: list[dict[str, Any]] = []
    for layer in plan.get("layers", []):
        path = str(layer["path"])
        try:
            module = _module_at_path(model, path)
            observed = _observed_module_device(module)
        except Exception as exc:
            observed = f"inspection_error: {exc}"
        entry = dict(layer)
        entry["planned_device"] = _device_label(device_map.get(path, "unknown"))
        entry["observed_device"] = observed
        if execution_device_map is not None and path in execution_device_map:
            entry["observed_execution_device"] = execution_device_map[path]
        observed_layers.append(entry)

    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        reported_map = {str(key): _device_label(value) for key, value in hf_device_map.items()}
    else:
        reported_map = {
            str(key): _device_label(value) for key, value in device_map.items()
        }
    observed_gpu_ids = {
        int(value.split(":", 1)[1])
        for value in (entry["observed_device"] for entry in observed_layers)
        if value.startswith("cuda:") and value.split(":", 1)[1].isdigit()
    }
    observed_execution_gpu_ids = {
        int(value.split(":", 1)[1])
        for value in (
            entry.get("observed_execution_device", entry["observed_device"])
            for entry in observed_layers
        )
        if str(value).startswith("cuda:") and str(value).split(":", 1)[1].isdigit()
    }
    reported_execution_map = {
        str(key): str(value)
        for key, value in (execution_device_map or {}).items()
    }
    return {
        "reported_module_map": reported_map,
        "reported_execution_map": reported_execution_map,
        "layers": observed_layers,
        "observed_gpu_ids": sorted(observed_gpu_ids),
        "observed_execution_gpu_ids": sorted(observed_execution_gpu_ids),
        "observed_all_requested_gpus": len(
            observed_execution_gpu_ids if execution_device_map is not None else observed_gpu_ids
        )
        >= 2,
    }


def dispatch_h3_transformer(
    transformer: Any,
    *,
    device_ids: Iterable[int],
    offload_dir: Path,
    gpu_limit_gib: float = DEFAULT_GPU_LIMIT_GIB,
    cpu_headroom_gib: float = DEFAULT_CPU_HEADROOM_GIB,
    preferred_layout: str = "h3_two_t4_preferred",
) -> tuple[Any, dict[str, Any]]:
    """Compute and apply an actual H3 hierarchical device map.

    Standard modules use Accelerate's hierarchical hooks.  ComfyUI's INT8
    ConvRot model is routed through the Comfy-native quantized branch because
    Accelerate cannot reconstruct its frozen ``QuantizedTensor`` parameters.
    """

    ids = [int(value) for value in device_ids]
    runtime = runtime_memory_budgets(
        ids,
        gpu_limit_gib=gpu_limit_gib,
        cpu_headroom_gib=cpu_headroom_gib,
    )
    plan = plan_layer_device_map(
        transformer,
        device_ids=ids,
        gpu_budgets={device_id: int(runtime[device_id]) for device_id in ids},
        cpu_budget_bytes=int(runtime["cpu"]),
        allow_disk=True,
        preferred_layout=preferred_layout,
    )
    planned_layer_devices = {
        item["device"]
        for item in plan["layers"]
        if str(item.get("device", "")).startswith("cuda:")
    }
    if len(planned_layer_devices) < 2:
        raise H3LayerShardingError("Automatic H3 dispatch produced a one-GPU map")

    if is_comfy_quantized_model(transformer):
        execution_map, router_report, handles = _install_comfy_quantized_router(
            transformer,
            plan,
            device_ids=ids,
        )
        try:
            observed = inspect_dispatched_map(
                transformer,
                plan,
                execution_device_map=execution_map,
            )
            report = {
                "status": "dispatched" if observed["observed_all_requested_gpus"] else "not_verified",
                "dispatch_backend": "comfy_native_quantized_apply_plus_activation_routing",
                "requested_device_ids": ids,
                "runtime_budget_observation": {
                    str(device_id): runtime[f"cuda:{device_id}"] for device_id in ids
                }
                | {"cpu": runtime["cpu_report"]},
                "planned": plan,
                "observed": observed,
                "cpu_third_tier": True,
                "disk_offload_used": False,
                "disk_offload_planned": "disk" in plan["device_map"].values(),
                "offload_dir": str(Path(offload_dir)) if "disk" in plan["device_map"].values() else None,
                "weight_residency": "Comfy-managed quantized weights; GPU entries moved with native Module._apply; overflow entries remain CPU-owned",
                "execution_routing": router_report,
            }
            if not observed["observed_all_requested_gpus"]:
                raise H3LayerShardingError(
                    "Comfy's quantized H3 router did not expose execution on every requested GPU"
                )
            # The phase runtime removes these hooks after denoising.  Keep the
            # handle out of the JSON report/runtime config.
            setattr(transformer, "_kaggle_h3_dispatch_handles", handles)
            return transformer, report
        except Exception:
            _remove_handles(handles)
            raise

    try:
        from accelerate import dispatch_model  # type: ignore
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise H3LayerShardingError(
            "The experimental layer-sharded backend requires accelerate.dispatch_model"
        ) from exc

    has_disk = "disk" in plan["device_map"].values()
    offload_dir = Path(offload_dir)
    if has_disk:
        offload_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "device_map": plan["device_map"],
        "main_device": f"cuda:{ids[0]}",
        "offload_buffers": True,
        "force_hooks": True,
    }
    if has_disk:
        kwargs["offload_dir"] = str(offload_dir)
    try:
        dispatched = dispatch_model(transformer, **kwargs)
    except Exception as exc:
        raise H3LayerShardingError(
            "Accelerate could not apply the computed H3 layer map. "
            f"Plan={plan['device_map']}; error={exc}"
        ) from exc

    observed = inspect_dispatched_map(dispatched, plan)
    report = {
        "status": "dispatched" if observed["observed_all_requested_gpus"] else "not_verified",
        "dispatch_backend": "accelerate.dispatch_model",
        "requested_device_ids": ids,
        "runtime_budget_observation": {
            str(device_id): runtime[f"cuda:{device_id}"] for device_id in ids
        }
        | {"cpu": runtime["cpu_report"]},
        "planned": plan,
        "observed": observed,
        "cpu_third_tier": True,
        "disk_offload_used": has_disk,
        "offload_dir": str(offload_dir) if has_disk else None,
    }
    if not observed["observed_all_requested_gpus"]:
        raise H3LayerShardingError(
            "Accelerate returned without observable transformer blocks on every requested GPU; refusing to continue"
        )
    return dispatched, report


def load_and_dispatch_h3_transformer(
    model_class: Any,
    *,
    repository: str,
    subfolder: str,
    dtype: Any,
    device_ids: Iterable[int],
    offload_dir: Path,
    gpu_limit_gib: float = DEFAULT_GPU_LIMIT_GIB,
    cpu_headroom_gib: float = DEFAULT_CPU_HEADROOM_GIB,
    preferred_layout: str = "h3_two_t4_preferred",
) -> tuple[Any, dict[str, Any]]:
    """Create a meta H3 model, plan its blocks, and load directly into devices.

    Planning before weight materialization is important on Kaggle: loading a
    full bfloat16 H3 transformer on CPU first can exceed the host RAM budget
    before Accelerate ever gets a chance to dispatch it.
    """

    ids = [int(value) for value in device_ids]
    if len(ids) < 2:
        raise H3LayerShardingError("H3 transformer loading requires at least two CUDA devices")
    try:
        import torch  # type: ignore
        from accelerate import init_empty_weights  # type: ignore
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise H3LayerShardingError(
            "Pre-materialization H3 dispatch requires torch and accelerate"
        ) from exc
    if not torch.cuda.is_available():
        raise H3LayerShardingError("Pre-materialization H3 dispatch requires CUDA")

    try:
        config = model_class.load_config(repository, subfolder=subfolder)
        with init_empty_weights():
            meta_model = model_class.from_config(config)
        # The meta parameters must reflect the requested load dtype when the
        # map is sized; otherwise a float32 meta model overestimates a bf16/fp16
        # checkpoint by 2x and needlessly sends layers to disk.
        meta_model = meta_model.to(dtype=dtype)
    except Exception as exc:
        raise H3LayerShardingError(
            f"Could not construct a meta H3 transformer from {repository}/{subfolder}: {exc}"
        ) from exc

    runtime = runtime_memory_budgets(
        ids,
        gpu_limit_gib=gpu_limit_gib,
        cpu_headroom_gib=cpu_headroom_gib,
    )
    plan = plan_layer_device_map(
        meta_model,
        device_ids=ids,
        gpu_budgets={device_id: int(runtime[device_id]) for device_id in ids},
        cpu_budget_bytes=int(runtime["cpu"]),
        allow_disk=True,
        preferred_layout=preferred_layout,
    )
    offload_dir = Path(offload_dir)
    has_disk = "disk" in plan["device_map"].values()
    if has_disk:
        offload_dir.mkdir(parents=True, exist_ok=True)
    try:
        loaded = model_class.from_pretrained(
            repository,
            subfolder=subfolder,
            dtype=dtype,
            device_map=plan["device_map"],
            offload_folder=str(offload_dir) if has_disk else None,
            offload_state_dict=True,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
    except Exception as exc:
        raise H3LayerShardingError(
            "H3 transformer weights could not be loaded with the precomputed layer map. "
            f"The loader failed before generation; plan={plan['device_map']}; error={exc}"
        ) from exc

    observed = inspect_dispatched_map(loaded, plan)
    report = {
        "status": "dispatched" if observed["observed_all_requested_gpus"] else "not_verified",
        "model_class": model_class.__name__,
        "repository": repository,
        "subfolder": subfolder,
        "requested_device_ids": ids,
        "runtime_budget_observation": {
            str(device_id): runtime[f"cuda:{device_id}"] for device_id in ids
        }
        | {"cpu": runtime["cpu_report"]},
        "planned": plan,
        "observed": observed,
        "cpu_third_tier": True,
        "disk_offload_used": has_disk,
        "offload_dir": str(offload_dir) if has_disk else None,
        "materialization": "meta_then_device_map_load",
    }
    if not observed["observed_all_requested_gpus"]:
        raise H3LayerShardingError(
            "H3 transformer loaded without observable blocks on every requested GPU; refusing one-GPU execution"
        )
    return loaded, report
