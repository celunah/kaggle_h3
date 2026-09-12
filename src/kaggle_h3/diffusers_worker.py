"""Optional official Diffusers Modular H3 component-split worker.

This module is imported lazily.  The base Kaggle requirements intentionally do
not install the large Diffusers/Transformers/TorchAO stack.  When an eligible
machine has that stack, this worker follows the H3 documentation's
ComponentsManager split and leaves CPU auto-offload enabled as a third tier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .layer_sharding import load_and_dispatch_h3_transformer
from .workflow import H3Request, build_prompt, dimensions, frame_length, select_mode


class DiffusersH3Worker:
    """Load H3 as two modular pipelines with explicit component ownership."""

    def __init__(
        self,
        *,
        repository: str = "MiniMaxAI/MiniMax-H3",
        workflow: str,
        dtype_name: str = "bfloat16",
        layer_sharded: bool = False,
        device_ids: tuple[int, ...] | None = None,
        offload_dir: Path | None = None,
    ) -> None:
        self.repository = repository
        self.workflow = workflow.lower()
        self.dtype_name = dtype_name
        self.layer_sharded = bool(layer_sharded)
        self.device_ids = tuple(device_ids or ())
        self.offload_dir = offload_dir
        self.conditioner: Any = None
        self.rest: Any = None
        self.dispatch_report: dict[str, Any] | None = None
        self._transformer_name: str | None = None
        self.actual_device_map: dict[str, Any] = {
            "text_encoder": "cuda:1 (ComponentsManager auto CPU offload)",
            "transformer": "cuda:0 (ComponentsManager auto CPU offload)",
            "video_vae": "cuda:0 (ComponentsManager auto CPU offload)",
            "audio_vae": "cuda:0 (ComponentsManager auto CPU offload)",
            "cpu": "host RAM third tier via both ComponentsManagers",
        }

    def load(self) -> "DiffusersH3Worker":
        if self.layer_sharded:
            return self._load_layer_sharded()

        import torch  # type: ignore
        from diffusers import ComponentsManager, ModularPipeline  # type: ignore

        dtype = getattr(torch, self.dtype_name)
        workflow_blocks = ModularPipeline.from_pretrained(self.repository).blocks.get_workflow(
            self.workflow
        )
        text_manager = ComponentsManager()
        _enable_cpu_offload(text_manager, "cuda:1")
        text_blocks = workflow_blocks.sub_blocks.pop("text_encoder")
        self.conditioner = text_blocks.init_pipeline(
            self.repository, components_manager=text_manager
        )
        self.conditioner.load_components(dtype=dtype)

        rest_manager = ComponentsManager()
        _enable_cpu_offload(rest_manager, "cuda:0")
        self.rest = workflow_blocks.init_pipeline(
            self.repository, components_manager=rest_manager
        )
        self.rest.load_components(dtype=dtype)
        self.actual_device_map["loaded"] = True
        return self

    def _load_layer_sharded(self) -> "DiffusersH3Worker":
        """Load H3 and dispatch intact transformer blocks across all GPUs."""

        import torch  # type: ignore
        from diffusers import ComponentsManager, ModularPipeline  # type: ignore

        ids = self.device_ids or tuple(range(min(int(torch.cuda.device_count()), 2)))
        if len(ids) < 2:
            raise RuntimeError(
                "diffusers_layer_sharded requires two visible CUDA devices; refusing one-GPU downgrade"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("diffusers_layer_sharded requires CUDA")
        if self.offload_dir is None:
            raise RuntimeError("diffusers_layer_sharded requires a writable layer offload directory")
        self.device_ids = tuple(ids)

        workflow_blocks = ModularPipeline.from_pretrained(self.repository).blocks.get_workflow(
            self.workflow
        )
        transformer_name = "transformer_ref" if self.workflow == "ref2va" else "transformer"

        # Keep conditioning on the second device, using the official modular
        # ComponentsManager CPU tier. Remove the large denoiser before loading
        # the remaining pipeline so it is never materialized as a CPU copy.
        text_manager = ComponentsManager()
        _enable_cpu_offload(text_manager, f"cuda:{ids[1]}")
        text_blocks = workflow_blocks.sub_blocks.pop("text_encoder")
        workflow_blocks.sub_blocks.pop(transformer_name, None)
        self.conditioner = text_blocks.init_pipeline(
            self.repository, components_manager=text_manager
        )
        self.conditioner.load_components(dtype=getattr(torch, self.dtype_name))

        rest_manager = ComponentsManager()
        _enable_cpu_offload(rest_manager, f"cuda:{ids[0]}")
        self.rest = workflow_blocks.init_pipeline(
            self.repository, components_manager=rest_manager
        )
        self.rest.load_components(dtype=getattr(torch, self.dtype_name))

        # Materialize the denoiser only after the conditioner has run.  This
        # follows the requested temporary Qwen tier: the text encoder can use
        # GPU 1 for conditioning, then its weights are released before the
        # two-GPU transformer map is loaded.  It prevents a large Qwen copy
        # from competing with transformer blocks during dispatch.
        self._transformer_name = transformer_name
        self.dispatch_report = None
        self.actual_device_map = {
            "text_encoder": f"cuda:{ids[1]} (temporary ComponentsManager auto CPU offload)",
            "transformer": "deferred until conditioner release",
            "video_vae": f"cuda:{ids[0]} (ComponentsManager auto CPU offload)",
            "audio_vae": f"cuda:{ids[0]} (ComponentsManager auto CPU offload)",
            "cpu": "host RAM third tier via ComponentsManager/Accelerate",
            "disk": str(self.offload_dir),
        }
        return self

    def _release_conditioner(self) -> None:
        """Release temporary Qwen weights while retaining generated state."""

        if self.conditioner is None:
            return
        self.conditioner = None
        import gc

        gc.collect()
        try:
            import torch  # type: ignore

            for device_id in self.device_ids:
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()
        except Exception:
            # Cache clearing is an optimization; Accelerate still owns the
            # actual offload lifecycle if a runtime does not expose CUDA.
            pass

    def _load_dispatched_transformer(self) -> None:
        """Load the denoiser with the preferred explicit two-T4 map."""

        import torch  # type: ignore
        from diffusers import MiniMaxH3Transformer3DModel  # type: ignore

        ids = self.device_ids or tuple(range(min(int(torch.cuda.device_count()), 2)))
        if self._transformer_name is None:
            self._transformer_name = "transformer_ref" if self.workflow == "ref2va" else "transformer"
        transformer_subfolder = "transformer_ref" if self.workflow == "ref2va" else "transformer"
        dispatched, report = load_and_dispatch_h3_transformer(
            MiniMaxH3Transformer3DModel,
            repository=self.repository,
            subfolder=transformer_subfolder,
            dtype=getattr(torch, self.dtype_name),
            device_ids=ids,
            offload_dir=self.offload_dir,
            preferred_layout="h3_two_t4_preferred",
        )
        update_components = getattr(self.rest, "update_components", None)
        if not callable(update_components):
            raise RuntimeError(
                "The installed Diffusers ModularPipeline cannot attach the dispatched H3 transformer"
            )
        update_components(**{self._transformer_name: dispatched})
        self.dispatch_report = report
        self.actual_device_map = {
            "text_encoder": f"cuda:{ids[1]} (released before denoising)",
            "transformer": report["observed"]["reported_module_map"],
            "transformer_blocks": report["observed"]["layers"],
            "video_vae": f"cuda:{ids[0]} (ComponentsManager auto CPU offload)",
            "audio_vae": f"cuda:{ids[0]} (ComponentsManager auto CPU offload)",
            "cpu": "host RAM third tier via ComponentsManager/Accelerate",
            "disk": report["offload_dir"] if report["disk_offload_used"] else "unused",
        }

    def generate(self, request: H3Request, output_path: Path) -> Path:
        if self.conditioner is None or self.rest is None:
            self.load()
        import torch  # type: ignore
        from diffusers.utils.export_utils import encode_video  # type: ignore

        mode = select_mode(request).lower()
        if mode != self.workflow:
            raise ValueError(f"Worker workflow is {self.workflow}, request selected {mode}")
        width, height = dimensions(request)
        frame_info = frame_length(request)
        conditioner_kwargs: dict[str, Any] = {
            "prompt": build_prompt(request, select_mode(request))
        }
        kwargs: dict[str, Any] = {
            "num_frames": frame_info["actual_frames"],
            "width": width,
            "height": height,
            "num_inference_steps": request.steps or {"quick": 4, "medium": 12, "full": 20}[request.quality_mode.lower()],
            "generator": torch.Generator(device=self._primary_device()).manual_seed(int(request.seed)),
            "output": ["videos", "audio", "sampling_rate"],
        }
        if mode == "fl2va":
            from diffusers.utils import load_image  # type: ignore

            if request.first_frame is not None:
                conditioner_kwargs["image"] = load_image(str(request.first_frame))
            if request.last_frame is not None:
                conditioner_kwargs["last_image"] = load_image(str(request.last_frame))
        elif mode == "ref2va":
            from diffusers.modular_pipelines.minimax_h3 import (  # type: ignore
                MiniMaxH3AudioReference,
                MiniMaxH3ImageReference,
                MiniMaxH3VideoReference,
            )

            references: list[Any] = []
            for asset in request.assets():
                if asset.media_type == "image":
                    references.append(MiniMaxH3ImageReference.from_file(str(asset.path)))
                elif asset.media_type == "video":
                    references.append(MiniMaxH3VideoReference.from_file(str(asset.path)))
                else:
                    references.append(MiniMaxH3AudioReference.from_file(str(asset.path)))
            conditioner_kwargs["references"] = references

        state = self.conditioner(**conditioner_kwargs)
        if self.layer_sharded and self.dispatch_report is None:
            self._release_conditioner()
            self._load_dispatched_transformer()
        # The official split passes the conditioner state to the denoising half.
        result = self.rest(state=state, **kwargs)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        encode_video(
            result["videos"][0],
            fps=24,
            output_path=str(output_path),
            audio=result["audio"][0],
            audio_sample_rate=result["sampling_rate"],
        )
        return output_path

    def _primary_device(self) -> str:
        return f"cuda:{self.device_ids[0] if self.device_ids else 0}"


def _enable_cpu_offload(manager: Any, device: str) -> None:
    try:
        manager.enable_auto_cpu_offload(device=device, memory_reserve_margin="14GB")
    except TypeError:
        manager.enable_auto_cpu_offload(device=device)
