"""Reusable MiniMax H3 + ComfyUI helpers for a Kaggle T4 notebook."""

from .pipeline import H3Request, run_generation, run_quick_validation
from .adapters import AdapterCache, H3AdapterMetadata, load_catalog
from .layer_sharding import (
    H3LayerShardingError,
    find_dispatchable_layers,
    plan_layer_device_map,
)
from .workflow import WORKFLOW_REGISTRY, build_workflow, select_mode

__all__ = [
    "H3Request",
    "WORKFLOW_REGISTRY",
    "build_workflow",
    "run_generation",
    "run_quick_validation",
    "select_mode",
    "AdapterCache",
    "H3AdapterMetadata",
    "load_catalog",
    "H3LayerShardingError",
    "find_dispatchable_layers",
    "plan_layer_device_map",
]
