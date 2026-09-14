"""V3 registration shim for the global MiniMax H3 conditioning node.

ComfyUI gives legacy ``NODE_CLASS_MAPPINGS`` precedence over
``comfy_entrypoint`` when both are exported by the same module.  The main H3
module intentionally keeps the legacy nodes, so this separate module exposes
the V3 autogrow conditioner through the modern extension contract.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

from comfy_api.latest import ComfyExtension, io


_LEGACY_PATH = Path(__file__).with_name("kaggle_h3_adapters.py").resolve()


def _load_legacy_module() -> Any:
    """Reuse the already-loaded legacy module when ComfyUI loaded it first."""

    for module in tuple(sys.modules.values()):
        if getattr(module, "__file__", None) is None:
            continue
        try:
            if Path(module.__file__).resolve() == _LEGACY_PATH:
                return module
        except (OSError, TypeError, ValueError):
            continue

    spec = importlib.util.spec_from_file_location(
        "kaggle_h3_adapters_shared", _LEGACY_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared H3 node module: {_LEGACY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_LEGACY_MODULE = _load_legacy_module()
KaggleH3Conditioning = getattr(_LEGACY_MODULE, "KaggleH3Conditioning", None)
if KaggleH3Conditioning is None:
    raise ImportError(
        "The installed ComfyUI does not expose comfy_api.latest, so the V3 "
        "Kaggle H3 Conditioning node cannot be registered."
    )


class KaggleH3ConditioningExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [KaggleH3Conditioning]


async def comfy_entrypoint() -> KaggleH3ConditioningExtension:
    return KaggleH3ConditioningExtension()
