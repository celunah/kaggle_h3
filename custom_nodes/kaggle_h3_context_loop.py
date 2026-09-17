"""ComfyUI registration shim for the separate Context Loop dependency.

The implementation remains in the Kaggle H3 adapter module so it shares the
existing sampler and runtime contracts. The external Context Loop package is
installed separately by the bootstrap and owns all scene/checkpoint/review
nodes; this file only registers the thin generation-edge compatibility node.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_LEGACY_PATH = Path(__file__).with_name("kaggle_h3_adapters.py").resolve()


def _load_legacy_module():
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


KaggleH3ContextLoopSampler = _load_legacy_module().KaggleH3ContextLoopSampler


NODE_CLASS_MAPPINGS = {
    "KaggleH3ContextLoopSampler": KaggleH3ContextLoopSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KaggleH3ContextLoopSampler": (
        "Kaggle H3 | Context Loop Sampler (Singularity Ref2VA)"
    ),
}
