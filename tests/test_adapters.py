import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

from kaggle_h3.adapters import (
    AdapterCache,
    H3AdapterError,
    H3AdapterCompatibilityError,
    H3AdapterMetadata,
    apply_adapter_stack,
    build_runtime_config,
    builtin_catalog,
    detect_h3_model_context,
)


class FakeH3Model:
    def __init__(self, *, variant="fl2va", fused_turbo=False):
        self.is_h3 = True
        self.model_variant = variant
        self.filename = f"minimax_h3_{variant}_pruned_int8_convrot.safetensors"
        self.fused_turbo = fused_turbo
        self.applied = []
        self.model_options = {}

    def clone(self):
        copied = FakeH3Model(variant=self.model_variant, fused_turbo=self.fused_turbo)
        copied.applied = list(self.applied)
        copied.model_options = dict(self.model_options)
        return copied

    def get_key_patches(self):
        return {f"patch.{index}": object() for index in range(len(self.applied))}


class AdapterTests(unittest.TestCase):
    def test_official_catalog_has_pinned_turbo_metadata_and_schedules(self):
        catalog = builtin_catalog()
        four = catalog["h3_fl2va_turbo_4step_v1_0_768p"]
        eight = catalog["h3_fl2va_turbo_8step_v1_0_768p"]
        self.assertEqual(len(four.revision), 40)
        self.assertEqual(four.schedule_for(4)["shift_video"], 6.0)
        self.assertEqual(eight.schedule_for(8)["steps"], 8)
        self.assertEqual(eight.schedule_for(8)["sampler_name"], "euler")

    def test_invalid_metadata_rejects_unsafe_or_unsupported_files(self):
        base = {
            "adapter_id": "bad",
            "name": "Bad",
            "repository": "org/repo",
            "filename": "bad.pt",
            "revision": "main",
            "model_variants": ["fl2va"],
            "conditioning_modes": ["T2VA"],
            "supported_steps": [4],
        }
        with self.assertRaises(ValueError):
            H3AdapterMetadata.from_mapping(base)
        base["filename"] = "../bad.safetensors"
        with self.assertRaises(ValueError):
            H3AdapterMetadata.from_mapping(base)

    def test_cache_downloads_once_and_writes_sidecar(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calls = []
            hub = types.ModuleType("huggingface_hub")

            def fake_download(**kwargs):
                calls.append(kwargs)
                path = root / kwargs["filename"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"safe adapter")
                return str(path)

            hub.hf_hub_download = fake_download
            old = sys.modules.get("huggingface_hub")
            sys.modules["huggingface_hub"] = hub
            try:
                metadata = builtin_catalog()["h3_fl2va_turbo_4step_v1_0_768p"]
                cache = AdapterCache(root)
                first = cache.resolve(metadata)
                second = cache.resolve(metadata)
            finally:
                if old is None:
                    sys.modules.pop("huggingface_hub", None)
                else:
                    sys.modules["huggingface_hub"] = old
            self.assertEqual(first, second)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["revision"], metadata.revision)
            self.assertTrue(first.with_name(first.name + ".h3.json").is_file())

    def test_cache_surfaces_actionable_download_error(self):
        with tempfile.TemporaryDirectory() as temp:
            hub = types.ModuleType("huggingface_hub")

            def failing_download(**kwargs):
                raise OSError("offline")

            hub.hf_hub_download = failing_download
            old = sys.modules.get("huggingface_hub")
            sys.modules["huggingface_hub"] = hub
            try:
                metadata = builtin_catalog()["h3_fl2va_turbo_4step_v1_0_768p"]
                with self.assertRaisesRegex(H3AdapterError, "repository=.*lightx2v"):
                    AdapterCache(temp).resolve(metadata)
            finally:
                if old is None:
                    sys.modules.pop("huggingface_hub", None)
                else:
                    sys.modules["huggingface_hub"] = old

    def test_wrong_variant_is_rejected_before_download_or_apply(self):
        model = FakeH3Model(variant="ref2va")
        metadata = builtin_catalog()["h3_fl2va_turbo_4step_v1_0_768p"]
        with self.assertRaises(H3AdapterCompatibilityError):
            build_runtime_config(
                model,
                {metadata.adapter_id: metadata},
                turbo_mode=True,
                turbo_steps=4,
                adapter_slots=((metadata.adapter_id, 1.0), ("None", 1.0), ("None", 1.0)),
            )
        self.assertEqual(model.applied, [])

    def test_fused_turbo_is_reported_and_not_selected_for_injection(self):
        model = FakeH3Model(fused_turbo=True)
        metadata = builtin_catalog()["h3_fl2va_turbo_4step_v1_0_768p"]
        config = build_runtime_config(
            model,
            {metadata.adapter_id: metadata},
            turbo_mode=True,
            turbo_steps=4,
            adapter_slots=((metadata.adapter_id, 1.0), ("None", 1.0), ("None", 1.0)),
        )
        self.assertTrue(config["turbo"]["injection_skipped"])
        self.assertEqual(config["adapters"][0]["status"], "skipped_fused_turbo")
        self.assertEqual(config["applied_order"], [])

    def test_turbo_step_count_is_rejected_by_metadata(self):
        model = FakeH3Model()
        metadata = builtin_catalog()["h3_fl2va_turbo_4step_v1_0_768p"]
        with self.assertRaisesRegex(H3AdapterCompatibilityError, "does not support Turbo 8-step"):
            build_runtime_config(
                model,
                {metadata.adapter_id: metadata},
                turbo_mode=True,
                turbo_steps=8,
                adapter_slots=((metadata.adapter_id, 1.0), ("None", 1.0), ("None", 1.0)),
            )

    def test_duplicate_adapter_selection_is_rejected(self):
        model = FakeH3Model()
        metadata = H3AdapterMetadata.from_mapping(
            {
                "adapter_id": "motion",
                "name": "Motion",
                "repository": "example/motion",
                "filename": "motion.safetensors",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "model_variants": ["fl2va"],
                "conditioning_modes": ["T2VA", "FL2VA"],
                "supported_steps": [4],
            }
        )
        with self.assertRaisesRegex(H3AdapterCompatibilityError, "more than once"):
            build_runtime_config(
                model,
                {metadata.adapter_id: metadata},
                turbo_mode=False,
                turbo_steps=4,
                adapter_slots=((metadata.adapter_id, 1.0), (metadata.adapter_id, 0.5), ("None", 1.0)),
            )

    def test_adapter_order_and_strength_are_preserved(self):
        model = FakeH3Model()
        first = H3AdapterMetadata.from_mapping(
            {
                "adapter_id": "first_motion",
                "name": "First motion",
                "repository": "example/first",
                "filename": "first.safetensors",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "model_variants": ["fl2va"],
                "conditioning_modes": ["T2VA", "FL2VA"],
                "supported_steps": [4, 8],
                "kind": "motion",
            }
        )
        second = H3AdapterMetadata.from_mapping(
            {
                "adapter_id": "second_motion",
                "name": "Second motion",
                "repository": "example/second",
                "filename": "second.safetensors",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "model_variants": ["fl2va"],
                "conditioning_modes": ["T2VA", "FL2VA"],
                "supported_steps": [4, 8],
                "kind": "motion",
            }
        )
        config = build_runtime_config(
            model,
            {first.adapter_id: first, second.adapter_id: second},
            turbo_mode=False,
            turbo_steps=4,
            adapter_slots=((first.adapter_id, 0.5), (second.adapter_id, 1.25), ("None", 1.0)),
        )
        self.assertEqual(config["applied_order"], [first.adapter_id, second.adapter_id])
        self.assertEqual([item["strength"] for item in config["adapters"]], [0.5, 1.25])

    def test_comfy_patch_api_is_called_in_order_without_mutating_input(self):
        model = FakeH3Model()
        first = H3AdapterMetadata.from_mapping(
            {
                "adapter_id": "first",
                "name": "First",
                "repository": "example/first",
                "filename": "first.safetensors",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "model_variants": ["fl2va"],
                "conditioning_modes": ["T2VA", "FL2VA"],
                "supported_steps": [4],
            }
        )
        second = H3AdapterMetadata.from_mapping(
            {
                "adapter_id": "second",
                "name": "Second",
                "repository": "example/second",
                "filename": "second.safetensors",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "model_variants": ["fl2va"],
                "conditioning_modes": ["T2VA", "FL2VA"],
                "supported_steps": [4],
            }
        )
        config = build_runtime_config(
            model,
            {first.adapter_id: first, second.adapter_id: second},
            turbo_mode=False,
            turbo_steps=4,
            adapter_slots=((first.adapter_id, 0.25), (second.adapter_id, 1.5), ("None", 1.0)),
        )
        with tempfile.TemporaryDirectory() as temp:
            cache = AdapterCache(temp)
            for metadata in (first, second):
                path = Path(temp) / metadata.filename
                path.write_bytes(metadata.adapter_id.encode())

            comfy = types.ModuleType("comfy")
            comfy.__path__ = []
            comfy_sd = types.ModuleType("comfy.sd")
            comfy_utils = types.ModuleType("comfy.utils")

            def load_torch_file(path, **kwargs):
                return ({"lora": Path(path).read_bytes()}, {"source": "test"})

            def load_lora_for_models(current, clip, state, strength_model, strength_clip, **kwargs):
                copied = current.clone()
                copied.applied.append((state["lora"].decode(), strength_model))
                return copied, None

            comfy_utils.load_torch_file = load_torch_file
            comfy_sd.load_lora_for_models = load_lora_for_models
            comfy.sd = comfy_sd
            comfy.utils = comfy_utils
            old = {name: sys.modules.get(name) for name in ("comfy", "comfy.sd", "comfy.utils")}
            sys.modules.update({"comfy": comfy, "comfy.sd": comfy_sd, "comfy.utils": comfy_utils})
            try:
                patched, applied_config = apply_adapter_stack(
                    model, config, cache, {first.adapter_id: first, second.adapter_id: second}
                )
            finally:
                for name, previous in old.items():
                    if previous is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = previous
        self.assertEqual(model.applied, [])
        self.assertEqual(
            patched.applied,
            [("first", 0.25), ("second", 1.5)],
        )
        self.assertEqual(applied_config["applied_order"], ["first", "second"])

    def test_context_reports_unknown_model_as_not_h3(self):
        context = detect_h3_model_context(object())
        self.assertFalse(context.is_h3)
        self.assertEqual(context.model_variant, "unknown")


if __name__ == "__main__":
    unittest.main()
