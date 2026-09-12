import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kaggle_h3.workflow import H3Request, build_workflow


ROOT = Path(__file__).parents[1]


def load_node_module():
    path = ROOT / "custom_nodes" / "kaggle_h3_adapters.py"
    spec = importlib.util.spec_from_file_location("kaggle_h3_adapters_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class ComfyNodeTests(unittest.TestCase):
    def test_standalone_copy_imports_without_package_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            node_dir = root / "custom_nodes"
            node_dir.mkdir()
            shutil.copy2(ROOT / "custom_nodes" / "kaggle_h3_adapters.py", node_dir / "kaggle_h3_adapters.py")
            shutil.copy2(ROOT / "src" / "kaggle_h3" / "adapters.py", node_dir / "kaggle_h3_adapter_core.py")
            shutil.copy2(ROOT / "src" / "kaggle_h3" / "phase_runtime.py", node_dir / "kaggle_h3_phase_runtime.py")
            shutil.copy2(ROOT / "src" / "kaggle_h3" / "layer_sharding.py", node_dir / "kaggle_h3_layer_sharding.py")
            shutil.copy2(ROOT / "custom_nodes" / "kaggle_h3_adapter_catalog.json", node_dir / "kaggle_h3_adapter_catalog.json")
            script = """
import importlib.util
import sys
from pathlib import Path

node_path = Path(sys.argv[1]) / "kaggle_h3_adapters.py"
spec = importlib.util.spec_from_file_location("standalone_h3_node", node_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert set(module.NODE_CLASS_MAPPINGS) == {
    "KaggleH3ShardedDiffusionLoader",
    "KaggleH3TextEncoderLoader",
    "KaggleH3VAELoader",
    "KaggleH3PhaseDispatch",
    "KaggleH3AdapterStack",
    "KaggleH3TurboSampler",
    "KaggleH3VAEDecode",
    "KaggleH3AudioVAEDecode",
}
"""
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            subprocess.run(
                [sys.executable, "-c", script, str(node_dir)],
                cwd=temporary,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )

    def test_registration_and_interface_are_present(self):
        module = load_node_module()
        self.assertIn("KaggleH3AdapterStack", module.NODE_CLASS_MAPPINGS)
        self.assertIn("KaggleH3ShardedDiffusionLoader", module.NODE_CLASS_MAPPINGS)
        self.assertIn("KaggleH3TextEncoderLoader", module.NODE_CLASS_MAPPINGS)
        self.assertIn("KaggleH3VAELoader", module.NODE_CLASS_MAPPINGS)
        self.assertIn("KaggleH3PhaseDispatch", module.NODE_CLASS_MAPPINGS)
        self.assertIn("KaggleH3TurboSampler", module.NODE_CLASS_MAPPINGS)
        self.assertIn("KaggleH3VAEDecode", module.NODE_CLASS_MAPPINGS)
        self.assertIn("KaggleH3AudioVAEDecode", module.NODE_CLASS_MAPPINGS)
        inputs = module.H3AdapterStack.INPUT_TYPES()
        for name in ("adapter_1", "strength_1", "adapter_2", "strength_2", "adapter_3", "strength_3"):
            self.assertIn(name, inputs["required"])
        self.assertEqual(inputs["required"]["turbo_steps"][0], ["4", "8"])
        self.assertEqual(module.H3AdapterStack.RETURN_TYPES, ("MODEL", "H3_RUNTIME_CONFIG"))
        self.assertEqual(module.H3TurboSampler.RETURN_TYPES, ("LATENT", "LATENT", "LATENT"))
        self.assertEqual(
            module.H3TurboSampler.RETURN_NAMES,
            ("video_latent", "audio_latent", "denoised_output"),
        )
        self.assertEqual(module.H3VAEDecode.INPUT_TYPES()["required"]["device_id"][1]["default"], 1)
        self.assertEqual(module.H3AudioVAEDecode.INPUT_TYPES()["required"]["device_id"][1]["default"], 0)

    def test_explicit_loader_interfaces_have_phase_targets(self):
        module = load_node_module()
        self.assertEqual(
            module.KaggleH3ShardedDiffusionLoader.INPUT_TYPES()["required"]["gpu_1"][1]["default"],
            1,
        )
        self.assertEqual(
            module.KaggleH3TextEncoderLoader.INPUT_TYPES()["required"]["device_id"][1]["default"],
            0,
        )
        self.assertEqual(
            module.KaggleH3VAELoader.INPUT_TYPES()["required"]["role"][0],
            ["video", "audio"],
        )
        self.assertEqual(
            module.KaggleH3PhaseDispatch.RETURN_TYPES,
            ("MODEL", "CONDITIONING", "LATENT", "H3_RUNTIME_CONFIG"),
        )

    def test_h3_video_decode_reconciles_latent_dtype_without_mutating_input(self):
        module = load_node_module()

        class Projection:
            weight = type("Parameter", (), {"dtype": "float16"})()

        class Decoder:
            x_embedder = Projection()

        class MiniMaxH3VideoVAE:
            decoder = Decoder()

            def parameters(self):
                # The wrapper's first parameter is FP32, while the projection
                # named by the real ComfyUI traceback is FP16.
                return iter([type("Parameter", (), {"dtype": "float32"})()])

        class FakeLatent:
            dtype = "float32"
            is_nested = False

            def to(self, *, dtype):
                converted = type(self)()
                converted.dtype = dtype
                return converted

        vae = type("VAE", (), {"first_stage_model": MiniMaxH3VideoVAE()})()
        latent = FakeLatent()
        samples = {"samples": latent, "batch_index": [0]}
        converted = module._coerce_h3_video_samples(vae, samples)
        self.assertIsNot(converted, samples)
        self.assertIs(converted["batch_index"], samples["batch_index"])
        self.assertEqual(converted["samples"].dtype, "float16")
        self.assertEqual(samples["samples"].dtype, "float32")

    def test_h3_video_decode_enables_comfyui_layer_casting(self):
        module = load_node_module()

        class Layer:
            comfy_cast_weights = False

        class Decoder:
            x_embedder = Layer()

        class MiniMaxH3VideoVAE:
            decoder = Decoder()

            def decode_temporal(self, _samples):
                return None

            def modules(self):
                return iter((self, self.decoder.x_embedder))

        model = MiniMaxH3VideoVAE()
        vae = type("VAE", (), {"first_stage_model": model})()
        changed = module._enable_h3_decode_weight_casting(vae)
        self.assertEqual(changed, 1)
        self.assertTrue(model.decoder.x_embedder.comfy_cast_weights)

    def test_h3_consumer_routing_splits_nested_video_and_audio_streams(self):
        module = load_node_module()
        import torch

        video = torch.zeros((1, 24, 2, 22, 38))
        audio = torch.zeros((1, 32, 2, 248))

        class Packed:
            is_nested = True

            def unbind(self):
                return video, audio

        samples = {"samples": Packed()}
        self.assertIs(module._h3_stream_tensor(samples, "video"), video)
        self.assertIs(module._h3_stream_tensor(samples, "audio"), audio)

    def test_h3_audio_codec_boundary_replaces_nonfinite_samples_and_moves_to_cpu(self):
        module = load_node_module()
        import torch

        audio = module._h3_finite_audio(
            {
                "waveform": torch.tensor([[[0.25, float("nan"), 2.0, float("-inf")]]]),
                "sample_rate": 32000,
            }
        )
        self.assertEqual(audio["waveform"].device.type, "cpu")
        self.assertTrue(torch.isfinite(audio["waveform"]).all())
        self.assertLessEqual(float(audio["waveform"].abs().max()), 1.0)

    def test_turbo_workflow_routes_both_outputs_into_dedicated_sampler(self):
        request = H3Request(
            "Turbo smoke",
            turbo_mode=True,
            turbo_steps=4,
            adapter_1="h3_fl2va_turbo_4step_v1_0_768p",
            adapter_1_strength=1.0,
            quality_mode="quick",
        )
        workflow = build_workflow(request)
        self.assertEqual(workflow["h3_adapter"]["class_type"], "KaggleH3AdapterStack")
        self.assertEqual(workflow["unet"]["class_type"], "KaggleH3ShardedDiffusionLoader")
        self.assertEqual(workflow["clip"]["class_type"], "KaggleH3TextEncoderLoader")
        self.assertEqual(workflow["phase"]["class_type"], "KaggleH3PhaseDispatch")
        self.assertEqual(workflow["sample"]["class_type"], "KaggleH3TurboSampler")
        self.assertEqual(workflow["sample"]["inputs"]["runtime_config"], ["phase", 3])
        self.assertEqual(workflow["sample"]["inputs"]["sampler_name"], "euler")
        self.assertNotIn("SamplerCustomAdvanced", {node["class_type"] for node in workflow.values()})

    def test_dedicated_sampler_consumes_selected_turbo_steps(self):
        module = load_node_module()
        config = {
            "schema_version": "kaggle_h3.runtime.v1",
            "turbo": {"enabled": True, "steps": 8},
            "sampler": {"steps": 8, "sampler_name": "euler", "scheduler": "simple"},
        }
        _, steps = module.H3TurboSampler._resolve_sampling_parameters(config, "euler")
        self.assertEqual(steps, 8)
        with self.assertRaisesRegex(RuntimeError, "requires sampler"):
            module.H3TurboSampler._resolve_sampling_parameters(config, "res_multistep")

    def test_phase_dispatch_hook_runs_after_comfy_managed_load(self):
        module = load_node_module()
        events = []

        class SamplerHelpers:
            def prepare_sampling(self, model, *args, **kwargs):
                events.append(("comfy_load", model))
                return "prepared"

        helpers = SamplerHelpers()
        holder = {}
        original_begin = module.begin_transformer_phase
        module.begin_transformer_phase = lambda model, config: (
            events.append(("h3_dispatch", model, config)) or "phase"
        )
        try:
            original_prepare = module._install_phase_dispatch_hook(
                helpers, {"strategy": "phase_aware_h3"}, holder
            )
            self.assertEqual(helpers.prepare_sampling("patcher"), "prepared")
            self.assertEqual(holder["state"], "phase")
            self.assertEqual(
                [event[0] for event in events], ["comfy_load", "h3_dispatch"]
            )
            helpers.prepare_sampling = original_prepare
            self.assertIs(helpers.prepare_sampling.__func__, original_prepare.__func__)
        finally:
            module.begin_transformer_phase = original_begin

    def test_catalog_is_valid_json_and_contains_extensible_metadata(self):
        catalog = json.loads((ROOT / "custom_nodes" / "kaggle_h3_adapter_catalog.json").read_text())
        self.assertEqual(catalog["schema_version"], "kaggle_h3.adapters.v1")
        required = {"name", "repository", "filename", "revision", "model_variants", "conditioning_modes", "supported_steps", "license", "attribution"}
        self.assertTrue(required.issubset(catalog["adapters"][0]))


if __name__ == "__main__":
    unittest.main()
