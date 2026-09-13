import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

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
            shutil.copy2(ROOT / "src" / "kaggle_h3" / "ref2va.py", node_dir / "kaggle_h3_ref2va.py")
            shutil.copy2(ROOT / "src" / "kaggle_h3" / "fp_diagnostics.py", node_dir / "kaggle_h3_fp_diagnostics.py")
            shutil.copy2(ROOT / "src" / "kaggle_h3" / "model_manager.py", node_dir / "kaggle_h3_model_manager.py")
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
    "KaggleH3SmokeReference",
    "KaggleH3Ref2VAConditioning",
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
        self.assertIn("KaggleH3Ref2VAConditioning", module.NODE_CLASS_MAPPINGS)
        self.assertIn("KaggleH3SmokeReference", module.NODE_CLASS_MAPPINGS)
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
        sampler_inputs = module.H3TurboSampler.INPUT_TYPES()["required"]
        self.assertEqual(sampler_inputs["synchronize_mode"][0], ["full", "safe", "fast"])
        self.assertEqual(sampler_inputs["synchronize_mode"][1]["default"], "full")
        self.assertEqual(
            module.H3TurboSampler.RETURN_NAMES,
            ("video_latent", "audio_latent", "denoised_output"),
        )
        self.assertEqual(module.H3VAEDecode.INPUT_TYPES()["required"]["device_id"][1]["default"], 1)
        self.assertEqual(module.H3AudioVAEDecode.INPUT_TYPES()["required"]["device_id"][1]["default"], 0)
        ref2va_inputs = module.KaggleH3Ref2VAConditioning.INPUT_TYPES()
        self.assertEqual(ref2va_inputs["required"]["size_preset"][0], ["240p", "360p", "480p", "720p"])
        self.assertEqual(ref2va_inputs["required"]["aspect_ratio"][0], ["16:9", "4:3"])
        self.assertAlmostEqual(ref2va_inputs["required"]["seconds"][1]["min"], 39 / 24)
        self.assertIn("ref_image_8", ref2va_inputs["optional"])
        self.assertIn("ref_video_audio_2", ref2va_inputs["optional"])

    def test_ref2va_conditioning_translates_controls_and_preserves_slot_names(self):
        module = load_node_module()
        captured = {}

        class Result:
            result = ("positive", "latent")

        class NativeRef2VA:
            @classmethod
            def execute(cls, **kwargs):
                captured.update(kwargs)
                return Result()

        original_loader = module._native_ref2va_conditioning
        module._native_ref2va_conditioning = lambda: NativeRef2VA
        try:
            result = module.KaggleH3Ref2VAConditioning().condition(
                clip="clip",
                vae="video_vae",
                audio_vae="audio_vae",
                prompt="animate the reference",
                seconds=5.0,
                size_preset="720p",
                aspect_ratio="16:9",
                ref_image_1="image-one",
                ref_audio_0="audio-one",
            )
        finally:
            module._native_ref2va_conditioning = original_loader

        self.assertEqual(result, ("positive", "latent"))
        self.assertEqual((captured["width"], captured["height"]), (1280, 704))
        self.assertEqual(captured["length"], 124)
        self.assertEqual(captured["ref_images"], {"ref_image_1": "image-one"})
        self.assertEqual(captured["ref_audios"], {"ref_audio_0": "audio-one"})
        with self.assertRaisesRegex(ValueError, "matching ref_video_N"):
            module.KaggleH3Ref2VAConditioning().condition(
                clip="clip",
                vae="video_vae",
                audio_vae="audio_vae",
                prompt="animate the reference",
                seconds=5.0,
                size_preset="360p",
                aspect_ratio="16:9",
                ref_image_0="image-one",
                ref_video_audio_1="orphan-audio",
            )

    def test_explicit_loader_interfaces_have_phase_targets(self):
        module = load_node_module()
        loader_inputs = module.KaggleH3ShardedDiffusionLoader.INPUT_TYPES()["required"]
        self.assertEqual(loader_inputs["model_variant"][0], ["FL2VA", "Ref2VA"])
        self.assertEqual(loader_inputs["model_variant"][1]["default"], "Ref2VA")
        self.assertEqual(
            loader_inputs["precision"][0], ["auto", "fp8_scaled", "int8_convrot"]
        )
        self.assertEqual(loader_inputs["precision"][1]["default"], "int8_convrot")
        self.assertEqual(
            loader_inputs["gpu_1"][1]["default"],
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

    def test_smoke_reference_loader_resolves_checked_in_assets(self):
        module = load_node_module()
        inputs = module.KaggleH3SmokeReference.INPUT_TYPES()["required"]
        self.assertEqual(inputs["reference"][0], ["character", "scene"])
        self.assertEqual(
            module._resolve_h3_smoke_reference("character").name,
            "CHARACTER_REFERENCE.png",
        )
        self.assertEqual(
            module._resolve_h3_smoke_reference("scene").name,
            "SCENE_REFERENCE.png",
        )

        with self.assertRaisesRegex(module.H3PhaseError, "choose character or scene"):
            module._resolve_h3_smoke_reference("unknown")

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

    def test_video_vae_validates_before_and_after_decode(self):
        module = load_node_module()
        import torch
        from unittest.mock import patch

        class VAE:
            bad_output = False

            def decode(self, samples):
                self.received = samples
                value = float("nan") if self.bad_output else 0.0
                return torch.full((1, 2, 2, 3), value, dtype=torch.float32)

        vae = VAE()
        with patch.object(module, "configure_vae_phase"), patch.object(
            module, "release_vae_phase"
        ), patch.object(module, "_enable_h3_decode_weight_casting"), patch.object(
            module, "_h3_move_for_consumer", side_effect=lambda tensor, **_: tensor
        ):
            result = module.H3VAEDecode().decode(
                vae,
                {"samples": torch.zeros((1, 4, 2, 2), dtype=torch.float32)},
                device_id=0,
            )
        self.assertEqual(result[0].shape, (1, 2, 2, 3))

        vae.bad_output = True
        with patch.object(module, "configure_vae_phase"), patch.object(
            module, "release_vae_phase"
        ), patch.object(module, "_enable_h3_decode_weight_casting"), patch.object(
            module, "_h3_move_for_consumer", side_effect=lambda tensor, **_: tensor
        ):
            with patch.dict(os.environ, {"KAGGLE_H3_FP_DIAGNOSTICS": "1"}), self.assertRaisesRegex(
                FloatingPointError, "stage 'vae_video_out'"
            ):
                module.H3VAEDecode().decode(
                    vae,
                    {"samples": torch.zeros((1, 4, 2, 2), dtype=torch.float32)},
                    device_id=0,
                )

        with patch.object(module, "configure_vae_phase"), patch.object(
            module, "release_vae_phase"
        ), patch.object(module, "_enable_h3_decode_weight_casting"), patch.object(
            module, "_h3_move_for_consumer", side_effect=lambda tensor, **_: tensor
        ):
            with patch.dict(os.environ, {"KAGGLE_H3_FP_DIAGNOSTICS": "1"}), self.assertRaisesRegex(
                FloatingPointError, "stage 'vae_video_in'"
            ):
                module.H3VAEDecode().decode(
                    vae,
                    {"samples": torch.tensor([[[[float("nan")]]]])},
                    device_id=0,
                )

    def test_audio_vae_validates_before_and_after_decode_before_sanitization(self):
        module = load_node_module()
        import sys
        import types
        import torch
        from unittest.mock import patch

        audio_module = types.ModuleType("comfy_extras.nodes_audio")
        audio_module.vae_decode_audio = lambda _vae, _samples: {
            "waveform": torch.tensor([[[float("inf")]]]),
            "sample_rate": 32000,
        }
        comfy_extras = types.ModuleType("comfy_extras")
        comfy_extras.__path__ = []
        with patch.dict(
            sys.modules,
            {"comfy_extras": comfy_extras, "comfy_extras.nodes_audio": audio_module},
        ), patch.object(module, "configure_vae_phase"), patch.object(
            module, "release_vae_phase"
        ), patch.object(module, "_h3_move_for_consumer", side_effect=lambda tensor, **_: tensor), patch.object(
            module, "_h3_finite_audio"
        ) as sanitize:
            with patch.dict(os.environ, {"KAGGLE_H3_FP_DIAGNOSTICS": "1"}), self.assertRaisesRegex(
                FloatingPointError, "stage 'vae_audio_out'"
            ):
                module.H3AudioVAEDecode().decode(
                    object(),
                    {
                        "samples": torch.zeros((1, 4, 2, 2), dtype=torch.float32),
                        "_kaggle_h3_stream": "audio",
                    },
                    device_id=0,
                )
        sanitize.assert_not_called()

        with patch.dict(
            sys.modules,
            {"comfy_extras": comfy_extras, "comfy_extras.nodes_audio": audio_module},
        ), patch.object(module, "configure_vae_phase"), patch.object(
            module, "release_vae_phase"
        ), patch.object(module, "_h3_move_for_consumer", side_effect=lambda tensor, **_: tensor), patch.object(
            module, "_h3_finite_audio"
        ) as sanitize:
            with patch.dict(os.environ, {"KAGGLE_H3_FP_DIAGNOSTICS": "1"}), self.assertRaisesRegex(
                FloatingPointError, "stage 'vae_audio_in'"
            ):
                module.H3AudioVAEDecode().decode(
                    object(),
                    {
                        "samples": torch.tensor([[[[float("nan")]]]]),
                        "_kaggle_h3_stream": "audio",
                    },
                    device_id=0,
                )
        sanitize.assert_not_called()

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

    def test_synchronization_modes_select_sampler_boundary_devices(self):
        module = load_node_module()
        with patch.object(module, "phase_device_ids", return_value=(0, 1)):
            self.assertEqual(module._h3_sampler_boundary_devices("full"), (0, 1))
            self.assertEqual(module._h3_sampler_boundary_devices("safe"), (0,))
            self.assertEqual(module._h3_sampler_boundary_devices("fast"), ())

    def test_base_res_multistep_is_wrapped_but_turbo_euler_is_untouched(self):
        module = load_node_module()

        euler_sampler = types.SimpleNamespace(sampler_function=object())
        original_euler_function = euler_sampler.sampler_function
        self.assertIs(
            module._h3_prepare_sampler("euler", euler_sampler),
            euler_sampler,
        )
        self.assertIs(euler_sampler.sampler_function, original_euler_function)

        res_sampler = types.SimpleNamespace(sampler_function=lambda *args: None)
        self.assertIs(
            module._h3_prepare_sampler("res_multistep", res_sampler),
            res_sampler,
        )
        self.assertIs(res_sampler.sampler_function, module._h3_res_multistep)

        with self.assertRaisesRegex(RuntimeError, "could not access"):
            module._h3_prepare_sampler("res_multistep", types.SimpleNamespace())

    def test_base_h3_rejects_euler(self):
        module = load_node_module()
        with self.assertRaisesRegex(RuntimeError, "reserved for Turbo"):
            module.H3TurboSampler._resolve_sampling_parameters(None, "euler")

    def test_res_multistep_history_isolated_from_reused_model_output_buffer(self):
        module = load_node_module()
        import torch
        from unittest.mock import patch

        sampling = types.ModuleType("comfy.k_diffusion.sampling")
        sampling.default_noise_sampler = lambda _x, seed=None: (
            lambda _sigma, _sigma_next: torch.zeros((1, 1))
        )
        sampling.get_ancestral_step = lambda sigma_from, sigma_to, eta: (
            sigma_to,
            0.0,
        )
        sampling.to_d = lambda x, sigma, denoised: (x - denoised) / sigma
        k_diffusion = types.ModuleType("comfy.k_diffusion")
        k_diffusion.__path__ = []
        k_diffusion.sampling = sampling
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        comfy.k_diffusion = k_diffusion

        class ModelSampling:
            noise_scale = 1.0

        class Patcher:
            def get_model_object(self, _name):
                return ModelSampling()

        class InnerModel:
            model_patcher = Patcher()

        class Model:
            inner_model = InnerModel()

            def __init__(self, reuse_buffer):
                self.reuse_buffer = reuse_buffer
                self.calls = 0

            def __call__(self, x, _sigma, **_kwargs):
                if self.calls == 0:
                    value = torch.tensor([[0.5]])
                    if self.reuse_buffer:
                        self.buffer = value
                else:
                    value = torch.tensor([[0.25]])
                    if self.reuse_buffer:
                        self.buffer.copy_(value)
                        value = self.buffer
                self.calls += 1
                return value

        with patch.dict(
            sys.modules,
            {
                "comfy": comfy,
                "comfy.k_diffusion": k_diffusion,
                "comfy.k_diffusion.sampling": sampling,
            },
        ), patch.object(module, "phase_device_ids", return_value=()), patch.object(
            module, "synchronize_h3_devices"
        ) as synchronize:
            sigmas = torch.tensor([2.0, 1.0, 0.0])
            expected = module._h3_res_multistep(
                Model(reuse_buffer=False),
                torch.zeros((1, 1)),
                sigmas,
            )
            actual = module._h3_res_multistep(
                Model(reuse_buffer=True),
                torch.zeros((1, 1)),
                sigmas,
            )

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(synchronize.call_count, 4)

    def test_repeated_seed_sampler_and_vae_outputs_remain_finite_and_stable(self):
        module = load_node_module()
        import torch
        from unittest.mock import patch

        class SamplingModel:
            def get_model_object(self, _name):
                return object()

        sampling_model = SamplingModel()

        class Guider:
            def __init__(self, _model):
                self.model_patcher = types.SimpleNamespace(
                    model=types.SimpleNamespace(process_latent_out=lambda value: value)
                )

            def set_conds(self, _conditioning):
                return None

            def sample(self, _noise, _latent, _sampler, _sigmas, **kwargs):
                generator = torch.Generator().manual_seed(int(kwargs["seed"]))
                video = torch.randn((1, 2, 3), generator=generator)
                audio = torch.randn((1, 2, 3), generator=generator)
                return torch.nested.nested_tensor([video, audio])

        class Noise:
            def __init__(self, _seed):
                pass

            def generate_noise(self, latent):
                return latent["samples"]

        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        comfy_sample = types.ModuleType("comfy.sample")
        comfy_sample.fix_empty_latent_channels = lambda _model, value, *_args: value
        comfy_samplers = types.ModuleType("comfy.samplers")
        comfy_samplers.calculate_sigmas = lambda *_args: torch.ones((3,))
        comfy_samplers.sampler_object = lambda _name: types.SimpleNamespace(
            sampler_function=lambda *args, **kwargs: None
        )
        comfy_helpers = types.ModuleType("comfy.sampler_helpers")
        comfy_helpers.prepare_sampling = lambda *_args, **_kwargs: None
        comfy_management = types.ModuleType("comfy.model_management")
        comfy_utils = types.ModuleType("comfy.utils")
        comfy_utils.PROGRESS_BAR_ENABLED = False
        comfy_extra = types.ModuleType("comfy_extras")
        comfy_extra.__path__ = []
        comfy_sampler_nodes = types.ModuleType("comfy_extras.nodes_custom_sampler")
        comfy_sampler_nodes.Guider_Basic = Guider
        comfy_sampler_nodes.Noise_RandomNoise = Noise
        comfy_audio_nodes = types.ModuleType("comfy_extras.nodes_audio")
        comfy_audio_nodes.vae_decode_audio = lambda _vae, _samples: {
            "waveform": torch.tensor([[[0.25, -0.5]]]),
            "sample_rate": 32000,
        }
        latent_preview = types.ModuleType("latent_preview")
        latent_preview.prepare_callback = lambda *_args: None
        modules = {
            "comfy": comfy,
            "comfy.sample": comfy_sample,
            "comfy.samplers": comfy_samplers,
            "comfy.sampler_helpers": comfy_helpers,
            "comfy.model_management": comfy_management,
            "comfy.utils": comfy_utils,
            "comfy_extras": comfy_extra,
            "comfy_extras.nodes_custom_sampler": comfy_sampler_nodes,
            "comfy_extras.nodes_audio": comfy_audio_nodes,
            "latent_preview": latent_preview,
        }
        for name, child in modules.items():
            if name.startswith("comfy."):
                setattr(comfy, name.split(".", 1)[1], child)
            if name.startswith("comfy_extras."):
                setattr(comfy_extra, name.split(".", 1)[1], child)

        conditioning = [[torch.ones((1,)), {"minimax_refs": True}]]
        latent = {"samples": torch.zeros((1, 2, 3))}
        with patch.dict(sys.modules, modules), patch.object(
            module.H3TurboSampler,
            "_shift_model",
            return_value=sampling_model,
        ), patch.object(
            module,
            "_install_phase_dispatch_hook",
            side_effect=lambda helpers, _config, _holder, **_kwargs: helpers.prepare_sampling,
        ), patch.object(module, "phase_device_ids", return_value=()), patch.object(
            module, "_h3_move_for_consumer", side_effect=lambda value, **_kwargs: value
        ), patch.object(module, "release_transformer_phase"), patch.object(
            module, "release_h3_runtime_resources"
        ), patch.object(module, "configure_vae_phase"), patch.object(
            module, "release_vae_phase"
        ):
            first = module.H3TurboSampler().sample(
                object(), None, conditioning, latent, 1234, "res_multistep", 2
            )
            second = module.H3TurboSampler().sample(
                object(), None, conditioning, latent, 1234, "res_multistep", 2
            )
            video_vae = types.SimpleNamespace(
                decode=lambda _samples: torch.ones((1, 2, 2, 3))
            )
            first_video = module.H3VAEDecode().decode(video_vae, first[0], device_id=1)
            second_video = module.H3VAEDecode().decode(video_vae, second[0], device_id=1)
            first_audio = module.H3AudioVAEDecode().decode(
                object(), first[1], device_id=0
            )
            second_audio = module.H3AudioVAEDecode().decode(
                object(), second[1], device_id=0
            )

        self.assertTrue(torch.equal(first[0]["samples"], second[0]["samples"]))
        self.assertTrue(torch.equal(first[1]["samples"], second[1]["samples"]))
        self.assertTrue(torch.isfinite(first_video[0]).all())
        self.assertTrue(torch.isfinite(second_video[0]).all())
        self.assertTrue(torch.isfinite(first_audio[0]["waveform"]).all())
        self.assertTrue(torch.isfinite(second_audio[0]["waveform"]).all())
        self.assertTrue(torch.equal(first_video[0], second_video[0]))
        self.assertTrue(torch.equal(first_audio[0]["waveform"], second_audio[0]["waveform"]))

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
        module.begin_transformer_phase = lambda model, config, **_kwargs: (
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
