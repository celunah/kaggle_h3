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
            shutil.copy2(ROOT / "src" / "kaggle_h3" / "diffusion_telemetry.py", node_dir / "kaggle_h3_diffusion_telemetry.py")
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

    def test_global_conditioning_routes_ref2va_autogrow_inputs_to_native(self):
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
            result = module._execute_global_h3_conditioning(
                clip="clip",
                vae="video_vae",
                audio_vae="audio_vae",
                prompt="preserve the subject",
                mode="Ref2VA",
                width=640,
                height=352,
                length=124,
                ref_image_size="match",
                ref_images={
                    "ref_image_0": "character",
                    "ref_image_1": None,
                },
                ref_videos={"ref_video_0": "motion"},
                ref_video_audios={"ref_video_audio_0": "motion-audio"},
                ref_audios={"ref_audio_0": "voice"},
            )
        finally:
            module._native_ref2va_conditioning = original_loader

        self.assertEqual(result, ("positive", "latent"))
        self.assertEqual(captured["ref_images"], {"ref_image_0": "character"})
        self.assertEqual(captured["ref_videos"], {"ref_video_0": "motion"})
        self.assertEqual(
            captured["ref_video_audios"],
            {"ref_video_audio_0": "motion-audio"},
        )
        self.assertEqual(captured["ref_audios"], {"ref_audio_0": "voice"})

    def test_inline_reference_rows_load_in_list_order_and_cache_duplicates(self):
        module = load_node_module()
        loaded = []
        original_loader = module._load_h3_inline_file

        def fake_loader(kind, filename):
            loaded.append((kind, filename))
            if kind == "video":
                return (f"frames:{filename}", f"audio:{filename}")
            return (f"{kind}:{filename}", None)

        module._load_h3_inline_file = fake_loader
        try:
            result = module._resolve_h3_inline_references(
                [
                    {"reference_0": "image", "file": "character.png"},
                    {"reference_1": "video", "file": "motion.mp4"},
                    {"reference_2": "audio", "file": "voice.wav"},
                    {"reference_3": "image", "file": "character.png"},
                    {"reference_4": "none", "file": "None"},
                ]
            )
        finally:
            module._load_h3_inline_file = original_loader

        self.assertEqual(
            result,
            (
                {"ref_image_0": "image:character.png", "ref_image_1": "image:character.png"},
                {"ref_video_0": "frames:motion.mp4"},
                {"ref_video_audio_0": "audio:motion.mp4"},
                {"ref_audio_0": "audio:voice.wav"},
            ),
        )
        self.assertEqual(
            loaded,
            [("image", "character.png"), ("video", "motion.mp4"), ("audio", "voice.wav")],
        )

    def test_inline_reference_sources_merge_before_socket_sources(self):
        module = load_node_module()
        self.assertEqual(
            module._merge_h3_reference_sources(
                {"ref_image_0": "inline"},
                {"ref_image_4": "socket"},
                "ref_images",
                "ref_image",
                offset=1,
            ),
            {"ref_image_0": "inline", "ref_image_1": "socket"},
        )

    def test_global_conditioning_uses_preset_geometry_for_zero_overrides(self):
        module = load_node_module()
        self.assertIsNone(module._h3_optional_override(None))
        self.assertIsNone(module._h3_optional_override(0))
        self.assertEqual(module._h3_optional_override(608), 608)

    def test_global_conditioning_routes_fl2va_start_and_end_frames(self):
        module = load_node_module()
        captured = {}

        class Result:
            result = ("positive", "latent")

        class NativeFL2VA:
            @classmethod
            def execute(cls, **kwargs):
                captured.update(kwargs)
                return Result()

        original_loader = module._native_fl2va_conditioning
        module._native_fl2va_conditioning = lambda: NativeFL2VA
        try:
            result = module._execute_global_h3_conditioning(
                clip="clip",
                vae="video_vae",
                audio_vae=None,
                prompt="move from start to end",
                mode="FL2VA",
                width=640,
                height=352,
                length=124,
                ref_image_size="match",
                start_frame="start",
                end_frame="end",
            )
        finally:
            module._native_fl2va_conditioning = original_loader

        self.assertEqual(result, ("positive", "latent"))
        self.assertEqual(captured["first_frame"], "start")
        self.assertEqual(captured["last_frame"], "end")

    def test_global_conditioning_rejects_refs_in_fl2va_mode(self):
        module = load_node_module()

        with self.assertRaisesRegex(ValueError, "FL2VA accepts start_frame/end_frame only"):
            module._execute_global_h3_conditioning(
                clip="clip",
                vae="video_vae",
                audio_vae=None,
                prompt="prompt",
                mode="FL2VA",
                width=640,
                height=352,
                length=124,
                ref_image_size="match",
                ref_images={"ref_image_0": "image"},
            )

    def test_global_conditioning_appends_ref2va_start_and_end_guides(self):
        module = load_node_module()
        guide_calls = []

        class Result:
            def __init__(self, *values):
                self.result = values

        class NativeRef2VA:
            @classmethod
            def execute(cls, **kwargs):
                return Result("base-positive", "latent")

        class NativeGuide:
            @classmethod
            def execute(cls, **kwargs):
                guide_calls.append(kwargs)
                return Result(f"guided-{len(guide_calls)}")

        original_ref_loader = module._native_ref2va_conditioning
        original_guide_loader = module._native_h3_add_guide
        module._native_ref2va_conditioning = lambda: NativeRef2VA
        module._native_h3_add_guide = lambda: NativeGuide
        try:
            result = module._execute_global_h3_conditioning(
                clip="clip",
                vae="video_vae",
                audio_vae="audio_vae",
                prompt="prompt",
                mode="Ref2VA",
                width=640,
                height=352,
                length=124,
                ref_image_size="match",
                start_frame="start",
                end_frame="end",
                ref_images={"ref_image_0": "image"},
            )
        finally:
            module._native_ref2va_conditioning = original_ref_loader
            module._native_h3_add_guide = original_guide_loader

        self.assertEqual(result, ("guided-2", "latent"))
        self.assertEqual([call["frame_idx"] for call in guide_calls], [0, 123])
        self.assertEqual(guide_calls[0]["image"], "start")
        self.assertEqual(guide_calls[1]["image"], "end")

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

    def test_h3_video_vae_converts_channel_first_video_to_comfy_frames(self):
        module = load_node_module()
        import torch
        from unittest.mock import patch

        # H3's native decoder returns [B, C, T, H, W]. Encode the indices into
        # the values so the test catches an accidental reshape as well as a
        # wrong output shape.
        images = torch.empty((1, 3, 2, 4, 5), dtype=torch.float32)
        for channel in range(3):
            for frame in range(2):
                images[0, channel, frame] = channel * 100 + frame

        class VAE:
            def decode(self, _samples):
                return images

        vae = VAE()
        with patch.object(module, "configure_vae_phase"), patch.object(
            module, "release_vae_phase"
        ), patch.object(module, "_enable_h3_decode_weight_casting"), patch.object(
            module, "_h3_move_for_consumer", side_effect=lambda tensor, **_: tensor
        ):
            result = module.H3VAEDecode().decode(
                vae,
                {"samples": torch.zeros((1, 24, 2, 1, 1), dtype=torch.float32)},
                device_id=0,
            )

        converted = result[0]
        self.assertEqual(converted.shape, (2, 4, 5, 3))
        self.assertTrue(torch.equal(converted[0, ..., 0], torch.zeros((4, 5))))
        self.assertTrue(torch.equal(converted[0, ..., 1], torch.full((4, 5), 100.0)))
        self.assertTrue(torch.equal(converted[0, ..., 2], torch.full((4, 5), 200.0)))
        self.assertTrue(torch.equal(converted[1, ..., 0], torch.ones((4, 5))))

    def test_h3_video_vae_accepts_comfy_channel_last_video(self):
        module = load_node_module()
        import torch

        images = torch.empty((1, 2, 4, 5, 3), dtype=torch.float32)
        for frame in range(2):
            images[0, frame] = frame

        converted = module._h3_video_images_to_comfy(images)

        self.assertEqual(converted.shape, (2, 4, 5, 3))
        self.assertTrue(torch.equal(converted[0], torch.zeros((4, 5, 3))))
        self.assertTrue(torch.equal(converted[1], torch.ones((4, 5, 3))))

    def test_h3_video_vae_rejects_invalid_channel_first_shape(self):
        module = load_node_module()
        import torch

        with self.assertRaisesRegex(RuntimeError, r"expected \[batch, time, height, width, 3\]"):
            module._h3_video_images_to_comfy(torch.zeros((1, 4, 2, 4, 5)))

    def test_vae_boundaries_are_recorded_with_checksums(self):
        module = load_node_module()
        import sys
        import types
        import torch
        from unittest.mock import patch

        runtime_config = {
            "execution": {
                "diffusion_telemetry": {
                    "generation_id": "telemetry-generation",
                    "output_path": None,
                }
            }
        }

        class VideoVAE:
            def decode(self, samples):
                self.received = samples
                return torch.ones((1, 2, 2, 3), dtype=torch.float32)

        with patch.object(module, "configure_vae_phase"), patch.object(
            module, "release_vae_phase"
        ), patch.object(module, "_enable_h3_decode_weight_casting"), patch.object(
            module, "_h3_move_for_consumer", side_effect=lambda tensor, **_: tensor
        ), patch.dict(
            os.environ, {"KAGGLE_H3_DIFFUSION_TELEMETRY": "1"}
        ):
            module.H3VAEDecode().decode(
                VideoVAE(),
                {"samples": torch.zeros((1, 4, 2, 2), dtype=torch.float32)},
                device_id=1,
                runtime_config=runtime_config,
            )

        video_telemetry = runtime_config["execution"]["vae_telemetry"]
        video_stages = [record["stage"] for record in video_telemetry["records"]]
        self.assertEqual(
            video_stages,
            [
                "vae_video_in_source",
                "vae_video_in",
                "vae_video_out",
                "vae_video_output",
            ],
        )
        self.assertEqual(video_telemetry["generation_id"], "telemetry-generation")
        self.assertTrue(all(record["checksum"] for record in video_telemetry["records"]))

        audio_module = types.ModuleType("comfy_extras.nodes_audio")
        audio_module.vae_decode_audio = lambda _vae, _samples: {
            "waveform": torch.ones((1, 1, 4), dtype=torch.float32),
            "sample_rate": 32000,
        }
        comfy_extras = types.ModuleType("comfy_extras")
        comfy_extras.__path__ = []
        with patch.dict(
            sys.modules,
            {"comfy_extras": comfy_extras, "comfy_extras.nodes_audio": audio_module},
        ), patch.object(module, "configure_vae_phase"), patch.object(
            module, "release_vae_phase"
        ), patch.object(
            module, "_h3_move_for_consumer", side_effect=lambda tensor, **_: tensor
        ), patch.dict(
            os.environ, {"KAGGLE_H3_DIFFUSION_TELEMETRY": "1"}
        ):
            module.H3AudioVAEDecode().decode(
                object(),
                {
                    "samples": torch.zeros((1, 4, 2, 2), dtype=torch.float32),
                    "_kaggle_h3_stream": "audio",
                },
                device_id=0,
                runtime_config=runtime_config,
            )

        audio_telemetry = runtime_config["execution"]["vae_telemetry"]
        audio_stages = [record["stage"] for record in audio_telemetry["records"]]
        self.assertEqual(
            audio_stages,
            video_stages
            + [
                "vae_audio_in_source",
                "vae_audio_in",
                "vae_audio_out",
                "vae_audio_output",
            ],
        )
        self.assertTrue(all(record["checksum"] for record in audio_telemetry["records"]))

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

    def test_base_res_multistep_and_turbo_euler_wrappers_preserve_disabled_path(self):
        module = load_node_module()

        sentinel = object()
        euler_sampler = types.SimpleNamespace(
            sampler_function=lambda *args, **kwargs: sentinel
        )
        original_euler_function = euler_sampler.sampler_function
        self.assertIs(module._h3_prepare_sampler("euler", euler_sampler), euler_sampler)
        self.assertIsNot(euler_sampler.sampler_function, original_euler_function)
        self.assertIs(euler_sampler.sampler_function(), sentinel)

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

    def test_res_multistep_records_all_requested_diffusion_boundaries(self):
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

            def __call__(self, _x, _sigma, **_kwargs):
                return torch.tensor([[0.5]])

        with patch.dict(
            sys.modules,
            {
                "comfy": comfy,
                "comfy.k_diffusion": k_diffusion,
                "comfy.k_diffusion.sampling": sampling,
            },
        ), patch.object(module, "phase_device_ids", return_value=()), patch.object(
            module, "synchronize_h3_devices"
        ):
            telemetry = module.H3DiffusionTelemetry(enabled=True)
            token = module._H3_DIFFUSION_TELEMETRY.set(telemetry)
            try:
                module._h3_res_multistep(
                    Model(),
                    torch.zeros((1, 1)),
                    torch.tensor([2.0, 1.0, 0.0]),
                )
            finally:
                module._H3_DIFFUSION_TELEMETRY.reset(token)

        expected = [
            f"diffusion_step_{step}_{boundary}"
            for step in range(2)
            for boundary in (
                "input",
                "model_output",
                "denoised",
                "history",
                "updated_latent",
            )
        ]
        self.assertEqual([record["stage"] for record in telemetry.records], expected)
        self.assertEqual(telemetry.records[0]["sigma"], 2.0)
        self.assertEqual(telemetry.records[5]["sigma"], 1.0)
        self.assertEqual(telemetry.records[2]["alias_of"], "model_output")

    def test_euler_records_all_requested_diffusion_boundaries(self):
        module = load_node_module()
        import torch
        from unittest.mock import patch

        sampling = types.ModuleType("comfy.k_diffusion.sampling")
        sampling.to_d = lambda x, sigma, denoised: (x - denoised) / sigma
        k_diffusion = types.ModuleType("comfy.k_diffusion")
        k_diffusion.__path__ = []
        k_diffusion.sampling = sampling
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        comfy.k_diffusion = k_diffusion

        class Model:
            def __call__(self, x, _sigma, **_kwargs):
                return x * 0.5

        with patch.dict(
            sys.modules,
            {
                "comfy": comfy,
                "comfy.k_diffusion": k_diffusion,
                "comfy.k_diffusion.sampling": sampling,
            },
        ):
            telemetry = module.H3DiffusionTelemetry(enabled=True)
            token = module._H3_DIFFUSION_TELEMETRY.set(telemetry)
            try:
                output = module._h3_euler_with_telemetry(
                    Model(),
                    torch.ones((1, 1)),
                    torch.tensor([2.0, 1.0, 0.0]),
                )
            finally:
                module._H3_DIFFUSION_TELEMETRY.reset(token)

        expected = [
            f"diffusion_step_{step}_{boundary}"
            for step in range(2)
            for boundary in (
                "input",
                "model_output",
                "denoised",
                "history",
                "updated_latent",
            )
        ]
        self.assertEqual([record["stage"] for record in telemetry.records], expected)
        self.assertEqual(telemetry.records[2]["timestep"], 2.0)
        self.assertEqual(telemetry.records[7]["timestep"], 1.0)
        self.assertFalse(telemetry.records[3]["present"])
        self.assertTrue(torch.isfinite(output).all())

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
