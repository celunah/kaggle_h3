import json
import unittest
from pathlib import Path

from kaggle_h3.bootstrap import required_model_files
from kaggle_h3.workflow import (
    H3Request,
    build_workflow,
    frame_length,
    select_mode,
    select_production_mode,
    validate_production_workflow_shape,
)
from kaggle_h3.ref2va import resolve_ref2va_dimensions, resolve_ref2va_length


class WorkflowTests(unittest.TestCase):
    def test_modes_are_explicit(self):
        self.assertEqual(select_mode(H3Request("text only")), "T2VA")
        self.assertEqual(select_mode(H3Request("endpoints", first_frame="a.png")), "FL2VA")
        self.assertEqual(select_mode(H3Request("refs", character_references=["a.png"])), "Ref2VA")
        with self.assertRaises(ValueError):
            select_mode(H3Request("mixed", first_frame="a.png", character_references=["b.png"]))
        self.assertEqual(
            select_production_mode(H3Request("refs", character_references=["a.png"])),
            "Ref2VA",
        )
        with self.assertRaisesRegex(ValueError, "Ref2VA references only"):
            select_production_mode(H3Request("text only"))

    def test_h3_frame_alignment(self):
        result = frame_length(H3Request("x", duration_seconds=5.0))
        self.assertEqual(result["fps"], 24)
        self.assertEqual((result["actual_frames"] - 5) % 17, 0)

    def test_ref2va_seconds_and_canvas_presets_are_h3_aligned(self):
        length = resolve_ref2va_length(5.0)
        self.assertEqual(length["actual_frames"], 124)
        self.assertEqual(resolve_ref2va_dimensions("240p", "16:9"), (448, 256))
        self.assertEqual(resolve_ref2va_dimensions("360p", "16:9"), (640, 352))
        self.assertEqual(resolve_ref2va_dimensions("480p", "4:3"), (640, 480))
        with self.assertRaises(ValueError):
            resolve_ref2va_dimensions("720p", "16:9")
        with self.assertRaises(ValueError):
            resolve_ref2va_length(1.0)

    def test_reference_workflow_has_role_inputs(self):
        request = H3Request(
            "A subject and their scene",
            character_references=["character.png"],
            scene_references=["scene.png"],
            quality_mode="quick",
        )
        workflow = build_workflow(request)
        self.assertEqual(workflow["h3"]["class_type"], "KaggleH3Conditioning")
        self.assertIn("audio_vae", workflow["h3"]["inputs"])
        self.assertEqual(workflow["h3"]["inputs"]["seconds"], 5.0)
        self.assertEqual(workflow["h3"]["inputs"]["size_preset"], "240p")
        self.assertEqual(workflow["h3"]["inputs"]["aspect_ratio"], "16:9")
        self.assertEqual(workflow["h3"]["inputs"]["reference_0"], "image")
        self.assertEqual(workflow["h3"]["inputs"]["reference_0.file"], "character.png")
        self.assertEqual(workflow["h3"]["inputs"]["reference_1"], "image")
        self.assertEqual(workflow["h3"]["inputs"]["reference_1.file"], "scene.png")
        self.assertIn("<Picture 1>", workflow["h3"]["inputs"]["prompt"])
        self.assertNotIn("ref_00", workflow)
        self.assertNotIn("ref_01", workflow)
        self.assertEqual(workflow["phase"]["inputs"]["conditioning"], ["h3", 0])
        self.assertEqual(workflow["phase"]["inputs"]["latent_image"], ["h3", 1])
        self.assertTrue(validate_production_workflow_shape(workflow)["valid"])

    def test_keyframe_workflow_uses_current_native_inputs(self):
        with self.assertRaisesRegex(ValueError, "Ref2VA references only"):
            build_workflow(H3Request("start to end", first_frame="first.png", last_frame="last.png"))

    def test_production_workflow_can_mute_generated_audio(self):
        workflow = build_workflow(
            H3Request(
                "video only",
                character_references=["character.png"],
                mute_generated_audio=True,
            )
        )
        self.assertNotIn("decode_audio", workflow)
        self.assertNotIn("audio", workflow["video"]["inputs"])
        self.assertTrue(validate_production_workflow_shape(workflow)["valid"])

    def test_production_guard_rejects_legacy_runtime_options(self):
        with self.assertRaisesRegex(ValueError, "requires Turbo mode"):
            select_production_mode(
                H3Request("base", character_references=["character.png"], turbo_mode=False)
            )
        self.assertEqual(
            select_production_mode(
                H3Request("sage", character_references=["character.png"], sage_attention=True)
            ),
            "Ref2VA",
        )
        workflow = build_workflow(
            H3Request("sage", character_references=["character.png"], sage_attention=True)
        )
        self.assertTrue(workflow["sample"]["inputs"]["use_sage_attention"])
        with self.assertRaisesRegex(ValueError, "explicit dimensions"):
            select_production_mode(
                H3Request("geometry", character_references=["character.png"], width=640)
            )

    def test_production_workflow_can_select_optional_dual_clock_euler(self):
        workflow = build_workflow(
            H3Request(
                "dual clock",
                character_references=["character.png"],
                sampler_name="euler_dualclock",
                steps=12,
            )
        )
        self.assertEqual(workflow["sample"]["inputs"]["sampler_name"], "euler_dualclock")
        self.assertIn("Euler Dual-clock", workflow["sample"]["_meta"]["title"])
        self.assertTrue(validate_production_workflow_shape(workflow)["valid"])

    def test_model_mode_names_are_case_insensitive(self):
        self.assertIn("ref2va", required_model_files("ref2va")["diffusion_model"][1].lower())

    def test_templates_are_valid_json(self):
        root = Path(__file__).parents[1]
        for path in (root / "workflows").glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(validate_production_workflow_shape(payload)["valid"], path)

    def test_native_h3_workflow_uses_dtype_safe_video_decode_node(self):
        workflow = build_workflow(
            H3Request("decode smoke", character_references=["character.png"])
        )
        self.assertTrue(all(node.get("_meta", {}).get("title") for node in workflow.values()))
        self.assertIn("Auto Loader", workflow["unet"]["_meta"]["title"])
        self.assertEqual(workflow["phase"]["_meta"]["title"], "Kaggle H3 | Transformer Dispatch Barrier")
        self.assertEqual(workflow["unet"]["class_type"], "KaggleH3ShardedDiffusionLoader")
        self.assertEqual(workflow["unet"]["inputs"]["model_variant"], "Ref2VA")
        self.assertEqual(workflow["unet"]["inputs"]["model_profile"], "singularity")
        self.assertEqual(workflow["unet"]["inputs"]["precision"], "int8_convrot")
        self.assertNotIn("unet_name", workflow["unet"]["inputs"])
        self.assertEqual(workflow["unet"]["inputs"]["gpu_0"], 0)
        self.assertEqual(workflow["unet"]["inputs"]["gpu_1"], 1)
        self.assertEqual(workflow["clip"]["class_type"], "KaggleH3TextEncoderLoader")
        self.assertEqual(workflow["clip"]["inputs"]["device_id"], 0)
        self.assertEqual(workflow["phase"]["class_type"], "KaggleH3PhaseDispatch")
        self.assertEqual(workflow["decode"]["class_type"], "KaggleH3VAEDecode")
        self.assertEqual(workflow["decode"]["inputs"]["device_id"], 1)
        self.assertEqual(workflow["decode_audio"]["class_type"], "KaggleH3AudioVAEDecode")
        self.assertEqual(workflow["decode_audio"]["inputs"]["device_id"], 0)
        self.assertEqual(workflow["decode"]["inputs"]["samples"], ["sample", 0])
        self.assertEqual(workflow["decode_audio"]["inputs"]["samples"], ["sample", 1])
        self.assertEqual(workflow["video"]["inputs"]["color_space"], "sRGB")
        self.assertEqual(workflow["sample"]["inputs"]["synchronize_mode"], "full")

    def test_default_turbo_smoke_template_matches_ref2va_active_mode(self):
        root = Path(__file__).parents[1]
        workflow = json.loads((root / "workflows" / "kaggle_h3_turbo_smoke.json").read_text(encoding="utf-8"))
        self.assertTrue(all(node.get("_meta", {}).get("title") for node in workflow.values()))
        self.assertEqual(workflow["sample"]["inputs"]["sampler_name"], "euler")
        self.assertNotIn("use_sage_attention", workflow["sample"]["inputs"])
        self.assertEqual(workflow["unet"]["inputs"]["model_variant"], "Ref2VA")
        self.assertNotIn("character_reference", workflow)
        self.assertNotIn("scene_reference", workflow)
        self.assertEqual(workflow["h3"]["class_type"], "KaggleH3Conditioning")
        self.assertEqual(workflow["h3_adapter"]["inputs"]["model_variant"], "ref2va")
        self.assertEqual(
            workflow["h3_adapter"]["inputs"]["adapter_1"],
            "auto_singularity_ref2va_turbo",
        )
        self.assertEqual(workflow["sample"]["inputs"]["synchronize_mode"], "full")
        self.assertEqual(workflow["h3"]["inputs"]["reference_0"], "image")
        self.assertEqual(
            workflow["h3"]["inputs"]["reference_0.file"],
            "CHARACTER_REFERENCE.png",
        )
        self.assertEqual(workflow["h3"]["inputs"]["seconds"], 5.0)

    def test_regular_template_exposes_optional_runtime_controls(self):
        root = Path(__file__).parents[1]
        workflow = json.loads((root / "workflows" / "kaggle_h3_ref2va.json").read_text(encoding="utf-8"))
        self.assertEqual(workflow["step_count"]["class_type"], "FirstInt")
        self.assertEqual(workflow["h3_adapter"]["inputs"]["turbo_steps"], ["step_count", 0])
        self.assertEqual(workflow["sample"]["inputs"]["steps"], ["step_count", 0])
        self.assertIn("Euler Dual-clock", workflow["sample"]["_meta"]["title"])
        self.assertFalse(workflow["sample"]["inputs"]["use_sage_attention"])
        self.assertEqual(workflow["upscale"]["class_type"], "KaggleH3LatentUpscale2x")
        self.assertFalse(workflow["upscale"]["inputs"]["enabled"])
        self.assertFalse(workflow["decode_audio"]["inputs"]["mute_generated_audio"])
        self.assertNotIn("core_bundle", workflow)
        self.assertNotIn("core_unbundle", workflow)
        self.assertEqual(workflow["phase"]["inputs"]["model"], ["h3_adapter", 0])
        self.assertEqual(workflow["phase"]["inputs"]["runtime_config"], ["h3_adapter", 1])
        self.assertEqual(workflow["phase"]["inputs"]["conditioning"], ["h3", 0])
        self.assertEqual(workflow["phase"]["inputs"]["latent_image"], ["h3", 1])

    def test_long_template_routes_context_loop_and_obvpm_bundle(self):
        root = Path(__file__).parents[1]
        workflow = json.loads((root / "workflows" / "kaggle_h3_long_context_loop.json").read_text(encoding="utf-8"))
        self.assertEqual(workflow["policy"]["class_type"], "MiniMaxH3GenerationProfile")
        self.assertEqual(workflow["plan"]["class_type"], "MiniMaxH3ChainPlanModern")
        self.assertEqual(workflow["loop_start"]["class_type"], "MiniMaxH3ChainLoopStart")
        self.assertEqual(workflow["context"]["class_type"], "MiniMaxH3ChainContext")
        self.assertEqual(workflow["sample"]["class_type"], "KaggleH3ContextLoopSampler")
        self.assertEqual(workflow["trim"]["class_type"], "MiniMaxH3LoopTrim")
        self.assertEqual(workflow["segment_save"]["class_type"], "MiniMaxH3ChainSegmentSave")
        self.assertEqual(workflow["review"]["class_type"], "MiniMaxH3ChainReview")
        self.assertEqual(workflow["loop_end"]["class_type"], "MiniMaxH3ChainLoopEnd")
        self.assertEqual(workflow["assemble"]["class_type"], "MiniMaxH3ChainAssemble")
        self.assertEqual(workflow["sample"]["inputs"]["sampler_name"], "euler_dualclock")
        self.assertEqual(workflow["loop_end"]["inputs"]["between_scene_cleanup"], "unload_models")
        self.assertEqual(workflow["decode"]["inputs"]["runtime_config"], ["phase", 3])
        self.assertEqual(workflow["decode_audio"]["inputs"]["runtime_config"], ["phase", 3])
        self.assertNotIn("core_bundle", workflow)
        self.assertNotIn("core_unbundle", workflow)
        self.assertEqual(workflow["phase"]["inputs"]["model"], ["h3_adapter", 0])
        self.assertEqual(workflow["phase"]["inputs"]["runtime_config"], ["h3_adapter", 1])
        self.assertEqual(workflow["phase"]["inputs"]["conditioning"], ["context", 0])
        self.assertEqual(workflow["phase"]["inputs"]["latent_image"], ["context", 3])
        plan = json.loads(workflow["plan"]["inputs"]["plan_json"])
        self.assertEqual(len(plan["shots"]), 4)


if __name__ == "__main__":
    unittest.main()
