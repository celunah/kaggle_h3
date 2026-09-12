import unittest

from kaggle_h3.backends import build_sglang_command, evaluate_backends, select_backend
from kaggle_h3.pipeline import _sglang_payload
from kaggle_h3.workflow import H3Request


FAKE_T4_HARDWARE = {
    "cuda_available": True,
    "cuda_device_count": 2,
    "cuda_devices": [
        {"index": 0, "name": "Tesla T4", "total_gib": 15.0, "free_gib": 14.5, "is_t4": True},
        {"index": 1, "name": "Tesla T4", "total_gib": 15.0, "free_gib": 14.5, "is_t4": True},
    ],
    "cpu_memory": {"total_bytes": 30 * 2**30, "available_gib": 28.0},
}


class BackendTests(unittest.TestCase):
    def test_t4_does_not_silently_select_single_gpu(self):
        evaluation = evaluate_backends(FAKE_T4_HARDWARE)
        selected = select_backend(evaluation, allow_single_gpu_fallback=False)
        self.assertIsNone(selected["selected"])
        self.assertEqual(selected["status"], "no_verified_dual_gpu_backend")

    def test_fallback_requires_explicit_flag(self):
        evaluation = evaluate_backends(FAKE_T4_HARDWARE)
        selected = select_backend(evaluation, allow_single_gpu_fallback=True)
        self.assertEqual(selected["selected"]["backend"], "comfyui_native_fallback")
        self.assertTrue(selected["fallback_used"])

    def test_all_multi_gpu_routes_are_reported(self):
        names = {item["backend"] for item in evaluate_backends(FAKE_T4_HARDWARE)["attempts"]}
        self.assertEqual(
            names,
            {
                "sglang",
                "diffusers_layer_sharded",
                "diffusers_balanced",
                "diffusers_component_split",
                "comfyui_sp",
                "comfyui_native_fallback",
            },
        )

    def test_layer_sharded_route_uses_live_safety_not_upstream_card_size(self):
        route = next(
            item
            for item in evaluate_backends(FAKE_T4_HARDWARE)["attempts"]
            if item["backend"] == "diffusers_layer_sharded"
        )
        self.assertEqual(route["planned_same_generation_multi_gpu"], True)
        self.assertEqual(route["requirements_evidence"]["min_free_gpu_gib"], 12.0)
        self.assertEqual(route["requirements_evidence"]["min_cpu_available_gib"], 24.0)
        self.assertEqual(route["observed_same_generation_multi_gpu"], False)

    def test_layer_sharded_route_advertises_preferred_h3_boundary(self):
        route = next(
            item
            for item in evaluate_backends(FAKE_T4_HARDWARE)["attempts"]
            if item["backend"] == "diffusers_layer_sharded"
        )
        self.assertIn("blocks[0:23]", route["actual_device_map"]["transformer"])
        self.assertEqual(
            route["dispatch_strategy"]["activation_boundary"],
            "one hidden-state transfer after block 22 before block 23 per denoising pass",
        )

    def test_sglang_payload_uses_current_video_api_fields(self):
        t2va = _sglang_payload(H3Request("x", quality_mode="quick"), "T2VA")
        self.assertEqual(t2va["task"], "t2va")
        self.assertEqual(t2va["conditions"], [])
        self.assertIn("num_inference_steps", t2va)
        self.assertIn("flow_shift", t2va)
        fl2va = _sglang_payload(H3Request("x", first_frame="first.png"), "FL2VA")
        self.assertEqual(fl2va["conditions"][0]["frame_index"], 0)
        self.assertEqual(fl2va["conditions"][0]["role"], "keyframe")
        ref2va = _sglang_payload(
            H3Request("x", character_references=["character.png"]), "Ref2VA"
        )
        self.assertEqual(ref2va["conditions"][0]["role"], "reference")
        self.assertEqual(ref2va["conditions"][0]["h3_role"], "character")

    def test_sglang_t2va_uses_fl2va_checkpoint_partition(self):
        command = build_sglang_command(variant="t2va")
        self.assertIn("fl2va", command)
        self.assertIn("--num-gpus", command)
        self.assertIn("2", command)


if __name__ == "__main__":
    unittest.main()
