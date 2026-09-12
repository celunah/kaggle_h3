import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kaggle_h3.phase_runtime import (
    _attach_dispatched_transformer,
    H3PhaseError,
    build_phase_runtime_config,
    phase_policy,
    prepare_model_for_h3_phase,
)


class PhaseRuntimeTests(unittest.TestCase):
    def test_accelerate_replacement_is_reattached_to_comfy_model(self):
        original = object()
        dispatched = object()
        model = type("Patcher", (), {})()
        model.model = original

        report = _attach_dispatched_transformer(model, original, dispatched, "model")

        self.assertIs(model.model, dispatched)
        self.assertEqual(report["status"], "reattached")

    def test_accelerate_replacement_fails_if_comfy_reference_changed(self):
        original = object()
        model = type("Patcher", (), {})()
        model.model = object()

        with self.assertRaisesRegex(H3PhaseError, "changed while H3 dispatch"):
            _attach_dispatched_transformer(model, original, object(), "model")

    def test_phase_plan_has_explicit_ownership_and_safety_tiers(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = build_phase_runtime_config(
                device_ids=(0, 1), offload_dir=Path(temporary) / "offload"
            )
        self.assertEqual(plan["strategy"], "phase_aware_h3")
        self.assertEqual(
            plan["text_encoder"],
            "automatic_contiguous_language_layers_cuda_0_cuda_1_then_cpu",
        )
        self.assertEqual(plan["audio_vae"], "cuda:0")
        self.assertEqual(plan["video_vae"], "cuda:1")
        self.assertEqual(plan["transformer"]["gpu_budget_gib"], 13.0)
        self.assertEqual(plan["transformer"]["cpu_headroom_gib"], 26.0)
        self.assertTrue(plan["release_transformer_before_decode"])

    def test_invalid_phase_policy_is_rejected(self):
        previous = os.environ.get("KAGGLE_H3_PHASE_SHARDING")
        os.environ["KAGGLE_H3_PHASE_SHARDING"] = "one_gpu_but_call_it_sharded"
        try:
            with self.assertRaises(H3PhaseError):
                phase_policy()
        finally:
            if previous is None:
                os.environ.pop("KAGGLE_H3_PHASE_SHARDING", None)
            else:
                os.environ["KAGGLE_H3_PHASE_SHARDING"] = previous

    def test_cpu_resident_loader_keeps_dynamic_patcher_on_registered_cuda_key(self):
        import torch

        cuda_zero = torch.device("cuda:0")
        patcher = SimpleNamespace(
            load_device=cuda_zero,
            offload_device=cuda_zero,
            model=SimpleNamespace(device=cuda_zero, dynamic_pins={cuda_zero: {}}),
        )

        def register_load_device(device):
            patcher.model.dynamic_pins.setdefault(device, {})

        patcher.register_load_device = register_load_device
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.device_count", return_value=2
        ):
            placement = prepare_model_for_h3_phase(patcher, device_ids=(0, 1))

        self.assertEqual(patcher.load_device, cuda_zero)
        self.assertIn(cuda_zero, patcher.model.dynamic_pins)
        self.assertEqual(patcher.offload_device, torch.device("cpu"))
        self.assertEqual(patcher.model.device, torch.device("cpu"))
        self.assertEqual(placement["model_residency"], "cpu")


if __name__ == "__main__":
    unittest.main()
