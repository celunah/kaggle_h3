import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kaggle_h3.phase_runtime import (
    _attach_dispatched_transformer,
    _install_transformer_activity_hooks,
    H3PhaseError,
    H3TransformerPhase,
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

    def test_transformer_activity_hooks_validate_sampler_segments(self):
        import torch

        class Transformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList(
                    [
                        torch.nn.Identity(),
                        torch.nn.Identity(),
                        torch.nn.Identity(),
                        torch.nn.Identity(),
                    ]
                )

        transformer = Transformer()
        state = H3TransformerPhase(
            transformer=transformer,
            device_ids=(0, 1),
            report={
                "planned": {
                    "group": {"container_path": "blocks"},
                    "layers": [
                        {"path": "blocks.0", "execution_device": "cuda:0"},
                        {"path": "blocks.1", "execution_device": "cuda:1"},
                        {"path": "blocks.2", "execution_device": "cuda:0"},
                        {"path": "blocks.3", "execution_device": "cuda:1"},
                    ],
                }
            },
        )
        _install_transformer_activity_hooks(state)
        try:
            value = torch.ones((1, 2), dtype=torch.float32)
            for block in transformer.blocks:
                block(value)
        finally:
            for handle in state.activity_handles:
                handle.remove()

        self.assertEqual(
            set(state.activity["diagnostic_checks"]),
            {
                "sampler_gpu0",
                "sampler_gpu0_out",
                "gpu0_to_gpu1",
                "sampler_gpu1_in",
                "sampler_gpu1",
                "sampler_gpu1_out",
            },
        )
        self.assertEqual(
            state.activity["validated_blocks"],
            {
                "input": ["blocks.0", "blocks.1", "blocks.2", "blocks.3"],
                "output": ["blocks.0", "blocks.1", "blocks.2", "blocks.3"],
            },
        )

    def test_transformer_activity_hooks_fail_at_intermediate_gpu1_block_output(self):
        import torch

        class BadBlock(torch.nn.Module):
            def forward(self, value):
                return torch.full_like(value, float("nan"))

        class Transformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList(
                    [torch.nn.Identity(), torch.nn.Identity(), BadBlock(), torch.nn.Identity()]
                )

        transformer = Transformer()
        state = H3TransformerPhase(
            transformer=transformer,
            device_ids=(0, 1),
            report={
                "planned": {
                    "group": {"container_path": "blocks"},
                    "layers": [
                        {"path": "blocks.0", "execution_device": "cuda:0"},
                        {"path": "blocks.1", "execution_device": "cuda:0"},
                        {"path": "blocks.2", "execution_device": "cuda:1"},
                        {"path": "blocks.3", "execution_device": "cuda:1"},
                    ],
                }
            },
        )
        _install_transformer_activity_hooks(state)
        try:
            transformer.blocks[0](torch.ones((1, 2), dtype=torch.float32))
            transformer.blocks[1](torch.ones((1, 2), dtype=torch.float32))
            with self.assertRaisesRegex(
                FloatingPointError,
                r"(?s)stage 'sampler_gpu1_out'.*tensor=blocks\.2\.output",
            ):
                transformer.blocks[2](torch.ones((1, 2), dtype=torch.float32))
        finally:
            for handle in state.activity_handles:
                handle.remove()

        self.assertEqual(state.activity["validated_blocks"]["output"], ["blocks.0", "blocks.1"])

    def test_transformer_activity_hooks_validate_all_internal_modules_on_both_gpus(self):
        import torch

        class NestedBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Sequential(
                    torch.nn.Identity(),
                    torch.nn.Identity(),
                )

            def forward(self, value):
                return self.proj(value)

        class Transformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([NestedBlock() for _ in range(4)])

        transformer = Transformer()
        state = H3TransformerPhase(
            transformer=transformer,
            device_ids=(0, 1),
            report={
                "planned": {
                    "group": {"container_path": "blocks"},
                    "layers": [
                        {"path": "blocks.0", "execution_device": "cuda:0"},
                        {"path": "blocks.1", "execution_device": "cuda:1"},
                        {"path": "blocks.2", "execution_device": "cuda:0"},
                        {"path": "blocks.3", "execution_device": "cuda:1"},
                    ],
                }
            },
        )
        _install_transformer_activity_hooks(state)
        try:
            value = torch.ones((1, 2), dtype=torch.float32)
            for block in transformer.blocks:
                block(value)
        finally:
            for handle in state.activity_handles:
                handle.remove()

        expected_internal_paths = [
            f"blocks.{block_index}{suffix}"
            for block_index in range(4)
            for suffix in (".proj", ".proj.0", ".proj.1")
        ]
        self.assertCountEqual(
            state.activity["validated_internal_ops"]["input"],
            expected_internal_paths,
        )
        self.assertCountEqual(
            state.activity["validated_internal_ops"]["output"],
            expected_internal_paths,
        )

    def test_transformer_activity_hooks_fail_at_internal_gpu1_operation(self):
        import torch

        class BadProjection(torch.nn.Module):
            def forward(self, value):
                return torch.full_like(value, float("nan"))

        class GoodBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Identity()

            def forward(self, value):
                return self.proj(value)

        class BadBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = BadProjection()

            def forward(self, value):
                return self.proj(value)

        class Transformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList(
                    [GoodBlock(), GoodBlock(), BadBlock(), GoodBlock()]
                )

        transformer = Transformer()
        state = H3TransformerPhase(
            transformer=transformer,
            device_ids=(0, 1),
            report={
                "planned": {
                    "group": {"container_path": "blocks"},
                    "layers": [
                        {"path": "blocks.0", "execution_device": "cuda:0"},
                        {"path": "blocks.1", "execution_device": "cuda:0"},
                        {"path": "blocks.2", "execution_device": "cuda:1"},
                        {"path": "blocks.3", "execution_device": "cuda:1"},
                    ],
                }
            },
        )
        _install_transformer_activity_hooks(state)
        try:
            transformer.blocks[0](torch.ones((1, 2), dtype=torch.float32))
            transformer.blocks[1](torch.ones((1, 2), dtype=torch.float32))
            with self.assertRaisesRegex(
                FloatingPointError,
                r"(?s)stage 'sampler_gpu1_out'.*tensor=blocks\.2\.proj\.output",
            ):
                transformer.blocks[2](torch.ones((1, 2), dtype=torch.float32))
        finally:
            for handle in state.activity_handles:
                handle.remove()

        self.assertIn("blocks.0.proj", state.activity["validated_internal_ops"]["output"])
        self.assertIn("blocks.1.proj", state.activity["validated_internal_ops"]["output"])
        self.assertNotIn("blocks.2.proj", state.activity["validated_internal_ops"]["output"])

    def test_transformer_activity_hooks_fail_at_gpu0_segment_output(self):
        import torch

        class BadBlock(torch.nn.Module):
            def forward(self, value):
                return torch.full_like(value, float("nan"))

        class Transformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([BadBlock(), torch.nn.Identity()])

        transformer = Transformer()
        state = H3TransformerPhase(
            transformer=transformer,
            device_ids=(0, 1),
            report={
                "planned": {
                    "group": {"container_path": "blocks"},
                    "layers": [
                        {"path": "blocks.0", "execution_device": "cuda:0"},
                        {"path": "blocks.1", "execution_device": "cuda:1"},
                    ],
                }
            },
        )
        _install_transformer_activity_hooks(state)
        try:
            with self.assertRaisesRegex(FloatingPointError, "stage 'sampler_gpu0_out'"):
                transformer.blocks[0](torch.ones((1, 2), dtype=torch.float32))
        finally:
            for handle in state.activity_handles:
                handle.remove()


if __name__ == "__main__":
    unittest.main()
