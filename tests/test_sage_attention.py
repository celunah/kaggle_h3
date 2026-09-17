import sys
import types
import unittest
from unittest.mock import patch

from kaggle_h3.sage_attention import (
    H3SageAttentionError,
    h3_sage_attention,
    install_h3_sage_attention,
    sage_attention_status,
)


class SageAttentionTests(unittest.TestCase):
    def _fake_modules(self):
        sageattention = types.ModuleType("sageattention")
        sageattention.__version__ = "test"
        sageattention.sageattn = lambda q, k, v, **kwargs: q

        attention = types.ModuleType("comfy.ldm.modules.attention")
        attention.attention_sage = lambda *args, **kwargs: args[0]
        attention.AttentionTensorContainer = type("AttentionTensorContainer", (), {})

        minimax_model = types.ModuleType("comfy.ldm.minimax.model")

        def original(*args, **kwargs):
            return "normal"

        def original_masked(*args, **kwargs):
            return "normal_masked"

        minimax_model.optimized_attention = original
        minimax_model.optimized_attention_masked = original_masked

        triton = types.ModuleType("triton")
        triton.__version__ = "3.2.0"

        packages = {
            "comfy": types.ModuleType("comfy"),
            "comfy.ldm": types.ModuleType("comfy.ldm"),
            "comfy.ldm.modules": types.ModuleType("comfy.ldm.modules"),
            "comfy.ldm.minimax": types.ModuleType("comfy.ldm.minimax"),
        }
        for package in packages.values():
            package.__path__ = []
        modules = {
            "sageattention": sageattention,
            "triton": triton,
            "comfy.ldm.modules.attention": attention,
            "comfy.ldm.minimax.model": minimax_model,
            **packages,
        }
        return modules, minimax_model, original, original_masked

    def test_status_reports_optional_backend(self):
        modules, _model, _original, _masked = self._fake_modules()
        with patch.dict(sys.modules, modules, clear=False):
            status = sage_attention_status()
        self.assertTrue(status["available"])
        self.assertTrue(status["supports_h3_packed_containers"])

    def test_install_patches_only_h3_model_and_restores(self):
        modules, model, original, original_masked = self._fake_modules()
        with patch.dict(sys.modules, modules, clear=False), patch(
            "kaggle_h3.sage_attention._cuda_runtime_status",
            return_value={"cuda_available": False, "gpu_architectures": [], "gpu_names": []},
        ):
            status, restore = install_h3_sage_attention()
            self.assertTrue(status["enabled"])
            self.assertIsNot(model.optimized_attention, original)
            self.assertIsNot(model.optimized_attention_masked, original_masked)
            restore()
            restore()
        self.assertIs(model.optimized_attention, original)
        self.assertIs(model.optimized_attention_masked, original_masked)

    def test_missing_backend_is_an_actionable_error(self):
        with patch.dict(sys.modules, {"sageattention": None}, clear=False), patch(
            "kaggle_h3.sage_attention._cuda_runtime_status",
            return_value={"cuda_available": False, "gpu_architectures": [], "gpu_names": []},
        ):
            with self.assertRaisesRegex(H3SageAttentionError, "not available"):
                install_h3_sage_attention()

    def test_requested_backend_falls_back_when_missing(self):
        with patch.dict(sys.modules, {"sageattention": None}, clear=False), patch(
            "kaggle_h3.sage_attention._cuda_runtime_status",
            return_value={
                "cuda_available": False,
                "gpu_architectures": [],
                "gpu_names": [],
            },
        ):
            with h3_sage_attention(True) as report:
                self.assertFalse(report["enabled"])
                self.assertTrue(report["fallback"])
                self.assertEqual(report["backend"], "comfyui_default")

    def test_same_device_ownership_is_required_for_sage_attention(self):
        modules, model, _original, _masked = self._fake_modules()

        class Tensor:
            def __init__(self, device, dtype="torch.float16"):
                self.device = device
                self.dtype = dtype

        class Container:
            def __init__(self, tensor):
                self._tensor = tensor

            def peek(self):
                return self._tensor

        with patch.dict(sys.modules, modules, clear=False), patch(
            "kaggle_h3.sage_attention._cuda_runtime_status",
            return_value={"cuda_available": False, "gpu_architectures": [], "gpu_names": []},
        ):
            status, restore = install_h3_sage_attention()
            try:
                self.assertEqual(
                    model.optimized_attention(
                        Tensor("cuda:0"), Tensor("cuda:1"), Tensor("cuda:0")
                    ),
                    "normal",
                )
                self.assertTrue(status["fallback"])
                self.assertIn("device ownership", status["fallback_reason"])
                self.assertEqual(
                    model.optimized_attention(
                        Container(Tensor("cuda:0")),
                        Container(Tensor("cuda:1")),
                        Container(Tensor("cuda:0")),
                    ),
                    "normal",
                )
            finally:
                restore()

    def test_incompatible_triton_falls_back_before_kernel_execution(self):
        modules, _model, _original, _masked = self._fake_modules()
        modules["triton"].__version__ = "3.3.0"
        runtime = {
            "cuda_available": True,
            "gpu_architectures": ["sm75"],
            "gpu_names": ["Tesla T4"],
        }
        with patch.dict(sys.modules, modules, clear=False), patch(
            "kaggle_h3.sage_attention._cuda_runtime_status",
            return_value=runtime,
        ):
            with h3_sage_attention(True) as report:
                self.assertTrue(report["fallback"])
                self.assertEqual(report["backend"], "comfyui_default")
                self.assertIn("Triton", report["fallback_reason"])

    def test_sage_v1_and_triton_versions_are_reported(self):
        modules, _model, _original, _masked = self._fake_modules()
        modules["sageattention"].__version__ = "1.0.6"
        with patch.dict(sys.modules, modules, clear=False), patch(
            "kaggle_h3.sage_attention._cuda_runtime_status",
            return_value={
                "cuda_available": True,
                "gpu_architectures": ["sm75"],
                "gpu_names": ["Tesla T4"],
            },
        ):
            status = sage_attention_status()
        self.assertEqual(status["implementation"], "SageAttention v1")
        self.assertEqual(status["triton_version"], "3.2.0")
        self.assertTrue(status["triton_compatible"])
        self.assertTrue(status["sage_v1_compatible"])


if __name__ == "__main__":
    unittest.main()
