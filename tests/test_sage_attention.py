import sys
import types
import unittest
from unittest.mock import patch

from kaggle_h3.sage_attention import (
    H3SageAttentionError,
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
        original = object()
        original_masked = object()
        minimax_model.optimized_attention = original
        minimax_model.optimized_attention_masked = original_masked

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
        with patch.dict(sys.modules, modules, clear=False):
            status, restore = install_h3_sage_attention()
            self.assertTrue(status["enabled"])
            self.assertIsNot(model.optimized_attention, original)
            self.assertIsNot(model.optimized_attention_masked, original_masked)
            restore()
            restore()
        self.assertIs(model.optimized_attention, original)
        self.assertIs(model.optimized_attention_masked, original_masked)

    def test_missing_backend_is_an_actionable_error(self):
        with patch.dict(sys.modules, {"sageattention": None}, clear=False):
            with self.assertRaisesRegex(H3SageAttentionError, "not available"):
                install_h3_sage_attention()


if __name__ == "__main__":
    unittest.main()
