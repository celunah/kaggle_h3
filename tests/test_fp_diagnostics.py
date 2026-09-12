import unittest

import torch

from kaggle_h3.fp_diagnostics import validate_finite


class FloatingPointDiagnosticsTests(unittest.TestCase):
    def test_finite_tensors_pass_and_are_not_replaced(self):
        value = {"nested": (torch.tensor([1.0, 2.0]), [torch.tensor(3.0)])}
        self.assertIs(validate_finite(value, "conditioning_in"), value)

    def test_nan_positive_and_negative_infinity_are_counted_separately(self):
        with self.assertRaises(FloatingPointError) as context:
            validate_finite(
                torch.tensor([float("nan"), float("inf"), float("-inf")]),
                "sampler_final_out",
                tensor_name="sampler_output",
            )
        message = str(context.exception)
        self.assertIn("FloatingPointError: stage 'sampler_final_out'", message)
        self.assertIn("nan=1, +inf=1, -inf=1", message)
        self.assertIn("tensor=sampler_output", message)
        self.assertIn("shape=(3,)", message)
        self.assertIn("dtype=torch.float32", message)
        self.assertIn("device=cpu", message)

    def test_recursive_nested_tensor_inspection(self):
        value = {"outer": [{"inner": torch.tensor([float("nan")])}]}
        with self.assertRaisesRegex(
            FloatingPointError,
            r"(?s)stage 'sampler_gpu1_in'.*tensor=sample\['outer'\]\[0\]\['inner'\]",
        ):
            validate_finite(value, "sampler_gpu1_in", tensor_name="sample")

    def test_integer_boolean_empty_and_metadata_values_are_skipped(self):
        value = {
            "integer": torch.tensor([1, 2]),
            "boolean": torch.tensor([True]),
            "empty": torch.empty((0,), dtype=torch.float32),
            "label": "metadata",
        }
        validate_finite(value, "vae_video_in")

    def test_complex_non_finite_values_are_checked(self):
        with self.assertRaisesRegex(FloatingPointError, r"nan=1, \+inf=1, -inf=1"):
            validate_finite(
                torch.tensor([complex(float("nan"), 0), complex(float("inf"), float("-inf"))]),
                "vae_audio_out",
            )

    def test_h3_nested_tensor_streams_are_checked_when_supported(self):
        if not hasattr(torch, "nested") or not hasattr(torch.nested, "nested_tensor"):
            self.skipTest("PyTorch nested tensors are unavailable")
        packed = torch.nested.nested_tensor(
            [torch.tensor([1.0]), torch.tensor([float("inf")])]
        )
        with self.assertRaisesRegex(FloatingPointError, r"stage 'gpu0_to_gpu1'"):
            validate_finite(packed, "gpu0_to_gpu1", tensor_name="av_latent")


if __name__ == "__main__":
    unittest.main()
