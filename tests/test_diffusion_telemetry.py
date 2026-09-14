import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from kaggle_h3.diffusion_telemetry import H3DiffusionTelemetry


class DiffusionTelemetryTests(unittest.TestCase):
    def test_disabled_by_default_does_not_record(self):
        with patch.dict(os.environ, {"KAGGLE_H3_DIFFUSION_TELEMETRY": "0"}):
            telemetry = H3DiffusionTelemetry()
            telemetry.record_boundary(
                step_index=0,
                sigma=torch.tensor(1.0),
                boundary="input",
                value=torch.ones((1, 2)),
                tensor_name="x",
            )
        self.assertFalse(telemetry.enabled)
        self.assertEqual(telemetry.records, [])

    def test_records_requested_statistics_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"KAGGLE_H3_DIFFUSION_TELEMETRY": "1"}
        ):
            path = Path(temporary) / "diffusion.jsonl"
            telemetry = H3DiffusionTelemetry(output_path=path)
            telemetry.record_boundary(
                step_index=3,
                sigma=torch.tensor(0.5),
                boundary="updated_latent",
                value=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
                tensor_name="latent",
            )
            self.assertEqual(len(telemetry.records), 1)
            record = telemetry.records[0]
            expected_checksum = hashlib.sha256(
                torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32
                ).contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            self.assertEqual(record["stage"], "diffusion_step_3_updated_latent")
            self.assertEqual(record["sigma"], 0.5)
            self.assertEqual(record["timestep"], 0.5)
            self.assertEqual(record["shape"], [2, 2])
            self.assertEqual(record["checksum_algorithm"], "sha256_raw_contiguous")
            self.assertEqual(record["checksum"], expected_checksum)
            self.assertEqual(record["min"], 1.0)
            self.assertEqual(record["max"], 4.0)
            self.assertEqual(record["mean"], 2.5)
            self.assertAlmostEqual(record["std"], 1.118033988749895, places=6)
            self.assertEqual(record["finite_count"], 4)
            self.assertEqual(record["dtype"], "torch.float32")
            self.assertEqual(record["device"], "cpu")
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)
            json.loads(path.read_text(encoding="utf-8"))

    def test_phase_record_uses_checksum_contract(self):
        with patch.dict(os.environ, {"KAGGLE_H3_DIFFUSION_TELEMETRY": "1"}):
            telemetry = H3DiffusionTelemetry()
            value = torch.arange(6, dtype=torch.float32).reshape(2, 3).transpose(0, 1)
            telemetry.record_phase(
                stage="vae_video_in",
                value=value,
                tensor_name="video_latent",
            )

        record = telemetry.records[0]
        expected_checksum = hashlib.sha256(
            value.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        self.assertEqual(record["stage"], "vae_video_in")
        self.assertIsNone(record["step_index"])
        self.assertEqual(record["shape"], [3, 2])
        self.assertEqual(record["checksum"], expected_checksum)
        self.assertEqual(record["checksum_algorithm"], "sha256_raw_contiguous")

    def test_nested_tensors_are_recorded_separately(self):
        with patch.dict(os.environ, {"KAGGLE_H3_DIFFUSION_TELEMETRY": "1"}):
            telemetry = H3DiffusionTelemetry()
            telemetry.record_boundary(
                step_index=0,
                sigma=1.0,
                boundary="model_output",
                value={"video": (torch.ones(1), [torch.zeros(2)])},
                tensor_name="output",
            )
        self.assertEqual(len(telemetry.records), 2)
        self.assertEqual(
            {record["tensor"] for record in telemetry.records},
            {"output['video'][0]", "output['video'][1][0]"},
        )

    def test_missing_history_is_explicit(self):
        with patch.dict(os.environ, {"KAGGLE_H3_DIFFUSION_TELEMETRY": "1"}):
            telemetry = H3DiffusionTelemetry()
            telemetry.record_boundary(
                step_index=0,
                sigma=1.0,
                boundary="history",
                value=None,
                tensor_name="history",
            )
        record = telemetry.records[0]
        self.assertEqual(record["stage"], "diffusion_step_0_history")
        self.assertFalse(record["present"])
        self.assertIsNone(record["dtype"])
        self.assertIsNone(record["device"])

    def test_model_output_alias_is_preserved(self):
        with patch.dict(os.environ, {"KAGGLE_H3_DIFFUSION_TELEMETRY": "1"}):
            telemetry = H3DiffusionTelemetry()
            value = torch.ones(1)
            telemetry.record_boundary(
                step_index=1,
                sigma=0.25,
                boundary="denoised",
                value=value,
                tensor_name="denoised",
                alias_of="model_output",
            )
        self.assertEqual(telemetry.records[0]["alias_of"], "model_output")
