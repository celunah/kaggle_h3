import unittest

from kaggle_h3.telemetry import summarize_samples


class TelemetryTests(unittest.TestCase):
    def test_summary_keeps_both_gpus_and_cpu_headroom(self):
        samples = [
            {
                "timestamp_utc": "a",
                "gpus": [
                    {"index": 0, "name": "T4", "used_mib": 1000, "total_mib": 15360, "utilization_percent": 30},
                    {"index": 1, "name": "T4", "used_mib": 2000, "total_mib": 15360, "utilization_percent": 40},
                ],
                "cpu": {"available_gib": 25.0},
            },
            {
                "timestamp_utc": "b",
                "gpus": [
                    {"index": 0, "name": "T4", "used_mib": 3000, "total_mib": 15360, "utilization_percent": 80},
                    {"index": 1, "name": "T4", "used_mib": 4000, "total_mib": 15360, "utilization_percent": 90},
                ],
                "cpu": {"available_gib": 24.5},
            },
        ]
        result = summarize_samples(samples)
        self.assertEqual(len(result["peak_by_gpu"]), 2)
        self.assertEqual(result["minimum_cpu_available_gib"], 24.5)
        self.assertEqual(result["cpu_headroom_limit_gib"], 24.0)


if __name__ == "__main__":
    unittest.main()
