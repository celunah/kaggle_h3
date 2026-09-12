"""Host and per-GPU telemetry for one H3 request."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sample_memory() -> dict[str, Any]:
    sample: dict[str, Any] = {"timestamp_utc": _now(), "gpus": [], "cpu": {}}
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            completed = subprocess.run(
                [
                    smi,
                    "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in completed.stdout.splitlines():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) != 6:
                    continue
                def number(value: str) -> float | int | None:
                    try:
                        return float(value) if "." in value else int(value)
                    except ValueError:
                        return None

                sample["gpus"].append(
                    {
                        "index": int(fields[0]),
                        "name": fields[1],
                        "used_mib": number(fields[2]),
                        "total_mib": number(fields[3]),
                        "utilization_percent": number(fields[4]),
                        "power_w": number(fields[5]),
                    }
                )
        except Exception as exc:
            sample["gpu_query_error"] = str(exc)

    try:
        import psutil  # type: ignore

        memory = psutil.virtual_memory()
        sample["cpu"] = {
            "available_bytes": int(memory.available),
            "available_gib": round(memory.available / 2**30, 2),
            "used_bytes": int(memory.used),
            "used_gib": round(memory.used / 2**30, 2),
            "total_bytes": int(memory.total),
        }
    except Exception as exc:  # pragma: no cover - optional package
        sample["cpu_query_error"] = str(exc)
    return sample


class TelemetryRecorder:
    """Sample nvidia-smi and host RAM while one backend job is running."""

    def __init__(self, output_path: Path, *, interval_seconds: float = 1.0):
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            while not self._stop.is_set():
                value = sample_memory()
                self.samples.append(value)
                handle.write(json.dumps(value) + "\n")
                handle.flush()
                self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="h3-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if not self.samples:
            self.samples.append(sample_memory())
        return summarize_samples(self.samples)

    def __enter__(self) -> "TelemetryRecorder":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    per_gpu: dict[str, dict[str, Any]] = {}
    for sample in samples:
        for gpu in sample.get("gpus", []):
            key = str(gpu.get("index"))
            current = per_gpu.setdefault(
                key,
                {
                    "index": gpu.get("index"),
                    "name": gpu.get("name"),
                    "peak_used_mib": 0,
                    "peak_utilization_percent": 0,
                    "total_mib": gpu.get("total_mib"),
                    "samples": 0,
                },
            )
            current["samples"] += 1
            if isinstance(gpu.get("used_mib"), (int, float)):
                current["peak_used_mib"] = max(
                    current["peak_used_mib"], gpu["used_mib"]
                )
            if isinstance(gpu.get("utilization_percent"), (int, float)):
                current["peak_utilization_percent"] = max(
                    current["peak_utilization_percent"], gpu["utilization_percent"]
                )
    for value in per_gpu.values():
        value["peak_used_gib"] = round(value["peak_used_mib"] / 1024, 3)
        value["safety_limit_gib"] = 14.0
        value["safety_headroom_pass"] = value["peak_used_gib"] <= 14.0
    cpu_available = [
        sample["cpu"]["available_gib"]
        for sample in samples
        if isinstance(sample.get("cpu", {}).get("available_gib"), (int, float))
    ]
    minimum_cpu = min(cpu_available) if cpu_available else None
    return {
        "sample_count": len(samples),
        "started_at_utc": samples[0].get("timestamp_utc"),
        "ended_at_utc": samples[-1].get("timestamp_utc"),
        "peak_by_gpu": list(per_gpu.values()),
        "minimum_cpu_available_gib": minimum_cpu,
        "cpu_headroom_limit_gib": 24.0,
        "cpu_headroom_target_gib": 26.0,
        "cpu_headroom_pass": minimum_cpu is not None and minimum_cpu >= 24.0,
    }
