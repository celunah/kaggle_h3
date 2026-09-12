"""Small, dependency-light client for the ComfyUI local API."""

from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

import requests


class ComfyAPIError(RuntimeError):
    """An actionable ComfyUI API failure."""


def _walk(value: Any) -> Iterator[Any]:
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


class ComfyClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.client_id = str(uuid.uuid4())

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", self.timeout)
        response = self.session.request(method, f"{self.base_url}{path}", **kwargs)
        if not response.ok:
            body = response.text[-2000:]
            raise ComfyAPIError(f"ComfyUI {method} {path} failed ({response.status_code}): {body}")
        return response

    def health(self) -> dict[str, Any]:
        response = self._request("GET", "/system_stats")
        return response.json()

    def object_info(self) -> dict[str, Any]:
        response = self._request("GET", "/object_info")
        return response.json()

    def check_required_nodes(self, node_ids: list[str]) -> dict[str, Any]:
        info = self.object_info()
        present = [node_id for node_id in node_ids if node_id in info]
        missing = [node_id for node_id in node_ids if node_id not in info]
        return {
            "required_nodes": node_ids,
            "present": present,
            "missing": missing,
            "all_present": not missing,
            "native_custom_node_dependencies": [],
        }

    def queue_prompt(self, workflow: dict[str, Any]) -> str:
        response = self._request(
            "POST",
            "/prompt",
            json={"prompt": workflow, "client_id": self.client_id},
        )
        payload = response.json()
        if payload.get("node_errors"):
            raise ComfyAPIError(json.dumps(payload, indent=2))
        prompt_id = payload.get("prompt_id")
        if not prompt_id:
            raise ComfyAPIError(f"ComfyUI did not return prompt_id: {payload}")
        return str(prompt_id)

    def history(self, prompt_id: str) -> dict[str, Any]:
        response = self._request("GET", f"/history/{prompt_id}")
        return response.json()

    def wait_for_completion(
        self,
        prompt_id: str,
        *,
        poll_interval: float = 2.0,
        timeout: float = 3600.0,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            history = self.history(prompt_id)
            record = history.get(prompt_id)
            if record:
                status = record.get("status", {})
                if status.get("status_str") == "error" or status.get("completed") is False:
                    if status.get("status_str") == "error":
                        raise ComfyAPIError(
                            f"ComfyUI prompt {prompt_id} failed: "
                            f"{json.dumps(status, default=str)[:4000]}"
                        )
                if status.get("completed") or record.get("outputs"):
                    return record
            time.sleep(poll_interval)
        raise TimeoutError(f"Timed out waiting for ComfyUI prompt {prompt_id}")

    def view_file(self, filename: str, subfolder: str = "", file_type: str = "output") -> bytes:
        response = self._request(
            "GET",
            "/view",
            params={"filename": filename, "subfolder": subfolder, "type": file_type},
        )
        return response.content


def wait_for_health(
    client: ComfyClient, *, timeout: float = 180.0, poll_interval: float = 2.0
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return client.health()
        except Exception as exc:  # server may still be importing nodes
            last_error = exc
            time.sleep(poll_interval)
    raise TimeoutError(f"ComfyUI did not become healthy: {last_error}")


def stage_input_files(
    paths: list[Path], comfy_root: Path, run_id: str
) -> dict[Path, str]:
    """Copy local references into ComfyUI/input without altering their pixels."""

    input_dir = comfy_root / "input" / "kaggle_h3" / run_id
    input_dir.mkdir(parents=True, exist_ok=True)
    result: dict[Path, str] = {}
    for index, source in enumerate(paths, start=1):
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.name).strip(".") or "asset"
        destination = input_dir / f"{index:02d}_{safe_name}"
        shutil.copy2(source, destination)
        result[source] = str(destination.relative_to(comfy_root / "input")).replace("\\", "/")
    return result


def iter_saved_results(history_record: dict[str, Any]) -> Iterator[dict[str, str]]:
    """Find SaveVideo's result metadata across ComfyUI version variations."""

    for value in _walk(history_record.get("outputs", history_record)):
        if not isinstance(value, dict):
            continue
        filename = value.get("filename")
        if not isinstance(filename, str) or not filename.lower().endswith(".mp4"):
            continue
        subfolder = value.get("subfolder", "")
        file_type = value.get("type", "output")
        yield {
            "filename": filename,
            "subfolder": str(subfolder or ""),
            "type": str(file_type or "output"),
        }


def collect_video_outputs(
    client: ComfyClient,
    history_record: dict[str, Any],
    *,
    comfy_root: Path,
    destination_dir: Path,
    run_id: str,
) -> list[Path]:
    """Materialize MP4 outputs in the notebook's predictable results folder."""

    destination_dir.mkdir(parents=True, exist_ok=True)
    results: list[Path] = []
    seen: set[tuple[str, str, str]] = set()
    for index, result in enumerate(iter_saved_results(history_record), start=1):
        key = (result["filename"], result["subfolder"], result["type"])
        if key in seen:
            continue
        seen.add(key)
        relative = Path(result["subfolder"]) / result["filename"]
        candidate = comfy_root / result["type"] / relative
        destination = destination_dir / f"{run_id}_{index:02d}.mp4"
        if candidate.is_file():
            shutil.copy2(candidate, destination)
        else:
            destination.write_bytes(
                client.view_file(
                    result["filename"], result["subfolder"], result["type"]
                )
            )
        results.append(destination)
    if not results:
        raise ComfyAPIError(
            "ComfyUI completed without an MP4 SaveVideo result. "
            f"History keys: {sorted(history_record.keys())}"
        )
    return results
