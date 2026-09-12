"""Local media validation and lightweight visual-continuity checks."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .workflow import H3Request, frame_length


def probe_media(path: Path) -> dict[str, Any]:
    """Use ffprobe when present; report uncertainty instead of inventing facts."""

    path = path.resolve()
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        result.update({"valid": False, "status": "missing"})
        return result
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        result.update(
            {
                "valid": True,
                "status": "file_present_ffprobe_unavailable",
                "validation_uncertain": True,
                "size_bytes": path.stat().st_size,
            }
        )
        return result
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        result.update({"valid": False, "status": "ffprobe_failed", "error": completed.stderr[-2000:]})
        return result
    payload = json.loads(completed.stdout or "{}")
    streams = payload.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    duration = _float_or_none((payload.get("format") or {}).get("duration"))
    fps = _parse_rate(video.get("r_frame_rate")) if video else None
    result.update(
        {
            "valid": bool(video),
            "status": "valid" if video else "no_video_stream",
            "size_bytes": path.stat().st_size,
            "duration_seconds": duration,
            "width": video.get("width") if video else None,
            "height": video.get("height") if video else None,
            "fps": fps,
            "frame_count": _int_or_none(video.get("nb_frames")) if video else None,
            "video_codec": video.get("codec_name") if video else None,
            "audio_present": bool(audio),
            "audio_codec": audio.get("codec_name") if audio else None,
        }
    )
    return result


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_rate(value: Any) -> float | None:
    if not isinstance(value, str) or "/" not in value:
        return _float_or_none(value)
    numerator, denominator = value.split("/", 1)
    try:
        return float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def extract_keyframes(video_path: Path, output_dir: Path, *, count: int = 3) -> list[Path]:
    """Extract a few frames for optional human/vision review."""

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not video_path.is_file():
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / f"keyframe_{index:02d}.jpg" for index in range(count)]
    duration = probe_media(video_path).get("duration_seconds") or 1.0
    for index, output in enumerate(outputs):
        timestamp = duration * index / max(count - 1, 1)
        completed = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(output),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if completed.returncode != 0 and output.exists():
            output.unlink()
    return [output for output in outputs if output.is_file()]


def validate_video(video_path: Path, request: H3Request) -> dict[str, Any]:
    """Validate hard media facts and explicitly label visual checks as uncertain."""

    probe = probe_media(video_path)
    expected = frame_length(request)
    checks: dict[str, Any] = {
        "file_present": probe.get("exists", False),
        "video_stream": probe.get("valid", False),
        "audio_stream": probe.get("audio_present"),
        "expected_frames": expected["actual_frames"],
        "frame_count_matches": None,
        "expected_fps": expected["fps"],
        "fps_matches": None,
        "expected_duration_seconds": expected["actual_seconds"],
        "duration_within_tolerance": None,
        "resolution_matches_request": None,
        "identity_continuity": "uncertain_without_vision_review",
        "soul_spirit_transition": "not_requested",
    }
    checks["consistency_report"] = {
        "character": "uncertain_without_vision_review" if request.character_references else "not_requested",
        "outfit": "uncertain_without_vision_review" if request.outfit_references else "not_requested",
        "scene": "uncertain_without_vision_review" if request.scene_references else "not_requested",
        "motion": "uncertain_without_vision_review" if request.motion_references else "not_requested",
    }
    duration = probe.get("duration_seconds")
    if isinstance(duration, (int, float)):
        checks["duration_within_tolerance"] = abs(duration - expected["actual_seconds"]) <= 1.0
    frame_count = probe.get("frame_count")
    if isinstance(frame_count, int):
        checks["frame_count_matches"] = abs(frame_count - expected["actual_frames"]) <= 1
    fps = probe.get("fps")
    if isinstance(fps, (int, float)):
        checks["fps_matches"] = abs(fps - expected["fps"]) <= 0.5
    try:
        from .workflow import dimensions

        width, height = dimensions(request)
        checks["expected_width"] = width
        checks["expected_height"] = height
        if probe.get("width") and probe.get("height"):
            checks["resolution_matches_request"] = probe["width"] == width and probe["height"] == height
    except Exception as exc:
        checks["resolution_error"] = str(exc)
    prompt_lower = request.prompt.lower()
    if "soul spirit" in prompt_lower or "personified soul" in prompt_lower:
        checks["soul_spirit_transition"] = "uncertain_without_vision_review"
    hard_checks = [checks["file_present"], checks["video_stream"]]
    for key in ("audio_stream", "frame_count_matches", "fps_matches"):
        if checks[key] is not None:
            hard_checks.append(checks[key])
    if checks["duration_within_tolerance"] is not None:
        hard_checks.append(checks["duration_within_tolerance"])
    if checks["resolution_matches_request"] is not None:
        hard_checks.append(checks["resolution_matches_request"])
    return {
        "status": "passed" if all(hard_checks) else "failed",
        "probe": probe,
        "checks": checks,
        "validation_uncertain": True,
        "note": "Identity, continuity, and transformation quality require human or vision-model review of extracted keyframes.",
    }


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path
