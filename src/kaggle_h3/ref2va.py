"""Dependency-light Ref2VA duration and canvas controls.

The native MiniMax H3 node uses a 32-pixel spatial canvas grid and a
``17*k + 5`` temporal grid.  Keeping these calculations outside ComfyUI
allows the custom node and workflow builder to use exactly the same rules.
"""

from __future__ import annotations

import math


H3_FPS = 24
H3_CANVAS_MULTIPLE = 32
H3_MIN_FRAMES = 39
H3_MAX_SECONDS = 15.0

# These are nominal quality tiers.  The actual canvas is the nearest safe
# H3-aligned canvas, so the 720p 16:9 tier is 1280x704 rather than 1280x720.
# 720 is not divisible by H3's required 32-pixel canvas multiple.
REF2VA_SIZE_PRESETS: dict[str, dict[str, tuple[int, int]]] = {
    "240p": {
        "16:9": (448, 256),
        "4:3": (352, 256),
    },
    "360p": {
        "16:9": (640, 352),
        "4:3": (480, 352),
    },
    "480p": {
        "16:9": (864, 480),
        "4:3": (640, 480),
    },
    "720p": {
        "16:9": (1280, 704),
        "4:3": (960, 704),
    },
}

QUALITY_MODE_TO_REF2VA_PRESET = {
    "quick": "360p",
    "medium": "480p",
    "full": "720p",
}


def resolve_ref2va_dimensions(size_preset: str, aspect_ratio: str) -> tuple[int, int]:
    """Return a safe H3 canvas for a named preset and aspect toggle."""

    preset = str(size_preset).strip().lower()
    ratio = str(aspect_ratio).strip()
    try:
        return REF2VA_SIZE_PRESETS[preset][ratio]
    except KeyError as exc:
        raise ValueError(
            "Ref2VA size_preset must be one of 240p, 360p, 480p, or 720p "
            "and aspect_ratio must be 16:9 or 4:3. "
            f"Got size_preset={size_preset!r}, aspect_ratio={aspect_ratio!r}."
        ) from exc


def resolve_ref2va_length(seconds: float) -> dict[str, float | int | str]:
    """Resolve seconds to H3's aligned frame count without silent mismatch."""

    try:
        requested_seconds = float(seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Ref2VA seconds must be numeric; got {seconds!r}") from exc
    if not math.isfinite(requested_seconds):
        raise ValueError("Ref2VA seconds must be finite")
    minimum_seconds = H3_MIN_FRAMES / H3_FPS
    if requested_seconds < minimum_seconds or requested_seconds > H3_MAX_SECONDS:
        raise ValueError(
            f"Ref2VA seconds must be between {minimum_seconds:.3f} and "
            f"{H3_MAX_SECONDS:g}; got {requested_seconds:g}."
        )

    requested_frames = max(H3_MIN_FRAMES, int(round(requested_seconds * H3_FPS)))
    # Align upward to 17*k+5, matching nodes_minimax_h3.align_frame_count.
    frame_count = requested_frames
    while frame_count % 17 != 5:
        frame_count += 1
    return {
        "requested_seconds": requested_seconds,
        "actual_frames": frame_count,
        "actual_seconds": round(frame_count / H3_FPS, 4),
        "fps": H3_FPS,
        "alignment": "17*k+5",
    }


def quality_mode_to_ref2va_preset(quality_mode: str) -> str:
    """Map the existing API quality names to the custom node's presets."""

    try:
        return QUALITY_MODE_TO_REF2VA_PRESET[str(quality_mode).strip().lower()]
    except KeyError as exc:
        raise ValueError(
            "Ref2VA quality_mode must be quick, medium, or full; "
            f"got {quality_mode!r}."
        ) from exc
