"""Just-in-time management of the mutually exclusive H3 diffusion models.

The Kaggle notebook downloads only the shared H3 assets.  This module is used
by the ComfyUI loader when a workflow is executed: it resolves the selected
variant, removes only the other known H3 diffusion checkpoint, streams the
pinned file directly into ComfyUI's diffusion-model directory, and then lets
ComfyUI's native ``UNETLoader`` load it.

The downloader intentionally does not use the Hugging Face cache.  Keeping
the temporary ``.part`` file beside the destination avoids a second 20 GiB
copy in Kaggle scratch storage and makes the one-variant storage policy
observable.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


H3_DIFFUSION_REPOSITORY = "Comfy-Org/MiniMax-H3"
H3_DIFFUSION_REPOSITORY_DIRECTORY = "diffusion_models"
H3_DIFFUSION_REVISION = os.environ.get(
    "KAGGLE_H3_MODEL_REVISION",
    "a98869194787969724c7425d95d0ed73ce9202af",
)


@dataclass(frozen=True)
class H3DiffusionModelSpec:
    variant: str
    filename: str
    repository: str = H3_DIFFUSION_REPOSITORY
    revision: str = H3_DIFFUSION_REVISION
    sha256: str | None = None


H3_DIFFUSION_MODEL_SPECS: dict[str, H3DiffusionModelSpec] = {
    "FL2VA": H3DiffusionModelSpec(
        variant="FL2VA",
        filename="minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    ),
    "Ref2VA": H3DiffusionModelSpec(
        variant="Ref2VA",
        filename="minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    ),
}


class H3DiffusionModelError(RuntimeError):
    """Raised when the selected H3 checkpoint cannot be prepared safely."""


@dataclass(frozen=True)
class H3DiffusionModelResult:
    spec: H3DiffusionModelSpec
    path: Path
    downloaded: bool
    removed: tuple[Path, ...]
    free_bytes: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "variant": self.spec.variant,
            "repository": self.spec.repository,
            "revision": self.spec.revision,
            "filename": self.spec.filename,
            "path": str(self.path),
            "downloaded": self.downloaded,
            "removed": [str(path) for path in self.removed],
            "free_bytes": self.free_bytes,
        }


def canonical_h3_diffusion_variant(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "")
    if normalized in {"fl2va", "fl2v", "t2va", "t2v"}:
        return "FL2VA"
    if normalized in {"ref2va", "ref2v", "r2va", "r2v"}:
        return "Ref2VA"
    raise H3DiffusionModelError(
        f"Unsupported H3 diffusion variant {value!r}; choose FL2VA or Ref2VA."
    )


def h3_diffusion_spec(value: str) -> H3DiffusionModelSpec:
    return H3_DIFFUSION_MODEL_SPECS[canonical_h3_diffusion_variant(value)]


def h3_diffusion_directory_from_comfy() -> Path:
    """Find the configured ComfyUI diffusion-model directory.

    ``KAGGLE_H3_MODEL_ROOT`` is set by the notebook to the external
    ``/kaggle/tmp`` model tree.  The folder_paths fallback keeps the node
    usable in a normal ComfyUI installation and honours extra_model_paths.
    """

    configured_root = os.environ.get("KAGGLE_H3_MODEL_ROOT", "").strip()
    if configured_root:
        candidate = Path(configured_root).expanduser()
        if candidate.name.lower() != "diffusion_models":
            candidate = candidate / "diffusion_models"
        return candidate.resolve()

    try:
        import folder_paths  # type: ignore

        candidates = [Path(value).expanduser() for value in folder_paths.get_folder_paths("diffusion_models")]
        if candidates:
            existing = [path for path in candidates if path.is_dir()]
            return (existing[0] if existing else candidates[0]).resolve()
        models_dir = getattr(folder_paths, "models_dir", None)
        if models_dir:
            return (Path(models_dir) / "diffusion_models").resolve()
    except Exception as exc:
        raise H3DiffusionModelError(
            "ComfyUI did not expose a diffusion_models folder; set "
            "KAGGLE_H3_MODEL_ROOT to the external H3 models root."
        ) from exc
    raise H3DiffusionModelError(
        "ComfyUI did not expose a diffusion_models folder; set "
        "KAGGLE_H3_MODEL_ROOT to the external H3 models root."
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hf_resolve_url(spec: H3DiffusionModelSpec) -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    repository = urllib.parse.quote(spec.repository, safe="/")
    # The local destination is already the diffusion_models directory, but
    # Hugging Face resolves paths relative to the repository root.
    repository_path = f"{H3_DIFFUSION_REPOSITORY_DIRECTORY}/{spec.filename}"
    filename = urllib.parse.quote(repository_path, safe="/")
    revision = urllib.parse.quote(spec.revision, safe="")
    return f"{endpoint}/{repository}/resolve/{revision}/{filename}?download=true"


Downloader = Callable[[H3DiffusionModelSpec, Path], None]


class H3DiffusionModelManager:
    """Prepare one H3 diffusion variant without retaining both checkpoints."""

    _switch_lock = threading.Lock()

    def __init__(self, diffusion_dir: Path, *, downloader: Downloader | None = None):
        self.diffusion_dir = Path(diffusion_dir).expanduser().resolve()
        self.downloader = downloader

    def _known_paths(self) -> dict[str, Path]:
        return {
            variant: (self.diffusion_dir / spec.filename).resolve()
            for variant, spec in H3_DIFFUSION_MODEL_SPECS.items()
        }

    def _validate_known_path(self, path: Path) -> None:
        if path.parent != self.diffusion_dir:
            raise H3DiffusionModelError(f"Refusing an H3 path outside the configured directory: {path}")

    def _verify(self, path: Path, spec: H3DiffusionModelSpec) -> None:
        self._validate_known_path(path)
        if not path.is_file() or path.stat().st_size <= 0:
            raise H3DiffusionModelError(f"Downloaded H3 diffusion checkpoint is empty or missing: {path}")
        if path.name not in {spec.filename, spec.filename + ".part"}:
            raise H3DiffusionModelError(
                f"Refusing an unexpected H3 diffusion checkpoint filename: {path.name}"
            )
        if spec.sha256:
            actual = _sha256(path)
            if actual.lower() != spec.sha256.lower():
                raise H3DiffusionModelError(
                    f"SHA-256 mismatch for {spec.variant} diffusion checkpoint: "
                    f"expected {spec.sha256}, got {actual}"
                )

    def _download(self, spec: H3DiffusionModelSpec, destination: Path) -> None:
        url = _hf_resolve_url(spec)
        headers: dict[str, str] = {}
        token = os.environ.get("HF_TOKEN", "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        print(
            f"[Kaggle H3] Downloading diffusion model {spec.variant}: "
            f"{spec.repository}/{H3_DIFFUSION_REPOSITORY_DIRECTORY}/{spec.filename} "
            f"@ {spec.revision}",
            flush=True,
        )
        request = urllib.request.Request(url, headers=headers)
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as handle:
                raw_length = response.headers.get("Content-Length")
                total = int(raw_length) if raw_length and raw_length.isdigit() else None
                completed = 0
                last_report = 0.0
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    completed += len(chunk)
                    now = time.monotonic()
                    if now - last_report >= 2.0:
                        if total:
                            print(
                                f"[Kaggle H3] Diffusion download {completed / 2**30:.2f}/"
                                f"{total / 2**30:.2f} GiB ({completed / total * 100:.1f}%)",
                                flush=True,
                            )
                        else:
                            print(
                                f"[Kaggle H3] Diffusion download {completed / 2**30:.2f} GiB",
                                flush=True,
                            )
                        last_report = now
        except urllib.error.HTTPError as exc:
            raise H3DiffusionModelError(
                f"Hugging Face returned HTTP {exc.code} for "
                f"{spec.repository}/{H3_DIFFUSION_REPOSITORY_DIRECTORY}/{spec.filename} "
                f"at pinned revision {spec.revision}. Check Internet access and HF_TOKEN."
            ) from exc
        except Exception as exc:
            raise H3DiffusionModelError(
                f"Could not download H3 {spec.variant} diffusion model from {url}: {exc}"
            ) from exc
        elapsed = max(time.monotonic() - started, 1e-6)
        print(
            f"[Kaggle H3] Diffusion model download complete: "
            f"{destination.stat().st_size / 2**30:.2f} GiB in {elapsed:.1f}s.",
            flush=True,
        )

    def switch(self, variant: str) -> H3DiffusionModelResult:
        spec = h3_diffusion_spec(variant)
        self.diffusion_dir.mkdir(parents=True, exist_ok=True)
        paths = self._known_paths()
        selected = paths[spec.variant]
        removed: list[Path] = []
        with self._switch_lock:
            for other_variant, path in paths.items():
                if other_variant == spec.variant:
                    continue
                self._validate_known_path(path)
                if path.is_dir() and not path.is_symlink():
                    raise H3DiffusionModelError(
                        f"Refusing to delete a directory named like an H3 checkpoint: {path}"
                    )
                if path.is_file() or path.is_symlink():
                    path.unlink()
                    removed.append(path)
                    print(f"[Kaggle H3] Removed inactive diffusion checkpoint: {path}", flush=True)

            if selected.is_dir() and not selected.is_symlink():
                raise H3DiffusionModelError(
                    f"Refusing to replace a directory named like an H3 checkpoint: {selected}"
                )
            if selected.is_file():
                if selected.stat().st_size <= 0:
                    selected.unlink()
                else:
                    self._verify(selected, spec)
                    print(f"[Kaggle H3] Diffusion model cached: {selected}", flush=True)
                    return H3DiffusionModelResult(
                        spec=spec,
                        path=selected,
                        downloaded=False,
                        removed=tuple(removed),
                        free_bytes=self._free_bytes(),
                    )

            temporary = selected.with_name(selected.name + ".part")
            self._validate_known_path(temporary)
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()
            try:
                if self.downloader is not None:
                    self.downloader(spec, temporary)
                else:
                    self._download(spec, temporary)
                self._verify(temporary, spec)
                temporary.replace(selected)
            except H3DiffusionModelError:
                if temporary.exists() or temporary.is_symlink():
                    temporary.unlink()
                raise
            except Exception as exc:
                if temporary.exists() or temporary.is_symlink():
                    temporary.unlink()
                raise H3DiffusionModelError(
                    f"H3 {spec.variant} diffusion checkpoint preparation failed: {exc}"
                ) from exc

            print(f"[Kaggle H3] Diffusion model ready for ComfyUI: {selected}", flush=True)
            return H3DiffusionModelResult(
                spec=spec,
                path=selected,
                downloaded=True,
                removed=tuple(removed),
                free_bytes=self._free_bytes(),
            )

    def _free_bytes(self) -> int | None:
        try:
            return int(shutil.disk_usage(self.diffusion_dir).free)
        except OSError:
            return None
