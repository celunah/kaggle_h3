import tempfile
import unittest
from pathlib import Path

from kaggle_h3.model_manager import (
    H3DiffusionModelError,
    H3DiffusionModelManager,
    _hf_resolve_url,
    h3_diffusion_spec,
)


class DiffusionModelManagerTests(unittest.TestCase):
    def test_hugging_face_url_includes_comfy_diffusion_directory(self):
        spec = h3_diffusion_spec("Ref2VA")
        url = _hf_resolve_url(spec)
        self.assertIn(
            f"/resolve/{spec.revision}/diffusion_models/{spec.filename}?download=true",
            url,
        )

    def test_switch_removes_only_known_inactive_checkpoint_and_caches_selected(self):
        with tempfile.TemporaryDirectory() as temporary:
            diffusion_dir = Path(temporary) / "diffusion_models"
            diffusion_dir.mkdir()
            inactive = diffusion_dir / h3_diffusion_spec("FL2VA").filename
            inactive.write_bytes(b"old diffusion")
            unrelated = diffusion_dir / "other_model.safetensors"
            unrelated.write_bytes(b"keep me")
            calls = []

            def downloader(spec, destination):
                calls.append((spec, destination))
                destination.write_bytes(b"new diffusion")

            manager = H3DiffusionModelManager(diffusion_dir, downloader=downloader)
            first = manager.switch("Ref2VA")
            second = manager.switch("Ref2VA")

            self.assertTrue(first.downloaded)
            self.assertFalse(second.downloaded)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0].variant, "Ref2VA")
            self.assertEqual(calls[0][0].revision, h3_diffusion_spec("Ref2VA").revision)
            self.assertFalse(inactive.exists())
            self.assertTrue(unrelated.exists())
            self.assertTrue(first.path.is_file())

    def test_invalid_variant_is_rejected_before_any_delete(self):
        with tempfile.TemporaryDirectory() as temporary:
            diffusion_dir = Path(temporary)
            inactive = diffusion_dir / h3_diffusion_spec("FL2VA").filename
            inactive.write_bytes(b"keep")
            manager = H3DiffusionModelManager(diffusion_dir)

            with self.assertRaisesRegex(H3DiffusionModelError, "choose FL2VA or Ref2VA"):
                manager.switch("unknown")
            self.assertTrue(inactive.exists())

    def test_failed_download_removes_partial_file_but_not_unrelated_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            diffusion_dir = Path(temporary)
            unrelated = diffusion_dir / "unrelated.safetensors"
            unrelated.write_bytes(b"keep")

            def failing_downloader(_spec, destination):
                destination.write_bytes(b"partial")
                raise OSError("offline")

            manager = H3DiffusionModelManager(diffusion_dir, downloader=failing_downloader)
            with self.assertRaisesRegex(H3DiffusionModelError, "preparation failed"):
                manager.switch("FL2VA")
            self.assertFalse(
                (diffusion_dir / h3_diffusion_spec("FL2VA").filename).exists()
            )
            self.assertFalse(
                (diffusion_dir / (h3_diffusion_spec("FL2VA").filename + ".part")).exists()
            )
            self.assertTrue(unrelated.exists())


if __name__ == "__main__":
    unittest.main()
