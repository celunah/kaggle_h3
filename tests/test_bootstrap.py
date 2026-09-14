import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kaggle_h3.bootstrap import (
    comfy_launch_command,
    configure_comfyui_model_paths,
    download_selected_models,
    ensure_github_checkout,
    install_h3_adapter_node,
    model_file_status,
    required_model_files,
    stage_smoke_assets,
    start_comfyui,
)


class BootstrapTests(unittest.TestCase):
    def test_h3_support_modules_are_not_installed_as_custom_nodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            comfy_root = root / "ComfyUI"
            result = install_h3_adapter_node(comfy_root, Path(__file__).parents[1])

            custom_nodes = comfy_root / "custom_nodes"
            self.assertEqual(
                sorted(path.name for path in custom_nodes.iterdir()),
                [
                    "kaggle_h3_adapter_catalog.json",
                    "kaggle_h3_adapters.py",
                    "kaggle_h3_conditioning.py",
                ],
            )
            support_dir = comfy_root / "kaggle_h3_support"
            self.assertTrue((support_dir / "__init__.py").is_file())
            self.assertTrue((support_dir / "phase_runtime.py").is_file())
            self.assertTrue((support_dir / "diffusion_telemetry.py").is_file())
            self.assertEqual(result["support_package"], str(support_dir))
            script = """
import importlib.util
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
node_path = root / "custom_nodes" / "kaggle_h3_adapters.py"
spec = importlib.util.spec_from_file_location("installed_h3_node", node_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert len(module.NODE_CLASS_MAPPINGS) == 10
"""
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            subprocess.run(
                [sys.executable, "-c", script, str(comfy_root)],
                cwd=comfy_root,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )

    def test_public_tunnel_mode_adds_explicit_cors_flag(self):
        command = comfy_launch_command(
            Path("/tmp/ComfyUI"),
            8188,
            listen_host="0.0.0.0",
            enable_cors_header="*",
        )
        self.assertEqual(command[-2:], ["--enable-cors-header", "*"])

    def test_external_model_tree_is_used_for_status_and_comfy_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            comfy_root = root / "ComfyUI"
            models_root = root / "scratch" / "models"
            for directory, filename in required_model_files("FL2VA").values():
                path = models_root / directory / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test")

            status = model_file_status(comfy_root, "FL2VA", models_root=models_root)
            self.assertTrue(status["all_present"])
            self.assertTrue(all(item["path"].startswith(str(models_root)) for item in status["files"].values()))

            result = configure_comfyui_model_paths(comfy_root, models_root)
            self.assertEqual(result["status"], "written")
            config = (comfy_root / "extra_model_paths.yaml").read_text(encoding="utf-8")
            self.assertIn("base_path: " + str(models_root.parent).replace("\\", "/"), config)
            self.assertIn("diffusion_models: models/diffusion_models/", config)
            self.assertIn("text_encoders: models/text_encoders/", config)
            self.assertIn("vae: models/vae/", config)

    def test_smoke_assets_are_staged_into_comfy_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            comfy_root = Path(temporary) / "ComfyUI"
            result = stage_smoke_assets(comfy_root, Path(__file__).parents[1])
            self.assertEqual(result["status"], "staged")
            self.assertTrue((comfy_root / "input" / "CHARACTER_REFERENCE.png").is_file())
            self.assertTrue((comfy_root / "input" / "SCENE_REFERENCE.png").is_file())

    def test_default_model_download_defers_diffusion_to_comfy_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models_root = root / "models"
            for directory, filename in required_model_files(
                "Ref2VA", include_diffusion=False
            ).values():
                path = models_root / directory / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"shared")

            result = download_selected_models(root / "ComfyUI", "Ref2VA", models_root=models_root)

            self.assertEqual(result["download_status"], "already_present")
            self.assertEqual(result["diffusion_model"]["status"], "deferred_to_comfy_loader")
            self.assertNotIn("diffusion_model", result["files"])

    def test_github_checkout_dry_run_does_not_need_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "kaggle_h3_github"
            result = ensure_github_checkout(
                "https://github.com/example/kaggle_h3.git",
                destination,
                ref="main",
                dry_run=True,
            )
        self.assertEqual(result["status"], "would_clone")
        self.assertEqual(result["requested_ref"], "main")

    def test_github_checkout_refuses_non_git_existing_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "kaggle_h3_github"
            destination.mkdir()
            (destination / "old.txt").write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "non-empty path"):
                ensure_github_checkout(
                    "https://github.com/example/kaggle_h3.git", destination
                )


    def test_phase_launch_keeps_gpu_vae_opt_in_and_fallback_cpu_vae_explicit(self):
        comfy_root = Path("/kaggle/working/ComfyUI")
        phase_command = comfy_launch_command(
            comfy_root,
            8188,
            visible_device_ids=[0, 1],
        )
        fallback_command = comfy_launch_command(
            comfy_root,
            8188,
            visible_device_ids=[0],
            cpu_vae=True,
        )
        self.assertNotIn("--cpu-vae", phase_command)
        self.assertIn("--cpu-vae", fallback_command)
        self.assertIn("--disable-cuda-malloc", phase_command)
        self.assertIn("--cuda-device", phase_command)
        self.assertIn("0,1", phase_command)

    def test_dry_run_records_two_gpu_visibility_for_comfyui(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = start_comfyui(
                Path(temporary) / "ComfyUI",
                execution_plan={"visible_device_ids": [0, 1]},
                log_dir=Path(temporary) / "logs",
                dry_run=True,
            )
        self.assertEqual(result["cuda_visible_devices"], "0,1")
        self.assertEqual(result["visible_device_ids"], [0, 1])
        cuda_index = result["command"].index("--cuda-device")
        self.assertEqual(result["command"][cuda_index + 1], "0,1")

    def test_comfyui_output_is_teeable_to_console_and_log(self):
        from unittest.mock import patch

        from kaggle_h3.bootstrap import _stream_comfyui_output

        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "comfyui.log"
            output = io.StringIO("[Kaggle H3][diffusion] step\nready\n")
            with patch("builtins.print") as print_mock:
                _stream_comfyui_output(output, log_path, show_console=True)

            self.assertEqual(log_path.read_text(encoding="utf-8"), output.getvalue())
            print_mock.assert_any_call(
                "[ComfyUI] [Kaggle H3][diffusion] step\n",
                end="",
                flush=True,
            )


if __name__ == "__main__":
    unittest.main()
