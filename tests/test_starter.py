"""Kiểm tra pipeline starter/: hợp đồng với eval.py, cú pháp, inference và notebook.

Chạy từ thư mục gốc repo:
    python -X utf8 -m unittest discover -s tests -v  # Windows console Unicode
Các module trong starter/ không import torch ở mức module nên test này chạy được không cần GPU.
"""
import ast
import base64
from io import BytesIO
import json
import py_compile
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STARTER = ROOT / "starter"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STARTER))

import eval as ev  # noqa: E402
import train  # noqa: E402

K = ev.NUM_CLASSES


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class TestPredictionContract(unittest.TestCase):
    def test_saved_predictions_are_accepted_by_eval(self):
        rng = np.random.default_rng(0)
        n = 200
        y = rng.integers(0, K, n)
        probs = softmax(rng.normal(size=(n, K)))
        names = [f"img{i}.jpg" for i in range(n)]
        with tempfile.TemporaryDirectory() as d:
            path = ev.save_predictions(Path(d) / "sub" / "F01_seed0_test.csv", names, y, probs)
            pred = ev.read_pred(str(path))
            self.assertEqual(pred.seed, 0)
            np.testing.assert_array_equal(pred.y_pred, probs.argmax(1))
            np.testing.assert_allclose(pred.probs, probs, atol=1e-6)

    def test_save_predictions_rejects_logits(self):
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(10, K))
        with self.assertRaisesRegex(ValueError, "chuẩn hoá"):
            ev.save_predictions("unused.csv", [f"{i}.jpg" for i in range(10)], np.zeros(10, int), logits)

    def test_save_predictions_rejects_wrong_shape(self):
        with self.assertRaisesRegex(ValueError, "dạng"):
            ev.save_predictions("unused.csv", ["a.jpg"], [0], np.ones((1, 5)) / 5)


class TestTrainHelpers(unittest.TestCase):
    def test_pred_path_follows_eval_naming(self):
        cfg = train.Config(exp_id="F01", seed=2, pred_dir="predictions")
        self.assertEqual(train.pred_path(cfg, "test"), Path("predictions/F01_seed2_test.csv"))
        self.assertEqual(ev.parse_seed(str(train.pred_path(cfg, "test"))), 2)

    def test_run_dir(self):
        cfg = train.Config(exp_id="T03", seed=1, out_dir="runs")
        self.assertEqual(train.run_dir(cfg), Path("runs/T03/seed1"))

    def test_test_predictions_off_by_default(self):
        self.assertFalse(train.Config().save_test_predictions)

    def test_baseline_defaults_match_guide(self):
        c = train.Config()
        self.assertEqual((c.epochs, c.batch_size, c.lr_backbone, c.lr_head, c.weight_decay),
                         (12, 64, 1e-4, 1e-3, 0.05))

    def test_parse_overrides_handles_typed_values_and_rejects_unknown_fields(self):
        parsed = train.parse_overrides(["seed=1", "amp=false", "lr_head=0.002"])
        self.assertEqual(parsed, {"seed": 1, "amp": False, "lr_head": 0.002})
        with self.assertRaisesRegex(ValueError, "không tồn tại"):
            train.parse_overrides(["not_a_config_field=1"])


class TestStarterFiles(unittest.TestCase):
    def test_all_python_files_compile(self):
        for f in sorted(STARTER.glob("*.py")) + [ROOT / "eval.py"]:
            py_compile.compile(str(f), doraise=True)

    def test_starter_modules_have_no_not_implemented_stubs(self):
        for name in ("dataset.py", "model.py", "losses.py", "train.py", "inference.py", "benchmark.py"):
            tree = ast.parse((STARTER / name).read_text(encoding="utf-8"))
            stubs = [node for node in ast.walk(tree) if isinstance(node, ast.Raise)
                     and isinstance(node.exc, ast.Call)
                     and getattr(node.exc.func, "id", "") == "NotImplementedError"]
            self.assertEqual(stubs, [], f"{name} still contains NotImplementedError")

    def test_starter_has_no_complete_helper_modules(self):
        """The starter implementation does not depend on an undocumented records module."""
        self.assertFalse((STARTER / "records.py").exists())
        for f in STARTER.glob("*.py"):
            self.assertNotIn("import records", f.read_text(encoding="utf-8"), f.name)

    def test_notebook_is_valid_and_clean(self):
        nb = json.loads((STARTER / "lab_day2.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
        text = "\n".join("".join(c["source"]) for c in nb["cells"])
        self.assertIn("eval.py score", text)
        self.assertIn("eval.py grade", text)
        self.assertIn("b7b30f96d466fba86016aa5a26606e0f", text)  # MD5 của images.zip
        self.assertIn("SOURCE_BUNDLE_BASE64", text)
        self.assertIn("batch 32", text)
        self.assertIn("run_inference_only", text)
        self.assertIn("CHECKPOINT_PATH = None", text)
        self.assertNotIn("assert torch.cuda.is_available()", text)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                code = "".join(cell["source"])
                code = "\n".join(line for line in code.splitlines() if not line.startswith("%"))
                ast.parse(code)
        match = re.search(r'SOURCE_BUNDLE_BASE64 = """([A-Za-z0-9+/=]+)"""', text)
        self.assertIsNotNone(match)
        with ZipFile(BytesIO(base64.b64decode(match.group(1)))) as archive:
            names = set(archive.namelist())
            self.assertIn("eval.py", names)
            self.assertIn("starter/lab_workflow.py", names)
            self.assertIn("starter/inference.py", names)


class TestInferenceHelpers(unittest.TestCase):
    def test_inference_only_uses_supplied_checkpoints_without_training(self):
        import lab_workflow
        with tempfile.TemporaryDirectory() as directory:
            primary = Path(directory) / "primary.pth"
            second = Path(directory) / "second.pth"
            third = Path(directory) / "third.pth"
            for path in (primary, second, third):
                path.write_bytes(b"checkpoint")
            expected = {"rows": [], "realtime_candidate": None}
            with patch.object(lab_workflow, "sweep_inference", return_value=expected) as sweep, \
                    patch.object(lab_workflow, "_cached_run", side_effect=AssertionError("training called")):
                result = lab_workflow.run_inference_only(
                    primary, "convnext_tiny", "images", "labels", directory,
                    ensemble_checkpoints=[
                        {"backbone": "swin_tiny_patch4_window7_224", "checkpoint": second},
                        {"backbone": "resnet50", "checkpoint": third},
                    ])
            self.assertIs(result, expected)
            self.assertEqual(len(sweep.call_args.args[1]), 3)  # primary + two ensemble checkpoints

    def test_inference_only_rejects_missing_checkpoint_without_training(self):
        import lab_workflow
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.pth"
            with patch.object(lab_workflow, "sweep_inference") as sweep:
                with self.assertRaisesRegex(FileNotFoundError, "Checkpoint không tồn tại"):
                    lab_workflow.run_inference_only(
                        missing, "convnext_tiny", "images", "labels", directory)
            sweep.assert_not_called()

    def test_probability_and_logit_aggregation_are_normalized(self):
        from inference import aggregate_views
        logits = [np.array([[3.0, 0.0, -1.0]]), np.array([[0.0, 2.0, -1.0]])]
        for space in ("prob", "logit"):
            probs = aggregate_views(logits, space)
            self.assertAlmostEqual(float(probs.sum()), 1.0)
            self.assertTrue(np.isfinite(probs).all())

    def test_temperature_is_positive_and_preserves_top1(self):
        from inference import apply_temperature, fit_temperature
        logits = np.array([[4.0, 0.0, -2.0], [0.0, 3.0, -1.0], [1.0, -1.0, 2.0]])
        labels = np.array([0, 1, 2])
        temperature = fit_temperature(logits, labels)
        probs = apply_temperature(logits, temperature)
        self.assertGreater(temperature, 0)
        np.testing.assert_array_equal(probs.argmax(1), logits.argmax(1))

    def test_multicrop_returns_four_corners_and_center(self):
        import torch
        from inference import views_multicrop
        views = views_multicrop(torch.zeros(2, 3, 256, 256), 224)
        self.assertEqual(len(views), 5)
        self.assertEqual(tuple(views[0].shape), (2, 3, 224, 224))

    def test_conv_bn_fusion_matches_eval_output_within_tolerance(self):
        import torch
        from inference import fuse_conv_bn
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1, bias=False),
                                   torch.nn.BatchNorm2d(4), torch.nn.ReLU()).eval()
        images = torch.randn(2, 3, 16, 16)
        fused = fuse_conv_bn(model, example=images, tolerance=1e-5)
        with torch.inference_mode():
            difference = (model(images) - fused(images)).abs().max().item()
        self.assertLessEqual(difference, 1e-5)

    def test_latency_benchmark_warms_up_and_synchronizes_each_measurement(self):
        from benchmark import bench
        calls, syncs = [], []
        timing = bench(lambda: calls.append(1), warmup=10, iters=50, sync=lambda: syncs.append(1))
        self.assertEqual(timing["warmup"], 10)
        self.assertEqual(timing["n"], 50)
        self.assertEqual(len(calls), 60)
        self.assertEqual(len(syncs), 100)
        self.assertLessEqual(timing["p50"], timing["p95"])
        self.assertLessEqual(timing["p95"], timing["p99"])

    def test_latency_report_supports_batch_32(self):
        import torch
        from benchmark import latency_report
        model = torch.nn.Conv2d(3, 2, kernel_size=1).eval()
        report = latency_report(model, batch_size=32, img_size=8, device="cpu", iters=50)
        self.assertEqual(report["batch"], 32)
        self.assertEqual(report["n"], 50)
        self.assertGreater(report["images_per_s"], 0)


if __name__ == "__main__":
    unittest.main()
