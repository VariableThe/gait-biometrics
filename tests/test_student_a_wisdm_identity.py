"""Boundary and train-only normalization tests with deliberately synthetic data."""

import argparse
import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "gait_pipeline.py"
if not PIPELINE.is_file():
    PIPELINE = ROOT / "student_a" / "gait_pipeline.py"
spec = importlib.util.spec_from_file_location("identity_pipeline_under_test", PIPELINE)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class WisdmPipelineIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = argparse.Namespace(
            hz=20.0, cutoff=8.0, window=4, step=2, wisdm_raw=None, out="processed"
        )

    def test_resample_rejects_reset_instead_of_sorting_sessions(self):
        with self.assertRaisesRegex(ValueError, "single segment"):
            pipeline.resample(np.zeros((4, 3)), [0, 0.05, 0, 0.05], 20)
        with self.assertRaisesRegex(ValueError, "single segment"):
            pipeline.resample(np.zeros((4, 3)), [0, 0.05, 0.05, 0.1], 20)

    def test_short_filter_segment_and_pure_windows(self):
        x = np.zeros((40, 3), dtype=np.float32)
        self.assertEqual(pipeline.lowpass(x, 20, 8).shape, x.shape)
        X, y, subjects = pipeline.windows(
            np.arange(24).reshape(8, 3), [0] * 3 + [1] * 5, 17, 4, 2
        )
        self.assertEqual(X.shape, (1, 4, 3))
        np.testing.assert_array_equal(y, [1])
        np.testing.assert_array_equal(subjects, [17])
        np.testing.assert_array_equal(X[0], np.arange(24).reshape(8, 3)[3:7])

    def test_real_subject_export_preserves_ids_and_train_only_statistics(self):
        raw = self.root / "WISDM_ar_v1.1_raw.txt"
        rows = []
        for subject in [1, 4, 11, 33, 36]:
            for j in range(12):
                rows.append(
                    f"{subject},Walking,{1_000_000_000 + j * 50_000_000},{subject + j},{subject * 2 + j},3;\n"
                )
        raw.write_text("".join(rows))
        self.args.wisdm_raw = str(raw)
        out = self.root / "result"
        pipeline.process_wisdm(self.args, out)
        with (
            np.load(out / "wisdm_train.npz", allow_pickle=False) as tr,
            np.load(out / "wisdm_test.npz", allow_pickle=False) as te,
        ):
            self.assertEqual(set(tr.files), {"X", "y", "subject"})
            self.assertEqual(tr["X"].shape[1:], (4, 3))
            self.assertTrue(set(tr["subject"]).isdisjoint(te["subject"]))
            self.assertEqual(
                set(tr["subject"]) | set(te["subject"]), {1, 4, 11, 33, 36}
            )
            np.testing.assert_allclose(tr["X"].mean(axis=(0, 1)), 0, atol=1e-6)
            self.assertGreater(abs(float(te["X"][..., 0].mean())), 0.01)
        with (
            np.load(out / "wisdm_train_normalization.npz", allow_pickle=False) as a,
            np.load(out / "wisdm_test_normalization.npz", allow_pickle=False) as b,
        ):
            np.testing.assert_array_equal(a["mean"], b["mean"])
            np.testing.assert_array_equal(a["std"], b["std"])
        metadata = json.loads((out / "wisdm_train.json").read_text())
        self.assertTrue(metadata["identity_available"])
        self.assertEqual(
            metadata["sha256"], hashlib.sha256(raw.read_bytes()).hexdigest()
        )
        self.assertEqual(metadata["activity_names"][0], "Walking")

    def test_legacy_unknown_identity_and_no_resampling_across_reset(self):
        dataset = self.root / "Dataset"
        dataset.mkdir()
        archive = dataset / "WISDM_Dataset.zip"
        old_root = pipeline.ROOT
        pipeline.ROOT = self.root
        self.addCleanup(setattr, pipeline, "ROOT", old_root)

        def write_archive(test_value):
            with zipfile.ZipFile(archive, "w") as z:
                for split, offset in [("train", 0), ("test", test_value)]:
                    df = pd.DataFrame(
                        {
                            "acc_x": np.r_[np.arange(6), np.arange(6) + 100] + offset,
                            "acc_y": [0] * 12,
                            "acc_z": [1] * 12,
                            "timestamp": list(np.arange(6) / 20) * 2,
                            "Activity": [0] * 12,
                        }
                    )
                    z.writestr(f"{split}.csv", df.to_csv(index=False))

        write_archive(1000)
        out = self.root / "first"
        pipeline.process_wisdm(self.args, out)
        with np.load(out / "wisdm_train.npz", allow_pickle=False) as train:
            before = train["X"].copy()
            self.assertEqual(len(before), 4)
            np.testing.assert_array_equal(train["subject"], [-1] * 4)
        meta = json.loads((out / "wisdm_train.json").read_text())
        self.assertFalse(meta["identity_available"])
        write_archive(5000)
        pipeline.process_wisdm(self.args, self.root / "second")
        with np.load(
            self.root / "second" / "wisdm_train.npz", allow_pickle=False
        ) as after:
            np.testing.assert_array_equal(before, after["X"])


if __name__ == "__main__":
    unittest.main()
