"""Tiny NumPy-only protocol tests; no archive, Torch, sklearn, or fitted jobs."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts import mentor_backend_study as study


class BackendProtocolTests(unittest.TestCase):
    def test_splits_are_unique_reproducible_and_preserve_original_five(self):
        folds = study.development_folds()
        self.assertEqual(folds, study.development_folds())
        self.assertEqual(len(folds), 20)
        self.assertEqual(len({tuple(sorted(f["val_subject_ids"])) for f in folds}), 20)
        validation_counts = {
            subject: sum(subject in fold["val_subject_ids"] for fold in folds)
            for subject in study.DEV_IDS
        }
        self.assertEqual(validation_counts, dict.fromkeys(study.DEV_IDS, 4))
        for start in range(0, 20, 5):
            partition_ids = [
                subject
                for fold in folds[start : start + 5]
                for subject in fold["val_subject_ids"]
            ]
            self.assertEqual(sorted(partition_ids), sorted(study.DEV_IDS))
        for i, fold in enumerate(folds):
            self.assertEqual(len(fold["train_subject_ids"]), 12)
            self.assertEqual(len(fold["val_subject_ids"]), 3)
            self.assertFalse(
                set(fold["train_subject_ids"]) & set(fold["val_subject_ids"])
            )
            self.assertEqual(
                set(fold["train_subject_ids"] + fold["val_subject_ids"]),
                set(study.DEV_IDS),
            )
            if i < 5:
                self.assertEqual(
                    fold["val_subject_ids"], list(study.DEV_IDS[3 * i : 3 * i + 3])
                )
            else:
                self.assertEqual(
                    fold["val_subject_ids"], sorted(fold["val_subject_ids"])
                )

    def test_duplicate_triple_rejects_whole_permutation(self):
        expected = study.development_folds()
        generator = np.random.default_rng(2026)
        # One original triple plus four new triples must all be rejected.
        rejected = np.r_[study.DEV_IDS[:3], np.roll(study.DEV_IDS[3:], 1)]
        first = True

        def permutation(ids):
            nonlocal first
            if first:
                first = False
                return rejected
            return generator.permutation(ids)

        rng = SimpleNamespace(permutation=permutation)
        with patch.object(study.np.random, "default_rng", return_value=rng):
            self.assertEqual(study.development_folds(), expected)
        self.assertFalse(first)

    def test_selection_requires_complete_evidence_and_ignores_baseline(self):
        rows = [
            {
                "fold": fold,
                "method": method,
                "metrics": {
                    "eer": 0 if method == study.BASELINE else 0.2,
                    "claim_graph_sha256": str(fold),
                },
            }
            for fold in range(20)
            for method in study.METHODS
        ]
        self.assertEqual(study.select_candidate(rows)["winner"], "control")
        with self.assertRaises(ValueError):
            study.select_candidate(rows[:-1])
        with self.assertRaises(ValueError):
            study.select_candidate(rows[:-1] + [rows[0]])
        rows[-1]["metrics"]["claim_graph_sha256"] = "different"
        with self.assertRaises(ValueError):
            study.select_candidate(rows)

    def test_roc_ties_and_calibration_have_attainable_decisions(self):
        labels = np.array([1, 0, 1, 0])
        scores = np.array([0.9, 0.8, 0.8, 0.1])
        self.assertAlmostEqual(study.eer(labels, scores), 0.25)
        result = study.calibrate(labels, scores)
        self.assertEqual(result["threshold"], 0.9)
        self.assertEqual(result["false_accepts"], 0)
        self.assertEqual(result["false_rejects"], 1)
        self.assertAlmostEqual(study.eer(np.array([1, 0]), np.ones(2)), 0.5)
        with self.assertRaises(ValueError):
            study.roc(np.ones(2), np.ones(2))

    def test_fit_gate_rejects_held_out_ids_before_backend_import(self):
        with self.assertRaisesRegex(ValueError, "DEV"):
            study.fit_metric(
                "nuisance", np.ones((2, 2)), {}, [study.CALIBRATION_IDS[0]]
            )

    def test_safe_serialization_and_exclusive_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "safe.npz"
            study.save_npz(path, {"x": np.arange(3)})
            np.testing.assert_array_equal(study.load_npz(path)["x"], np.arange(3))
            with self.assertRaises(FileExistsError):
                study.save_npz(path, {"x": np.arange(3)})
            with self.assertRaises(ValueError):
                study.save_npz(
                    Path(directory) / "unsafe.npz", {"x": np.array([{}], dtype=object)}
                )

    def test_real_numpy_snorm_interface_preserves_graph_and_person_reports(self):
        class Identity:
            def transform(self, values):
                return values

        data = {
            "subjects": np.repeat([1, 3, 5], 2),
            "days": np.tile([1, 2], 3),
            "trials": np.ones(6, dtype=np.int64),
        }
        values = np.array(
            [
                [1, 0.1, 0.2],
                [1, 0.2, 0.1],
                [0.1, 1, 0.2],
                [0.2, 1, 0.1],
                [0.1, 0.2, 1],
                [0.2, 0.1, 1],
            ]
        )
        cohort = {
            "subjects": np.repeat([7, 9], 2),
            "days": np.tile([1, 2], 2),
            "trials": np.array([2, 3, 4, 5]),
            "embeddings": np.array(
                [[1, 1, 0.1], [1, 0.8, 0.2], [0.2, 1, 1], [0.1, 0.8, 1]]
            ),
        }
        reports = [
            study.report(
                study.score(method, Identity(), cohort, values, data, [1, 3, 5])
            )
            for method in ("control", "snorm", study.BASELINE)
        ]
        study.matched_graphs(reports)
        self.assertEqual(set(reports[1]["per_subject_eer"]), {"1", "3", "5"})

    def test_evaluate_reserves_attempt_before_test_load_and_consumes_failure(self):
        # Gate test with tiny synthetic evidence. All persistence and the actual
        # exclusive TEST reservation run; source/data/replay inputs are stubs.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = {"winner": "control"}
            evidence = {"selection": selection}
            study.write_new(root / "registration.json", {})
            study.write_new(root / "development.json", evidence)
            study.save_npz(root / "model.npz", {"x": np.ones(1)})
            model = study._artifact(root, root / "model.npz")
            labels, scores = np.array([1, 0]), np.array([0.9, 0.1])
            scored = {"labels": labels, "scores": scores, "metadata": {}}
            study.save_npz(root / "cal.npz", {"labels": labels, "scores": scores})
            frozen = {
                "scores": study._artifact(root, root / "cal.npz"),
                "metadata": {},
                "threshold": study.calibrate(labels, scores),
                "metrics": {},
            }
            lock = {
                "registration_sha256": study.digest(root / "registration.json"),
                "development_sha256": study.digest(root / "development.json"),
                "selection": selection,
                "models": {"control": model, study.BASELINE: model},
                "calibration": {},
                "test_subject_ids": list(study.TEST_IDS),
                "calibration_results": {"control": frozen, study.BASELINE: frozen},
            }
            lock["lock_sha256"] = study.object_hash(lock)
            study.write_new(root / "protocol_lock.json", lock)
            args = SimpleNamespace(out_root=root, archive_path=Path("unused.zip"))
            loads = []

            def loader(archive, ids):
                loads.append(tuple(ids))
                if tuple(ids) == study.TEST_IDS:
                    self.assertTrue((root / "test_attempt.json").exists())
                    raise RuntimeError("simulated failed TEST load")
                return {}

            with (
                patch.object(study, "_registration", return_value=(root, {})),
                patch.object(
                    study,
                    "verify_development",
                    return_value=(
                        {},
                        {"base": np.ones(1), study.BASELINE: np.ones(1)},
                        evidence,
                    ),
                ),
                patch.object(study, "load_data", side_effect=loader),
                patch.object(study, "_manifest", return_value={}),
                patch.object(
                    study,
                    "features",
                    return_value={"base": np.ones(1), study.BASELINE: np.ones(1)},
                ),
                patch.object(study, "load_model", return_value=(None, {})),
                patch.object(study, "verify_cohort"),
                patch.object(study, "score", return_value=scored),
                patch.object(study, "report", return_value={}),
            ):
                with self.assertRaisesRegex(RuntimeError, "failed TEST"):
                    study.evaluate_stage(args)
                with self.assertRaisesRegex(ValueError, "already attempted"):
                    study.evaluate_stage(args)
            self.assertEqual(loads.count(study.TEST_IDS), 1)
            self.assertFalse((root / "test_result.json").exists())


if __name__ == "__main__":
    unittest.main()
