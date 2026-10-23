import unittest
import numpy as np
import torch
from engine.evaluate import compute_eer, verification_metrics


class TestEvaluateMetrics(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        torch.set_num_threads(2)

    def test_compute_eer_perfect_separation(self):
        # Perfect separation: genuine > impostor
        labels = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        # FPR = [0, 0, 0, 1/3, 2/3, 1], FNR = [1, 1, 1, 0, 0, 0]
        # crossing at FAR=0, FRR=0 (approx)
        self.assertAlmostEqual(compute_eer(labels, scores), 0.0, places=4)

    def test_compute_eer_non_separable(self):
        # Non-separable: identical scores
        np.random.seed(42)
        labels = np.array([1] * 100 + [0] * 100)
        scores = np.random.rand(200)
        eer = compute_eer(labels, scores)
        self.assertTrue(0.4 <= eer <= 0.6)

    def test_verification_metrics_keys_and_ranges(self):
        # Synthetic data: 5 subjects, 4 windows each
        n_subs = 5
        win_per_sub = 4
        embeddings = np.random.randn(n_subs * win_per_sub, 128).astype(np.float32)
        subject_ids = np.repeat(np.arange(n_subs), win_per_sub)

        res = verification_metrics(embeddings, subject_ids)

        expected_keys = {
            "eer",
            "eer_threshold",
            "far_at_frr_0.01",
            "frr_at_far_0.01",
            "roc_auc",
            "n_pairs",
        }
        self.assertTrue(expected_keys.issubset(res))

        for k in ["eer", "far_at_frr_0.01", "frr_at_far_0.01", "roc_auc"]:
            self.assertTrue(0.0 <= res[k] <= 1.0, f"{k} out of range: {res[k]}")
        self.assertTrue(np.isfinite(res["eer_threshold"]))
        self.assertEqual(res["n_pairs"], 10000)

    def test_verification_metrics_separable(self):
        # Create highly separable embeddings
        # Each subject has a unique, distant center.
        n_subs = 10
        win_per_sub = 4
        embeddings = np.zeros((n_subs * win_per_sub, 128), dtype=np.float32)
        for i in range(n_subs):
            center = np.random.randn(128) * 100.0
            embeddings[i * win_per_sub : (i + 1) * win_per_sub] = (
                center + np.random.randn(win_per_sub, 128) * 0.1
            )

        subject_ids = np.repeat(np.arange(n_subs), win_per_sub)
        res = verification_metrics(embeddings, subject_ids)

        self.assertLess(res["eer"], 0.1)
        self.assertGreater(res["roc_auc"], 0.9)
        self.assertLess(res["far_at_frr_0.01"], 0.1)
        self.assertLess(res["frr_at_far_0.01"], 0.1)

    def test_verification_metrics_independence(self):
        # Use a moderately separable case to check independence
        n_subs = 10
        win_per_sub = 10
        embeddings = np.random.randn(n_subs * win_per_sub, 128).astype(np.float32)
        # Add some signal
        for i in range(n_subs):
            embeddings[i * win_per_sub : (i + 1) * win_per_sub] += i * 0.1

        subject_ids = np.repeat(np.arange(n_subs), win_per_sub)
        res = verification_metrics(embeddings, subject_ids)

        # In general, FAR@FRR=0.01 should not be exactly EER unless the ROC is a straight line
        self.assertNotAlmostEqual(res["far_at_frr_0.01"], res["eer"], places=5)
        self.assertNotAlmostEqual(res["frr_at_far_0.01"], res["eer"], places=5)

    def test_verification_metrics_errors(self):
        # Length mismatch
        with self.assertRaises(ValueError):
            verification_metrics(np.random.randn(10, 128), np.arange(5))

        # < 2 subjects
        with self.assertRaises(ValueError):
            verification_metrics(np.random.randn(10, 128), np.zeros(10))

        # No subject with >= 2 windows
        # 10 subjects, 1 window each
        with self.assertRaises(ValueError):
            verification_metrics(np.random.randn(10, 128), np.arange(10))

        # All same label
        # Since verification_metrics generates both genuine and impostor pairs
        # if there are >=2 subjects and each has >=2 windows, we test compute_eer directly.
        with self.assertRaises(ValueError):
            compute_eer(np.ones(10), np.random.rand(10))
        with self.assertRaises(ValueError):
            compute_eer(np.zeros(10), np.random.rand(10))

    def test_train_siamese_compatibility(self):
        # Mock the calls that train_siamese makes
        embeddings = np.random.randn(20, 128).astype(np.float32)
        subject_ids = np.repeat(np.arange(5), 4)

        metrics = verification_metrics(embeddings, subject_ids)
        # train_siamese reads: val_metrics["eer"] and val_metrics["roc_auc"]
        self.assertIn("eer", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertTrue(0.0 <= metrics["eer"] <= 1.0)
        self.assertTrue(0.0 <= metrics["roc_auc"] <= 1.0)

    def test_boundary_conditions(self):
        # Case where EER crossing is at the very end
        # labels = [1, 0], scores = [0.1, 0.9] -> genuine < impostor (worst case)
        # roc_curve: fpr=[0, 1], tpr=[0, 1], thresholds=[0.9, 0.1]
        # fnr=[1, 0]. diff = [0-1, 1-0] = [-1, 1]. idx=1.
        # idx=1 is the last index of diff.
        labels = np.array([1, 0])
        scores = np.array([0.1, 0.9])
        # Manually check compute_eer to avoid verification_metrics overhead
        self.assertTrue(np.isfinite(compute_eer(labels, scores)))

        # Tiny dataset
        labels = np.array([1, 0])
        scores = np.array([0.5, 0.4])
        self.assertTrue(np.isfinite(compute_eer(labels, scores)))

        # Tied scores
        labels = np.array([1, 1, 0, 0])
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        self.assertTrue(np.isfinite(compute_eer(labels, scores)))

        # Verify verification_metrics doesn't crash on small inputs
        res = verification_metrics(
            np.random.randn(4, 128),  # 2 subs, 2 windows each
            np.array([0, 0, 1, 1]),
            n_pairs=10,
        )
        self.assertTrue(np.isfinite(res["eer_threshold"]))


if __name__ == "__main__":
    unittest.main()
