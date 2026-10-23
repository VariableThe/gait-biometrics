"""Protocol invariants on known score tables and real network inference."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from data.adapter import WindowData
from engine.evaluate import (
    _eval_verification,
    _evaluation_data,
    _verification_pairs,
    _roc,
    calibrate_threshold,
    compute_eer,
    fixed_threshold_metrics,
    verification_metrics,
)
from models.siamese import EmbeddingNet


def windows(subjects, seed=0):
    ids = torch.tensor(np.repeat(subjects, 4))
    return WindowData(
        torch.randn(len(ids), 3, 16, generator=torch.Generator().manual_seed(seed)),
        torch.zeros(len(ids), dtype=torch.long),
        ids,
        ["walking"],
        list(subjects),
        20.0,
    )


class ScoreProtocolTests(unittest.TestCase):
    def test_ties_are_not_order_dependent(self):
        for labels in ([1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1]):
            self.assertEqual(compute_eer(labels, [0.5] * 4), 0.5)

    def test_extreme_rankings(self):
        self.assertEqual(compute_eer([1, 0], [0.8, 0.2]), 0)
        self.assertEqual(compute_eer([1, 0], [0.2, 0.8]), 1)

    def test_exact_one_percent_frr_survives_float_subtraction(self):
        far, frr, thresholds = _roc([1] * 100 + [0], [1.0] * 99 + [0.2, 0.5])
        index = int(np.flatnonzero(thresholds == 1.0)[0])
        self.assertEqual(frr[index], 0.01)
        self.assertEqual(far[index], 0)

    def test_actual_tied_decisions(self):
        result = fixed_threshold_metrics([1, 0, 1, 0], [0.5, 0.5, 0.2, 0.1], 0.5)
        self.assertEqual((result["false_accepts"], result["false_rejects"]), (1, 1))
        self.assertEqual((result["far"], result["frr"]), (0.5, 0.5))

    def test_calibration_never_interpolates_tied_scores(self):
        result = calibrate_threshold([1, 1, 0, 0], [0.0] * 4, 0.01)
        self.assertTrue(np.isfinite(result["threshold"]))
        self.assertGreater(result["threshold"], 0)
        self.assertEqual((result["far"], result["frr"]), (0, 1))
        json.dumps(result, allow_nan=False)

    def test_empirical_far_target_is_a_constraint(self):
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.5, 0.6, 0.2]
        result = calibrate_threshold(labels, scores, 0.01)
        self.assertEqual(result["threshold"], 0.9)
        self.assertEqual(result["far"], 0)
        self.assertEqual(result["frr"], 0.5)

    def test_threshold_fixed_when_test_scores_shift(self):
        calibration = calibrate_threshold([1, 1, 0, 0], [0.9, 0.8, 0.4, 0.1])
        test = fixed_threshold_metrics(
            [1, 1, 0, 0], [0.2, 0.1, 0.9, 0.85], calibration["threshold"]
        )
        self.assertEqual(test["threshold"], calibration["threshold"])
        self.assertEqual((test["far"], test["frr"]), (1, 1))

    def test_invalid_score_arrays(self):
        for labels, scores in (
            ([], []),
            ([0, 0], [0, 1]),
            ([1, 2], [0, 1]),
            ([0, 1], [0]),
            ([[0, 1]], [[0, 1]]),
            ([0, 1], [0, np.nan]),
            ([0, 1], [0, np.inf]),
        ):
            with (
                self.subTest(labels=labels, scores=scores),
                self.assertRaises(ValueError),
            ):
                compute_eer(labels, scores)

    def test_invalid_threshold_targets(self):
        for target in (-0.1, 1.1, np.nan, np.inf):
            with self.subTest(target=target), self.assertRaises(ValueError):
                calibrate_threshold([1, 0], [1, 0], target)
        with self.assertRaises(ValueError):
            fixed_threshold_metrics([1, 0], [1, 0], np.inf)


class PairProtocolTests(unittest.TestCase):
    def test_reproducible_balanced_no_self_pairs(self):
        ids = np.repeat([10, 25, 70], 4)
        pairs, labels, context = _verification_pairs(ids, 101, 4)
        again = _verification_pairs(ids, 101, 4)
        np.testing.assert_array_equal(pairs, again[0])
        self.assertEqual(context, again[2])
        self.assertEqual(int(labels.sum()), 50)
        self.assertTrue(np.all(pairs[:, 0] < pairs[:, 1]))
        np.testing.assert_array_equal(
            ids[pairs[:, 0]] == ids[pairs[:, 1]], labels.astype(bool)
        )
        self.assertFalse(np.array_equal(pairs, _verification_pairs(ids, 101, 5)[0]))

    def test_repeated_pairs_are_reported(self):
        _, _, context = _verification_pairs(np.array([1, 1, 2, 2]), 1000, 9)
        self.assertEqual(context["n_unique_pairs"], 6)
        self.assertEqual(context["n_repeated_pairs"], 994)
        self.assertEqual(context["n_unique_genuine_pairs"], 2)
        self.assertEqual(context["n_windows_used"], 4)
        self.assertEqual(set(context["per_subject"]), {"1", "2"})

    def test_activity_cannot_be_an_easy_impostor_cue(self):
        subs = np.tile(np.repeat([0, 1], 3), 2)
        activities = np.repeat([0, 1], 6)
        pairs, labels, _ = _verification_pairs(subs, 100, 2, activities)
        self.assertTrue(np.all(activities[pairs[:, 0]] == activities[pairs[:, 1]]))
        for activity in (0, 1):
            for label in (0, 1):
                self.assertEqual(
                    int(
                        np.sum(
                            (activities[pairs[:, 0]] == activity) & (labels == label)
                        )
                    ),
                    25,
                )
        # Embeddings encode only activity: all within-activity comparisons tie.
        z = np.eye(2)[activities]
        metrics = verification_metrics(z, subs, 100, activity_ids=activities)
        self.assertEqual(metrics["eer"], 0.5)
        self.assertEqual(metrics["roc_auc"], 0.5)
        self.assertEqual(metrics["far_at_frr_0.01"], 1)
        self.assertEqual(metrics["frr_at_far_0.01"], 1)

    def test_ineligible_activities_reported_and_empty_fails_actionably(self):
        _, _, context = _verification_pairs(
            np.array([0, 0, 1, 1, 2, 2]), 20, 2, np.array([0, 0, 0, 0, 1, 1])
        )
        self.assertEqual(context["excluded_activity_ids"], [1])
        with self.assertRaisesRegex(ValueError, "enlarge subset or check activity_ids"):
            _verification_pairs(np.array([0, 0, 1, 1]), 20, 2, np.array([0, 0, 1, 1]))

    def test_invalid_pair_budgets(self):
        for n in (0, 1, -1, 2.5, True, "10"):
            with self.subTest(n=n), self.assertRaises(ValueError):
                _verification_pairs(np.array([0, 0, 1, 1]), n, 0)

    def test_invalid_embeddings_and_labels(self):
        for z in (
            np.zeros((4, 2)),
            np.full((4, 2), np.nan),
            np.ones(4),
            np.ones((0, 2)),
        ):
            with self.subTest(shape=z.shape), self.assertRaises(ValueError):
                verification_metrics(z, np.array([0, 0, 1, 1]))
        for ids in ([0.1, 0.1, 1.9, 1.9], [[0, 0, 1, 1]], [0, 1]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                verification_metrics(np.ones((4, 2)), ids)

    def test_tensor_detaches_and_fixed_threshold_counts_are_exposed(self):
        z = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], requires_grad=True
        )
        result = verification_metrics(z, [0, 0, 1, 1], 20, threshold=-0.5)
        self.assertEqual(result["eer"], 0)
        self.assertEqual(result["fixed_threshold"]["far"], 0)
        self.assertEqual(result["fixed_threshold"]["frr"], 0)
        json.dumps(result, allow_nan=False)


class InferenceProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.config = {
            "source": "wisdm_v1",
            "data_root": "/real/path",
            "activity_ids": [0],
            "window_size": 16,
            "batch_size": 4,
            "verification_pairs": 100,
            "model_args": {"embed_dim": 4, "dropout": 0.0},
        }
        self.protocol = {
            "source": "wisdm_v1",
            "train_subject_ids": [10, 11],
            "val_subject_ids": [20, 21],
            "activity_ids": [0],
            "n_channels": 3,
            "window_size": 16,
        }

    def run_evaluation(self, protocol, test=None, calibrate=True):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = str(Path(tmp, "best.pt"))
            torch.manual_seed(1)
            model = EmbeddingNet(3, embed_dim=4, dropout=0)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metrics": {"protocol": protocol},
                },
                checkpoint,
            )

            def loader(source, split, **kwargs):
                self.assertEqual(kwargs["data_root"], "/real/path")
                return (
                    windows([10, 11, 20, 21], 5)
                    if split == "train"
                    else (test or windows([30, 31], 6))
                )

            with mock.patch("engine.evaluate.load_windows", side_effect=loader):
                result = _eval_verification(
                    self.config, checkpoint, None, tmp, calibrate=calibrate
                )
            self.assertEqual(
                json.loads(Path(tmp, "eval_verification.json").read_text()), result
            )
            return result

    def test_real_network_inference_with_validation_only_calibration(self):
        result = self.run_evaluation(self.protocol)
        self.assertEqual(result["calibration"]["subject_ids"], [20, 21])
        self.assertEqual(result["calibration"]["n_samples"], 8)
        self.assertEqual(result["n_samples"], 8)
        self.assertEqual(
            result["fixed_threshold"]["threshold"], result["calibration"]["threshold"]
        )
        self.assertFalse(result["deployment_ready"])

    def test_legacy_checkpoint_is_descriptive_only(self):
        result = self.run_evaluation({}, calibrate=False)
        self.assertIsNone(result["calibration"])
        self.assertNotIn("fixed_threshold", result)
        with self.assertRaisesRegex(ValueError, "requires checkpoint protocol"):
            self.run_evaluation({})

    def test_reject_subject_leakage(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.run_evaluation(self.protocol, windows([10, 31]))
        protocol = {**self.protocol, "val_subject_ids": [10, 21]}
        with self.assertRaisesRegex(ValueError, "disjoint"):
            self.run_evaluation(protocol)

    def test_reject_activity_and_channel_mismatch(self):
        for protocol in (
            {**self.protocol, "activity_ids": [1]},
            {**self.protocol, "n_channels": 9},
        ):
            with self.subTest(protocol=protocol), self.assertRaises(ValueError):
                self.run_evaluation(protocol)

    def test_empty_activity_filter_fails_actionably(self):
        with mock.patch("engine.evaluate.load_windows", return_value=windows([1, 2])):
            with self.assertRaisesRegex(ValueError, "existing zero-indexed"):
                _evaluation_data({**self.config, "activity_ids": [6]}, "test", {})

    def test_pairing_inherited_from_checkpoint_and_mismatch_rejected(self):
        protocol = {**self.protocol, "verification_pairing": "unrestricted"}
        result = self.run_evaluation(protocol)
        self.assertFalse(result["same_activity_pairs"])
        self.config["verification_pairing"] = "same_activity"
        with self.assertRaisesRegex(ValueError, "verification_pairing differs"):
            self.run_evaluation(protocol)

    def test_unsupported_normalization_fails(self):
        with self.assertRaisesRegex(ValueError, "normalization"):
            self.run_evaluation({**self.protocol, "normalization": "unknown"})


if __name__ == "__main__":
    unittest.main()
