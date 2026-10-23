"""Tiny deterministic controls for the opt-in protocol and feature baseline."""

from __future__ import annotations

import json
import unittest

import numpy as np
import torch

from engine.enrollment import enrollment_scores, enrollment_split
from models.gait_features import GaitFeatureBaseline, extract_gait_features


class EnrollmentSplitTests(unittest.TestCase):
    def test_prefix_purge_and_excluded_subject_coverage(self):
        ids = np.repeat([10, 25, 90], [9, 8, 7])
        enroll, probe, coverage = enrollment_split(ids)
        np.testing.assert_array_equal(enroll, [0, 1, 2, 3, 4, 9, 10, 11, 12, 13])
        np.testing.assert_array_equal(probe, [7, 8, 16])
        self.assertEqual(coverage["purged_indices"], [5, 6, 14, 15])
        self.assertEqual(coverage["included_subject_ids"], [10, 25])
        self.assertEqual(coverage["excluded_subject_ids"], [90])
        self.assertEqual(coverage["n_excluded_windows"], 7)
        self.assertEqual(coverage["per_subject"]["25"]["n_probes"], 1)
        self.assertFalse(coverage["raw_offset_nonoverlap_verified"])
        json.dumps(coverage, allow_nan=False)
        again = enrollment_split(ids)
        np.testing.assert_array_equal(enroll, again[0])
        np.testing.assert_array_equal(probe, again[1])
        self.assertEqual(coverage, again[2])

    def test_purge_disjointness_with_known_raw_offsets(self):
        ids = np.repeat([10, 25], 10)
        enroll, probe, _ = enrollment_split(ids)
        offsets = np.tile(np.arange(10) * 64, 2)
        for e in enroll:
            for p in probe:
                if ids[e] == ids[p]:
                    self.assertGreaterEqual(offsets[p] - offsets[e], 128)
        self.assertFalse(np.intersect1d(enroll, probe).size)

    def test_interleaved_subjects_preserve_per_subject_source_order(self):
        ids = np.tile([10, 25], 8)
        enroll, probe, _ = enrollment_split(ids)
        np.testing.assert_array_equal(enroll, np.arange(10))
        np.testing.assert_array_equal(probe, [14, 15])

    def test_invalid_budgets(self):
        ids = np.repeat([0, 1], 8)
        for n in (0, -1, 2.5, True, "5"):
            with self.subTest(enrollment=n), self.assertRaises(ValueError):
                enrollment_split(ids, n)
        for gap in (-1, 2.5, True, "2"):
            with self.subTest(purge=gap), self.assertRaises(ValueError):
                enrollment_split(ids, purge_windows=gap)
        enroll, probe, _ = enrollment_split(ids, 1, 0)
        self.assertEqual((len(enroll), len(probe)), (2, 14))

    def test_singletons_and_empty_inputs_exclude_without_inventing_probes(self):
        for ids in (np.array([], dtype=np.int64), np.array([0, 1])):
            enroll, probe, coverage = enrollment_split(ids)
            self.assertEqual((len(enroll), len(probe)), (0, 0))
            self.assertEqual(coverage["excluded_subject_ids"], ids.tolist())
        for ids in ([0.1, 1.1], [[0, 1]], [-1, 0], [True, False]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                enrollment_split(ids)


class EnrollmentScoresTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.ids = np.array([10, 10, 10, 25, 25, 25])
        self.z = np.array(
            [[2, 0], [0, 20], [1, 1], [-2, 0], [-40, 0], [-1, 0]], dtype=float
        )
        self.enroll = np.array([0, 1, 3, 4])
        self.probe = np.array([2, 5])

    def test_known_normalized_templates_and_probe_major_scores(self):
        labels, scores, metadata = enrollment_scores(
            self.z, self.ids, self.enroll, self.probe
        )
        np.testing.assert_array_equal(labels, [1, 0, 0, 1])
        np.testing.assert_allclose(
            scores, [0, -1 - 2**-0.5, -1 - 2**-0.5, 0], atol=1e-14
        )
        np.testing.assert_allclose(metadata["template_norms"], [1, 1])
        self.assertEqual(metadata["claimed_subject_ids"], [10, 25])
        self.assertEqual((metadata["n_genuine"], metadata["n_impostor"]), (2, 2))
        self.assertEqual(metadata["n_repeated_comparisons"], 0)
        self.assertFalse(metadata["threshold_calibrated"])
        json.dumps(metadata, allow_nan=False)

    def test_embedding_magnitude_and_enrollment_order_do_not_change_scores(self):
        expected = enrollment_scores(self.z, self.ids, self.enroll, self.probe)
        scaled = self.z * np.arange(1, 7)[:, None]
        actual = enrollment_scores(scaled, self.ids, self.enroll[::-1], self.probe)
        np.testing.assert_array_equal(expected[0], actual[0])
        np.testing.assert_allclose(expected[1], actual[1])

    def test_reproducible_exhaustive_counts_without_fake_balance(self):
        ids = np.repeat([10, 25, 90], [8, 8, 9])
        z = np.eye(3)[np.searchsorted([10, 25, 90], ids)]
        enroll, probe, _ = enrollment_split(ids)
        labels, scores, meta = enrollment_scores(z, ids, enroll, probe)
        self.assertEqual((meta["n_probes"], meta["n_comparisons"]), (4, 12))
        self.assertEqual((meta["n_genuine"], meta["n_impostor"]), (4, 8))
        np.testing.assert_array_equal(scores[labels == 1], np.zeros(4))
        np.testing.assert_array_equal(scores[labels == 0], -np.ones(8))
        repeated = enrollment_scores(z, ids, enroll, probe)
        np.testing.assert_array_equal(labels, repeated[0])
        np.testing.assert_array_equal(scores, repeated[1])
        self.assertEqual(meta, repeated[2])

    def test_reject_self_pairs_duplicates_uncovered_subjects_and_single_identity(self):
        for enroll, probe in (
            ([0, 1, 3], [1, 5]),
            ([0, 0, 3], [2, 5]),
            ([0, 1], [2]),
            ([0, 3], [2]),
            ([0, 3], [2, 5, 5]),
            ([], [2, 5]),
            ([0.0, 3.0], [2, 5]),
            ([0, 3], [-1, 2]),
            ([0, 3], [2, 6]),
        ):
            with (
                self.subTest(enroll=enroll, probe=probe),
                self.assertRaises(ValueError),
            ):
                enrollment_scores(self.z, self.ids, enroll, probe)

    def test_reject_nonfinite_zero_embeddings_and_cancelling_templates(self):
        for value in (np.nan, np.inf, 0):
            z = self.z.copy()
            z[0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                enrollment_scores(z, self.ids, self.enroll, self.probe)
        z = self.z.copy()
        z[1] = -z[0]
        with self.assertRaisesRegex(ValueError, "cancel"):
            enrollment_scores(z, self.ids, self.enroll, self.probe)

    def test_tensor_inputs_detach_without_training(self):
        z = torch.tensor(self.z, requires_grad=True)
        labels, scores, _ = enrollment_scores(
            z, torch.tensor(self.ids), self.enroll, self.probe
        )
        self.assertEqual((labels.shape, scores.shape), ((4,), (4,)))
        self.assertIsNone(z.grad)


class FeatureBaselineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.x = np.random.default_rng(42).normal(size=(8, 3, 128)).astype(np.float32)

    def test_shapes_constants_empty_and_tensor_input(self):
        self.assertEqual(extract_gait_features(self.x, 20).shape, (8, 104))
        self.assertEqual(
            extract_gait_features(np.ones((2, 9, 128)), 50).shape, (2, 312)
        )
        constant = extract_gait_features(np.ones((2, 3, 16)), 20)
        self.assertTrue(np.isfinite(constant).all())
        self.assertEqual(extract_gait_features(self.x[:0], 20).shape, (0, 104))
        np.testing.assert_allclose(
            extract_gait_features(torch.tensor(self.x), 20),
            extract_gait_features(self.x, 20),
        )

    def test_known_spectral_frequencies_and_vector_norm_rotation(self):
        t = np.arange(128) / 20
        x = np.stack([np.sin(2 * np.pi * f * t) for f in (1.25, 3.125)])[:, None, :]
        features = extract_gait_features(x, 20)
        np.testing.assert_allclose(features[:, -2], [1.25, 3.125], atol=1e-6)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        rotated = np.einsum("ij,njt->nit", rotation, self.x)
        a = extract_gait_features(self.x, 20).reshape(8, 4, 26)
        b = extract_gait_features(rotated, 20).reshape(8, 4, 26)
        np.testing.assert_allclose(a[:, 3], b[:, 3], atol=1e-6)

    def test_train_only_scaling_frozen_state_and_round_trip(self):
        model = GaitFeatureBaseline(20).fit(self.x[:5])
        before = json.dumps(model.state_dict(), allow_nan=False, sort_keys=True)
        validation = model.transform(self.x[5:] + 100)
        self.assertEqual(
            before, json.dumps(model.state_dict(), allow_nan=False, sort_keys=True)
        )
        restored = GaitFeatureBaseline.from_state_dict(json.loads(before))
        np.testing.assert_array_equal(validation, restored.transform(self.x[5:] + 100))
        self.assertEqual(model.n_fit_windows_, 5)
        self.assertGreater(float(np.abs(validation).mean()), 1)

    def test_pca_and_raw_controls(self):
        model = GaitFeatureBaseline(20, pca_components=3)
        train = model.fit_transform(self.x[:5])
        self.assertEqual(train.shape, (5, 3))
        restored = GaitFeatureBaseline.from_state_dict(model.state_dict())
        np.testing.assert_array_equal(
            model.transform(self.x[5:]), restored.transform(self.x[5:])
        )
        raw = GaitFeatureBaseline(20, standardize=False).fit_transform(self.x)
        np.testing.assert_array_equal(raw, extract_gait_features(self.x, 20))

    def test_invalid_inputs_and_unfitted_transform(self):
        with self.assertRaisesRegex(ValueError, "fit"):
            GaitFeatureBaseline(20).transform(self.x)
        for rate in (0, -1, np.nan, np.inf, True):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                extract_gait_features(self.x, rate)
        for x in (
            np.ones((2, 3)),
            np.ones((2, 0, 16)),
            np.ones((2, 3, 2)),
            np.full((2, 3, 16), np.nan),
        ):
            with self.subTest(shape=x.shape), self.assertRaises(ValueError):
                extract_gait_features(x, 20)
        for components in (0, -1, True, 1.5, 8):
            with self.subTest(components=components), self.assertRaises(ValueError):
                GaitFeatureBaseline(20, pca_components=components).fit(self.x[:5])
        with self.assertRaises(ValueError):
            GaitFeatureBaseline(20).fit(self.x[:0])
        with self.assertRaises(ValueError):
            GaitFeatureBaseline(20).fit(self.x).transform(self.x[:, :, :64])


if __name__ == "__main__":
    unittest.main()
