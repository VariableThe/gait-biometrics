"""Synthetic NumPy-only checks; no dataset, Torch, fitting, or GPU work."""

from __future__ import annotations

import builtins
import json
import unittest
from unittest import mock

import numpy as np

_IMPORT = builtins.__import__


def _without_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("template scoring must remain Torch-free")
    return _IMPORT(name, *args, **kwargs)


with mock.patch("builtins.__import__", side_effect=_without_torch):
    from engine.template_scoring import (
        CoverageError,
        paired_subject_diagnostics,
        score_trials,
    )
    from scripts import cross_day_verification as legacy


def _score(rows, vectors, **kwargs):
    subjects, days, trials = np.asarray(rows, dtype=np.int64).T
    return score_trials(vectors, subjects, days, trials, **kwargs)


def _base():
    return [(7, 1, 1), (7, 2, 1), (16, 1, 1), (16, 2, 1)], [
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
    ]


class TemplateScoringTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("builtins.__import__", side_effect=_without_torch)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_known_scores_dtypes_sorted_claim_graph_and_json(self):
        rows, vectors = _base()
        labels, scores, meta = _score(rows[::-1], vectors[::-1])
        np.testing.assert_array_equal(labels, [1, 0, 0, 1])
        np.testing.assert_array_equal(scores, [0, -1, -1, 0])
        self.assertEqual(labels.dtype, np.dtype("int64"))
        self.assertEqual(scores.dtype, np.dtype("float64"))
        self.assertEqual(meta["claim_subject_ids"], [7, 16, 7, 16])
        self.assertEqual(meta["probe_subject_ids"], [7, 7, 16, 16])
        self.assertEqual(meta["probe_trial_ids"], [1, 1, 1, 1])
        self.assertEqual(meta["coverage"]["n_genuine"], 2)
        self.assertEqual(meta["coverage"]["n_impostor"], 2)
        self.assertEqual(meta["coverage"]["far_resolution"], 0.5)
        self.assertFalse(meta["calibration"]["threshold_calibrated"])
        json.dumps(meta, allow_nan=False)

    def test_mean_exactly_matches_legacy_scorer(self):
        rng = np.random.default_rng(12)
        rows = [
            (s, d, t)
            for s in (16, 7, 29)
            for d in (1, 2)
            for t in (3, 1, 2)
            for _ in range(t)
        ]
        subjects, days, trials = np.asarray(rows, dtype=np.int64).T
        z = rng.normal(size=(len(rows), 17))
        starts = np.concatenate(
            [np.arange(t) * 256 for _ in (16, 7, 29) for _ in (1, 2) for t in (3, 1, 2)]
        )
        data = {
            "X": np.zeros((len(rows), 3, 256)),
            "subjects": subjects,
            "days": days,
            "trials": trials,
            "starts": starts,
            "metadata": {},
        }
        expected = legacy.cross_day_scores(z, data, (7, 16, 29))
        labels, scores, meta = score_trials(z, subjects, days, trials)
        np.testing.assert_array_equal(labels, expected["labels"])
        np.testing.assert_array_equal(scores, expected["scores"])
        for key, old in (
            ("claim_subject_ids", "claim_subjects"),
            ("probe_subject_ids", "probe_subjects"),
            ("probe_trial_ids", "probe_trials"),
        ):
            self.assertEqual(meta[key], expected[old].tolist())
        self.assertEqual(
            {k: v for k, v in expected["coverage"].items() if k != "adapter_metadata"},
            meta["coverage"],
        )

    def test_equal_trial_weight_and_unit_window_weight(self):
        rows = [(7, 1, 1)] * 3 + [(7, 1, 2), (7, 2, 1), (16, 1, 1), (16, 2, 1)]
        vectors = [[100, 0], [2, 0], [1, 0], [0, 1], [1, 0], [-1, 0], [-1, 0]]
        _, scores, _ = _score(rows, vectors)
        self.assertAlmostEqual(scores[0], 1 / np.sqrt(2) - 1)

    def test_probe_values_never_change_templates_or_other_probes(self):
        rng = np.random.default_rng(9)
        rows = [(s, d, t) for s in (7, 16, 29) for d in (1, 2) for t in (1, 2, 3)]
        z = rng.normal(size=(len(rows), 6))
        changed = z.copy()
        mask = np.array([s == 7 and d == 2 for s, d, _ in rows])
        changed[mask] *= -1
        for method in ("mean", "robust", "multi"):
            with self.subTest(method=method):
                _, a, ma = _score(rows, z, template_method=method)
                _, b, mb = _score(rows, changed, template_method=method)
                self.assertEqual(ma["templates"], mb["templates"])
                other = np.array(ma["probe_subject_ids"]) != 7
                np.testing.assert_array_equal(a[other], b[other])

    def test_cancellations_and_missing_days_reported(self):
        rows, vectors = _base()
        rows += [
            (7, 2, 2),
            (7, 2, 2),
            (29, 1, 1),
            (30, 2, 1),
            (31, 1, 1),
            (31, 1, 2),
            (31, 2, 1),
        ]
        vectors += [[1, 0], [-1, 0], [1, 0], [0, 1], [1, 0], [-1, 0], [0, 1]]
        _, _, meta = _score(rows, vectors)
        coverage = meta["coverage"]
        self.assertEqual(coverage["eligible_subject_ids"], [7, 16])
        self.assertIn(
            {
                "subject": 7,
                "day": 2,
                "trial": 2,
                "reason": "zero_norm_trial_mean",
                "windows": 2,
            },
            coverage["exclusions"],
        )
        self.assertEqual(
            {r["subject"] for r in coverage["exclusions"]}, {7, 29, 30, 31}
        )
        # Robust candidates may retain a cancelled-mean enrollment subject;
        # that changed coverage must be disclosed, never compared by row offset.
        _, _, robust = _score(rows, vectors, template_method="robust")
        self.assertEqual(robust["coverage"]["eligible_subject_ids"], [7, 16, 31])

    def test_insufficient_coverage_keeps_exclusion_details(self):
        with self.assertRaises(CoverageError) as error:
            _score([(7, 1, 1), (7, 2, 1), (16, 1, 1)], [[1, 0]] * 3)
        self.assertEqual(error.exception.coverage["eligible_subject_ids"], [7])
        json.dumps(error.exception.coverage, allow_nan=False)

    def test_angular_medoid_resists_single_distant_trial_and_ties_choose_first(self):
        angles = np.deg2rad([0, 10, 20, 180])
        rows = [(7, 1, t) for t in (1, 2, 3, 4)] + [(7, 2, 1), (16, 1, 1), (16, 2, 1)]
        z = np.vstack(
            (np.column_stack((np.cos(angles), np.sin(angles))), [1, 0], [0, 1], [0, 1])
        )
        _, _, meta = _score(rows, z, template_method="robust")
        # Both central points minimize angular sum mathematically; actual finite
        # costs decide unless exactly tied. Either is an enrollment medoid.
        self.assertIn(meta["templates"][0]["prototype_trial_ids"], ([2], [3]))
        tie_rows = [(7, 1, 2), (7, 1, 1), (7, 2, 1), (16, 1, 1), (16, 2, 1)]
        _, _, tied = _score(
            tie_rows,
            [[0, 1], [1, 0], [1, 0], [-1, 0], [-1, 0]],
            template_method="robust",
        )
        self.assertEqual(tied["templates"][0]["prototype_trial_ids"], [1])

    def test_multi_exact_pair_objective_and_max_fusion(self):
        rows = [(7, 1, 1), (7, 1, 2), (7, 1, 3), (7, 2, 1), (16, 1, 1), (16, 2, 1)]
        z = [[1, 0], [1, 0], [0, 1], [0, 1], [-1, 0], [-1, 0]]
        _, scores, meta = _score(rows, z, template_method="multi")
        first = meta["templates"][0]
        self.assertEqual(first["prototype_trial_ids"], [1, 3])
        self.assertEqual(first["trial_prototype_indices"], [0, 0, 1])
        self.assertEqual(first["n_prototypes"], 2)
        self.assertEqual(scores[0], 0.0)
        np.testing.assert_allclose(
            np.linalg.norm(first["prototype_vectors"], axis=1), 1
        )

    def test_multi_duplicate_and_single_trials_collapse_to_one(self):
        rows = [(7, 1, 1), (7, 1, 2), (7, 2, 1), (16, 1, 1), (16, 2, 1)]
        _, _, meta = _score(
            rows, [[1, 1], [1, 1], [1, 1], [-1, 0], [-1, 0]], template_method="multi"
        )
        self.assertEqual([row["n_prototypes"] for row in meta["templates"]], [1, 1])

    def test_deterministic_row_permutation_and_no_mutation(self):
        rows = [(s, d, t) for s in (7, 16) for d in (1, 2) for t in (1, 2, 3)]
        z = np.random.default_rng(4).normal(size=(len(rows), 7))
        original = z.copy()
        permutation = np.arange(len(rows))[::-1]
        for method in ("mean", "robust", "multi"):
            a = _score(rows, z, template_method=method)
            b = _score(
                [rows[i] for i in permutation], z[permutation], template_method=method
            )
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])
            self.assertEqual(a[2], b[2])
        np.testing.assert_array_equal(z, original)

    def test_bad_inputs_fail(self):
        rows, z = _base()
        subjects, days, trials = np.asarray(rows).T
        cases = [
            ([[]] * 4, subjects, days, trials),
            ([1, 2, 3, 4], subjects, days, trials),
            (z, subjects.astype(float), days, trials),
            (z, subjects, [1, 2], trials),
            (z, subjects, [1, 3, 1, 2], trials),
            (z, subjects, days, [1, 1, 11, 1]),
            (z, [-1, 7, 16, 16], days, trials),
            ([[0, 0]] * 4, subjects, days, trials),
            ([[np.nan, 0]] * 4, subjects, days, trials),
            (np.array(z, dtype=complex), subjects, days, trials),
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                score_trials(*args)
        for kwargs in ({"template_method": "trimmed"}, {"probe_method": "robust"}):
            with self.assertRaises(ValueError):
                score_trials(z, subjects, days, trials, **kwargs)

    def test_finite_extreme_inputs_still_have_cosine_direction(self):
        rows, _ = _base()
        _, scores, _ = _score(rows, [[1e308, 0], [1e308, 0], [0, 1e308], [0, 1e308]])
        np.testing.assert_array_equal(scores, [0, -1, -1, 0])


def _diagnostic_data(n=5):
    claims = np.tile(np.arange(n), n)
    probes = np.repeat(np.arange(n), n)
    labels = (claims == probes).astype(np.int64)
    baseline = np.full(n * n, -1.0)
    new = np.where(labels == 1, 0.0, -1.0)
    return labels, baseline, new, claims, probes


class PairedDiagnosticsTests(unittest.TestCase):
    def test_five_subject_exact_reference_and_counts(self):
        report = paired_subject_diagnostics(
            *_diagnostic_data(), {"baseline": -0.5, "new": -0.5}
        )
        self.assertEqual(report["genuine_macro_acceptance_difference"], 1)
        self.assertEqual(report["n_positive_subject_differences"], 5)
        self.assertEqual(report["exact_sign_flip"]["n_assignments"], 32)
        self.assertEqual(report["exact_sign_flip"]["two_sided_tail_fraction"], 2 / 32)
        for row in report["per_subject"]:
            self.assertEqual(row["genuine"]["n"], 1)
            self.assertEqual(row["impostor_by_claim"]["n"], 4)
            self.assertEqual(row["impostor_by_probe"]["n"], 4)
        json.dumps(report, allow_nan=False)

    def test_candidate_specific_thresholds_and_tied_acceptance(self):
        labels, baseline, new, claims, probes = _diagnostic_data(2)
        # Each candidate independently has zero genuine rejections at its own threshold.
        report = paired_subject_diagnostics(
            labels, baseline, new, claims, probes, {"baseline": -1.0, "new": 0.0}
        )
        self.assertEqual(report["genuine_macro_acceptance_difference"], 0)
        self.assertEqual(report["exact_sign_flip"]["two_sided_tail_fraction"], 1)
        self.assertEqual(
            report["per_subject"][0]["impostor_by_claim"]["acceptance_difference"], -1
        )

    def test_macro_weights_users_equally_and_preserves_negative_differences(self):
        labels = [1, 1, 1, 1, 0, 0]
        claims = [7, 7, 7, 16, 7, 16]
        probes = [7, 7, 7, 16, 16, 7]
        report = paired_subject_diagnostics(
            labels,
            [-1, -1, -1, 0, -1, -1],
            [0, -1, -1, -1, -1, -1],
            claims,
            probes,
            {"baseline": -0.5, "new": -0.5},
        )
        self.assertAlmostEqual(
            report["genuine_macro_acceptance_difference"], (1 / 3 - 1) / 2
        )
        self.assertEqual(report["n_positive_subject_differences"], 1)
        self.assertEqual(report["n_negative_subject_differences"], 1)

    def test_exact_enumeration_is_bounded(self):
        report = paired_subject_diagnostics(
            *_diagnostic_data(3), {"baseline": -0.5, "new": -0.5}, max_exact_subjects=2
        )
        self.assertEqual(
            report["exact_sign_flip"]["status"], "not_enumerated_subject_limit"
        )
        self.assertIsNone(report["exact_sign_flip"]["two_sided_tail_fraction"])

    def test_float32_scores_do_not_round_the_calibrated_threshold(self):
        labels, _, _, claims, probes = _diagnostic_data(2)
        scores = np.full(4, -0.5, dtype=np.float32)
        report = paired_subject_diagnostics(
            labels,
            scores,
            scores,
            claims,
            probes,
            {"baseline": -0.49999999, "new": -0.5},
        )
        self.assertEqual(report["genuine_macro_acceptance_difference"], 1.0)
        self.assertEqual(report["per_subject"][0]["genuine"]["baseline"]["accepted"], 0)

    @unittest.skipUnless(
        np.finfo(np.longdouble).max > np.finfo(np.float64).max,
        "platform has no extended-precision floats",
    )
    def test_extended_precision_overflow_cannot_enter_json_or_decisions(self):
        args = list(_diagnostic_data(2))
        huge = np.longdouble("1e400")
        self.assertTrue(np.isfinite(huge))
        with self.assertRaises(ValueError):
            paired_subject_diagnostics(*args, {"baseline": huge, "new": -0.5})
        args[1] = np.full(4, huge, dtype=np.longdouble)
        with self.assertRaises(ValueError):
            paired_subject_diagnostics(*args, {"baseline": -0.5, "new": -0.5})

    def test_invalid_alignment_graph_thresholds_and_limits_fail(self):
        args = list(_diagnostic_data(2))
        for index, invalid in (
            (0, [0, 0, 0, 0]),
            (1, [np.nan] * 4),
            (2, [0, 0]),
            (3, [0.0] * 4),
            (4, [1, 1]),
        ):
            bad = args.copy()
            bad[index] = invalid
            with self.subTest(index=index), self.assertRaises(ValueError):
                paired_subject_diagnostics(*bad, {"baseline": -0.5, "new": -0.5})
        for thresholds in (
            {"baseline": -0.5},
            {"baseline": True, "new": -0.5},
            {"baseline": -0.5, "new": np.inf},
            -0.5,
        ):
            with self.assertRaises(ValueError):
                paired_subject_diagnostics(*args, thresholds)
        for limit in (0, 21, True, 1.5):
            with self.assertRaises(ValueError):
                paired_subject_diagnostics(
                    *args, {"baseline": -0.5, "new": -0.5}, max_exact_subjects=limit
                )


if __name__ == "__main__":
    unittest.main()
