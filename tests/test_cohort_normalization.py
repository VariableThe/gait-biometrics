"""Tiny synthetic NumPy-only checks; no real data, Torch, fitting or GPU."""

from __future__ import annotations

import builtins
import json
import unittest
from unittest import mock

import numpy as np

_IMPORT = builtins.__import__


def _without_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("cohort normalization must remain Torch-free")
    return _IMPORT(name, *args, **kwargs)


with mock.patch("builtins.__import__", side_effect=_without_torch):
    from engine import cohort_normalization as module
    from engine.template_scoring import CoverageError, score_trials


def _data(rows, embeddings):
    subjects, days, trials = np.asarray(rows, dtype=np.int64).T
    return np.asarray(embeddings, dtype=np.float64), subjects, days, trials


def _simple():
    target = _data(
        [(7, 1, 1), (7, 2, 1), (16, 1, 1), (16, 2, 1)],
        [[1, 0], [1, 0], [0, 1], [0, 1]],
    )
    cohort = _data(
        [(100, 1, 1), (100, 2, 1), (101, 1, 1), (101, 2, 1)],
        [[1, 0], [1, 0], [0, 1], [0, 1]],
    )
    return target, cohort


def _varied():
    rng = np.random.default_rng(43)
    rows = [
        (s, d, t)
        for s in (16, 7)
        for d in (1, 2)
        for t in ((3, 1, 2) if d == 1 else (2, 1))
        for _ in range(2 if t == 2 else 1)
    ]
    cohort_rows = [
        (s, d, t)
        for s in (102, 100, 101)
        for d in (1, 2)
        for t in ((1, 3, 2) if d == 1 else ((1,) if s == 101 else (2, 1)))
        for _ in range(2 if t == 2 else 1)
    ]
    return (
        _data(rows, rng.normal(size=(len(rows), 3))),
        _data(cohort_rows, rng.normal(size=(len(cohort_rows), 3))),
    )


def _normalize(target, cohort, **kwargs):
    return module.cohort_normalized_scores(
        *target,
        cohort_embeddings=cohort[0],
        cohort_subjects=cohort[1],
        cohort_days=cohort[2],
        cohort_trials=cohort[3],
        **kwargs,
    )


def _manual_probes(data):
    embeddings, subjects, days, trials = data
    windows = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    result = {}
    for subject in sorted(set(subjects.tolist())):
        for trial in sorted(set(trials[(subjects == subject) & (days == 2)].tolist())):
            mean = windows[
                (subjects == subject) & (days == 2) & (trials == trial)
            ].mean(axis=0)
            result[(subject, trial)] = mean / np.linalg.norm(mean)
    return result


def _manual_score(vector, prototypes):
    return float(np.clip(np.max(np.asarray(prototypes) @ vector), -1, 1) - 1)


class CohortNormalizationTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("builtins.__import__", side_effect=_without_torch)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_known_formula_dtypes_graph_and_json(self):
        target, cohort = _simple()
        labels, scores, metadata = _normalize(target, cohort)
        np.testing.assert_array_equal(labels, [1, 0, 0, 1])
        np.testing.assert_array_equal(scores, [1, -1, -1, 1])
        self.assertEqual(labels.dtype, np.dtype("int64"))
        self.assertEqual(scores.dtype, np.dtype("float64"))
        _, raw, base = score_trials(*target, template_method="multi")
        for key in (
            "claim_subject_ids",
            "probe_subject_ids",
            "probe_trial_ids",
            "coverage",
            "templates",
        ):
            self.assertEqual(metadata[key], base[key])
        np.testing.assert_array_equal(metadata["raw_scores"], raw)
        self.assertFalse(metadata["calibration"]["threshold_calibrated"])
        self.assertEqual(metadata["normalization"]["cohort_subject_ids"], [100, 101])
        self.assertFalse(metadata["normalization"]["adaptive"])
        self.assertEqual(metadata["normalization"]["std_ddof"], 0)
        self.assertIn("not cosine", metadata["score_range"])
        json.dumps(metadata, allow_nan=False)

    def test_manual_formula_all_methods_and_population_weights(self):
        target, cohort = _varied()
        target_probes, cohort_probes = _manual_probes(target), _manual_probes(cohort)
        for method in ("mean", "robust", "multi"):
            with self.subTest(method=method):
                labels, scores, metadata = _normalize(
                    target, cohort, template_method=method
                )
                base_labels, raw_scores, base = score_trials(
                    *target, template_method=method
                )
                _, _, cohort_base = score_trials(*cohort, template_method=method)
                np.testing.assert_array_equal(labels, base_labels)
                self.assertEqual(metadata["templates"], base["templates"])
                norm = metadata["normalization"]
                self.assertEqual(norm["cohort_templates"], cohort_base["templates"])
                templates = {
                    r["subject"]: r["prototype_vectors"] for r in base["templates"]
                }
                cohort_templates = [
                    r["prototype_vectors"] for r in cohort_base["templates"]
                ]
                expected = []
                for index, (claim, subject, trial) in enumerate(
                    zip(
                        metadata["claim_subject_ids"],
                        metadata["probe_subject_ids"],
                        metadata["probe_trial_ids"],
                        strict=True,
                    )
                ):
                    probe = target_probes[(subject, trial)]
                    raw = _manual_score(probe, templates[claim])
                    self.assertAlmostEqual(raw, raw_scores[index])
                    z = np.array(
                        [
                            _manual_score(p, templates[claim])
                            for p in cohort_probes.values()
                        ]
                    )
                    t = np.array([_manual_score(probe, p) for p in cohort_templates])
                    expected.append(
                        0.5
                        * (
                            (raw - z.mean()) / max(z.std(ddof=0), 1e-3)
                            + (raw - t.mean()) / max(t.std(ddof=0), 1e-3)
                        )
                    )
                np.testing.assert_allclose(scores, expected, rtol=1e-13, atol=1e-13)
                self.assertTrue(all(r["n_scores"] == 5 for r in norm["z_stats"]))
                self.assertTrue(all(r["n_scores"] == 3 for r in norm["t_stats"]))
                json.dumps(metadata, allow_nan=False)

    def test_both_sets_call_baseline_with_same_method(self):
        with mock.patch.object(module, "score_trials", wraps=score_trials) as scorer:
            _normalize(*_simple(), template_method="robust")
        self.assertEqual(scorer.call_count, 2)
        self.assertEqual(
            [c.kwargs for c in scorer.call_args_list],
            [{"template_method": "robust"}] * 2,
        )

    def test_std_floor_and_custom_floor(self):
        target, cohort = _simple()
        cohort[0][:] = [1, 0]
        _, scores, metadata = _normalize(target, cohort)
        np.testing.assert_array_equal(scores, [0, -500, -500, 1000])
        for stats in (
            metadata["normalization"]["z_stats"] + metadata["normalization"]["t_stats"]
        ):
            self.assertEqual(stats["std"], 0)
            self.assertEqual(stats["denominator"], 1e-3)
            self.assertTrue(stats["std_floored"])
        _, larger_floor, _ = _normalize(target, cohort, std_floor=0.25)
        np.testing.assert_array_equal(larger_floor, [0, -2, -2, 4])

    def test_no_target_probe_leak_into_enrollment_or_other_probe_scores(self):
        target, cohort = _varied()
        changed = tuple(array.copy() for array in target)
        mask = (changed[1] == 7) & (changed[2] == 2) & (changed[3] == 1)
        changed[0][mask] *= -1
        for method in ("mean", "robust", "multi"):
            with self.subTest(method=method):
                _, a, ma = _normalize(target, cohort, template_method=method)
                _, b, mb = _normalize(changed, cohort, template_method=method)
                self.assertEqual(ma["templates"], mb["templates"])
                for key in ("cohort_templates", "z_stats", "cohort_probe_order"):
                    self.assertEqual(ma["normalization"][key], mb["normalization"][key])
                other = ~(
                    (np.array(ma["probe_subject_ids"]) == 7)
                    & (np.array(ma["probe_trial_ids"]) == 1)
                )
                np.testing.assert_array_equal(a[other], b[other])
                self.assertTrue(np.any(a[~other] != b[~other]))
                old_t = ma["normalization"]["t_stats"]
                new_t = mb["normalization"]["t_stats"]
                for old, new in zip(old_t, new_t, strict=True):
                    if (old["subject"], old["trial"]) != (7, 1):
                        self.assertEqual(old, new)

    def test_row_permutation_and_no_mutation(self):
        target, cohort = _varied()
        originals = [array.copy() for array in (*target, *cohort)]
        for method in ("mean", "robust", "multi"):
            a = _normalize(target, cohort, template_method=method)
            b = _normalize(
                tuple(v[::-1] for v in target),
                tuple(v[::-1] for v in cohort),
                template_method=method,
            )
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])
            self.assertEqual(a[2], b[2])
        for original, current in zip(originals, (*target, *cohort), strict=True):
            np.testing.assert_array_equal(original, current)

    def test_window_direction_and_trial_mean_weight_match_baseline(self):
        target, cohort = _varied()
        scaled_target = (
            target[0] * np.arange(1, len(target[0]) + 1)[:, None],
            *target[1:],
        )
        scaled_cohort = (
            cohort[0] * np.arange(1, len(cohort[0]) + 1)[:, None],
            *cohort[1:],
        )
        for method in ("mean", "robust", "multi"):
            a = _normalize(target, cohort, template_method=method)
            b = _normalize(scaled_target, scaled_cohort, template_method=method)
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_allclose(a[1], b[1], rtol=1e-12, atol=1e-12)

    def test_target_exclusions_and_graph_are_exactly_baseline(self):
        target, cohort = _simple()
        embeddings = np.vstack([target[0], [1, 0], [1, 0], [-1, 0]])
        extra = [(29, 1, 1), (7, 2, 2), (7, 2, 2)]
        rows = list(zip(*target[1:], strict=True)) + extra
        target = _data(rows, embeddings)
        labels, _, metadata = _normalize(target, cohort)
        base_labels, _, base = score_trials(*target, template_method="multi")
        np.testing.assert_array_equal(labels, base_labels)
        for key in (
            "claim_subject_ids",
            "probe_subject_ids",
            "probe_trial_ids",
            "coverage",
            "templates",
        ):
            self.assertEqual(metadata[key], base[key])

    def test_self_cohort_including_excluded_target_identity_rejected(self):
        target, _ = _simple()
        with self.assertRaisesRegex(ValueError, "disjoint"):
            _normalize(target, target)
        target, cohort = _simple()
        target = (
            np.vstack([target[0], [1, 0]]),
            np.append(target[1], 100),
            np.append(target[2], 1),
            np.append(target[3], 1),
        )
        with self.assertRaisesRegex(ValueError, "disjoint"):
            _normalize(target, cohort)

    def test_every_cohort_id_requires_eligible_both_days(self):
        target, cohort = _simple()
        for extra_rows, extra_vectors in (
            ([(102, 1, 1)], [[1, 0]]),
            ([(102, 2, 1)], [[1, 0]]),
            ([(102, 1, 1), (102, 2, 1), (102, 2, 1)], [[1, 0], [1, 0], [-1, 0]]),
        ):
            rows = list(zip(*cohort[1:], strict=True)) + extra_rows
            incomplete = _data(rows, np.vstack([cohort[0], extra_vectors]))
            with self.assertRaisesRegex(ValueError, "every cohort ID"):
                _normalize(target, incomplete)

    def test_each_population_needs_two_baseline_eligible_ids(self):
        target, cohort = _simple()
        for a, b in (
            (tuple(v[:2] for v in target), cohort),
            (target, tuple(v[:2] for v in cohort)),
        ):
            with self.assertRaises(CoverageError):
                _normalize(a, b)

    def test_bad_floor_inputs_and_overflow_fail_explicitly(self):
        target, cohort = _simple()
        for floor in (True, False, 0, -1, np.nan, np.inf, "0.001", [0.001], 10**400):
            with self.subTest(floor=floor), self.assertRaises(ValueError):
                _normalize(target, cohort, std_floor=floor)
        cohort[0][:] = [1, 0]
        with self.assertRaisesRegex(ValueError, "overflow|finite"):
            _normalize(target, cohort, std_floor=np.nextafter(0.0, 1.0))

    def test_nonfinite_zero_dimension_and_bad_metadata_inputs_fail(self):
        target, cohort = _simple()
        for value in (np.nan, np.inf, 0.0):
            broken = (np.full_like(cohort[0], value), *cohort[1:])
            with self.subTest(value=value), self.assertRaises(ValueError):
                _normalize(target, broken)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            _normalize(target, (np.column_stack([cohort[0], np.ones(4)]), *cohort[1:]))
        for broken in (
            (cohort[0].astype(complex), *cohort[1:]),
            (cohort[0], cohort[1].astype(float), *cohort[2:]),
            (cohort[0], cohort[1], [1, 3, 1, 2], cohort[3]),
            (cohort[0], *cohort[1:3], [1, 11, 1, 1]),
        ):
            with self.assertRaises(ValueError):
                _normalize(target, broken)

    def test_finite_extreme_windows_use_safe_baseline_direction(self):
        target, cohort = _simple()
        extreme_target = (target[0] * 1e308, *target[1:])
        extreme_cohort = (cohort[0] * 1e308, *cohort[1:])
        ordinary = _normalize(target, cohort)
        extreme = _normalize(extreme_target, extreme_cohort)
        np.testing.assert_array_equal(extreme[0], ordinary[0])
        np.testing.assert_array_equal(extreme[1], ordinary[1])
        json.dumps(extreme[2], allow_nan=False)


if __name__ == "__main__":
    unittest.main()
