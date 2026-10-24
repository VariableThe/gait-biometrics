"""Synthetic cross-day protocol tests; no dataset, Torch, or GPU is required."""

from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock
from zipfile import ZipFile

import numpy as np
from sklearn.metrics import roc_curve

_IMPORT = builtins.__import__


def _without_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("cross-day synthetic tests must not import Torch")
    return _IMPORT(name, *args, **kwargs)


with mock.patch("builtins.__import__", side_effect=_without_torch):
    from scripts import cross_day_verification as study


ROOT = Path(__file__).resolve().parents[1]
DEV = (7, 16, 11, 9, 29, 27, 5, 13, 3, 1, 25, 24, 14, 19, 17)
CALIBRATION = (21, 18, 23, 12, 30)
TEST = (26, 28, 22, 15, 6)


def _evaluation_functions():
    """Compile the actual pure metric functions without executing Torch imports."""
    names = {
        "_finite_real_scalar",
        "_validated_scores",
        "_roc",
        "_eer_operating_point",
        "compute_eer",
        "calibrate_threshold",
        "fixed_threshold_metrics",
    }
    source = ast.parse((ROOT / "engine/evaluate.py").read_text())
    functions = [
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != names:
        raise AssertionError("the production score metric helpers changed")
    namespace = {"np": np, "roc_curve": roc_curve}
    exec(  # noqa: S102 -- compile only the explicitly selected local pure functions
        compile(
            ast.Module(body=list[ast.stmt](functions), type_ignores=[]),
            "engine/evaluate.py",
            "exec",
        ),
        namespace,
    )
    return tuple(
        namespace[name]
        for name in (
            "compute_eer",
            "calibrate_threshold",
            "fixed_threshold_metrics",
        )
    )


def _data(rows):
    """Rows are (authentic subject, day, trial, within-recording start)."""
    labels = np.asarray(rows, dtype=np.int64).reshape(-1, 4)
    return {
        "X": np.zeros((len(rows), 3, 256), dtype=np.float32),
        **{
            key: labels[:, index].copy()
            for index, key in enumerate(("subjects", "days", "trials", "starts"))
        },
        "recording_indices": np.arange(len(rows), dtype=np.int64),
        "metadata": {
            "sampling_rate_hz": 100,
            "units": "m/s^2",
            "recordings": [],
            "exclusions": [],
            "placement_side": "unknown",
        },
    }


def _cohort(subject_ids):
    rows = [
        (subject, day, trial, start)
        for subject in subject_ids
        for day in (1, 2)
        for trial in (1, 2)
        for start in (0, 256)
    ]
    result = _data(rows)
    # Stable per-subject signals: calls for the same cohort reproduce every byte.
    for index, (subject, day, trial, start) in enumerate(rows):
        rng = np.random.default_rng(subject * 10_000 + day * 100 + trial * 10 + start)
        result["X"][index] = rng.normal(size=(3, 256)) + subject / 30
    return result


def _score_table(scored):
    return {
        (int(claim), int(probe), int(trial)): (int(label), float(score))
        for claim, probe, trial, label, score in zip(
            scored["claim_subjects"],
            scored["probe_subjects"],
            scored["probe_trials"],
            scored["labels"],
            scored["scores"],
            strict=True,
        )
    }


class _TorchFreeTestCase(unittest.TestCase):
    def setUp(self):
        self.addCleanup(mock.patch.stopall)
        mock.patch("builtins.__import__", side_effect=_without_torch).start()


class CrossDayTests(_TorchFreeTestCase):
    def test_exact_population_and_five_consecutive_development_folds(self):
        self.assertEqual(
            (study.DEV_IDS, study.CALIBRATION_IDS, study.TEST_IDS),
            (DEV, CALIBRATION, TEST),
        )
        self.assertEqual(len({*DEV, *CALIBRATION, *TEST}), 25)
        expected = [
            {
                "fold": index,
                "val_subject_ids": list(DEV[index * 3 : index * 3 + 3]),
                "train_subject_ids": [
                    s for s in DEV if s not in DEV[index * 3 : index * 3 + 3]
                ],
            }
            for index in range(5)
        ]
        self.assertEqual(study.development_folds(), expected)
        config = study.protocol_config()
        self.assertEqual(config["development_subject_ids"], list(DEV))
        self.assertEqual(config["calibration_subject_ids"], list(CALIBRATION))
        self.assertEqual(config["test_subject_ids"], list(TEST))
        self.assertEqual(
            (config["sampling_rate_hz"], config["window_size"], config["stride"]),
            (100, 256, 256),
        )
        self.assertEqual(config["calibration_target_far"], 0.01)
        json.dumps(config, allow_nan=False)

    def test_validation_accepts_missing_coverage_but_rejects_invalid_windows(self):
        valid = _cohort((7, 16))
        study.validate_windows(valid, (7, 16, 11))
        alterations = {
            "channel_count": ("X", valid["X"][:, :2]),
            "window_length": ("X", valid["X"][:, :, :-1]),
            "nonfinite": ("X", np.full_like(valid["X"], np.nan)),
            "string_signal": ("X", valid["X"].astype(str)),
            "float_subjects": ("subjects", valid["subjects"].astype(float)),
            "wrong_label_shape": ("trials", valid["trials"][:, None]),
            "short_labels": ("starts", valid["starts"][:-1]),
            "wrong_day": ("days", np.zeros_like(valid["days"])),
            "wrong_trial": ("trials", np.full_like(valid["trials"], 11)),
            "unexpected_identity": ("subjects", np.full_like(valid["subjects"], 21)),
            "nonmapping_metadata": ("metadata", []),
            "nonfinite_metadata": ("metadata", {"sampling_rate_hz": np.nan}),
        }
        for name, (key, value) in alterations.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                study.validate_windows({**valid, key: value}, (7, 16))
        duplicate = copy.deepcopy(valid)
        for key in ("subjects", "days", "trials", "starts"):
            duplicate[key][1] = duplicate[key][0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            study.validate_windows(duplicate, (7, 16))
        for key in ("X", "subjects", "days", "trials", "starts", "metadata"):
            with self.subTest(missing=key), self.assertRaises(ValueError):
                study.validate_windows(
                    {k: v for k, v in valid.items() if k != key}, (7, 16)
                )

    def test_fingerprint_covers_all_arrays_and_nested_metadata(self):
        data = _cohort((7, 16))
        original = study.data_fingerprint(data)
        self.assertEqual(original, study.data_fingerprint(copy.deepcopy(data)))
        reordered = dict(reversed(list(data.items())))
        reordered["metadata"] = dict(reversed(list(data["metadata"].items())))
        self.assertEqual(original, study.data_fingerprint(reordered))
        strided = copy.deepcopy(data)
        strided["X"] = np.asfortranarray(strided["X"])
        self.assertEqual(original, study.data_fingerprint(strided))
        for key in ("X", "subjects", "days", "trials", "starts", "recording_indices"):
            altered = copy.deepcopy(data)
            altered[key].flat[0] += 1
            with self.subTest(array=key):
                self.assertNotEqual(original, study.data_fingerprint(altered))
        for key, value in (
            ("units", "g"),
            ("placement_side", "left"),
            ("exclusions", [{"reason": "short_trial", "n_rows": 255}]),
            ("recordings", [{"header": ["frame", "acc x"], "first_frame": 99}]),
        ):
            altered = copy.deepcopy(data)
            altered["metadata"][key] = value
            with self.subTest(metadata=key):
                self.assertNotEqual(original, study.data_fingerprint(altered))
        altered = copy.deepcopy(data)
        altered["X"] = altered["X"].astype(np.float64)
        self.assertNotEqual(original, study.data_fingerprint(altered))

    def test_cosine_minus_one_and_exhaustive_directed_claims(self):
        data = _data([(7, 1, 1, 0), (7, 2, 1, 0), (16, 1, 1, 0), (16, 2, 1, 0)])
        scored = study.cross_day_scores(
            [[1, 0], [1, 0], [-1, 0], [0, 1]], data, (7, 16)
        )
        self.assertEqual(
            _score_table(scored),
            {
                (7, 7, 1): (1, 0.0),
                (16, 7, 1): (0, -2.0),
                (7, 16, 1): (0, -1.0),
                (16, 16, 1): (1, -1.0),
            },
        )
        self.assertEqual(
            (scored["coverage"]["n_genuine"], scored["coverage"]["n_impostor"]), (2, 2)
        )

    def test_windows_and_trials_are_unit_normalized_before_equal_trial_enrollment(self):
        # Trial 1 has three windows and unequal magnitudes; trial 2 has one.
        rows = [
            (7, 1, 1, 0),
            (7, 1, 1, 256),
            (7, 1, 1, 512),
            (7, 1, 2, 0),
            (7, 2, 1, 0),
            (7, 2, 1, 256),
            (7, 2, 2, 0),
            (16, 1, 1, 0),
            (16, 2, 1, 0),
        ]
        z = np.asarray(
            [
                [100, 0],
                [0, 1],
                [0, 2],
                [0, 10],
                [10, 0],
                [0, 1],
                [-1, 0],
                [0, 1],
                [1, 0],
            ],
            dtype=float,
        )
        data = _data(rows)
        table = _score_table(study.cross_day_scores(z, data, (7, 16)))
        trial1 = np.array([1, 2]) / np.sqrt(5)
        template = trial1 + [0, 1]
        template /= np.linalg.norm(template)
        np.testing.assert_allclose(
            table[(7, 7, 1)][1], template @ (np.ones(2) / np.sqrt(2)) - 1
        )
        np.testing.assert_allclose(table[(7, 7, 2)][1], -template[0] - 1)
        # A single enrollment trial with three windows must get no extra weight.
        same_direction = z.copy()
        same_direction[:3] = [1, 0]
        same_direction[4:6] = [0, 1]
        equal = _score_table(study.cross_day_scores(same_direction, data, (7, 16)))
        self.assertAlmostEqual(equal[(7, 7, 1)][1], 1 / np.sqrt(2) - 1)

    def test_day_roles_do_not_leak_and_window_order_does_not_change_scores(self):
        rows = [
            (s, day, trial, start)
            for s in (7, 16, 11)
            for day in (1, 2)
            for trial in range(1, (1 if day == 1 else (7, 16, 11).index(s) + 1) + 1)
            for start in (0, 256)
        ]
        data = _data(rows)
        rng = np.random.default_rng(8)
        z = rng.normal(size=(len(rows), 4))
        scored = study.cross_day_scores(z, data, (7, 16, 11))
        self.assertEqual(
            (scored["coverage"]["n_genuine"], scored["coverage"]["n_impostor"]), (6, 12)
        )
        permutation = rng.permutation(len(rows))
        shuffled = {
            key: value[permutation] if isinstance(value, np.ndarray) else value
            for key, value in data.items()
        }
        actual = _score_table(
            study.cross_day_scores(z[permutation], shuffled, (7, 16, 11))
        )
        for key, (label, score) in _score_table(scored).items():
            self.assertEqual(actual[key][0], label)
            self.assertAlmostEqual(actual[key][1], score)
        # Changing another day2 probe cannot change this subject's enrollment.
        changed = z.copy()
        changed[(data["subjects"] == 7) & (data["days"] == 2)] *= -1
        after = _score_table(study.cross_day_scores(changed, data, (7, 16, 11)))
        for key, value in _score_table(scored).items():
            if key[1] != 7:
                self.assertEqual(after[key], value)

    def test_missing_day_and_cancelled_trials_are_excluded_with_coverage(self):
        rows = [
            (7, 1, 1, 0),
            (7, 2, 1, 0),
            (16, 1, 1, 0),
            (16, 2, 1, 0),
            (16, 2, 2, 0),
            (16, 2, 2, 256),
            (11, 1, 1, 0),
            (9, 2, 1, 0),
            (29, 1, 1, 0),
            (29, 1, 2, 0),
            (29, 2, 1, 0),
        ]
        z = [
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
            [1, 0],
            [-1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [-1, 0],
            [0, 1],
        ]
        scored = study.cross_day_scores(z, _data(rows), (7, 16, 11, 9, 29, 27))
        coverage = scored["coverage"]
        self.assertEqual(coverage["eligible_subject_ids"], [7, 16])
        self.assertEqual((coverage["n_genuine"], coverage["n_impostor"]), (2, 2))
        self.assertEqual(set(scored["probe_subjects"]), {7, 16})
        self.assertTrue(
            any(
                row.get("reason") == "zero_norm_trial_mean" and row["subject"] == 16
                for row in coverage["exclusions"]
            )
        )
        self.assertTrue(
            {11, 9, 29, 27}.issubset({row["subject"] for row in coverage["exclusions"]})
        )
        with self.assertRaises(ValueError):
            study.cross_day_scores([[1, 0], [1, 0]], _data(rows[:2]), (7, 16))

    def test_invalid_embeddings_fail_instead_of_manufacturing_scores(self):
        data = _cohort((7, 16))
        for z in (
            np.ones((len(data["X"]) - 1, 2)),
            np.ones(len(data["X"])),
            np.zeros((len(data["X"]), 2)),
            np.full((len(data["X"]), 2), np.inf),
        ):
            with self.subTest(shape=z.shape), self.assertRaises(ValueError):
                study.cross_day_scores(z, data, (7, 16))

    def test_candidate_selection_uses_mean_eer_then_fixed_order(self):
        rows = [
            {"name": name, "mean_eer": 0.25}
            for name in ("wccn_0.2", "wccn_auto", "standard")
        ]
        self.assertEqual(study.select_candidate(rows)["name"], "standard")
        rows[-1]["mean_eer"] = 0.5
        self.assertEqual(study.select_candidate(rows)["name"], "wccn_auto")
        rows[0]["mean_eer"] = 0.1
        self.assertIs(study.select_candidate(rows), rows[0])
        for bad in (
            rows[:-1],
            rows + rows[:1],
            [{"name": "unknown", "mean_eer": 0.1}, *rows[1:]],
            [{**row, "mean_eer": np.nan} for row in rows],
        ):
            with self.assertRaises(ValueError):
                study.select_candidate(bad)

    def test_calibration_ties_use_attainable_rank_and_accept_equal_scores_together(
        self,
    ):
        _, calibrate, fixed = _evaluation_functions()
        labels = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.6, 0.6, 0.6, 0.3, 0.1])
        calibration = calibrate(labels, scores, target_far=0.01)
        self.assertEqual(calibration["threshold"], 0.9)
        self.assertEqual((calibration["far"], calibration["frr"]), (0.0, 2 / 3))
        at_tie = fixed(labels, scores, 0.6)
        self.assertEqual((at_tie["false_accepts"], at_tie["false_rejects"]), (1, 0))
        self.assertEqual(
            calibrate(labels[::-1], scores[::-1], target_far=0.01), calibration
        )


class CrossDayArchiveTests(_TorchFreeTestCase):
    def setUp(self):
        super().setUp()
        temp_root = ROOT / "artifacts/tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="cross-day-zip-", dir=temp_root)
        self.addCleanup(temporary.cleanup)
        self.archive = Path(temporary.name) / "synthetic.zip"

    def _write_archive(self, *, omit=(), extra=()):
        rows = [(s, 1, 1) for s in study.ALL_IDS if s not in omit]
        rows.append((7, 2, 2))
        self.payloads = {
            f"fixture/processed_data/user{s:02d}/capture_user{s:02d}_day{d}_{t:04d}_imu_walk.csv": f"synthetic only: subject={s}, day={d}, trial={t}".encode()
            for s, d, t in rows
        }
        self.payloads.update(extra)
        with ZipFile(self.archive, "w") as archive:
            for name, content in self.payloads.items():
                archive.writestr(name, content)
            archive.writestr("fixture/README.txt", "synthetic inventory fixture")
            archive.writestr(
                "fixture/raw_data/user26/capture_user26_day1_0001_imu.csv", "ignored"
            )
        content = self.archive.read_bytes()
        return {
            "ARCHIVE_SIZE": len(content),
            "ARCHIVE_MD5": hashlib.md5(content, usedforsecurity=False).hexdigest(),
            "ARCHIVE_SHA256": hashlib.sha256(content).hexdigest(),
        }

    def _audit_without_member_reads(self):
        with mock.patch.object(
            study.ZipFile,
            "open",
            side_effect=AssertionError("inventory audit opened payload"),
        ) as opened:
            audit = study.audit_archive(self.archive)
        opened.assert_not_called()
        return audit

    def test_archive_audit_hashes_inventory_and_missing_trials_without_payload_reads(
        self,
    ):
        pinned = self._write_archive()
        with mock.patch.multiple(study, **pinned):
            audit = self._audit_without_member_reads()
        self.assertEqual(
            (audit["size"], audit["md5"], audit["sha256"]),
            (pinned["ARCHIVE_SIZE"], pinned["ARCHIVE_MD5"], pinned["ARCHIVE_SHA256"]),
        )
        self.assertEqual(
            {row["subject"] for row in audit["members"]}, set(study.ALL_IDS)
        )
        self.assertEqual(
            [row["member"] for row in audit["members"]], sorted(self.payloads)
        )
        present = {(s, 1, 1) for s in study.ALL_IDS} | {(7, 2, 2)}
        expected_missing = {
            (s, d, t) for s in study.ALL_IDS for d in (1, 2) for t in range(1, 11)
        } - present
        self.assertEqual(
            {(r["subject"], r["day"], r["trial"]) for r in audit["missing_recordings"]},
            expected_missing,
        )
        for row in audit["members"]:
            self.assertEqual(row["size"], len(self.payloads[row["member"]]))

    def test_archive_integrity_failures_precede_inventory_access(self):
        pinned = self._write_archive()
        for field, wrong, message in (
            ("ARCHIVE_SIZE", pinned["ARCHIVE_SIZE"] + 1, "byte count"),
            ("ARCHIVE_MD5", "0" * 32, "MD5"),
            ("ARCHIVE_SHA256", "0" * 64, "SHA-256"),
        ):
            overrides: dict[str, Any] = {**pinned, field: wrong}
            with (
                self.subTest(field=field),
                mock.patch.multiple(study, **overrides),
                mock.patch.object(
                    study,
                    "ZipFile",
                    side_effect=AssertionError(
                        "inventory accessed before integrity check"
                    ),
                ),
                self.assertRaisesRegex(ValueError, message),
            ):
                study.audit_archive(self.archive)

    def test_archive_rejects_wrong_population_and_inconsistent_or_duplicate_names(self):
        cases = (
            ({"omit": (TEST[0],)}, "inventory"),
            (
                {
                    "extra": [
                        (
                            "fixture/processed_data/user07/capture_user16_day1_0002_imu_walk.csv",
                            b"synthetic",
                        )
                    ]
                },
                "inconsistent",
            ),
            (
                {
                    "extra": [
                        (
                            "fixture/processed_data/user07/capture_user07_day1_0011_imu_walk.csv",
                            b"synthetic",
                        )
                    ]
                },
                "inconsistent",
            ),
            (
                {
                    "extra": [
                        (
                            "alias/processed_data/user07/capture_user07_day1_0001_imu_walk.csv",
                            b"synthetic",
                        )
                    ]
                },
                "duplicate",
            ),
        )
        for options, message in cases:
            with self.subTest(options=options):
                pinned = self._write_archive(**options)
                with (
                    mock.patch.multiple(study, **pinned),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    self._audit_without_member_reads()

    def test_selected_member_hashes_open_only_explicit_cohort_payloads(self):
        pinned = self._write_archive()
        with mock.patch.multiple(study, **pinned):
            audit = self._audit_without_member_reads()
        original_audit = copy.deepcopy(audit)
        real_open = ZipFile.open
        for ids in (DEV, CALIBRATION, ()):
            expected = {
                row["member"] for row in audit["members"] if row["subject"] in ids
            }
            reads = []

            def permitted_open(
                archive, name, *args, permitted=expected, observed=reads, **kwargs
            ):
                self.assertIn(
                    name, permitted, "opened a payload outside the explicit cohort"
                )
                observed.append(name)
                return real_open(archive, name, *args, **kwargs)

            with (
                self.subTest(ids=ids),
                mock.patch.object(ZipFile, "open", new=permitted_open),
            ):
                selected = study.selected_member_hashes(self.archive, audit, ids)
            self.assertEqual(set(reads), expected)
            self.assertEqual(len(reads), len(expected))
            self.assertEqual({row["member"] for row in selected}, expected)
            for row in selected:
                self.assertEqual(
                    row["sha256"],
                    hashlib.sha256(self.payloads[row["member"]]).hexdigest(),
                )
        self.assertEqual(audit, original_audit)


class CrossDayStageTests(_TorchFreeTestCase):
    """Real metric fits and artifacts, synthetic descriptors and archive boundaries."""

    def setUp(self):
        super().setUp()
        temp_root = ROOT / "artifacts/tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(
            prefix="cross-day-tests-", dir=temp_root
        )
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)
        self.args = SimpleNamespace(
            archive_path=self.out / "not-a-real-dataset.zip",
            out_root=self.out / "study",
            target_far=0.01,
            prior_test_inspection=None,
        )
        self.data = {tuple(ids): _cohort(ids) for ids in (DEV, CALIBRATION, TEST)}
        self.loads = []
        self.test_load_error = None

        def load(_path, ids):
            cohort = tuple(ids)
            self.loads.append(cohort)
            if cohort == TEST:
                self.assertTrue((self.args.out_root / "protocol_lock.json").is_file())
                self.assertTrue((self.args.out_root / "test_attempt.json").is_file())
                if self.test_load_error is not None:
                    raise self.test_load_error
            return copy.deepcopy(self.data[cohort])

        self.loader = mock.patch.object(study, "load_windows", side_effect=load).start()
        mock.patch.object(
            study, "audit_archive", return_value={"sha256": "synthetic", "members": []}
        ).start()
        self.sources = {"synthetic.py": "frozen-source-hash"}
        mock.patch.object(
            study, "source_hashes", side_effect=lambda: self.sources.copy()
        ).start()
        self.members = mock.patch.object(
            study, "selected_member_hashes", return_value=[]
        ).start()
        # Stage behavior is independent of descriptor equations, tested elsewhere.
        mock.patch.object(
            study,
            "extract_gait_features",
            side_effect=lambda x, _rate: x[:, 0, :8].copy(),
        ).start()
        self.metric_functions = _evaluation_functions()
        mock.patch.object(
            study, "evaluation_functions", return_value=self.metric_functions
        ).start()
        self.fits = []
        fit = study.WithinClassMetric.fit

        def checked_fit(metric, features, subjects):
            identities = set(map(int, subjects))
            self.assertTrue(identities.issubset(DEV))
            self.fits.append(identities)
            return fit(metric, features, subjects)

        mock.patch.object(study.WithinClassMetric, "fit", new=checked_fit).start()

    def _lock(self):
        study.train_stage(self.args)
        return study.lock_stage(self.args)

    def test_stages_fit_development_only_and_use_frozen_calibration_threshold(self):
        registry = study.train_stage(self.args)
        self.assertEqual(self.loads, [DEV])
        self.assertEqual(len(self.fits), 16)
        expected = [
            set(fold["train_subject_ids"]) for fold in study.development_folds()
        ]
        self.assertEqual(self.fits[:-1], expected * 3)
        self.assertEqual(self.fits[-1], set(DEV))
        self.assertFalse(registry["test_values_loaded"])
        lock = study.lock_stage(self.args)
        self.assertEqual(self.loads, [DEV, DEV, CALIBRATION])
        self.assertEqual(len(self.fits), 16)
        self.assertLessEqual(lock["calibration"]["far"], 0.01)
        self.assertFalse(lock["test_values_loaded"])
        # Neither candidate selection by scores nor calibration may run on TEST.
        compute_eer, _, fixed = self.metric_functions
        with mock.patch.object(
            study,
            "evaluation_functions",
            return_value=(
                compute_eer,
                mock.Mock(side_effect=AssertionError("TEST recalibration")),
                fixed,
            ),
        ):
            result = study.evaluate_stage(self.args)
        self.assertEqual(self.loads, [DEV, DEV, CALIBRATION, DEV, CALIBRATION, TEST])
        self.assertEqual(len(self.fits), 16)
        self.assertEqual(result["fixed_threshold"]["threshold"], lock["threshold"])
        self.assertEqual(
            (
                result["fixed_threshold"]["n_genuine"],
                result["fixed_threshold"]["n_impostor"],
            ),
            (10, 40),
        )
        errors = result["per_subject_errors"]
        for role in ("by_claim_subject", "by_probe_subject"):
            self.assertEqual(set(errors[role]["per_subject"]), set(map(str, TEST)))
        self.assertEqual(errors["threshold"], lock["threshold"])
        self.assertEqual(
            [tuple(call.args[2]) for call in self.members.call_args_list],
            [DEV, DEV, CALIBRATION, DEV, CALIBRATION, TEST],
        )
        for filename in (
            "registry.json",
            "metric.npz",
            "protocol_lock.json",
            "test_attempt.json",
            "test_result.json",
        ):
            self.assertTrue((self.args.out_root / filename).is_file())
        before = self.loader.call_count
        with self.assertRaisesRegex(ValueError, "attempt"):
            study.evaluate_stage(self.args)
        self.assertEqual(self.loader.call_count, before)

    def test_evaluation_requires_lock_before_any_loader_call(self):
        with self.assertRaisesRegex(ValueError, "lock"):
            study.evaluate_stage(self.args)
        self.loader.assert_not_called()

    def test_tied_scores_preserve_finite_reject_all_threshold(self):
        original = study.cross_day_scores

        def tied_scores(*args):
            scored = original(*args)
            scored["scores"].fill(-1.0)
            return scored

        with mock.patch.object(study, "cross_day_scores", side_effect=tied_scores):
            lock = self._lock()
            self.assertTrue(np.isfinite(lock["threshold"]))
            self.assertGreater(lock["threshold"], -1.0)
            result = study.evaluate_stage(self.args)
        fixed = result["fixed_threshold"]
        self.assertEqual(fixed["threshold"], lock["threshold"])
        self.assertEqual((fixed["far"], fixed["frr"]), (0.0, 1.0))
        self.assertEqual((fixed["false_accepts"], fixed["false_rejects"]), (0, 10))

    def test_training_revalidates_loader_population_before_fitting(self):
        self.data[DEV]["subjects"][0] = CALIBRATION[0]
        with self.assertRaisesRegex(ValueError, "requested cohort"):
            study.train_stage(self.args)
        self.assertEqual(self.fits, [])
        self.assertFalse((self.args.out_root / "metric.npz").exists())

    def test_lock_revalidates_calibration_population(self):
        study.train_stage(self.args)
        self.data[CALIBRATION]["subjects"][0] = TEST[0]
        with self.assertRaisesRegex(ValueError, "requested cohort"):
            study.lock_stage(self.args)
        self.assertNotIn(TEST, self.loads)
        self.assertEqual(len(self.fits), 16)
        self.assertFalse((self.args.out_root / "protocol_lock.json").exists())

    def test_failed_test_load_consumes_attempt_and_prevents_retry(self):
        self._lock()
        self.test_load_error = RuntimeError("synthetic TEST read failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic TEST"):
            study.evaluate_stage(self.args)
        attempt = self.args.out_root / "test_attempt.json"
        self.assertTrue(attempt.is_file())
        saved_attempt = attempt.read_bytes()
        self.assertFalse((self.args.out_root / "test_result.json").exists())
        before = self.loader.call_count
        with self.assertRaisesRegex(ValueError, "attempt"):
            study.evaluate_stage(self.args)
        self.assertEqual(self.loader.call_count, before)
        self.assertEqual(attempt.read_bytes(), saved_attempt)

    def test_metadata_tampering_fails_before_test_load_or_attempt(self):
        self._lock()
        for cohort in (DEV, CALIBRATION):
            with self.subTest(cohort=cohort):
                original = copy.deepcopy(self.data[cohort])
                self.data[cohort]["metadata"]["exclusions"] = [{"reason": "tampered"}]
                with self.assertRaisesRegex(ValueError, "data/metadata"):
                    study.evaluate_stage(self.args)
                self.assertNotIn(TEST, self.loads)
                self.assertFalse((self.args.out_root / "test_attempt.json").exists())
                self.data[cohort] = original

    def test_lock_tampering_fails_before_any_loader_call(self):
        self._lock()
        path = self.args.out_root / "protocol_lock.json"
        lock = json.loads(path.read_text())
        lock["threshold"] += 0.01
        path.write_text(json.dumps(lock))
        self.loader.reset_mock()
        with self.assertRaisesRegex(ValueError, "lock"):
            study.evaluate_stage(self.args)
        self.loader.assert_not_called()

    def test_metric_and_source_tampering_fail_before_test_loading(self):
        self._lock()
        self.sources["synthetic.py"] = "changed-source-hash"
        with self.assertRaisesRegex(ValueError, "source"):
            study.evaluate_stage(self.args)
        self.assertNotIn(TEST, self.loads)
        self.sources["synthetic.py"] = "frozen-source-hash"
        with (self.args.out_root / "metric.npz").open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "metric"):
            study.evaluate_stage(self.args)
        self.assertNotIn(TEST, self.loads)

    def test_development_metadata_must_match_before_calibration_load(self):
        study.train_stage(self.args)
        self.data[DEV]["recording_indices"][0] += 1
        with self.assertRaisesRegex(ValueError, "data/metadata"):
            study.lock_stage(self.args)
        self.assertEqual(self.loads, [DEV, DEV])
        self.assertFalse((self.args.out_root / "protocol_lock.json").exists())


if __name__ == "__main__":
    unittest.main()
