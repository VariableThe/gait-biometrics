"""Bounded synthetic runner guards: no Torch, GPU, or dataset loading."""

from __future__ import annotations

import builtins
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

_IMPORT = builtins.__import__


def _without_training_or_data(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch.") or name == "data.santos":
        raise AssertionError("synthetic study guards must not load Torch or Santos")
    return _IMPORT(name, *args, **kwargs)


with mock.patch("builtins.__import__", side_effect=_without_training_or_data):
    from scripts import multisensor_study as study


DESCRIPTORS = (
    ("accel", "axes", "auto"),
    ("accel_gyro", "axes", "auto"),
    ("accel", "invariant", "auto"),
    ("accel_gyro", "invariant", "auto"),
    ("accel_gyro", "fusion", "auto"),
    ("accel_gyro", "fusion", 0.2),
)
NEURAL = (
    "accel_supcon",
    "accel_cross_day_supcon",
    "accel_gyro_supcon",
    "accel_gyro_cross_day_supcon",
)
TEMPLATES = ("mean", "robust", "multi")


def _candidates():
    descriptors = [
        {
            "name": "descriptor_" + "_".join(map(str, spec)) + "/" + template,
            "family": "descriptors",
            "descriptor": list(spec),
            "template": template,
        }
        for spec in DESCRIPTORS
        for template in TEMPLATES
    ]
    return descriptors + [
        {"name": family + "/" + template, "family": family, "template": template}
        for family in NEURAL
        for template in TEMPLATES
    ]


def _rows():
    return [
        {
            **candidate,
            "folds": [{"fold": f, "eer": 0.5} for f in range(5)],
            "mean_eer": 0.5,
        }
        for candidate in _candidates()
    ]


def _record(family, ids, seed, n=100):
    """Canonical effective fit_cross_day_encoder config, without importing it."""
    channels, objective = {
        "accel_supcon": ("accel", "supcon"),
        "accel_cross_day_supcon": ("accel", "cross_day_supcon"),
        "accel_gyro_supcon": ("accel_gyro", "supcon"),
        "accel_gyro_cross_day_supcon": ("accel_gyro", "cross_day_supcon"),
    }[family]
    count = 3 if channels == "accel" else 6
    p = min(8, len(ids))
    steps = max((n + p * 4 - 1) // (p * 4), (len(ids) + p - 1) // p)
    config = {
        "n_channels": count,
        "model_args": {
            "architecture": "multiscale",
            "embed_dim": 64,
            "width": 32,
            "dropout": 0.1,
            "stats_fusion": True,
            "normalization": True,
            "vector_groups": [list(range(i, i + 3)) for i in range(0, count, 3)],
            "add_magnitudes": True,
            "stats_include_mean": False,
            "norm_eps": 1e-5,
        },
        "seed": seed,
        "epochs": 40,
        "epoch_selection": "fixed_final_epoch",
        "device": "cpu",
        "precision": "float32",
        "torch_threads": 2,
        "objective": objective,
        "temperature": 0.1,
        "ce_weight": 0.25,
        "classifier": "linear_training_id_only",
        "label_smoothing": 0.0,
        "optimizer": "AdamW",
        "optimizer_args": {
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "amsgrad": False,
            "foreach": None,
            "maximize": False,
            "capturable": False,
            "differentiable": False,
            "fused": False,
        },
        "lr": 0.001,
        "weight_decay": 0.0001,
        "gradient_clip": 1.0,
        "scheduler": "none",
        "augment": {},
        "sampler": {
            "requested_identities_per_batch": 8,
            "identities_per_batch": p,
            "windows_per_identity": 4,
            "days_per_identity_per_batch": 2,
            "windows_per_selected_day": 2,
            "steps_per_epoch": steps,
            "batch_size": p * 4,
            "subject_coverage_per_epoch_guaranteed": True,
            "row_replacement": "only_if_selected_day_has_fewer_than_K_over_2_rows",
            "original_row_ids_preserved": True,
        },
    }
    return {
        "config": config,
        "training_window_count": n,
        "epochs_completed": 40,
        "selected_epoch": 40,
        "history": [{"epoch": e, "steps": steps} for e in range(1, 41)],
        "training_subject_ids": list(ids),
        "normalization_fit_calls": 1,
        "all_finite": True,
        "input_shape": [n, count, 256],
    }


def _data(ids):
    return {
        "X": np.zeros((len(ids), 6, 256), dtype=np.float32),
        "subjects": np.asarray(ids, dtype=np.int64),
        "days": np.ones(len(ids), dtype=np.int64),
        "metadata": {"members_sha256": "synthetic-members"},
    }


class StudyGuards(unittest.TestCase):
    def setUp(self):
        guard = mock.patch("builtins.__import__", side_effect=_without_training_or_data)
        guard.start()
        self.addCleanup(guard.stop)

    def test_fixed_plan_budget_cohorts_and_prior_inspection(self):
        plan = study.plan()
        self.assertEqual(plan["protocol"], "santos_multisensor_exploratory_v2")
        self.assertEqual(
            plan["reproduction"],
            "lossless validation embedding caches; model replay atol=rtol=1e-5; "
            "metrics exactly recomputed from frozen caches",
        )
        self.assertEqual(
            (study.DESCRIPTORS, study.NEURAL, study.TEMPLATES),
            (DESCRIPTORS, NEURAL, TEMPLATES),
        )
        self.assertEqual(
            (
                plan["n_candidates"],
                plan["epochs"],
                plan["seeds"],
                plan["n_fold_neural_runs"],
            ),
            (30, 40, [42, 43], 40),
        )
        self.assertIs(plan["prior_test_inspection"], True)
        self.assertEqual(plan["baseline"], "descriptor_accel_axes_auto/mean")
        self.assertEqual(
            plan["final_neural_runs"], "two only if the selected method is neural"
        )
        self.assertEqual(
            (plan["window_size"], plan["stride"], plan["sampling_rate"]),
            (256, 256, 100),
        )
        cohorts = plan["cohorts"]
        self.assertEqual(
            [len(cohorts[k]) for k in ("development", "calibration", "test")],
            [15, 5, 5],
        )
        self.assertEqual(len({s for ids in cohorts.values() for s in ids}), 25)
        held = []
        for index, fold in enumerate(plan["folds"]):
            self.assertEqual(fold["fold"], index)
            self.assertEqual(len(fold["train_subject_ids"]), 12)
            self.assertEqual(len(fold["val_subject_ids"]), 3)
            self.assertFalse(
                set(fold["train_subject_ids"]) & set(fold["val_subject_ids"])
            )
            self.assertEqual(
                set(fold["train_subject_ids"] + fold["val_subject_ids"]),
                set(cohorts["development"]),
            )
            held.extend(fold["val_subject_ids"])
        self.assertEqual(held, cohorts["development"])

    def test_exact_candidate_canonical_binding(self):
        expected = _candidates()
        self.assertEqual(study.expected_names(), [r["name"] for r in expected])
        for candidate in expected:
            with self.subTest(name=candidate["name"]):
                self.assertEqual(
                    study.canonical_candidate(candidate["name"]), candidate
                )
        for name in (
            "accel_supcon/median",
            "descriptor_accel_axes_0.2/mean",
            "unknown",
        ):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "unregistered"),
            ):
                study.canonical_candidate(name)
        self.assertFalse(any(path.startswith("docs/") for path in study.SOURCES))

    def test_exact_artifact_inventories(self):
        descriptors = {
            "descriptors/descriptor_" + "_".join(map(str, spec)) + f"_fold{f}.npz"
            for spec in DESCRIPTORS
            for f in range(5)
        }
        descriptor_caches = {
            "descriptors/descriptor_"
            + "_".join(map(str, spec))
            + f"_fold{f}_validation.npz"
            for spec in DESCRIPTORS
            for f in range(5)
        }
        self.assertEqual(
            study.expected_artifacts("descriptors"), descriptors | descriptor_caches
        )
        self.assertEqual(len(descriptors), 30)
        self.assertEqual(len(descriptor_caches), 30)
        self.assertEqual(len(study.expected_artifacts("descriptors")), 60)
        neural = set()
        for family in NEURAL:
            checkpoints = {
                f"{family}/fold{f}_seed{s}.pt" for f in range(5) for s in (42, 43)
            }
            caches = {f"{family}/fold{f}_validation.npz" for f in range(5)}
            expected = checkpoints | caches
            self.assertEqual(study.expected_artifacts(family), expected)
            self.assertEqual(
                (len(checkpoints), len(caches), len(expected)), (10, 5, 15)
            )
            neural.update(expected)
        self.assertEqual(len(neural), 60)

    def test_validation_cache_is_lossless_and_small_replay_drift_returns_original(self):
        data = object()  # No real or synthetic dataset is needed for this API.
        z = np.asarray([[0.125 + np.finfo(np.float64).eps, -0.25], [0.5, -0.75]])
        replayed = z + 1e-7
        with (
            tempfile.TemporaryDirectory() as temp,
            mock.patch.object(
                study.previous, "data_fingerprint", return_value="exact-input-hash"
            ) as fingerprint,
        ):
            path = Path(temp) / "validation.npz"
            study.cache_validation(path, z, data)
            before = path.read_bytes()
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(set(saved.files), {"embeddings", "data_fingerprint"})
                self.assertEqual(str(saved["data_fingerprint"]), "exact-input-hash")
                self.assertEqual(saved["embeddings"].dtype, z.dtype)
                np.testing.assert_array_equal(saved["embeddings"], z)
            verified = study.verified_validation(path, replayed, data)
            np.testing.assert_array_equal(verified, z)
            self.assertFalse(np.array_equal(verified, replayed))
            self.assertFalse(np.shares_memory(verified, replayed))
            np.testing.assert_array_equal(replayed, z + 1e-7)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(
                fingerprint.call_args_list, [mock.call(data), mock.call(data)]
            )

    def test_validation_cache_rejects_large_drift_shape_and_nonfinite_values(self):
        z = np.asarray([[0.1, -0.2], [0.3, -0.4]], dtype=np.float32)
        cases = {
            "large_drift": (z, z + 1e-3),
            "shape": (z, z[:, :1]),
            "broadcastable_shape": (z, z[None, :, :]),
            "cached_nan": (np.full_like(z, np.nan), z),
            "cached_inf": (np.full_like(z, np.inf), z),
            "replay_nan": (z, np.full_like(z, np.nan)),
            "replay_inf": (z, np.full_like(z, np.inf)),
        }
        with (
            tempfile.TemporaryDirectory() as temp,
            mock.patch.object(
                study.previous, "data_fingerprint", return_value="input-hash"
            ),
        ):
            for name, (cached, replayed) in cases.items():
                path = Path(temp) / f"{name}.npz"
                study.cache_validation(path, cached, None)
                before = path.read_bytes()
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(ValueError, "model replay differs"),
                ):
                    study.verified_validation(path, replayed, None)
                self.assertEqual(path.read_bytes(), before)

    def test_validation_cache_requires_exact_input_hash_even_for_identical_embeddings(
        self,
    ):
        z = np.eye(2, dtype=np.float32)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "validation.npz"
            with mock.patch.object(
                study.previous, "data_fingerprint", return_value="original"
            ):
                study.cache_validation(path, z, None)
            with (
                mock.patch.object(
                    study.previous, "data_fingerprint", return_value="changed"
                ),
                self.assertRaisesRegex(ValueError, "cached validation inputs"),
            ):
                study.verified_validation(path, z.copy(), None)

    def test_validation_cache_cannot_overwrite_frozen_embeddings(self):
        z = np.eye(2, dtype=np.float32)
        with (
            tempfile.TemporaryDirectory() as temp,
            mock.patch.object(
                study.previous, "data_fingerprint", return_value="input-hash"
            ),
        ):
            path = Path(temp) / "validation.npz"
            study.cache_validation(path, z, None)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                study.cache_validation(path, z + 1e-7, None)
            self.assertEqual(path.read_bytes(), before)

    def test_select_minimum_equal_fold_mean_and_fixed_order_ties(self):
        rows = _rows()
        self.assertEqual(
            study.select(list(reversed(rows)))["name"], study.plan()["baseline"]
        )
        # A pooled claim-count average would prefer the other candidate.
        rows[20]["folds"] = [
            {"fold": f, "eer": e, "n_claims": n}
            for f, (e, n) in enumerate(((0, 1), (0, 1), (0, 1), (0, 1), (1, 1000)))
        ]
        rows[20]["mean_eer"] = 0.2
        rows[21]["folds"] = [
            {"fold": f, "eer": 0.3, "n_claims": 1000} for f in range(5)
        ]
        rows[21]["mean_eer"] = 0.3
        self.assertIs(study.select(rows), rows[20])
        rows[23]["folds"] = copy.deepcopy(rows[20]["folds"])
        rows[23]["mean_eer"] = 0.2
        self.assertIs(study.select(list(reversed(rows))), rows[20])

    def test_select_rejects_incomplete_registry_fold_and_false_summary(self):
        for change in ("missing", "duplicate", "extra", "fold", "fold_order", "mean"):
            rows = _rows()
            if change == "missing":
                rows.pop()
            elif change == "duplicate":
                rows[-1] = copy.deepcopy(rows[0])
            elif change == "extra":
                rows.append(copy.deepcopy(rows[0]))
            elif change == "fold":
                rows[0]["folds"].pop()
            elif change == "fold_order":
                rows[0]["folds"].reverse()
            else:
                rows[0]["mean_eer"] = 0.01
            with self.subTest(change=change), self.assertRaises(ValueError):
                study.select(rows)

    def test_select_rejects_method_mislabels(self):
        for index, key, value in (
            (0, "descriptor", ["accel_gyro", "axes", "auto"]),
            (0, "family", "accel_supcon"),
            (0, "template", "multi"),
            (18, "family", "accel_cross_day_supcon"),
            (18, "template", "robust"),
        ):
            rows = _rows()
            rows[index][key] = value
            with (
                self.subTest(index=index, key=key),
                self.assertRaisesRegex(ValueError, "method identity"),
            ):
                study.select(rows)

    def test_training_records_all_families_seeds_and_sampler_sizes(self):
        for family in NEURAL:
            for seed in (42, 43):
                for ids, n in (
                    (list(range(12)), 100),
                    ([1, 2, 3], 12),
                    (list(range(15)), 16),
                ):
                    record = _record(family, ids, seed, n)
                    before = copy.deepcopy(record)
                    with self.subTest(family=family, seed=seed, n=n):
                        self.assertEqual(
                            study.validate_training_record(record, family, ids, seed),
                            record["config"]["model_args"],
                        )
                        self.assertEqual(record, before)

    def test_training_record_rejects_every_effective_config_field_and_extra_optimizer(
        self,
    ):
        ids, family = list(range(12)), "accel_gyro_cross_day_supcon"
        valid = _record(family, ids, 42)
        # Nested changes and extra fields cannot be hidden in a valid sidecar.
        paths = [(key,) for key in valid["config"] if key != "device"]
        paths += [
            (section, key)
            for section in ("model_args", "optimizer_args", "sampler")
            for key in valid["config"][section]
        ]
        paths += [("optimizer_args", "unregistered_flag"), ("unregistered_flag",)]
        for path in paths:
            record = copy.deepcopy(valid)
            mapping = record["config"]
            for key in path[:-1]:
                mapping = mapping[key]
            mapping[path[-1]] = "altered"
            with (
                self.subTest(path=path),
                self.assertRaisesRegex(ValueError, "complete checkpoint configuration"),
            ):
                study.validate_training_record(record, family, ids, 42)

    def test_training_record_rejects_history_epochs_ids_steps_shape_and_finiteness(
        self,
    ):
        ids, family = list(range(12)), "accel_supcon"
        alterations = {
            "history_short": lambda r: r["history"].pop(),
            "history_order": lambda r: r["history"].reverse(),
            "history_duplicate": lambda r: r["history"][1].update(epoch=1),
            "sampler_step": lambda r: r["history"][10].update(steps=1),
            "completed": lambda r: r.update(epochs_completed=39),
            "selected": lambda r: r.update(selected_epoch=1),
            "identity": lambda r: r["training_subject_ids"].__setitem__(0, 99),
            "normalization": lambda r: r.update(normalization_fit_calls=2),
            "finite": lambda r: r.update(all_finite=False),
            "shape": lambda r: r.update(input_shape=[100, 6, 256]),
        }
        for name, alter in alterations.items():
            record = _record(family, ids, 42)
            alter(record)
            with self.subTest(name=name), self.assertRaises(ValueError):
                study.validate_training_record(record, family, ids, 42)
        with self.assertRaises(ValueError):
            study.validate_training_record(_record(family, ids, 43), family, ids, 42)

    def _registration(self, root):
        args = SimpleNamespace(
            out_root=root, archive_path=Path("synthetic.zip"), device="cpu"
        )
        sources = mock.patch.object(
            study, "source_hashes", return_value={"source.py": "frozen"}
        )
        audit = mock.patch.object(
            study.previous, "audit_archive", return_value={"sha256": "archive"}
        )
        loader = mock.patch.object(
            study, "load_data", side_effect=lambda archive, ids: _data(ids)
        )
        for patch in (sources, audit, loader):
            patch.start()
            self.addCleanup(patch.stop)
        study.register(args)
        return args

    def test_registration_rejects_orphan_root_before_data_or_hashing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "orphan.pt").write_bytes(b"synthetic")
            args = SimpleNamespace(out_root=root, archive_path=Path("unused"))
            with (
                mock.patch.object(study, "load_data") as loader,
                mock.patch.object(study, "source_hashes") as sources,
            ):
                with self.assertRaisesRegex(ValueError, "completely empty"):
                    study.register(args)
                loader.assert_not_called()
                sources.assert_not_called()

    def test_registration_freezes_source_archive_plan_and_development_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self._registration(Path(temp))
            root, registration = study.check_registration(args)
            self.assertEqual(root, args.out_root)
            study.development(args, registration)
            for target, replacement, message in (
                ("source_hashes", {"source.py": "changed"}, "implementation"),
                ("plan", {"epochs": 1}, "plan"),
            ):
                with (
                    mock.patch.object(study, target, return_value=replacement),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    study.check_registration(args)
            with (
                mock.patch.object(
                    study.previous, "audit_archive", return_value={"sha256": "changed"}
                ),
                self.assertRaisesRegex(ValueError, "archive"),
            ):
                study.check_registration(args)
            altered = _data(study.previous.DEV_IDS)
            altered["X"][0, 3, 0] = 1
            with (
                mock.patch.object(study, "load_data", return_value=altered),
                self.assertRaisesRegex(ValueError, "DEV values"),
            ):
                study.development(args, registration)
            with self.assertRaisesRegex(ValueError, "completely empty"):
                study.register(args)

    def _results(self):
        root = Path("synthetic-root")
        values = {}
        for family in ("descriptors", *NEURAL):
            values[str(root / family / "result.json")] = {
                "family": family,
                "registration_sha256": "digest:registration.json",
                "rows": [r for r in _rows() if r["family"] == family],
                "artifacts": {
                    p: "digest:" + p for p in study.expected_artifacts(family)
                },
            }
        return root, values

    def test_verified_results_checks_every_artifact_hash_and_inventory(self):
        root, values = self._results()

        def digest(path):
            return "digest:" + str(path.relative_to(root))

        with (
            mock.patch.object(
                study.previous,
                "_read",
                side_effect=lambda p: copy.deepcopy(values[str(p)]),
            ),
            mock.patch.object(study.previous, "digest", side_effect=digest) as hasher,
        ):
            rows, hashes = study.verified_results(root)
            self.assertEqual(len(rows), 30)
            # 70 fitted artifacts, 50 validation caches, and five result files.
            self.assertEqual(len(hashes), 125)
            self.assertEqual(len({str(c.args[0]) for c in hasher.call_args_list}), 126)
            for change in (
                "registration",
                "family",
                "missing",
                "missing_validation_cache",
                "extra",
                "hash",
                "validation_cache_hash",
                "coverage",
            ):
                broken = copy.deepcopy(values)
                result = broken[str(root / "descriptors/result.json")]
                if change == "registration":
                    result["registration_sha256"] = "changed"
                elif change == "family":
                    result["family"] = "accel_supcon"
                elif change == "missing":
                    result["artifacts"].pop(next(iter(result["artifacts"])))
                elif change == "missing_validation_cache":
                    cache = next(
                        p for p in result["artifacts"] if p.endswith("_validation.npz")
                    )
                    result["artifacts"].pop(cache)
                elif change == "extra":
                    result["artifacts"]["descriptors/orphan.npz"] = "extra"
                elif change == "hash":
                    result["artifacts"][next(iter(result["artifacts"]))] = "changed"
                elif change == "validation_cache_hash":
                    cache = next(
                        p for p in result["artifacts"] if p.endswith("_validation.npz")
                    )
                    result["artifacts"][cache] = "changed"
                else:
                    result["rows"].pop()
                with (
                    mock.patch.object(
                        study.previous,
                        "_read",
                        side_effect=lambda p, fixture=broken: fixture[str(p)],
                    ),
                    self.subTest(change=change),
                    self.assertRaises(ValueError),
                ):
                    study.verified_results(root)

    def _locked(self, root):
        args = self._registration(root)
        (root / "fitted.npz").write_bytes(b"synthetic fitted artifact")
        value = {
            "artifact_hashes": {
                p: study.previous.digest(root / p)
                for p in ("registration.json", "fitted.npz")
            },
            "calibration_fingerprint": study.previous.data_fingerprint(
                _data(study.previous.CALIBRATION_IDS)
            ),
            "selected": {},
            "prior_test_inspection": True,
        }
        value["self_hash"] = study.previous._object_hash(value)
        study.write(root / "lock.json", value)
        return args

    def test_evaluate_verifies_lock_and_artifacts_before_any_cohort_load(self):
        for corrupt in ("lock", "artifact"):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                args = self._locked(root)
                if corrupt == "lock":
                    value = study.previous._read(root / "lock.json")
                    value["prior_test_inspection"] = False
                    (root / "lock.json").unlink()
                    study.write(root / "lock.json", value)
                else:
                    (root / "fitted.npz").write_bytes(b"altered")
                with mock.patch.object(study, "load_data") as loader:
                    with self.assertRaisesRegex(
                        ValueError, "study lock|locked artifact"
                    ):
                        study.evaluate(args)
                    loader.assert_not_called()
                self.assertFalse((root / "exploratory_test_attempt.json").exists())

    def test_evaluate_rejects_changed_dev_or_calibration_before_test(self):
        for changed_ids in (study.previous.DEV_IDS, study.previous.CALIBRATION_IDS):
            with self.subTest(ids=changed_ids), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                args = self._locked(root)
                seen = []

                def load(archive, ids, seen=seen, changed_ids=changed_ids):
                    seen.append(ids)
                    self.assertNotEqual(ids, study.previous.TEST_IDS)
                    data = _data(ids)
                    if ids == changed_ids:
                        data["X"][0, 0, 0] = 1
                    return data

                with (
                    mock.patch.object(study, "load_data", side_effect=load),
                    self.assertRaisesRegex(ValueError, "DEV values|CALIB values"),
                ):
                    study.evaluate(args)
                self.assertNotIn(study.previous.TEST_IDS, seen)
                self.assertFalse((root / "exploratory_test_attempt.json").exists())

    def test_evaluate_attempt_is_exclusive_before_test_and_lock_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self._locked(root)
            before = (root / "lock.json").read_bytes()
            seen = []

            def load(archive, ids):
                seen.append(ids)
                if ids == study.previous.TEST_IDS:
                    attempt = study.previous._read(
                        root / "exploratory_test_attempt.json"
                    )
                    self.assertIs(attempt["prior_test_inspection"], True)
                    self.assertEqual(
                        attempt["lock_hash"],
                        study.previous._read(root / "lock.json")["self_hash"],
                    )
                    raise RuntimeError("synthetic stop before TEST values")
                return _data(ids)

            with mock.patch.object(study, "load_data", side_effect=load):
                with self.assertRaisesRegex(RuntimeError, "synthetic stop"):
                    study.evaluate(args)
                self.assertEqual(
                    seen,
                    [
                        study.previous.DEV_IDS,
                        study.previous.CALIBRATION_IDS,
                        study.previous.TEST_IDS,
                    ],
                )
                seen.clear()
                with self.assertRaises(FileExistsError):
                    study.evaluate(args)
                self.assertNotIn(study.previous.TEST_IDS, seen)
            self.assertEqual((root / "lock.json").read_bytes(), before)

    def test_mocked_evaluate_discloses_inspection_and_requires_matched_test_graph(self):
        from engine import template_scoring

        for change in (
            None,
            "claim_subject_ids",
            "probe_subject_ids",
            "probe_trial_ids",
            "labels",
        ):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                args = self._locked(root)
                value = study.previous._read(root / "lock.json")
                value.pop("self_hash")
                value["selected"] = {
                    key: {
                        "row": {"name": name, "template": "mean"},
                        "threshold": threshold,
                    }
                    for key, name, threshold in (
                        ("baseline", "descriptor_accel_axes_auto/mean", -0.2),
                        ("winner", "accel_supcon/mean", -0.4),
                    )
                }
                value["self_hash"] = study.previous._object_hash(value)
                (root / "lock.json").unlink()
                study.write(root / "lock.json", value)
                before = (root / "lock.json").read_bytes()
                baseline = {
                    "labels": np.asarray([1, 0]),
                    "scores": np.asarray([-0.1, -0.8]),
                    "metadata": {
                        "claim_subject_ids": [26, 26],
                        "probe_subject_ids": [26, 28],
                        "probe_trial_ids": [1, 1],
                    },
                }
                winner = copy.deepcopy(baseline)
                if change == "labels":
                    winner["labels"][0] = 0
                elif change:
                    winner["metadata"][change][0] += 1
                with (
                    mock.patch.object(
                        study, "final_embeddings", return_value=np.zeros((2, 2))
                    ),
                    mock.patch.object(study, "score", side_effect=[baseline, winner]),
                    mock.patch.object(study, "metrics", return_value={"eer": 0.0}),
                    mock.patch.object(
                        template_scoring,
                        "paired_subject_diagnostics",
                        return_value={"synthetic": True},
                    ) as paired,
                ):
                    if change:
                        with self.assertRaisesRegex(
                            ValueError, "matched TEST claim graph|comparison support"
                        ):
                            study.evaluate(args)
                        paired.assert_not_called()
                        self.assertFalse(
                            (root / "exploratory_test_result.json").exists()
                        )
                    else:
                        result = study.evaluate(args)
                        self.assertIs(result["prior_test_inspection"], True)
                        self.assertIn(
                            "Exploratory reused-TEST", result["interpretation"]
                        )
                        self.assertEqual(
                            paired.call_args.args[-1], {"baseline": -0.2, "new": -0.4}
                        )
                        with np.load(
                            root / "exploratory_test_scores.npz", allow_pickle=False
                        ) as saved:
                            np.testing.assert_array_equal(
                                saved["labels"], baseline["labels"]
                            )
                        self.assertIs(
                            study.previous._read(root / "exploratory_test_result.json")[
                                "prior_test_inspection"
                            ],
                            True,
                        )
                self.assertEqual((root / "lock.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
