"""Epoch-boundary resume contracts on generated TRAIN tensors; no dataset reads."""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch

from data.adapter import WindowData
from engine import train_metric as metric


def _data():
    subjects = torch.tensor([3, 11, 29, 41, 53, 67]).repeat_interleave(4)
    windows = torch.randn(24, 9, 16, generator=torch.Generator().manual_seed(73))
    windows += subjects[:, None, None].float() / 40
    return WindowData(
        windows,
        torch.zeros(24, dtype=torch.long),
        subjects,
        ["Walking"],
        subjects.unique().tolist(),
        50.0,
    )


def _config(**extra):
    return {
        "source": "har",
        "device": "cpu",
        "torch_threads": 1,
        "epochs": 4,
        "window_size": 16,
        "steps_per_epoch": 2,
        "validation_pairs": 20,
        "identities_per_batch": 2,
        "windows_per_identity": 2,
        "train_subject_ids": [3, 11, 29],
        "val_subject_ids": [41, 53],
        "objective": "hybrid",
        "augment": {"jitter_std": 0.02},
        "model_args": {"width": 4, "embed_dim": 6, "dropout": 0.25},
        **extra,
    }


class InterruptedTraining(Exception):
    pass


class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.loader = mock.patch(
            "engine.train_metric.load_windows", side_effect=lambda *a, **k: _data()
        ).start()
        self.addCleanup(mock.patch.stopall)
        # Timings are observational, not replayable. A fixed clock lets us
        # compare the entire numeric history without discarding any fields.
        mock.patch("engine.train_metric.time.monotonic", return_value=123.0).start()
        mock.patch(
            "engine.train_metric._git_commit", return_value="synthetic-test"
        ).start()

    def run_training(self, name, config=None, **options):
        return metric.train_metric(
            config or _config(), str(self.root / name), **options
        )

    def interrupt_at(self, name, completed, config=None):
        save = metric._atomic_torch_save

        def interrupted_save(payload, path):
            save(payload, path)
            if (
                Path(path).name == "last.pt"
                and payload["completed_epochs"] == completed
            ):
                raise InterruptedTraining

        with (
            mock.patch(
                "engine.train_metric._atomic_torch_save", side_effect=interrupted_save
            ),
            self.assertRaises(InterruptedTraining),
        ):
            self.run_training(name, config, resumable=True)

    def load(self, name, artifact="last.pt"):
        return torch.load(
            self.root / name / artifact, map_location="cpu", weights_only=True
        )

    def assert_state_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            self.assertIsInstance(actual, torch.Tensor)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_state_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_state_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    def artifacts(self, name):
        return {path.name: path.read_bytes() for path in (self.root / name).iterdir()}

    def test_split_run_matches_uninterrupted_with_dropout_augmentation_and_adam(self):
        uninterrupted = self.run_training("full", resumable=True)
        self.interrupt_at("split", 2)
        saved_history = self.load("split")["history"]
        # Include ahead/partial logs and a broken best copy: last.pt alone
        # must recover committed epochs and the selected inference state.
        with (self.root / "split" / "per_epoch.jsonl").open("a") as stream:
            stream.write('{"epoch": 999}\n{"partial":')
        (self.root / "split" / "best.pt").write_bytes(b"interrupted best write")
        optimizer_step = torch.optim.AdamW.step
        steps = []

        def counted_step(optimizer, *args, **kwargs):
            steps.append(1)
            return optimizer_step(optimizer, *args, **kwargs)

        with mock.patch.object(torch.optim.AdamW, "step", counted_step):
            resumed = self.run_training("split", resume=True)
        self.assertEqual(len(steps), 4)  # Two remaining epochs, two batches each.
        self.assertEqual(resumed["per_epoch"][:2], saved_history)
        self.assertEqual(resumed["per_epoch"], uninterrupted["per_epoch"])
        self.assertEqual(resumed["selected_epoch"], uninterrupted["selected_epoch"])
        self.assertEqual(resumed["selected_val_eer"], uninterrupted["selected_val_eer"])
        left, right = self.load("split"), self.load("full")
        for key in (
            "model_state_dict",
            "classifier_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "scaler_state_dict",
            "rng_state",
            "sampler_state",
            "history",
            "completed_epochs",
            "best_epoch",
            "best_val_eer",
            "best_checkpoint",
        ):
            self.assert_state_equal(left[key], right[key])
        self.assert_state_equal(
            self.load("split", "best.pt"), self.load("full", "best.pt")
        )
        logs = [
            json.loads(line)
            for line in (self.root / "split" / "per_epoch.jsonl")
            .read_text()
            .splitlines()
        ]
        self.assertEqual(logs, resumed["per_epoch"])
        for call in self.loader.call_args_list:
            self.assertEqual(call.args[:2], ("har", "train"))

    def test_final_selection_resumes_before_any_best_exists_and_without_classifier(
        self,
    ):
        config = _config(epoch_selection="final", ce_weight=0)
        full = self.run_training("full", config, resumable=True)
        self.interrupt_at("split", 2, config)
        self.assertFalse((self.root / "split" / "best.pt").exists())
        self.assertIsNone(self.load("split")["best_checkpoint"])
        resumed = self.run_training("split", config, resume=True)
        self.assertEqual(resumed["per_epoch"], full["per_epoch"])
        self.assertEqual(resumed["selected_epoch"], 4)
        self.assert_state_equal(
            self.load("split", "best.pt"), self.load("full", "best.pt")
        )

    def test_earliest_tied_best_is_retained_across_interruption(self):
        with mock.patch(
            "engine.train_metric.verification_metrics",
            return_value={"eer": 0.2, "roc_auc": 0.8},
        ):
            self.interrupt_at("split", 2)
            selected = self.load("split")["best_checkpoint"]["model_state_dict"]
            resumed = self.run_training("split", resume=True)
        best = self.load("split", "best.pt")
        self.assertEqual(resumed["best_epoch"], 1)
        self.assertEqual(best["epoch"], 1)
        self.assertEqual(best["completed_epochs"], 4)
        self.assertEqual(best["metrics"]["per_epoch"], resumed["per_epoch"])
        self.assert_state_equal(best["model_state_dict"], selected)
        metric.load_metric_encoder(self.root / "split" / "best.pt")

    def test_completed_resume_does_no_optimization_or_validation(self):
        full = self.run_training("complete", resumable=True)
        last_before = (self.root / "complete" / "last.pt").read_bytes()
        with (
            mock.patch.object(
                torch.optim.AdamW, "step", side_effect=AssertionError("retrained")
            ),
            mock.patch(
                "engine.train_metric.verification_metrics",
                side_effect=AssertionError("revalidated"),
            ),
        ):
            resumed = self.run_training("complete", resume=True)
        self.assertEqual(resumed, full)
        self.assertEqual((self.root / "complete" / "last.pt").read_bytes(), last_before)
        self.assertEqual(
            len((self.root / "complete" / "per_epoch.jsonl").read_text().splitlines()),
            4,
        )

    def test_resume_from_initial_boundary(self):
        full = self.run_training("full", resumable=True)
        self.interrupt_at("initial", 0)
        self.assertEqual(self.load("initial")["history"], [])
        resumed = self.run_training("initial", resume=True)
        self.assertEqual(resumed["per_epoch"], full["per_epoch"])
        self.assert_state_equal(
            self.load("initial")["model_state_dict"],
            self.load("full")["model_state_dict"],
        )

    def test_uncommitted_epoch_is_replayed_from_previous_boundary(self):
        full = self.run_training("full", resumable=True)
        save = metric._atomic_torch_save

        def fail_before_commit(payload, path):
            if Path(path).name == "last.pt" and payload["completed_epochs"] == 3:
                raise InterruptedTraining
            save(payload, path)

        with (
            mock.patch(
                "engine.train_metric._atomic_torch_save", side_effect=fail_before_commit
            ),
            self.assertRaises(InterruptedTraining),
        ):
            self.run_training("split", resumable=True)
        self.assertEqual(self.load("split")["completed_epochs"], 2)
        self.assertEqual(
            len((self.root / "split" / "per_epoch.jsonl").read_text().splitlines()), 3
        )
        resumed = self.run_training("split", resume=True)
        self.assertEqual(resumed["per_epoch"], full["per_epoch"])
        self.assert_state_equal(
            self.load("split")["optimizer_state_dict"],
            self.load("full")["optimizer_state_dict"],
        )
        self.assert_state_equal(
            self.load("split")["model_state_dict"],
            self.load("full")["model_state_dict"],
        )

    def test_changed_config_or_total_budget_fails_before_data_or_artifact_writes(self):
        self.interrupt_at("split", 2)
        original = self.artifacts("split")
        for changed in (
            {"epochs": 5},
            {"epochs": 3},
            {"lr": 0.003},
            {"seed": 43},
            {"objective": "supcon"},
            {"train_subject_ids": [3, 11, 67]},
            {"data_root": "other-root"},
            {"augment": {}},
        ):
            with self.subTest(changed=changed):
                self.loader.reset_mock()
                with self.assertRaisesRegex(ValueError, "config mismatch"):
                    self.run_training("split", _config(**changed), resume=True)
                self.loader.assert_not_called()
                self.assertEqual(self.artifacts("split"), original)

    def test_changed_train_val_unused_data_order_and_interpretation_are_refused(self):
        self.interrupt_at("split", 2)
        original = self.artifacts("split")
        for change in (
            "train",
            "validation",
            "unused",
            "order",
            "sampling_rate",
            "activity_names",
        ):
            data = _data()
            if change in {"train", "validation", "unused"}:
                index = {"train": 0, "validation": 12, "unused": 20}[change]
                data.X[index, 0, 0] += 0.125
            elif change == "order":
                data.X[[0, 1]] = data.X[[1, 0]]
            elif change == "sampling_rate":
                data.sampling_rate = 25.0
            else:
                data.activity_names = ["renamed"]
            with self.subTest(change=change):
                with (
                    mock.patch("engine.train_metric.load_windows", return_value=data),
                    self.assertRaisesRegex(ValueError, "provenance mismatch"),
                ):
                    self.run_training("split", resume=True)
                self.assertEqual(self.artifacts("split"), original)

    def test_copied_output_source_change_and_legacy_checkpoint_are_refused(self):
        self.interrupt_at("split", 2)
        shutil.copytree(self.root / "split", self.root / "copy")
        with self.assertRaisesRegex(ValueError, "output directory provenance"):
            self.run_training("copy", resume=True)
        with (
            mock.patch(
                "engine.train_metric._source_fingerprints",
                return_value={"changed.py": "other"},
            ),
            self.assertRaisesRegex(ValueError, "source fingerprint"),
        ):
            self.run_training("split", resume=True)
        self.run_training("legacy")
        with self.assertRaisesRegex(ValueError, "best.pt is inference-only"):
            self.run_training("legacy", resume=True)
        shutil.copyfile(
            self.root / "legacy" / "best.pt", self.root / "legacy" / "last.pt"
        )
        with self.assertRaisesRegex(ValueError, "legacy best.pt cannot resume"):
            self.run_training("legacy", resume=True)

    def test_opt_in_refuses_artifact_overwrite_and_default_keeps_legacy_format(self):
        default = self.run_training("default")
        enabled = self.run_training("enabled", resumable=True)
        self.assertEqual(default["per_epoch"], enabled["per_epoch"])
        self.assert_state_equal(
            self.load("default", "best.pt"), self.load("enabled", "best.pt")
        )
        self.assertFalse((self.root / "default" / "last.pt").exists())
        self.assertEqual(
            self.load("default", "best.pt")["checkpoint_format"], "gait_metric_v1"
        )
        before = self.artifacts("default")
        with self.assertRaisesRegex(ValueError, "already exist"):
            self.run_training("default", resumable=True)
        self.assertEqual(self.artifacts("default"), before)
        self.interrupt_at("split", 2)
        before = self.artifacts("split")
        with self.assertRaisesRegex(ValueError, "already exist"):
            self.run_training("split")
        self.assertEqual(self.artifacts("split"), before)

    def test_rng_round_trip_includes_python_numpy_cached_gaussian_and_torch(self):
        random.seed(3)
        np.random.seed(5)
        np.random.normal()  # Populate the legacy generator's cached Gaussian.
        torch.manual_seed(7)
        state = metric._rng_state(torch.device("cpu"))
        expected = (random.random(), np.random.normal(), torch.rand(4))
        metric._restore_rng(state, torch.device("cpu"))
        actual = (random.random(), np.random.normal(), torch.rand(4))
        self.assert_state_equal(actual, expected)

    def test_cuda_rng_capture_restore_and_device_count_validation_without_gpu(self):
        cuda_states = [torch.tensor([1, 2], dtype=torch.uint8)]
        with mock.patch("torch.cuda.get_rng_state_all", return_value=cuda_states):
            state = metric._rng_state(torch.device("cuda:0"))
        self.assert_state_equal(state["torch_cuda"], cuda_states)
        with (
            mock.patch("torch.cuda.device_count", return_value=1),
            mock.patch("torch.cuda.set_rng_state_all") as restore,
        ):
            metric._restore_rng(state, torch.device("cuda:0"))
        restore.assert_called_once_with(cuda_states)
        with (
            mock.patch("torch.cuda.device_count", return_value=2),
            self.assertRaisesRegex(ValueError, "CUDA RNG device count"),
        ):
            metric._restore_rng(state, torch.device("cuda:0"))
        with self.assertRaisesRegex(ValueError, "CUDA training RNG on CPU"):
            metric._restore_rng(state, torch.device("cpu"))

    def test_changed_runtime_is_refused_without_artifact_writes(self):
        self.interrupt_at("split", 2)
        before = self.artifacts("split")
        with (
            mock.patch(
                "engine.train_metric.platform.python_version", return_value="different"
            ),
            self.assertRaisesRegex(ValueError, "runtime provenance mismatch"),
        ):
            self.run_training("split", resume=True)
        self.assertEqual(self.artifacts("split"), before)

    def test_interrupted_atomic_last_write_preserves_previous_boundary(self):
        self.interrupt_at("split", 2)
        path = self.root / "split" / "last.pt"
        previous = path.read_bytes()
        with (
            mock.patch(
                "engine.train_metric.os.replace",
                side_effect=OSError("interrupted replace"),
            ),
            self.assertRaisesRegex(OSError, "interrupted replace"),
        ):
            metric._atomic_torch_save({"incomplete": True}, path)
        self.assertEqual(path.read_bytes(), previous)
        self.assertFalse(list(path.parent.glob(".last.pt.*")))

    def test_corrupt_history_selection_and_sampler_state_are_refused(self):
        self.interrupt_at("split", 2)
        path = self.root / "split" / "last.pt"
        original = path.read_bytes()
        for field, value, message in (
            ("completed_epochs", 3, "epoch/history"),
            ("best_epoch", 99, "selection/history"),
            ("sampler_state", {}, "sampler RNG"),
        ):
            with self.subTest(field=field):
                path.write_bytes(original)
                state = self.load("split")
                state[field] = value
                torch.save(state, path)
                before = self.artifacts("split")
                with self.assertRaisesRegex(ValueError, message):
                    self.run_training("split", resume=True)
                self.assertEqual(self.artifacts("split"), before)

    def test_fractional_boolean_and_string_integer_budgets_are_rejected(self):
        for key in (
            "epochs",
            "steps_per_epoch",
            "identities_per_batch",
            "windows_per_identity",
            "torch_threads",
        ):
            for value in (1.5, True, "2"):
                with (
                    self.subTest(key=key, value=value),
                    self.assertRaisesRegex(ValueError, "integer"),
                ):
                    metric._resolved_config(_config(**{key: value}))


if __name__ == "__main__":
    unittest.main()
