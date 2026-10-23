"""Budget, data, resume and subject isolation guards for the GPU study."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from data.adapter import WindowData
from engine.train_metric import _resolved_config
from scripts.subject_scaling_study import (
    EPOCHS,
    NEURAL,
    SEEDS,
    cohort,
    fingerprint,
    neural_fold,
    register,
    training_config,
    validate_run,
)


class StudyGuards(unittest.TestCase):
    def data(self):
        subjects = torch.arange(12).repeat_interleave(3)
        return WindowData(
            X=torch.arange(36 * 3 * 8, dtype=torch.float32).reshape(36, 3, 8),
            y_subject=subjects,
            y_activity=torch.zeros(36, dtype=torch.long),
            activity_names=["Walking"],
            subject_ids=list(range(12)),
            sampling_rate=20.0,
        )

    def test_factorial_is_complete_and_budget_fixed(self):
        self.assertEqual(
            {(head, width) for _, head, width in NEURAL},
            {(h, w) for h in ("linear", "arcface") for w in (32, 96)},
        )
        self.assertEqual((EPOCHS, SEEDS), (40, [42, 43]))

    def test_registered_data_values_and_plan_are_checked_on_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(
                out_root=Path(temp), data_root=Path("datasets"), device="cpu"
            )
            data = self.data()
            plan = register(args, "wisdm_v1", data)
            self.assertEqual(register(args, "wisdm_v1", data), plan)
            self.assertEqual(len(plan["training_configs"]), 24)
            calibration = set(plan["split"]["calibration_subject_ids"])
            for config in plan["training_configs"]:
                self.assertFalse(
                    calibration
                    & set(config["train_subject_ids"] + config["val_subject_ids"])
                )
                self.assertEqual(config["epoch_selection"], "final")
            data.X[0, 0, 0] += 1
            with self.assertRaisesRegex(ValueError, "registered"):
                register(args, "wisdm_v1", data)

    def test_cohort_order_and_fingerprint(self):
        data = self.data()
        selected = cohort(data, [7, 2])
        self.assertEqual(selected.y_subject.tolist(), [2, 2, 2, 7, 7, 7])
        self.assertNotEqual(fingerprint(data), fingerprint(selected))

    def test_actual_checkpoint_config_epoch_history_and_protocol(self):
        fold = {"train_subject_ids": [1, 2], "val_subject_ids": [3, 4]}
        expected = training_config("har", "datasets", fold, "arcface", 96, 42, "cpu")
        config = _resolved_config(expected)
        protocol = {**fold, "config": config, "test_split_loaded": False}
        completed = {
            "best_epoch": 40,
            "per_epoch": [{"epoch": i} for i in range(1, 41)],
            "protocol": protocol,
        }
        validate_run(expected, config, protocol, completed, 40)
        for alteration in (
            "seed",
            "history",
            "epoch",
            "cohort",
            "test",
            "steps",
            "softplus",
        ):
            actual, p, done, epoch = (
                copy.deepcopy(config),
                copy.deepcopy(protocol),
                copy.deepcopy(completed),
                40,
            )
            if alteration == "seed":
                actual["seed"] = 43
            elif alteration == "steps":
                actual["steps_per_epoch"] = 1
            elif alteration == "softplus":
                actual["softplus"] = True
            elif alteration == "history":
                done["per_epoch"] = done["per_epoch"][:-1]
            elif alteration == "epoch":
                epoch = 1
            elif alteration == "cohort":
                p["train_subject_ids"] = [1, 9]
                done["protocol"] = p
            else:
                p["test_split_loaded"] = True
                done["protocol"] = p
            if alteration in {"seed", "steps", "softplus"}:
                # Keep all supplied actual metadata mutually consistent. Only
                # the complete registered config can reject this replacement.
                p["config"] = actual
                done["protocol"] = p
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                validate_run(expected, actual, p, done, epoch)

    def test_cannot_re_register_orphaned_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(
                out_root=Path(temp), data_root=Path("datasets"), device="cpu"
            )
            source = args.out_root / "har"
            source.mkdir()
            (source / "best.pt").write_bytes(b"old run")
            with self.assertRaisesRegex(
                ValueError, "original pretraining registration"
            ):
                register(args, "har", self.data())

    def test_lock_cannot_start_training_missing_run(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("engine.train_metric.train_metric") as trainer,
        ):
            args = SimpleNamespace(
                out_root=Path(temp), data_root=Path("datasets"), device="cpu"
            )
            fold = {"fold": 0, "train_subject_ids": [1, 2], "val_subject_ids": [3, 4]}
            with self.assertRaisesRegex(ValueError, "every registered checkpoint"):
                neural_fold(
                    args,
                    "har",
                    self.data(),
                    "linear_w32",
                    "linear",
                    32,
                    fold,
                    42,
                    require_existing=True,
                )
            trainer.assert_not_called()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
