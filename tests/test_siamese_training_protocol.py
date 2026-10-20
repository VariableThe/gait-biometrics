"""Siamese training protocol: triplet budget and validation-EER selection.

Two config keys were added to `engine/train_siamese.py`:

  - `triplets_per_epoch` (default 1) scales the per-epoch triplet sampler so
    the historical one-dataset-pass budget is unchanged by default.
  - `val_subjects` (default 0 = disabled) carves a subject-disjoint validation
    split out of the TRAIN split and selects `best.pt` on validation EER.

Everything here uses synthetic tensors; no dataset is touched.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from data.adapter import WindowData, build_loader
from engine.train_siamese import (
    _n_triplets,
    _select_best_epoch,
    _selection,
    _split_validation,
    train_siamese,
)


def _window_data(subject_ids, per_subject: int = 4, n_channels: int = 9,
                 window: int = 16, seed: int = 0) -> WindowData:
    """Synthetic WindowData: `per_subject` windows per subject id."""
    g = torch.Generator().manual_seed(seed)
    subjects: list[int] = []
    for s in subject_ids:
        subjects.extend([s] * per_subject)
    n = len(subjects)
    return WindowData(
        X=torch.randn(n, n_channels, window, generator=g),
        y_activity=torch.arange(n, dtype=torch.long) % 6,
        y_subject=torch.tensor(subjects, dtype=torch.long),
        activity_names=[f"a{i}" for i in range(6)],
        subject_ids=sorted(set(subject_ids)),
        sampling_rate=50.0,
    )


class TripletBudgetTests(unittest.TestCase):
    def test_default_is_one_dataset_pass(self):
        self.assertEqual(_n_triplets({}, 2000), 2000)

    def test_missing_key_and_explicit_one_match(self):
        self.assertEqual(_n_triplets({}, 123), _n_triplets({"triplets_per_epoch": 1}, 123))

    def test_multiplier_scales_the_budget(self):
        self.assertEqual(_n_triplets({"triplets_per_epoch": 5}, 100), 500)


class ValidationCarveTests(unittest.TestCase):
    def _data(self) -> WindowData:
        return _window_data([3, 5, 8, 9, 12], per_subject=4, seed=1)

    def test_validation_subjects_are_subject_disjoint(self):
        data = self._data()
        train, val, val_ids = _split_validation(data, 2, seed=7)

        train_subjects = set(torch.unique(train.y_subject).tolist())
        val_subjects = set(torch.unique(val.y_subject).tolist())

        self.assertEqual(val_subjects, set(val_ids))
        self.assertTrue(train_subjects.isdisjoint(val_subjects))
        self.assertEqual(
            train_subjects | val_subjects,
            set(torch.unique(data.y_subject).tolist()),
        )

    def test_carve_is_deterministic_under_a_fixed_seed(self):
        data = self._data()
        _, val_a, ids_a = _split_validation(data, 2, seed=7)
        _, val_b, ids_b = _split_validation(data, 2, seed=7)

        self.assertEqual(ids_a, ids_b)
        self.assertTrue(torch.equal(val_a.X, val_b.X))
        self.assertTrue(torch.equal(val_a.y_subject, val_b.y_subject))

    def test_windows_partition_and_counts(self):
        data = self._data()
        train, val, val_ids = _split_validation(data, 2, seed=7)

        self.assertEqual(len(train.X) + len(val.X), len(data.X))
        self.assertEqual(len(val_ids), 2)
        self.assertEqual(len(val.X), 2 * 4)
        self.assertEqual(set(val.y_activity.tolist()).issubset(set(data.y_activity.tolist())), True)

    def test_val_data_is_build_loader_compatible(self):
        data = self._data()
        _, val, _ = _split_validation(data, 2, seed=7)
        batch = next(iter(build_loader(val, batch_size=4, shuffle=False)))
        self.assertEqual(batch[0].shape[1:], data.X.shape[1:])
        self.assertEqual(len(batch), 3)  # X, y_activity, y_subject

    def test_invalid_val_subjects_raise(self):
        data = self._data()
        with self.assertRaises(ValueError):
            _split_validation(data, 5, seed=7)  # >= number of subjects
        with self.assertRaises(ValueError):
            _split_validation(data, 0, seed=7)


class SelectionRuleTests(unittest.TestCase):
    def test_lower_eer_epoch_wins_not_the_last_epoch(self):
        records = [
            {"epoch": 1, "val_eer": 0.40, "val_roc_auc": 0.55},
            {"epoch": 2, "val_eer": 0.25, "val_roc_auc": 0.78},
            {"epoch": 3, "val_eer": 0.33, "val_roc_auc": 0.70},
        ]
        best = _select_best_epoch(records, "val_eer")
        self.assertEqual(best, 2)
        self.assertNotEqual(best, records[-1]["epoch"])

    def test_ties_resolve_to_the_earliest_epoch(self):
        records = [{"epoch": 1, "val_eer": 0.2}, {"epoch": 2, "val_eer": 0.2}]
        self.assertEqual(_select_best_epoch(records, "val_eer"), 1)

    def test_legacy_loss_metric_still_selects_the_minimum(self):
        records = [{"epoch": 1, "loss": 0.7}, {"epoch": 2, "loss": 0.9}]
        self.assertEqual(_select_best_epoch(records, "loss"), 1)

    def test_incremental_rule_matches_strict_improvement(self):
        # The trainer saves on a new running minimum; equal values keep the
        # earlier epoch. Reproduce that directly on synthetic metrics.
        records: list[dict] = []
        saved: list[int] = []
        for epoch, loss in enumerate([0.9, 0.5, 0.5, 0.3, 0.4], start=1):
            records.append({"epoch": epoch, "loss": loss})
            if _select_best_epoch(records, "loss") == epoch:
                saved.append(epoch)
        self.assertEqual(saved, [1, 2, 4])

    def test_selection_guard_defaults_to_legacy_metric(self):
        self.assertEqual(_selection(None), ("train_loss", "loss"))
        self.assertEqual(_selection(object()), ("val_eer", "val_eer"))


class TrainSiameseProtocolTests(unittest.TestCase):
    """Drive the real training loop on synthetic data (no dataset dependency)."""

    def _run(self, extra_config: dict, epochs: int = 1):
        data = _window_data([3, 5, 8, 9], per_subject=4, seed=2)
        config = {
            "source": "har",
            "seed": 1,
            "torch_threads": 1,
            "batch_size": 4,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "model_args": {"embed_dim": 16, "dropout": 0.0, "margin": 1.0},
        }
        config.update(extra_config)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("engine.train_siamese.load_windows", return_value=data):
                metrics = train_siamese(config, epochs, None, "triplet", tmp)
            checkpoint = torch.load(
                os.path.join(tmp, "best.pt"), weights_only=True
            )
        return metrics, checkpoint

    def test_default_config_keeps_legacy_keys_and_selection(self):
        metrics, checkpoint = self._run({})

        self.assertEqual(metrics["selection_metric"], "train_loss")
        self.assertNotIn("n_val_subjects", metrics)
        self.assertNotIn("n_val_windows", metrics)
        self.assertNotIn("val_window_fraction", metrics)
        self.assertEqual(set(metrics["per_epoch"][0]), {"epoch", "loss"})

        self.assertEqual(checkpoint["metrics"]["selection_metric"], "train_loss")
        self.assertEqual(set(checkpoint["metrics"]), {"loss", "selection_metric"})

    def test_val_subjects_zero_preserves_legacy_metric(self):
        metrics, _ = self._run({"val_subjects": 0})
        self.assertEqual(metrics["selection_metric"], "train_loss")
        self.assertNotIn("n_val_subjects", metrics)
        self.assertEqual(set(metrics["per_epoch"][0]), {"epoch", "loss"})

    def test_val_subjects_carve_selects_on_validation_eer(self):
        metrics, checkpoint = self._run({"val_subjects": 2})

        self.assertEqual(metrics["selection_metric"], "val_eer")
        self.assertEqual(metrics["n_val_subjects"], 2)
        self.assertEqual(metrics["n_val_windows"], 8)
        self.assertEqual(metrics["val_subject_fraction"], 0.5)
        self.assertEqual(metrics["val_window_fraction"], 0.5)

        record = metrics["per_epoch"][0]
        self.assertIn("val_eer", record)
        self.assertIn("val_roc_auc", record)
        self.assertEqual(
            metrics["best_epoch"],
            _select_best_epoch(metrics["per_epoch"], "val_eer"),
        )

        self.assertEqual(checkpoint["metrics"]["selection_metric"], "val_eer")
        self.assertIn("val_eer", checkpoint["metrics"])
        self.assertIn("val_roc_auc", checkpoint["metrics"])


if __name__ == "__main__":
    unittest.main()
