"""Leakage, selection and identity coverage tests using real small train loops."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn

from data.adapter import WindowData
from engine.train import (
    _chronological_validation,
    _filter_activities,
    _split_validation,
    _validate_windows,
    train,
)
from engine.train_siamese import _identity_subset, _IdentitySampler, train_siamese


def windows(subjects=(3, 8, 12, 20, 31), per_subject=8, channels=3):
    labels = torch.tensor([s for s in subjects for _ in range(per_subject)])
    count = len(labels)
    return WindowData(
        torch.randn(count, channels, 16, generator=torch.Generator().manual_seed(9)),
        torch.arange(count) % 2,
        labels,
        ["walk", "sit"],
        list(subjects),
        20.0,
    )


class TinyModel(nn.Module):
    def __init__(self, multitask=False):
        super().__init__()
        self.activity = nn.Linear(3, 2)
        self.subject = nn.Linear(3, 4) if multitask else None

    def forward(self, x):
        x = x.mean(-1)
        activity = self.activity(x)
        return (activity, self.subject(x)) if self.subject is not None else activity


class ValidationTests(unittest.TestCase):
    def config(self):
        return {
            "model": "cnn_baseline",
            "source": "har",
            "val_subjects": 1,
            "batch_size": 8,
            "torch_threads": 1,
            "seed": 11,
            "lr": 0.01,
        }

    def test_split_is_deterministic_and_entirely_subject_disjoint(self):
        data = windows()
        train_a, val_a, ids = _split_validation(data, 2, 8)
        train_b, val_b, repeated = _split_validation(data, 2, 8)
        self.assertEqual(ids, repeated)
        self.assertTrue(torch.equal(train_a.X, train_b.X))
        self.assertTrue(torch.equal(val_a.X, val_b.X))
        self.assertTrue(set(train_a.subject_ids).isdisjoint(val_a.subject_ids))
        self.assertEqual(len(train_a.X) + len(val_a.X), len(data.X))
        self.assertEqual(
            set(train_a.subject_ids) | set(val_a.subject_ids), set(data.subject_ids)
        )

    def test_best_checkpoint_uses_validation_even_when_later_train_loss_improves(self):
        data = windows()
        validation = [
            {"val_activity_loss": 0.1, "val_activity_acc": 0.9},
            {"val_activity_loss": 1.0, "val_activity_acc": 0.5},
        ]
        with (
            tempfile.TemporaryDirectory() as folder,
            mock.patch("engine.train.load_windows", return_value=data) as load,
            mock.patch("engine.train.build_model", return_value=TinyModel()),
            mock.patch("engine.train._activity_validation", side_effect=validation),
        ):
            metrics = train(self.config(), 2, None, folder)
            checkpoint = torch.load(os.path.join(folder, "best.pt"), weights_only=True)
        self.assertEqual(load.call_count, 1)
        self.assertEqual(load.call_args.args[1], "train")
        self.assertLess(
            metrics["per_epoch"][1]["train_loss"], metrics["per_epoch"][0]["train_loss"]
        )
        self.assertEqual(metrics["best_epoch"], 1)
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertEqual(checkpoint["metrics"]["selection_metric"], "val_activity_loss")
        self.assertEqual(checkpoint["metrics"]["protocol"], metrics["protocol"])

    def test_validation_partition_unchanged_by_training_subset(self):
        config = self.config()
        ids = []
        for subset in (None, 16):
            with (
                tempfile.TemporaryDirectory() as folder,
                mock.patch("engine.train.load_windows", return_value=windows()),
                mock.patch("engine.train.build_model", return_value=TinyModel()),
            ):
                ids.append(
                    train(config, 1, subset, folder)["protocol"]["val_subject_ids"]
                )
        self.assertEqual(ids[0], ids[1])

    def test_multitask_validation_never_scores_unseen_subject_head(self):
        config = {
            **self.config(),
            "model": "multitask",
            "model_args": {"n_subjects": 4},
        }
        with (
            tempfile.TemporaryDirectory() as folder,
            mock.patch("engine.train.load_windows", return_value=windows()),
            mock.patch(
                "engine.train.build_model", return_value=TinyModel(multitask=True)
            ),
        ):
            metrics = train(config, 1, None, folder)
        protocol = metrics["protocol"]
        self.assertEqual(
            set(protocol["subject_label_map"]),
            {str(s) for s in protocol["train_subject_ids"]},
        )
        self.assertTrue(
            set(protocol["train_subject_ids"]).isdisjoint(protocol["val_subject_ids"])
        )
        self.assertNotIn("val_subject_acc", metrics["per_epoch"][0])

    def test_missing_id_and_too_few_subjects_fail_usefully(self):
        data = windows()
        data.y_subject = None
        with self.assertRaisesRegex(ValueError, "real y_subject"):
            _split_validation(data, 1, 1)
        with self.assertRaisesRegex(ValueError, "val_subjects"):
            _split_validation(windows(subjects=(3,)), 1, 1)

    def test_chronological_purge_keeps_order_and_separates_overlap(self):
        data = windows()
        data.y_subject = None
        data.X[:, 0, 0] = torch.arange(len(data.X))
        train_data, val_data, purge = _chronological_validation(
            data, {"stride": 8, "val_fraction": 0.25}
        )
        self.assertEqual(purge, 2)
        self.assertEqual(len(train_data.X) + len(val_data.X) + purge, len(data.X))
        self.assertEqual(int(val_data.X[0, 0, 0] - train_data.X[-1, 0, 0]), purge + 1)

    def test_explicit_chronological_legacy_activity_training(self):
        data = windows()
        data.y_subject = None
        config = {**self.config(), "validation_strategy": "chronological", "stride": 8}
        with (
            tempfile.TemporaryDirectory() as folder,
            mock.patch("engine.train.load_windows", return_value=data),
            mock.patch("engine.train.build_model", return_value=TinyModel()),
        ):
            metrics = train(config, 1, None, folder)
        self.assertEqual(metrics["protocol"]["split"], "chronological")
        self.assertEqual(metrics["protocol"]["train_subject_ids"], [])
        self.assertEqual(metrics["protocol"]["purged_windows"], 2)

    def test_activity_filter_preserves_original_label_meaning(self):
        data = _filter_activities(windows(), [0])
        self.assertEqual(data.y_activity.unique().tolist(), [0])
        self.assertEqual(data.activity_names, ["walk", "sit"])
        with self.assertRaisesRegex(ValueError, "activity_ids"):
            _filter_activities(data, [8])

    def test_channel_and_nonfinite_data_are_rejected_before_training(self):
        data = windows()
        with self.assertRaisesRegex(ValueError, "n_channels"):
            _validate_windows(data, {"model_args": {"n_channels": 9}})
        data.X[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            _validate_windows(data, {})


class IdentitySamplingTests(unittest.TestCase):
    def test_triplets_are_valid_deterministic_and_cover_all_eligible_identities(self):
        data = windows()
        sampler = _IdentitySampler(data)
        triples = list(sampler.indices(200, 4, True))
        self.assertEqual(triples, list(sampler.indices(200, 4, True)))
        self.assertNotEqual(triples, list(sampler.indices(200, 5, True)))
        for anchor, positive, negative in triples:
            self.assertNotEqual(anchor, positive)
            self.assertEqual(data.y_subject[anchor], data.y_subject[positive])
            self.assertNotEqual(data.y_subject[anchor], data.y_subject[negative])
        self.assertEqual(
            {int(data.y_subject[a]) for a, _, _ in triples}, set(data.subject_ids)
        )

    def test_contrastive_has_both_pair_classes_and_correct_targets(self):
        data = windows()
        pairs = list(_IdentitySampler(data).indices(100, 7, False))
        self.assertEqual({label for _, _, label in pairs}, {0, 1})
        for a, b, label in pairs:
            self.assertNotEqual(a, b)
            self.assertEqual(int(data.y_subject[a] == data.y_subject[b]), label)

    def test_subset_preserves_real_identity_coverage_or_fails(self):
        data = windows()
        subset = _identity_subset(data, 10, 5)
        self.assertEqual(
            subset.y_subject.unique(return_counts=True)[1].tolist(), [2] * 5
        )
        self.assertTrue(torch.equal(subset.X, _identity_subset(data, 10, 5).X))
        with self.assertRaisesRegex(ValueError, "identity coverage"):
            _identity_subset(data, 9, 5)

    def test_undefined_genuine_or_impostor_coverage_fails(self):
        with self.assertRaisesRegex(ValueError, "2 subjects"):
            _IdentitySampler(windows(subjects=(7,)))
        with self.assertRaisesRegex(ValueError, "2 windows"):
            _IdentitySampler(windows(per_subject=1))

    def test_validation_requires_two_subjects_before_model_allocation(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            mock.patch("engine.train_siamese.load_windows", return_value=windows()),
            mock.patch("engine.train_siamese.SiameseNet") as model,
            self.assertRaisesRegex(ValueError, "at least 2"),
        ):
            train_siamese(
                {"source": "har", "val_subjects": 1}, 1, None, "triplet", folder
            )
        model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
