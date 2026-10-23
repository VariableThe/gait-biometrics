"""Exact TRAIN cohorts and checkpoint/precision contracts on tiny CPU tensors."""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from data.adapter import WindowData
from engine.train import _filter_activities
from engine.train_metric import (
    _build_classifier,
    _embed,
    _resolved_config,
    _split_cohorts,
    _training_losses,
    _validate_precision,
    load_metric_encoder,
    train_metric,
)


def _data():
    subjects = torch.tensor([3, 11, 29, 41, 53, 67, 99]).repeat_interleave(3)
    x = torch.randn(21, 9, 16, generator=torch.Generator().manual_seed(4))
    x += subjects[:, None, None].float()
    activities = torch.zeros(21, dtype=torch.long)
    activities[-3:] = 1  # ID 99 is unavailable after walking is selected.
    return WindowData(
        x, activities, subjects, ["Walking", "Other"], subjects.unique().tolist(), 50.0
    )


def _config(**extra):
    return {
        "source": "har",
        "device": "cpu",
        "torch_threads": 1,
        "epochs": 2,
        "window_size": 16,
        "steps_per_epoch": 1,
        "validation_pairs": 20,
        "identities_per_batch": 2,
        "windows_per_identity": 2,
        "train_subject_ids": [29, 3, 11],
        "val_subject_ids": [53, 41],
        "model_args": {"width": 4, "embed_dim": 6, "dropout": 0},
        **extra,
    }


class CohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_explicit_ids_require_both_valid_unique_disjoint_lists(self):
        for extra in (
            {"train_subject_ids": [3, 11]},
            {"val_subject_ids": [41, 53]},
            {"train_subject_ids": None, "val_subject_ids": [41, 53]},
            {"train_subject_ids": [], "val_subject_ids": [41, 53]},
            {"train_subject_ids": [3], "val_subject_ids": [41, 53]},
            {"train_subject_ids": [3, 3], "val_subject_ids": [41, 53]},
            {"train_subject_ids": [3, 11], "val_subject_ids": [11, 53]},
            {"train_subject_ids": [True, 11], "val_subject_ids": [41, 53]},
            {"train_subject_ids": [3.5, 11], "val_subject_ids": [41, 53]},
            {"train_subject_ids": ["3", 11], "val_subject_ids": [41, 53]},
            {"train_subject_ids": [-3, 11], "val_subject_ids": [41, 53]},
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                _resolved_config({"source": "har", **extra})

    def test_membership_is_exact_preserves_rows_and_excludes_unused_ids(self):
        config = _resolved_config(
            _config(val_subjects=1)
        )  # Explicit lists take precedence.
        walking = _filter_activities(_data(), [0])
        train, val, val_ids = _split_cohorts(walking, config)
        self.assertEqual(config["train_subject_ids"], [3, 11, 29])
        self.assertEqual(train.subject_ids, [3, 11, 29])
        self.assertEqual(val_ids, [41, 53])
        torch.testing.assert_close(train.X, walking.X[:9], atol=0, rtol=0)
        torch.testing.assert_close(val.X, walking.X[9:15], atol=0, rtol=0)
        self.assertNotIn(67, train.subject_ids + val.subject_ids)

    def test_missing_or_nonwalking_requested_identity_fails_before_model(self):
        for key, ids in (
            ("train_subject_ids", [3, 99]),
            ("val_subject_ids", [41, 100]),
        ):
            with (
                self.subTest(key=key),
                tempfile.TemporaryDirectory() as folder,
                mock.patch(
                    "engine.train_metric.load_windows", return_value=_data()
                ) as loader,
                mock.patch("engine.train_metric.build_gait_encoder") as factory,
                self.assertRaisesRegex(
                    ValueError, "unavailable after activity filtering"
                ),
            ):
                train_metric(_config(**{key: ids}), folder)
            factory.assert_not_called()
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(loader.call_args.args[:2], ("har", "train"))

    def test_explicit_cohorts_require_real_subject_labels(self):
        data = _data()
        data.y_subject = None
        with self.assertRaisesRegex(ValueError, "real y_subject"):
            _split_cohorts(data, _resolved_config(_config()))

    def _run(self, folder, **extra):
        with mock.patch(
            "engine.train_metric.load_windows", return_value=_data()
        ) as loader:
            metrics = train_metric(_config(**extra), folder)
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(loader.call_args.args[:2], ("har", "train"))
        return metrics

    def test_final_checkpoint_ignores_better_prior_eer_and_keeps_all_history(self):
        states = []

        def capture(encoder, *args):
            states.append(
                {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
            )
            return _embed(encoder, *args)

        with tempfile.TemporaryDirectory() as folder:
            with (
                mock.patch("engine.train_metric._embed", side_effect=capture),
                mock.patch(
                    "engine.train_metric.verification_metrics",
                    side_effect=[
                        {"eer": 0.1, "roc_auc": 0.9},
                        {"eer": 0.4, "roc_auc": 0.6},
                    ],
                ),
            ):
                metrics = self._run(folder, epoch_selection="final", objective="hybrid")
            payload = torch.load(Path(folder) / "best.pt", weights_only=True)
            model, protocol, config = load_metric_encoder(Path(folder) / "best.pt")
            self.assertEqual(payload["epoch"], 2)
            self.assertEqual(metrics["selected_epoch"], 2)
            self.assertEqual(metrics["selected_val_eer"], 0.4)
            self.assertEqual(metrics["observed_min_val_eer"], 0.1)
            self.assertEqual(metrics["selection_metric"], "final_epoch")
            self.assertFalse(protocol["validation_used_for_selection"])
            self.assertEqual(payload["completed_epochs"], 2)
            self.assertEqual(protocol["completed_epochs"], 2)
            self.assertEqual(payload["metrics"]["per_epoch"], metrics["per_epoch"])
            self.assertEqual(protocol, metrics["protocol"])
            self.assertEqual(protocol["unused_subject_ids"], [67])
            self.assertEqual(protocol["train_subject_ids"], [3, 11, 29])
            self.assertEqual(protocol["val_subject_ids"], [41, 53])
            self.assertEqual(protocol["classifier_subject_ids"], [3, 11, 29])
            self.assertFalse(protocol["test_split_loaded"])
            self.assertEqual(config["precision"], "float32")
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, states[-1][key], atol=0, rtol=0)
            torch.testing.assert_close(
                model.channel_mean, _data().X[:9].mean((0, 2), keepdim=True)
            )
            self.assertEqual(
                len((Path(folder) / "per_epoch.jsonl").read_text().splitlines()), 2
            )

    def test_default_earliest_tie_retains_selected_state_and_complete_history(self):
        states = []

        def capture(encoder, *args):
            states.append(
                {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
            )
            return _embed(encoder, *args)

        with tempfile.TemporaryDirectory() as folder:
            with (
                mock.patch("engine.train_metric._embed", side_effect=capture),
                mock.patch(
                    "engine.train_metric.verification_metrics",
                    return_value={"eer": 0.2, "roc_auc": 0.8},
                ),
            ):
                metrics = self._run(folder)
            payload = torch.load(Path(folder) / "best.pt", weights_only=True)
        self.assertEqual(metrics["selection_metric"], "val_eer")
        self.assertEqual(payload["epoch"], 1)
        self.assertEqual(payload["completed_epochs"], 2)
        self.assertEqual(len(payload["metrics"]["per_epoch"]), 2)
        self.assertEqual(metrics["protocol"]["selection_ties"], "earliest_epoch")
        self.assertTrue(metrics["protocol"]["validation_used_for_selection"])
        for key, value in payload["model_state_dict"].items():
            torch.testing.assert_close(value, states[0][key], atol=0, rtol=0)

    def test_arcface_only_receives_remapped_train_labels_and_is_optimized(self):
        from models.angular_margin import AngularMarginClassifier

        heads, targets, initial_weights = [], [], []

        class TrackingHead(AngularMarginClassifier):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                heads.append(self)
                initial_weights.append(self.weight.detach().clone())

            def forward(self, z, labels=None):
                self_training = self.training
                targets.append((labels.detach().clone(), self_training))
                return super().forward(z, labels)

        with tempfile.TemporaryDirectory() as folder:
            with mock.patch(
                "models.angular_margin.AngularMarginClassifier", TrackingHead
            ):
                metrics = self._run(
                    folder, classifier_type="arcface", objective="hybrid"
                )
            self.assertEqual(
                len(targets), 2
            )  # One training call per epoch; no validation CE.
            for labels, training in targets:
                self.assertTrue(training)
                self.assertTrue(set(labels.tolist()) <= {0, 1, 2})
            self.assertFalse(torch.equal(initial_weights[0], heads[0].weight.detach()))
            with mock.patch(
                "engine.train_metric._build_classifier",
                side_effect=AssertionError("inference head constructed"),
            ):
                _, protocol, _ = load_metric_encoder(Path(folder) / "best.pt")
            self.assertEqual(
                protocol["classifier_config"]["classifier_type"], "arcface"
            )
            self.assertEqual(metrics["classifier_config"]["n_classes"], 3)

    def test_zero_ce_does_not_construct_arcface(self):
        with mock.patch("models.angular_margin.AngularMarginClassifier") as factory:
            self.assertIsNone(
                _build_classifier(
                    _resolved_config(_config(ce_weight=0, classifier_type="arcface")), 3
                )
            )
        factory.assert_not_called()

    def test_new_config_defaults_and_invalid_settings(self):
        config = _resolved_config({"source": "har"})
        self.assertEqual(config["epoch_selection"], "validation_eer")
        self.assertEqual(config["classifier_type"], "linear")
        self.assertEqual(config["precision"], "float32")
        for extra in (
            {"classifier_type": "cosface"},
            {"epoch_selection": "test_eer"},
            {"precision": "float16"},
            {"angular_scale": 0},
            {"angular_margin": -0.1},
            {"angular_margin": math.pi / 2},
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                _resolved_config(_config(**extra))

    def test_bfloat16_requires_supported_requested_cuda_device(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            mock.patch("engine.train_metric.load_windows") as loader,
            self.assertRaisesRegex(ValueError, "requires CUDA"),
        ):
            train_metric(_config(precision="bfloat16"), folder)
        loader.assert_not_called()
        config = _resolved_config(_config(precision="bfloat16"))
        for supported in (False, True):
            with (
                mock.patch("torch.cuda.device", return_value=nullcontext()) as context,
                mock.patch("torch.cuda.is_bf16_supported", return_value=supported),
            ):
                if supported:
                    _validate_precision(config, torch.device("cuda:3"))
                else:
                    with self.assertRaisesRegex(ValueError, "BF16-supported"):
                        _validate_precision(config, torch.device("cuda:3"))
            context.assert_called_once_with(torch.device("cuda:3"))

    def test_bfloat16_embeddings_losses_and_evaluation_are_float32(self):
        config = _resolved_config(_config(objective="hybrid"))
        labels = torch.tensor([0, 0, 1, 1])
        z = torch.randn(4, 6, dtype=torch.bfloat16, requires_grad=True)
        classifier = torch.nn.Linear(6, 2)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            losses = _training_losses(
                z, labels, torch.arange(4), classifier, labels, config
            )
        self.assertTrue(all(loss.dtype == torch.float32 for loss in losses))
        losses[-1].backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        encoder = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(9 * 16, 6))
        x = _data().X[:4]
        expected = _embed(encoder, x, 2, torch.device("cpu"))
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = _embed(encoder, x, 2, torch.device("cpu"))
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
