"""Tiny analytic/synthetic CPU checks; no public datasets or GPU training."""

from __future__ import annotations

import json
import math
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from data.adapter import WindowData
from engine.train import _filter_activities, _split_validation
from engine.train_metric import _resolved_config, load_metric_encoder, train_metric
from models.metric_learning import (
    IdentityBalancedBatchSampler,
    batch_hard_triplet_loss,
    supervised_contrastive_loss,
)
from torch.nn import functional as F


class _SingleThreadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)


class LossTests(_SingleThreadTests):
    def test_batch_hard_known_euclidean_math_and_softplus(self):
        z = torch.tensor([[0.0], [2.0], [1.0], [3.0]], requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        loss = batch_hard_triplet_loss(z, labels, margin=0.2)
        self.assertAlmostEqual(loss.item(), 1.2, places=6)
        self.assertAlmostEqual(
            batch_hard_triplet_loss(z, labels, softplus=True).item(),
            F.softplus(torch.tensor(1.0)).item(),
            places=6,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_supcon_known_outside_log_average(self):
        z = torch.tensor([[1.0, 0], [1.0, 0], [0, 1.0], [0, 1.0]])
        loss = supervised_contrastive_loss(z, [0, 0, 1, 1], temperature=1)
        self.assertAlmostEqual(loss.item(), math.log(math.e + 2) - 1, places=6)

    def test_supcon_multiple_positives_matches_explicit_scalar_reference(self):
        z = F.normalize(
            torch.tensor(
                [[1.0, 0], [1.0, 1], [0.0, 1], [-1.0, 0], [-1.0, -1], [0.0, -1]]
            ),
            dim=1,
        )
        labels = [0, 0, 0, 1, 1, 1]
        temperature = 0.7
        expected = []
        for i in range(6):
            logits = [float(z[i] @ z[j]) / temperature for j in range(6) if j != i]
            positives = [
                float(z[i] @ z[j]) / temperature
                for j in range(6)
                if j != i and labels[j] == labels[i]
            ]
            expected.append(
                math.log(sum(math.exp(value) for value in logits))
                - sum(positives) / len(positives)
            )
        self.assertAlmostEqual(
            supervised_contrastive_loss(z, labels, temperature).item(),
            sum(expected) / 6,
            places=6,
        )

    def test_no_self_positives_or_duplicate_source_row_positives(self):
        for loss_function in (batch_hard_triplet_loss, supervised_contrastive_loss):
            with self.subTest(loss=loss_function.__name__):
                z = torch.randn(4, 3, requires_grad=True)
                self.assertEqual(loss_function(z, [0, 1, 2, 3]).item(), 0)
                loss = loss_function(z, [0, 0, 1, 1], sample_ids=[10, 10, 20, 20])
                self.assertEqual(loss.item(), 0)
                loss.backward()
                torch.testing.assert_close(z.grad, torch.zeros_like(z))

    def test_zero_and_random_embeddings_have_finite_loss_and_gradients(self):
        for loss_function in (batch_hard_triplet_loss, supervised_contrastive_loss):
            for all_zero in (False, True):
                with self.subTest(loss=loss_function.__name__, zero=all_zero):
                    z = (
                        torch.zeros(6, 4) if all_zero else torch.randn(6, 4)
                    ).requires_grad_()
                    loss = loss_function(z, [0, 0, 1, 1, 2, 2])
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    self.assertTrue(torch.isfinite(z.grad).all())
                    if all_zero:
                        expected = (
                            0.2
                            if loss_function is batch_hard_triplet_loss
                            else math.log(5)
                        )
                        self.assertAlmostEqual(loss.item(), expected, places=6)

    def test_invalid_margin_temperature_and_label_shapes_fail(self):
        z = torch.randn(4, 3)
        for margin in (-1, math.nan):
            with self.assertRaises(ValueError):
                batch_hard_triplet_loss(z, [0, 0, 1, 1], margin)
        for temperature in (0, -1, math.nan):
            with self.assertRaises(ValueError):
                supervised_contrastive_loss(z, [0, 0, 1, 1], temperature)
        with self.assertRaises(ValueError):
            batch_hard_triplet_loss(z, [0])


class SamplerTests(_SingleThreadTests):
    def test_pk_shape_class_coverage_unique_rows_and_determinism(self):
        labels = torch.arange(7).repeat_interleave(5)
        a = IdentityBalancedBatchSampler(labels, 3, 4, steps_per_epoch=6, seed=5)
        b = IdentityBalancedBatchSampler(labels, 3, 4, steps_per_epoch=6, seed=5)
        self.assertEqual(list(a), list(b))
        covered = set()
        for rows in a:
            self.assertEqual(len(rows), 12)
            self.assertEqual(len(set(rows)), 12)
            subjects, counts = labels[rows].unique(return_counts=True)
            self.assertEqual(len(subjects), 3)
            self.assertTrue((counts == 4).all())
            covered.update(subjects.tolist())
        self.assertEqual(covered, set(range(7)))
        a.set_epoch(1)
        self.assertNotEqual(list(a), list(b))

    def test_small_cohort_cap_singleton_report_and_replacement(self):
        labels = torch.tensor([4, 4, 9, 9, 9, 12])
        sampler = IdentityBalancedBatchSampler(labels, 8, 4, seed=3)
        self.assertEqual(sampler.p, 2)
        self.assertEqual(sampler.context["excluded_singleton_subject_ids"], [12])
        self.assertEqual(sampler.context["excluded_singleton_windows"], 1)
        self.assertEqual(
            sampler.context["negative_identity_coverage"],
            "nonsingleton_optimizer_train_identities_only",
        )
        for rows in sampler:
            self.assertNotIn(5, rows)
            for subject in (4, 9):
                selected = [i for i in rows if labels[i] == subject]
                self.assertEqual(len(selected), 4)
                self.assertEqual(
                    set(selected), set((labels == subject).nonzero().flatten().tolist())
                )

    def test_invalid_cohorts_or_pk_fail(self):
        for labels, p, k in (
            ([1, 1, 2], 2, 2),
            ([1, 2], 2, 2),
            ([1, 1, 2, 2], 1, 2),
            ([1, 1, 2, 2], 2, 1),
        ):
            with self.assertRaises(ValueError):
                IdentityBalancedBatchSampler(labels, p, k)


def _data():
    subjects = torch.tensor([2, 5, 11, 19, 23, 29, 31, 41]).repeat_interleave(3)
    x = torch.randn(24, 9, 32, generator=torch.Generator().manual_seed(15))
    x += subjects[:, None, None] / 10
    # A source identity without walking must not enter the walking split.
    subjects = torch.cat((subjects, torch.tensor([99, 99])))
    x = torch.cat((x, torch.ones(2, 9, 32) * 100))
    activities = torch.cat(
        (torch.zeros(24, dtype=torch.long), torch.ones(2, dtype=torch.long))
    )
    return WindowData(
        x,
        activities,
        subjects,
        ["Walking", "Other"],
        sorted(subjects.unique().tolist()),
        50.0,
    )


class TrainerTests(_SingleThreadTests):
    def _run(self, tmp, **extra):
        config = {
            "source": "har",
            "epochs": 1,
            "window_size": 32,
            "torch_threads": 1,
            "device": "cpu",
            "steps_per_epoch": 1,
            "validation_pairs": 20,
            "identities_per_batch": 2,
            "windows_per_identity": 2,
            "model_args": {
                "architecture": "multiscale",
                "width": 4,
                "embed_dim": 8,
                "dropout": 0,
            },
            **extra,
        }
        with mock.patch(
            "engine.train_metric.load_windows", return_value=_data()
        ) as loader:
            metrics = train_metric(config, str(tmp))
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(loader.call_args.args[:2], ("har", "train"))
        return metrics

    def test_tiny_hybrid_train_checkpoint_and_train_only_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            metrics = self._run(
                tmp, objective="hybrid", augment={"roll_fraction": 0.05}
            )
            model, protocol, config = load_metric_encoder(Path(tmp) / "best.pt")
            train, val, val_ids = _split_validation(
                _filter_activities(_data(), [0]), 4, 42
            )
            self.assertEqual(protocol["val_subject_ids"], val_ids)
            self.assertEqual(protocol["train_subject_ids"], train.subject_ids)
            self.assertNotIn(99, protocol["train_subject_ids"] + val_ids)
            self.assertFalse(set(train.subject_ids) & set(val.subject_ids))
            torch.testing.assert_close(
                model.channel_mean, train.X.mean((0, 2), keepdim=True)
            )
            torch.testing.assert_close(
                model.channel_std, train.X.std((0, 2), unbiased=False, keepdim=True)
            )
            self.assertFalse(
                torch.allclose(model.channel_mean, _data().X.mean((0, 2), keepdim=True))
            )
            original = {k: v.clone() for k, v in model.named_buffers()}
            with torch.inference_mode():
                self.assertEqual(tuple(model(val.X).shape), (len(val.X), 8))
            for key, buffer in model.named_buffers():
                torch.testing.assert_close(original[key], buffer)
            self.assertFalse(model.training)
            self.assertEqual(config["objective"], "hybrid")
            self.assertEqual(protocol["val_seed"], 42)
            self.assertFalse(protocol["test_split_loaded"])
            self.assertTrue(
                all(
                    math.isfinite(value)
                    for key, value in metrics["per_epoch"][0].items()
                    if isinstance(value, (float, int))
                )
            )
            self.assertEqual(
                len((Path(tmp) / "per_epoch.jsonl").read_text().splitlines()), 1
            )
            json.loads((Path(tmp) / "metrics.json").read_text())
            payload = torch.load(Path(tmp) / "best.pt", weights_only=True)
            self.assertEqual(set(payload["model_state_dict"]), set(model.state_dict()))

    def test_validation_cohort_pairs_fixed_across_train_seeds_and_earliest_tie(self):
        cohorts = []
        for seed in (42, 43, 44):
            with tempfile.TemporaryDirectory() as tmp:
                with mock.patch(
                    "engine.train_metric.verification_metrics",
                    return_value={"eer": 0.25, "roc_auc": 0.75},
                ) as scorer:
                    metrics = self._run(tmp, seed=seed, epochs=2, ce_weight=0)
                self.assertTrue(
                    all(call.kwargs["seed"] == 42 for call in scorer.call_args_list)
                )
                cohorts.append(metrics["protocol"]["val_subject_ids"])
                self.assertEqual(metrics["best_epoch"], 1)
                payload = torch.load(Path(tmp) / "best.pt", weights_only=True)
                self.assertEqual(payload["epoch"], 1)
                self.assertIsNone(payload["classifier_state_dict"])
        self.assertEqual(cohorts, [cohorts[0]] * 3)

    def test_supcon_can_disable_input_normalization_and_fit_stats_fusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run(
                tmp,
                objective="supcon",
                ce_weight=0,
                model_args={
                    "architecture": "temporal",
                    "normalization": False,
                    "stats_fusion": True,
                    "width": 4,
                    "embed_dim": 8,
                    "dropout": 0,
                },
            )
            model, protocol, _ = load_metric_encoder(Path(tmp) / "best.pt")
            self.assertEqual(protocol["normalization"], "none")
            torch.testing.assert_close(
                model.channel_mean, torch.zeros_like(model.channel_mean)
            )
            self.assertTrue(model.normalization_fitted.item())
            self.assertGreater(model.stats_mean.abs().sum().item(), 0)

    def test_normalization_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp)
            path = Path(tmp) / "best.pt"
            payload = torch.load(path, weights_only=True)
            payload["model_state_dict"]["channel_mean"] += 1
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_metric_encoder(path)

    def test_weights_only_rejects_unsafe_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unsafe.pt"
            torch.save({"unsafe": Path("arbitrary")}, path)
            with self.assertRaises(pickle.UnpicklingError):
                load_metric_encoder(path)

    def test_config_rejects_nonplain_or_invalid_values(self):
        for overrides in (
            {"temperature": 0},
            {"val_subjects": 1},
            {"seed": 42, "seedmodel": 43},
            {"objective": "test_selected"},
        ):
            with self.assertRaises(ValueError):
                _resolved_config({"source": "har", **overrides})
        with self.assertRaises(TypeError):
            _resolved_config({"source": "har", "data_root": Path("datasets")})


if __name__ == "__main__":
    unittest.main()
