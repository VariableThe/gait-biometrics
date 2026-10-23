"""Small synthetic CPU checks only; no public data, GPU, or experiment runs."""

from __future__ import annotations

import io
import json
import math
import os
import sys
import unittest
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from torch.nn import functional as F

from engine.cross_day_metric import (
    CrossDayBatchSampler,
    cross_day_pair_masks,
    cross_day_supervised_contrastive_loss,
    fit_cross_day_encoder,
)
from models.gait_encoders import GaitEncoder, build_gait_encoder
from models.metric_learning import supervised_contrastive_loss


def _synthetic(channels=3):
    subjects = torch.tensor([3] * 4 + [19] * 4)
    days = torch.tensor([1, 1, 2, 2] * 2)
    x = torch.randn(8, channels, 16, generator=torch.Generator().manual_seed(10))
    x[subjects == 19] *= 2
    return x, subjects, days


class CrossDayMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_masks_exclude_same_day_self_and_duplicate_original_rows(self):
        subjects = torch.tensor([3, 3, 3, 19, 19])
        days = torch.tensor([1, 1, 2, 1, 2])
        rows = torch.tensor([10, 10, 11, 20, 21])
        positive, negative = cross_day_pair_masks(subjects, days, rows)
        self.assertFalse(positive.diag().any())
        self.assertFalse(positive[0, 1])
        self.assertFalse(negative[0, 1])
        self.assertTrue(positive[0, 2])
        self.assertTrue(negative[0, 3])
        torch.testing.assert_close(positive, positive.T)
        self.assertFalse((positive & negative).any())
        # Even inconsistent duplicate metadata cannot turn a self view positive.
        positive, _ = cross_day_pair_masks(
            subjects, days, torch.tensor([10, 10, 10, 20, 21])
        )
        self.assertFalse(positive[:3].any())

    def test_cross_day_loss_matches_scalar_outside_log_reference(self):
        z = F.normalize(
            torch.tensor(
                [[1.0, 0], [1.0, 1], [0.0, 1], [-1.0, 0], [-1.0, -1], [0.0, -1]]
            ),
            dim=1,
        ).requires_grad_()
        subjects = torch.tensor([3, 3, 3, 19, 19, 19])
        days = torch.tensor([1, 1, 2, 1, 2, 2])
        tau = 0.7
        expected = []
        for i in range(len(z)):
            positives = [
                j
                for j in range(len(z))
                if subjects[i] == subjects[j] and days[i] != days[j]
            ]
            allowed = positives + [
                j for j in range(len(z)) if subjects[i] != subjects[j]
            ]
            expected.append(
                math.log(
                    sum(
                        math.exp(float(z[i].detach() @ z[j].detach()) / tau)
                        for j in allowed
                    )
                )
                - sum(float(z[i].detach() @ z[j].detach()) / tau for j in positives)
                / len(positives)
            )
        loss = cross_day_supervised_contrastive_loss(z, subjects, days, tau)
        self.assertAlmostEqual(float(loss.detach()), sum(expected) / len(z), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(float(z.grad.abs().sum()), 0)

    def test_no_valid_anchor_returns_finite_zero_and_baseline_agrees_when_applicable(
        self,
    ):
        z = torch.randn(4, 3, requires_grad=True)
        subjects = torch.tensor([3, 3, 19, 19])
        same_day = torch.ones(4, dtype=torch.long)
        loss = cross_day_supervised_contrastive_loss(z, subjects, same_day)
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        torch.testing.assert_close(z.grad, torch.zeros_like(z))
        days = torch.tensor([1, 2, 1, 2])
        torch.testing.assert_close(
            cross_day_supervised_contrastive_loss(z, subjects, days),
            supervised_contrastive_loss(z, subjects),
        )

    def test_sampler_balances_days_covers_subjects_and_preserves_row_identity(self):
        subjects = torch.arange(5).repeat_interleave(8)
        days = torch.tensor([1, 1, 1, 1, 2, 2, 2, 2] * 5)
        sampler = CrossDayBatchSampler(subjects, days, identities_per_batch=3, seed=7)
        batches = list(sampler)
        self.assertEqual(batches, list(sampler))
        covered = set()
        for rows in batches:
            self.assertEqual(len(rows), 12)
            self.assertEqual(len(set(rows)), 12)
            identities, counts = subjects[rows].unique(return_counts=True)
            self.assertTrue((counts == 4).all())
            covered.update(identities.tolist())
            for subject in identities:
                selected = [row for row in rows if subjects[row] == subject]
                self.assertEqual(days[selected].tolist().count(1), 2)
                self.assertEqual(days[selected].tolist().count(2), 2)
        self.assertEqual(covered, set(range(5)))
        sampler.set_epoch(1)
        self.assertNotEqual(batches, list(sampler))
        tiny_subjects = torch.tensor([3, 3, 19, 19])
        tiny_days = torch.tensor([1, 2, 1, 2])
        rows = next(iter(CrossDayBatchSampler(tiny_subjects, tiny_days)))
        self.assertEqual(set(rows), set(range(4)))
        positive, _ = cross_day_pair_masks(
            tiny_subjects[rows], tiny_days[rows], torch.tensor(rows)
        )
        self.assertTrue(positive.any(1).all())
        self.assertFalse(
            (
                positive & (torch.tensor(rows)[:, None] == torch.tensor(rows)[None, :])
            ).any()
        )

    def test_invalid_dimensions_dtypes_cohorts_and_settings_fail(self):
        x, subjects, days = _synthetic()
        base = {
            "n_channels": 3,
            "model_args": {"width": 4},
            "epochs": 1,
            "torch_threads": 1,
        }
        cases = [
            (x.long(), subjects, days, {}),
            (x[:, :2], subjects, days, {}),
            (x, subjects.float(), days, {}),
            (x, subjects, days.float(), {}),
            (x, torch.zeros_like(subjects), days, {}),
            (x, subjects, torch.ones_like(days), {}),
            (x, subjects[:-1], days, {}),
            (x, subjects, days, {"seed": 1.5}),
            (x, subjects, days, {"objective": "other"}),
            (x, subjects, days, {"model_args": {"embed_dim": 2.5}}),
            (x, subjects, days, {"model_args": {"surprise": True}}),
            (x, subjects, days, {"epochs": 0}),
        ]
        for values, labels, day_ids, overrides in cases:
            with (
                self.subTest(overrides=overrides, shape=values.shape),
                self.assertRaises((ValueError, TypeError)),
            ):
                fit_cross_day_encoder(values, labels, day_ids, **{**base, **overrides})
        bad = x.clone()
        bad[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_cross_day_encoder(bad, subjects, days, **base)

    def test_fit_is_finite_train_only_final_epoch_and_numpy_compatible(self):
        x, subjects, days = _synthetic()
        originals = [v.clone() for v in (x, subjects, days)]
        original_fit = GaitEncoder.fit_normalization
        original_adam = torch.optim.AdamW
        calls = []

        def fit_once(model, windows, **kwargs):
            calls.append("normalization")
            torch.testing.assert_close(windows, x)
            return original_fit(model, windows, **kwargs)

        def optimizer_after_fit(*args, **kwargs):
            self.assertEqual(calls, ["normalization"])
            calls.append("optimizer")
            return original_adam(*args, **kwargs)

        with (
            mock.patch.object(GaitEncoder, "fit_normalization", fit_once),
            mock.patch(
                "engine.cross_day_metric.torch.optim.AdamW",
                side_effect=optimizer_after_fit,
            ),
        ):
            model, record = fit_cross_day_encoder(
                x.numpy(),
                subjects.numpy(),
                days.numpy(),
                n_channels=3,
                model_args={"width": 4, "embed_dim": 6, "dropout": 0.0},
                epochs=2,
                torch_threads=1,
            )
        self.assertFalse(model.training)
        self.assertEqual((model.n_channels, model.embed_dim), (3, 6))
        self.assertEqual(record["epochs_completed"], 2)
        self.assertEqual(record["selected_epoch"], 2)
        self.assertEqual(record["normalization_fit_calls"], 1)
        self.assertEqual(len(record["history"]), 2)
        self.assertTrue(record["all_finite"])
        self.assertGreater(record["wall_seconds"], 0)
        self.assertEqual(record["config"]["ce_weight"], 0.25)
        self.assertEqual(record["config"]["model_args"]["vector_groups"], [[0, 1, 2]])
        json.dumps(record, allow_nan=False)
        self.assertTrue(all(np.isfinite(row["loss"]) for row in record["history"]))
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
        torch.testing.assert_close(model.channel_mean, x.mean((0, 2), keepdim=True))
        for value, original in zip((x, subjects, days), originals):
            torch.testing.assert_close(value, original, atol=0, rtol=0)
        before = {key: value.clone() for key, value in model.named_buffers()}
        with torch.no_grad():
            self.assertTrue(torch.isfinite(model(x * 10 + 100)).all())
        for key, value in model.named_buffers():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def test_reproducible_cpu_optimization_restores_rng_and_threads(self):
        x, subjects, days = _synthetic()
        kwargs = {
            "n_channels": 3,
            "model_args": {"width": 4, "embed_dim": 6, "dropout": 0.1},
            "epochs": 2,
            "torch_threads": 1,
            "seed": 7,
        }
        state = torch.random.get_rng_state().clone()
        a, record_a = fit_cross_day_encoder(x, subjects, days, **kwargs)
        torch.testing.assert_close(torch.random.get_rng_state(), state, atol=0, rtol=0)
        self.assertEqual(torch.get_num_threads(), 1)
        b, record_b = fit_cross_day_encoder(x, subjects, days, **kwargs)
        self.assertEqual(record_a["history"], record_b["history"])
        self.assertEqual(
            record_a["training_input_sha256"], record_b["training_input_sha256"]
        )
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], atol=0, rtol=0)
        initial = build_gait_encoder(3, record_a["config"]["model_args"])
        initial.fit_normalization(x)
        self.assertTrue(
            any(
                not torch.equal(value, initial.state_dict()[key])
                for key, value in a.named_parameters()
            )
        )

    def test_both_objectives_and_six_axis_stats_reload_via_factory(self):
        x, subjects, days = _synthetic(channels=6)
        configs = []
        for objective in ("cross_day_supcon", "supcon"):
            model, record = fit_cross_day_encoder(
                x.double(),
                subjects,
                days,
                n_channels=6,
                model_args={
                    "architecture": "temporal",
                    "width": 4,
                    "embed_dim": 6,
                    "dropout": 0.0,
                    "stats_fusion": True,
                    "vector_groups": [[0, 1, 2], [3, 4, 5]],
                },
                epochs=1,
                torch_threads=1,
                objective=objective,
            )
            configs.append(record["config"]["model_args"])
            self.assertEqual(model.signal_channels, 8)
            stream = io.BytesIO()
            torch.save(model.state_dict(), stream)
            stream.seek(0)
            restored = build_gait_encoder(6, record["config"]["model_args"])
            restored.load_state_dict(torch.load(stream, weights_only=True))
            restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(restored(x), model(x), atol=0, rtol=0)
            self.assertTrue(restored.normalization_fitted.item())
            self.assertFalse(hasattr(model, "classifier"))
        self.assertEqual(configs[0], configs[1])


if __name__ == "__main__":
    unittest.main()
