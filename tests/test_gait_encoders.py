"""Tiny synthetic CPU checks; no datasets, training runs, or GPU allocation."""

from __future__ import annotations

import io
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from data.gait_transforms import augment_gait
from models.gait_encoders import build_gait_encoder


class EncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_shapes_dynamic_lengths_unit_norm_and_finite_backward(self):
        for architecture in ("legacy", "multiscale", "temporal"):
            for channels in (3, 9):
                for fusion in (False,) if architecture == "legacy" else (False, True):
                    with self.subTest(
                        architecture=architecture, channels=channels, fusion=fusion
                    ):
                        model = build_gait_encoder(
                            channels,
                            {
                                "architecture": architecture,
                                "stats_fusion": fusion,
                                "width": 8,
                                "embed_dim": 11,
                                "dropout": 0,
                            },
                        )
                        train = torch.randn(4, channels, 32)
                        model.fit_normalization(train, chunk_size=2)
                        for length in (32, 47, 128):
                            x = torch.randn(2, channels, length, requires_grad=True)
                            y = model(x)
                            self.assertEqual(y.shape, (2, 11))
                            torch.testing.assert_close(
                                y.norm(dim=1), torch.ones(2), atol=1e-6, rtol=1e-5
                            )
                            (y * torch.arange(11)).sum().backward()
                            self.assertTrue(torch.isfinite(x.grad).all())
                            self.assertGreater(float(x.grad.abs().sum()), 0)
                            self.assertTrue(
                                all(
                                    torch.isfinite(p.grad).all()
                                    for p in model.parameters()
                                    if p.grad is not None
                                )
                            )

    def test_train_channel_moments_clamps_and_chunk_invariance(self):
        x = torch.randn(5, 3, 17) * 2 + 7
        x[:, 0] = 4
        a = build_gait_encoder(3, {"architecture": "multiscale"})
        b = build_gait_encoder(3, {"architecture": "multiscale"})
        a.fit_normalization(x, chunk_size=1)
        b.fit_normalization(x, chunk_size=4)
        torch.testing.assert_close(a.channel_mean, x.mean((0, 2), keepdim=True))
        torch.testing.assert_close(
            a.channel_std, x.std((0, 2), unbiased=False, keepdim=True).clamp_min(1e-5)
        )
        torch.testing.assert_close(a.channel_mean, b.channel_mean)
        torch.testing.assert_close(a.channel_std, b.channel_std)
        self.assertTrue(torch.isfinite(a(x)).all())

    def test_normalization_never_refits_in_forward_or_eval(self):
        model = build_gait_encoder(
            9, {"architecture": "temporal", "stats_fusion": True}
        )
        model.fit_normalization(torch.randn(4, 9, 32))
        state = {key: value.clone() for key, value in model.named_buffers()}
        model(torch.randn(2, 9, 32) * 10 + 90)
        model.eval()
        model(torch.randn(2, 9, 47) * 20 - 90)
        for key, value in model.named_buffers():
            torch.testing.assert_close(value, state[key])
        with self.assertRaisesRegex(RuntimeError, "train-only"):
            model.fit_normalization(torch.randn(2, 9, 32))

    def test_weights_only_checkpoint_restores_all_normalization(self):
        for architecture in ("legacy", "multiscale", "temporal"):
            args = {
                "architecture": architecture,
                "normalization": True,
                "stats_fusion": architecture != "legacy",
                "width": 8,
            }
            model = build_gait_encoder(3, args)
            model.fit_normalization(torch.randn(4, 3, 32) + 2)
            model.eval()
            stream = io.BytesIO()
            torch.save(model.state_dict(), stream)
            stream.seek(0)
            restored = build_gait_encoder(3, args)
            restored.load_state_dict(torch.load(stream, weights_only=True))
            restored.eval()
            x = torch.randn(2, 3, 41)
            torch.testing.assert_close(restored(x), model(x), atol=0, rtol=0)
            self.assertTrue(restored.normalization_fitted.item())

    def test_legacy_raw_wrapper_is_exact_and_does_not_change_class(self):
        from models.siamese import EmbeddingNet

        model = build_gait_encoder(
            3, {"architecture": "legacy", "normalization": False}
        )
        self.assertIsInstance(model.encoder, EmbeddingNet)
        model.fit_normalization(torch.randn(4, 3, 32) + 10)
        torch.testing.assert_close(
            model.channel_mean, torch.zeros_like(model.channel_mean)
        )
        torch.testing.assert_close(
            model.channel_std, torch.ones_like(model.channel_std)
        )
        model.eval()
        x = torch.randn(2, 3, 32)
        torch.testing.assert_close(model(x), model.encoder(x), atol=0, rtol=0)

    def test_statistics_dimensions_and_constant_signal_gradients(self):
        for channels, signals, dimension in ((3, 4, 42), (9, 12, 174)):
            model = build_gait_encoder(
                channels, {"architecture": "multiscale", "stats_fusion": True}
            )
            self.assertEqual(model.signal_channels, signals)
            self.assertEqual(model.stats_dim, dimension)
            model.fit_normalization(torch.ones(3, channels, 32))
            x = torch.ones(2, channels, 32, requires_grad=True)
            model(x).sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
        means = build_gait_encoder(
            3,
            {
                "architecture": "temporal",
                "stats_fusion": True,
                "stats_include_mean": True,
            },
        )
        self.assertEqual(means.stats_dim, 46)

    def test_magnitudes_use_raw_physical_axes(self):
        model = build_gait_encoder(
            3, {"architecture": "multiscale", "normalization": False}
        )
        x = torch.tensor([[[3.0, 0.0], [4.0, 0.0], [0.0, 2.0]]])
        signals = model._signals(x)
        torch.testing.assert_close(signals[:, 3], torch.tensor([[5.0, 2.0]]))
        generic = build_gait_encoder(4, {"architecture": "temporal"})
        self.assertEqual(generic.signal_channels, 4)

    def test_temporal_bins_preserve_order_and_parameter_budget(self):
        torch.manual_seed(6)
        for bins in (4, 8):
            model = build_gait_encoder(
                9,
                {
                    "architecture": "temporal",
                    "temporal_bins": bins,
                    "normalization": False,
                    "dropout": 0,
                },
            )
            model.eval()
            x = torch.randn(1, 9, 64)
            self.assertGreater(
                float((model(x) - model(x.roll(16, -1))).abs().max().detach()), 1e-4
            )
        for architecture in ("multiscale", "temporal"):
            model = build_gait_encoder(
                9, {"architecture": architecture, "stats_fusion": True}
            )
            self.assertLess(sum(p.numel() for p in model.parameters()), 500_000)

    def test_invalid_inputs_and_nonfinite_fit_rejected(self):
        model = build_gait_encoder(3, {"architecture": "temporal"})
        for x in (
            torch.ones(2, 4, 8),
            torch.ones(2, 3, 0),
            torch.ones(2, 3, 8, dtype=torch.long),
        ):
            with self.assertRaises(ValueError):
                model(x)
        with self.assertRaisesRegex(ValueError, "finite"):
            model.fit_normalization(torch.full((2, 3, 8), float("nan")))
        with self.assertRaises(ValueError):
            build_gait_encoder(9, {"architecture": "legacy", "stats_fusion": True})


class TransformTests(unittest.TestCase):
    def test_identity_is_default_and_input_never_mutates(self):
        x = torch.randn(2, 9, 16)
        before = x.clone()
        out = augment_gait(x, {})
        torch.testing.assert_close(out, before, atol=0, rtol=0)
        self.assertNotEqual(out.data_ptr(), x.data_ptr())
        augment_gait(
            x,
            {
                "jitter_std": 0.01,
                "scale_std": 0.02,
                "roll_fraction": 0.1,
                "rotation_degrees": 10,
            },
        )
        torch.testing.assert_close(x, before, atol=0, rtol=0)

    def test_shared_rotation_preserves_norms_dots_handedness_and_vector_relations(self):
        torch.manual_seed(1)
        a = torch.randn(2, 3, 12)
        b = torch.randn(2, 3, 12)
        c = a + 2 * b
        x = torch.cat((a, b, c), dim=1)
        y = augment_gait(x, {"rotation_degrees": 120}, torch.Generator().manual_seed(3))
        self.assertFalse(torch.allclose(x, y))
        for offset in (0, 3, 6):
            torch.testing.assert_close(
                y[:, offset : offset + 3].norm(dim=1),
                x[:, offset : offset + 3].norm(dim=1),
            )
        torch.testing.assert_close(
            (y[:, :3] * y[:, 3:6]).sum(1), (a * b).sum(1), atol=2e-6, rtol=1e-5
        )
        torch.testing.assert_close(y[:, 6:9], y[:, :3] + 2 * y[:, 3:6])
        basis = torch.eye(3).unsqueeze(0).repeat(1, 3, 1)
        rotated = augment_gait(basis, {"rotation_degrees": 90})
        torch.testing.assert_close(torch.linalg.det(rotated[:, :3]), torch.ones(1))

    def test_seeded_reproducible_preserves_shape_and_gradients(self):
        config = {
            "jitter_std": 0.01,
            "scale_std": 0.02,
            "roll_fraction": 0.2,
            "rotation_degrees": 10,
        }
        x = torch.randn(2, 9, 17, requires_grad=True)
        a = augment_gait(x, config, torch.Generator().manual_seed(9))
        b = augment_gait(x, config, torch.Generator().manual_seed(9))
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        a.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(augment_gait(x[0], config).shape, (9, 17))

    def test_scaling_uses_same_factor_for_every_axis_of_a_vector(self):
        x = torch.ones(3, 9, 8)
        y = augment_gait(x, {"scale_std": 0.1})
        for start in (0, 3, 6):
            torch.testing.assert_close(y[:, start], y[:, start + 1], atol=0, rtol=0)
            torch.testing.assert_close(y[:, start], y[:, start + 2], atol=0, rtol=0)
        self.assertTrue((y > 0).all())

    def test_temporal_roll_is_one_shared_shift_and_never_mixes_rows(self):
        x = torch.arange(16.0).view(1, 1, 16).expand(3, 9, -1).clone()
        x += torch.arange(3.0).view(3, 1, 1) * 100
        y = augment_gait(x, {"roll_fraction": 0.25})
        for row in range(3):
            torch.testing.assert_close(y[row].sort(-1).values, x[row].sort(-1).values)
            torch.testing.assert_close(y[row, :1].expand(9, -1), y[row])

    def test_bad_config_and_unknown_layout_rotation_rejected(self):
        x = torch.randn(2, 4, 16)
        for config in (
            {"rotation_degrees": 10},
            {"jitter_std": -1},
            {"roll_fraction": 0.8},
            {"rotation_degrees": 181},
            {"typo": 1},
        ):
            with self.assertRaises(ValueError):
                augment_gait(x, config)


if __name__ == "__main__":
    unittest.main()
