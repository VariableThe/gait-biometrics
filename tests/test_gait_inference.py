"""Tiny CPU serving-equivalence checks; no data loading or normalization fitting."""

from __future__ import annotations

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from torch import nn

from models.gait_encoders import build_gait_encoder
from models.gait_inference import pack_multiscale_branches


class PackedInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_equivalence_preserves_weights_buffers_and_original(self):
        for channels, length, fusion in ((3, 32, False), (9, 47, True), (3, 128, True)):
            with self.subTest(channels=channels, length=length, fusion=fusion):
                torch.manual_seed(17)
                raw = build_gait_encoder(
                    channels,
                    {
                        "architecture": "multiscale",
                        "stats_fusion": fusion,
                        "width": 32,
                        "embed_dim": 11,
                        "dropout": 0.2,
                    },
                ).eval()
                # Stand in for saved training moments, without fitting or
                # updating learned statistics on any test input.
                raw.channel_mean.fill_(0.3)
                raw.channel_std.fill_(1.7)
                raw.magnitude_mean.fill_(0.8)
                raw.magnitude_std.fill_(1.2)
                raw.stats_mean.fill_(0.1)
                raw.stats_std.fill_(1.3)
                raw.normalization_fitted.fill_(True)
                state = {
                    name: value.clone() for name, value in raw.state_dict().items()
                }
                trainability = [p.requires_grad for p in raw.parameters()]
                packed = pack_multiscale_branches(raw)
                self.assertFalse(packed.training)
                self.assertTrue(all(not p.requires_grad for p in packed.parameters()))
                self.assertTrue(all(not m.training for m in packed.modules()))
                for name, value in raw.named_buffers():
                    copied = dict(packed.model.named_buffers())[name]
                    torch.testing.assert_close(copied, value, atol=0, rtol=0)
                    if value.numel():
                        self.assertNotEqual(copied.data_ptr(), value.data_ptr())
                for before, after in zip(raw.encoder, packed.model.encoder):
                    self.assertIsInstance(after.packed, nn.Conv1d)
                    self.assertEqual(after.packed.weight.shape, (32, 8, 31))
                    for name, value in before.bottleneck.state_dict().items():
                        torch.testing.assert_close(
                            after.bottleneck.state_dict()[name], value
                        )
                    for name, value in before.mix.state_dict().items():
                        torch.testing.assert_close(after.mix.state_dict()[name], value)
                    offset = 0
                    for branch in before.branches:
                        pad = (31 - branch.kernel_size[0]) // 2
                        weights = after.packed.weight[
                            offset : offset + branch.out_channels
                        ]
                        torch.testing.assert_close(
                            weights[..., pad : 31 - pad], branch.weight
                        )
                        if pad:
                            self.assertEqual(
                                int(torch.count_nonzero(weights[..., :pad])), 0
                            )
                            self.assertEqual(
                                int(torch.count_nonzero(weights[..., 31 - pad :])), 0
                            )
                        offset += branch.out_channels
                self.assertGreater(
                    sum(p.numel() for p in packed.parameters()),
                    sum(p.numel() for p in raw.parameters()),
                )
                with torch.inference_mode():
                    for inputs in (
                        torch.randn(2, channels, length),
                        torch.full((2, channels, length), 0.5),
                    ):
                        expected, actual = raw(inputs), packed(inputs)
                        self.assertTrue(torch.isfinite(actual).all())
                        torch.testing.assert_close(
                            actual, expected, atol=1e-5, rtol=1e-5
                        )
                        self.assertLessEqual(
                            float((actual - expected).abs().max()), 1e-5
                        )
                for name, value in raw.state_dict().items():
                    torch.testing.assert_close(value, state[name], atol=0, rtol=0)
                for name, value in packed.model.named_buffers():
                    torch.testing.assert_close(value, state[name], atol=0, rtol=0)
                self.assertEqual(
                    [p.requires_grad for p in raw.parameters()], trainability
                )
                with self.assertRaisesRegex(RuntimeError, "inference-only"):
                    packed.train()
                self.assertFalse(packed.training)
                with self.assertRaisesRegex(RuntimeError, "normalization is frozen"):
                    packed.fit_normalization(torch.zeros(1, channels, length))

    def test_rejects_training_and_other_architectures(self):
        with self.assertRaisesRegex(ValueError, "eval-only"):
            pack_multiscale_branches(
                build_gait_encoder(3, {"architecture": "multiscale"})
            )
        for architecture in ("legacy", "temporal"):
            with (
                self.subTest(architecture=architecture),
                self.assertRaisesRegex(ValueError, "multiscale GaitEncoder"),
            ):
                pack_multiscale_branches(
                    build_gait_encoder(3, {"architecture": architecture}).eval()
                )


if __name__ == "__main__":
    unittest.main()
