"""Analytic/synthetic CPU checks; no datasets, training runs, or GPU use."""

from __future__ import annotations

import io
import math
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from torch.nn import functional as F

from models.angular_margin import AngularMarginClassifier


class AngularMarginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.generator = torch.Generator().manual_seed(73)

    def _random(self, *shape):
        return torch.randn(*shape, generator=self.generator)

    def _angle_head(self, **kwargs):
        head = AngularMarginClassifier(2, 3, **kwargs)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[1.0, 0], [0, 2.0], [-3.0, 0]]))
        return head

    def test_inference_is_scaled_normalized_cosine_with_no_bias(self):
        head = self._angle_head(scale=7.0)
        z = torch.tensor([[3.0, 4], [-2.0, 0], [0, -5.0]])
        expected = torch.tensor([[0.6, 0.8, -0.6], [-1, 0, 1], [0, -1, 0]]) * 7
        torch.testing.assert_close(head(z), expected)
        self.assertEqual(list(dict(head.named_parameters())), ["weight"])
        self.assertEqual(head.weight.shape, (3, 2))
        self.assertTrue(head.weight.requires_grad)

    def test_known_angles_match_independent_scalar_formula_target_only(self):
        scale, margin = 9.0, 0.3
        head = self._angle_head(scale=scale, margin=margin)
        angles = [
            0.0,
            0.4,
            math.pi / 2,
            math.pi - margin - 0.1,
            math.pi - margin + 0.1,
            math.pi,
        ]
        z = torch.tensor([[math.cos(a), math.sin(a)] for a in angles])
        labels = torch.zeros(len(angles), dtype=torch.long)
        expected = []
        for angle in angles:
            target = (
                math.cos(angle + margin)
                if angle < math.pi - margin
                else math.cos(angle) - math.sin(math.pi - margin) * margin
            )
            expected.append(
                [scale * target, scale * math.sin(angle), -scale * math.cos(angle)]
            )
        actual = head(z, labels)
        torch.testing.assert_close(actual, torch.tensor(expected), atol=2e-6, rtol=1e-6)
        torch.testing.assert_close(actual[:, 1:], head(z)[:, 1:], atol=0, rtol=0)
        # The monotonic fallback is intentionally allowed below -scale.
        self.assertLess(actual[-1, 0].item(), -scale)

    def test_nonuniform_labels_change_only_their_own_targets(self):
        head = self._angle_head()
        z = torch.tensor([[1.0, 1], [1.0, -1], [-1.0, 1], [0, 0.0]])
        labels = torch.tensor([0, 2, 1, 2])
        before, after = head(z), head(z, labels)
        mask = F.one_hot(labels, 3).bool()
        torch.testing.assert_close(before[~mask], after[~mask], atol=0, rtol=0)
        self.assertTrue((after[mask] < before[mask]).all())

    def test_threshold_equality_uses_fallback(self):
        # At m=pi/3 the fp32 threshold and this exact cosine both equal -0.5.
        head = self._angle_head(scale=1.0, margin=math.pi / 3)
        z = torch.tensor([[-0.5, math.sqrt(0.75)]])
        actual = head(z, torch.tensor([0]))[0, 0]
        expected = -0.5 - math.sin(math.pi / 3) * math.pi / 3
        self.assertAlmostEqual(actual.item(), expected, places=6)
        self.assertLess(actual.item(), -1.0)

    def test_target_is_monotonic_across_both_pieces_and_threshold_jump(self):
        margin = 0.3
        angles = sorted(
            [i * math.pi / 128 for i in range(129)]
            + [math.pi - margin - 1e-4, math.pi - margin + 1e-4]
        )
        z = torch.tensor([[math.cos(a), math.sin(a)] for a in angles])
        actual = self._angle_head(scale=1.0, margin=margin)(
            z, torch.zeros(len(z), dtype=torch.long)
        )[:, 0]
        self.assertTrue((actual[1:] <= actual[:-1] + 1e-6).all())

    def test_zero_margin_is_exactly_inference_including_endpoints_and_zeros(self):
        head = self._angle_head(margin=0)
        z = torch.tensor([[1.0, 0], [-1.0, 0], [0, 0.0], [1.0, 1]])
        labels = torch.tensor([0, 0, 1, 2])
        torch.testing.assert_close(head(z, labels), head(z), atol=0, rtol=0)

    def test_label_presence_controls_margin_in_both_train_and_eval_modes(self):
        head = self._angle_head()
        z = self._random(4, 2)
        labels = torch.tensor([0, 1, 2, 0])
        inference, training = head(z), head(z, labels)
        head.eval()
        torch.testing.assert_close(head(z), inference, atol=0, rtol=0)
        torch.testing.assert_close(head(z, labels), training, atol=0, rtol=0)
        self.assertFalse(torch.equal(inference, training))

    def test_positive_feature_and_per_class_weight_scaling_invariance(self):
        head = self._angle_head()
        z = self._random(4, 2)
        labels = torch.tensor([0, 1, 2, 1])
        before = head(z, labels)
        with torch.no_grad():
            head.weight.mul_(torch.tensor([[2.0], [0.5], [4.0]]))
        after = head(z * torch.tensor([[2.0], [4.0], [0.5], [8.0]]), labels)
        torch.testing.assert_close(after, before, atol=0, rtol=0)

    def test_cross_entropy_backpropagates_to_features_and_learned_weights(self):
        head = AngularMarginClassifier(5, 3)
        z = self._random(6, 5).requires_grad_()
        loss = F.cross_entropy(
            head(z, torch.tensor([0, 1, 2, 0, 2, 1])), torch.tensor([0, 1, 2, 0, 2, 1])
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for gradient in (z.grad, head.weight.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_large_finite_rows_keep_cosine_directions_and_finite_gradients(self):
        head = self._angle_head()
        ordinary = torch.tensor([[1.0, 0], [0.0, 1], [0.6, 0.8]])
        labels = torch.tensor([0, 1, 2])
        expected, expected_margin = head(ordinary), head(ordinary, labels)
        huge = (ordinary * 1e20).requires_grad_()
        with torch.no_grad():
            head.weight.mul_(1e20)
        logits = head(huge, labels)
        torch.testing.assert_close(head(huge), expected)
        torch.testing.assert_close(logits, expected_margin)
        F.cross_entropy(logits, labels).backward()
        self.assertTrue(torch.isfinite(huge.grad).all())
        self.assertTrue(torch.isfinite(head.weight.grad).all())
        self.assertGreater(huge.grad.abs().sum().item(), 0)
        self.assertGreater(head.weight.grad.abs().sum().item(), 0)

    def test_exact_and_near_endpoints_have_finite_forward_and_backward(self):
        for labels_present in (False, True):
            with self.subTest(labels_present=labels_present):
                head = self._angle_head()
                z = torch.tensor(
                    [
                        [1.0, 0],
                        [-1.0, 0],
                        [1.0, 1e-4],
                        [-1.0, 1e-4],
                        [1.0, 1e-3],
                        [-1.0, 1e-3],
                    ],
                    requires_grad=True,
                )
                labels = torch.tensor([0, 0, 0, 0, 0, 0])
                logits = head(z, labels if labels_present else None)
                self.assertTrue(torch.isfinite(logits).all())
                F.cross_entropy(logits, labels).backward()
                self.assertTrue(torch.isfinite(z.grad).all())
                self.assertTrue(torch.isfinite(head.weight.grad).all())
                if labels_present:
                    self.assertAlmostEqual(
                        logits[0, 0].item(),
                        head.scale * math.cos(head.margin),
                        places=5,
                    )

    def test_zero_features_zero_weight_rows_and_ties_are_finite(self):
        for zero_features, zero_weights in ((True, False), (False, True), (True, True)):
            with self.subTest(zero_features=zero_features, zero_weights=zero_weights):
                head = AngularMarginClassifier(3, 3)
                with torch.no_grad():
                    head.weight.copy_(
                        torch.zeros(3, 3)
                        if zero_weights
                        else torch.tensor([[1.0, 0, 0]]).expand(3, -1)
                    )
                z = (
                    torch.zeros(3, 3)
                    if zero_features
                    else torch.tensor([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0]])
                ).requires_grad_()
                labels = torch.arange(3)
                baseline, logits = head(z), head(z, labels)
                torch.testing.assert_close(
                    baseline[:, 0], baseline[:, 1], atol=0, rtol=0
                )
                self.assertTrue(torch.isfinite(logits).all())
                F.cross_entropy(logits, labels).backward()
                self.assertTrue(torch.isfinite(z.grad).all())
                self.assertTrue(torch.isfinite(head.weight.grad).all())

    def test_float32_output_and_finite_gradients_for_supported_float_dtypes(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                head = self._angle_head()
                z = torch.tensor(
                    [[1.0, 0.7], [-0.2, 1.0]], dtype=dtype, requires_grad=True
                )
                labels = torch.tensor([0, 1])
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    logits = head(z, labels)
                    self.assertEqual(logits.dtype, torch.float32)
                    F.cross_entropy(logits, labels).backward()
                self.assertTrue(torch.isfinite(z.grad).all())
                self.assertTrue(torch.isfinite(head.weight.grad).all())
                torch.testing.assert_close(
                    logits.detach(), head(z.detach().float(), labels), atol=0, rtol=0
                )

    def test_noncontiguous_embeddings_and_integer_label_dtypes(self):
        head = self._angle_head()
        z = self._random(2, 4).T
        self.assertFalse(z.is_contiguous())
        labels = torch.tensor([0, 1, 2, 0])
        expected = head(z.contiguous(), labels)
        for dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            with self.subTest(dtype=dtype):
                torch.testing.assert_close(head(z, labels.to(dtype)), expected)

    def test_weights_only_checkpoint_roundtrip_with_constructor_configuration(self):
        args = {"in_features": 5, "n_classes": 3, "scale": 17.0, "margin": 0.2}
        head = AngularMarginClassifier(**args)
        z = self._random(4, 5)
        labels = torch.tensor([0, 2, 1, 0])
        stream = io.BytesIO()
        torch.save({"config": args, "classifier_state_dict": head.state_dict()}, stream)
        stream.seek(0)
        checkpoint = torch.load(stream, weights_only=True, map_location="cpu")
        restored = AngularMarginClassifier(**checkpoint["config"])
        restored.load_state_dict(checkpoint["classifier_state_dict"], strict=True)
        self.assertEqual(set(checkpoint["classifier_state_dict"]), {"weight"})
        torch.testing.assert_close(restored(z), head(z), atol=0, rtol=0)
        torch.testing.assert_close(restored(z, labels), head(z, labels), atol=0, rtol=0)

    def test_invalid_constructor_dimensions(self):
        for name in ("in_features", "n_classes"):
            for invalid in (0, -1, True, 1.5, "3", None):
                with self.subTest(name=name, invalid=invalid):
                    args = {"in_features": 2, "n_classes": 3, name: invalid}
                    with self.assertRaisesRegex((TypeError, ValueError), name):
                        AngularMarginClassifier(**args)

    def test_invalid_constructor_scale_and_margin(self):
        for name, invalids in (
            ("scale", (0, -1, math.nan, math.inf, -math.inf, True, "30", None)),
            (
                "margin",
                (
                    -0.01,
                    math.pi / 2,
                    math.pi,
                    math.nan,
                    math.inf,
                    -math.inf,
                    True,
                    "0.3",
                    None,
                ),
            ),
        ):
            for invalid in invalids:
                with (
                    self.subTest(name=name, invalid=invalid),
                    self.assertRaisesRegex((TypeError, ValueError), name),
                ):
                    AngularMarginClassifier(2, 3, **{name: invalid})

    def test_invalid_embedding_types_dimensions_and_nonfinite_values(self):
        head = self._angle_head()
        for z in (
            None,
            [[1.0, 2.0]],
            torch.ones(2, 2, dtype=torch.long),
            torch.ones(2, 2, dtype=torch.bool),
            torch.ones(2, 2, dtype=torch.complex64),
            torch.ones(2),
            torch.ones(1, 2, 1),
            torch.ones(2, 3),
            torch.empty(0, 2),
            torch.tensor([[math.nan, 0]]),
            torch.tensor([[math.inf, 0]]),
            torch.tensor([[1e100, 0]], dtype=torch.float64),
        ):
            with (
                self.subTest(value=str(z)),
                self.assertRaisesRegex((TypeError, ValueError), "embeddings"),
            ):
                head(z)

    def test_invalid_labels_fail_before_computation(self):
        head = self._angle_head()
        z = torch.ones(2, 2)
        for labels in (
            [0, 1],
            torch.tensor([0.0, 1.0]),
            torch.tensor([True, False]),
            torch.tensor([0]),
            torch.tensor([[0], [1]]),
            torch.empty(0, dtype=torch.long),
            torch.tensor([-1, 0]),
            torch.tensor([0, 3]),
        ):
            with (
                self.subTest(labels=str(labels)),
                self.assertRaisesRegex((TypeError, ValueError), "labels"),
            ):
                head(z, labels)

    def test_device_mismatch_errors_without_allocating_gpu(self):
        head = self._angle_head()
        with self.assertRaisesRegex(ValueError, "labels and embeddings"):
            head(torch.ones(2, 2), torch.empty(2, dtype=torch.long, device="meta"))
        head.to("meta")
        with self.assertRaisesRegex(ValueError, "weight must share a device"):
            head(torch.ones(2, 2))


if __name__ == "__main__":
    unittest.main()
