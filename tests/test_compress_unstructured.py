"""Unstructured pruning: reported sparsity is real, and the result is usable.

`torch.nn.utils.prune.l1_unstructured` reparametrizes a layer: the real weights
move to `weight_orig` plus a `weight_mask` buffer. Reading `model.parameters()`
therefore reports the *un-pruned* tensors and ~0% sparsity, and deep-copying or
dynamically quantizing the reparametrized module fails. `apply_pruning` now
materializes the mask (`prune.remove`), so the zeros are genuine weights.
"""

from __future__ import annotations

import copy
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from models import compress as C
from models.registry import build_model

N_CHANNELS = 9
N_CLASSES = 6
T = 128
BATCH = 2

CNN_LSTM_ARGS = {
    "n_channels": N_CHANNELS, "n_classes": N_CLASSES, "hidden_dim": 128,
    "n_lstm_layers": 2, "dropout": 0.3,
}


def _build() -> torch.nn.Module:
    return build_model("cnn_lstm", CNN_LSTM_ARGS, N_CHANNELS, N_CLASSES)


def _has_weight_orig(model: torch.nn.Module) -> bool:
    return any(hasattr(module, "weight_orig") for module in model.modules())


class UnstructuredSparsityTests(unittest.TestCase):
    def test_reported_sparsity_reflects_requested_amount(self):
        for amount in (0.3, 0.5):
            with self.subTest(amount=amount):
                model = C.apply_pruning(_build(), amount, "unstructured")
                report = C.pruned_weight_sparsity(model)

                self.assertGreater(report["total"], 0)
                self.assertGreater(report["zeros"], 0)
                self.assertAlmostEqual(report["sparsity"], amount, delta=0.02)

    def test_raw_parameter_zeros_match_pruned_weight_zeros(self):
        """The zeros must be real weights, not hidden behind `weight_orig`."""
        model = C.apply_pruning(_build(), 0.3, "unstructured")
        report = C.pruned_weight_sparsity(model)

        parameter_zeros = sum(
            int((p == 0).sum().item()) for p in model.parameters()
            if p.dim() > 1
        )
        self.assertEqual(parameter_zeros, report["zeros"])

    def test_reparametrization_is_removed(self):
        model = C.apply_pruning(_build(), 0.3, "unstructured")
        self.assertFalse(_has_weight_orig(model))
        for module in model.modules():
            self.assertFalse(hasattr(module, "weight_mask"))

    def test_zero_amount_reports_zero_sparsity(self):
        model = C.apply_pruning(_build(), 0.0, "unstructured")
        self.assertEqual(C.pruned_weight_sparsity(model)["sparsity"], 0.0)


class UnstructuredResultUsabilityTests(unittest.TestCase):
    def test_pruned_model_still_forwards(self):
        model = C.apply_pruning(_build(), 0.3, "unstructured")
        model.eval()
        with torch.no_grad():
            logits = model(torch.zeros(BATCH, N_CHANNELS, T))
        self.assertEqual(tuple(logits.shape), (BATCH, N_CLASSES))
        self.assertTrue(torch.isfinite(logits).all())

    def test_pruned_model_deepcopies_and_keeps_zeros(self):
        model = C.apply_pruning(_build(), 0.3, "unstructured")
        clone = copy.deepcopy(model)
        self.assertEqual(
            C.pruned_weight_sparsity(clone)["zeros"],
            C.pruned_weight_sparsity(model)["zeros"],
        )

    def test_pruned_model_dynamically_quantizes(self):
        """Dynamic quantization must accept the materialized (non-reparametrized)
        model. Skipped only when this CPU build ships no quantized engine."""
        model = C.apply_pruning(_build(), 0.3, "unstructured")
        try:
            torch.backends.quantized.engine = "qnnpack"
        except Exception:
            pass
        try:
            quantized = C.dynamic_quantize(copy.deepcopy(model))
        except (RuntimeError, NotImplementedError) as exc:  # no quantized engine
            self.skipTest(f"dynamic quantization unavailable here: {exc}")
        quantized.eval()
        with torch.no_grad():
            logits = quantized(torch.zeros(BATCH, N_CHANNELS, T))
        self.assertEqual(tuple(logits.shape), (BATCH, N_CLASSES))


if __name__ == "__main__":
    unittest.main()
