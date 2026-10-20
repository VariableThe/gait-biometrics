"""Structured pruning: DeepConvLSTM prunes, every other topology fails closed.

Paper-level context: `models/compress.py` (B6) physically removes Conv1d
channels and LSTM hidden units by propagating one channel keep-set along a
linear module order. That is implemented and index-exact only for the
DeepConvLSTM chain (B3, the model `configs/lightweight.yaml` selects). B1
(per-sensor branches), B2 (two heads), B4 (projection head), B5 (depthwise +
SE bottleneck) and B7 (patch width tied to positional embeddings) must raise
`NotImplementedError` *before* any mutation rather than return a model whose
logits are computed from misaligned weights.

Correct slicing of the DeepConvLSTM chain is asserted separately in
`tests/test_compress_deepconvlstm_alignment.py`.
"""

from __future__ import annotations

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from models import compress as C
from models.registry import build_model
from models.siamese import EmbeddingNet

N_CHANNELS = 9
N_CLASSES = 6
T = 128
BATCH = 2

SUPPORTED_CASES = {
    "cnn_lstm": {
        "n_channels": N_CHANNELS, "n_classes": N_CLASSES, "hidden_dim": 32,
        "n_lstm_layers": 2, "dropout": 0.3,
    },
}

UNSUPPORTED_CASES = {
    "cnn_baseline": {
        "n_channels": N_CHANNELS, "n_classes": N_CLASSES, "fusion": "concat",
        "dropout": 0.3,
    },
    "se_convnet": {
        "n_channels": N_CHANNELS, "n_classes": N_CLASSES, "use_depthwise": True,
        "dropout": 0.3, "reduction": 16,
    },
    "multitask": {
        "n_channels": N_CHANNELS, "n_classes": N_CLASSES, "n_subjects": 21,
        "hidden_dim": 32, "dropout": 0.3,
    },
    "gait_vit": {
        "n_channels": N_CHANNELS, "n_classes": N_CLASSES, "patch_len": 16,
        "embed_dim": 64, "depth": 2, "n_heads": 4,
    },
}


def _forward_logits(model, batch: int = BATCH) -> torch.Tensor:
    """Eval-mode forward on zeros; unwraps the multitask tuple if needed."""
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(batch, N_CHANNELS, T))
    return out[0] if isinstance(out, tuple) else out


def _snapshot(model) -> dict:
    return {k: v.clone() for k, v in model.state_dict().items()}


def _assert_same_state(test: unittest.TestCase, before: dict, model) -> None:
    after = model.state_dict()
    test.assertEqual(set(before), set(after))
    for key, value in before.items():
        test.assertTrue(
            torch.equal(value, after[key]), msg=f"parameter {key!r} changed"
        )


class StructuredPruneSupportedTests(unittest.TestCase):
    """DeepConvLSTM (B3) is the one supported topology."""

    def test_only_deepconvlstm_is_allowlisted(self):
        self.assertEqual(C.STRUCTURED_PRUNE_SUPPORTED, ("DeepConvLSTM",))

    def test_registry_model_prunes_and_keeps_output_shape(self):
        for name, args in SUPPORTED_CASES.items():
            with self.subTest(model=name):
                model = build_model(name, args, N_CHANNELS, N_CLASSES)
                params_before = C.count_parameters(model)

                pruned = C.apply_pruning(model, 0.3, "structured")

                self.assertLess(C.count_parameters(pruned), params_before)
                logits = _forward_logits(pruned)
                self.assertEqual(tuple(logits.shape), (BATCH, N_CLASSES))
                self.assertTrue(torch.isfinite(logits).all())

    def test_supported_class_name_reports_support(self):
        for name, args in SUPPORTED_CASES.items():
            with self.subTest(model=name):
                model = build_model(name, args, N_CHANNELS, N_CLASSES)
                ok, reason = C.structured_prune_support(model)
                self.assertTrue(ok)
                self.assertEqual(reason, "")

    def test_input_model_is_not_mutated(self):
        model = build_model("cnn_lstm", SUPPORTED_CASES["cnn_lstm"],
                            N_CHANNELS, N_CLASSES)
        before = _snapshot(model)

        C.apply_pruning(model, 0.3, "structured")

        _assert_same_state(self, before, model)


class StructuredPruneUnsupportedTests(unittest.TestCase):
    """B1/B2/B4/B5/B7 must raise before any mutation."""

    def test_unsupported_models_fail_early_without_mutation(self):
        for name, args in UNSUPPORTED_CASES.items():
            with self.subTest(model=name):
                model = build_model(name, args, N_CHANNELS, N_CLASSES)
                self._assert_fails_closed(model)

    def test_embeddingnet_fails_early_without_mutation(self):
        self._assert_fails_closed(EmbeddingNet(N_CHANNELS, embed_dim=64))

    def _assert_fails_closed(self, model) -> None:
        before = _snapshot(model)
        params_before = C.count_parameters(model)

        with self.assertRaises(NotImplementedError) as ctx:
            C.apply_pruning(model, 0.3, "structured")

        self.assertIn(type(model).__name__, str(ctx.exception))
        # No partial pruning: structure and values are untouched ...
        _assert_same_state(self, before, model)
        self.assertEqual(C.count_parameters(model), params_before)
        # ... and the model still runs.
        self.assertEqual(tuple(_forward_logits(model).shape[:1]), (BATCH,))

    def test_unsupported_models_report_a_concrete_reason(self):
        cases = [build_model(n, a, N_CHANNELS, N_CLASSES)
                 for n, a in UNSUPPORTED_CASES.items()]
        cases.append(EmbeddingNet(N_CHANNELS, embed_dim=64))
        for model in cases:
            with self.subTest(model=type(model).__name__):
                ok, reason = C.structured_prune_support(model)
                self.assertFalse(ok)
                self.assertTrue(reason.strip())

    def test_unknown_topology_fails_closed(self):
        class MysteryTopology(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = torch.nn.Linear(4, 2)

            def forward(self, x):  # pragma: no cover - never reached
                return self.fc(x)

        with self.assertRaises(NotImplementedError):
            C.apply_pruning(MysteryTopology(), 0.3, "structured")


class PrunableStructureGuardTests(unittest.TestCase):
    """The structural guard rejects shapes the index propagation cannot follow."""

    @staticmethod
    def _guard(model) -> None:
        C._check_prunable_structure(model)

    def test_depthwise_conv_is_rejected(self):
        class Depthwise(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv1d(8, 8, 3, padding=1, groups=8)
                self.lstm = torch.nn.LSTM(8, 8)

        with self.assertRaises(NotImplementedError):
            self._guard(Depthwise())

    def test_bidirectional_lstm_is_rejected(self):
        class Bidirectional(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv1d(8, 8, 3, padding=1)
                self.lstm = torch.nn.LSTM(8, 8, bidirectional=True)

        with self.assertRaises(NotImplementedError):
            self._guard(Bidirectional())

    def test_conv_only_stack_is_rejected(self):
        class NoLSTM(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv1d(8, 8, 3, padding=1)

        with self.assertRaises(NotImplementedError):
            self._guard(NoLSTM())


if __name__ == "__main__":
    unittest.main()
