"""DeepConvLSTM structured pruning must select the exact intended tensors.

`models/compress.py` (B6) physically removes Conv1d channels and LSTM hidden
units. Forward-shape checks are not enough to claim that is correct: every
downstream tensor has to be sliced by the *producer's* kept-channel indices —
conv input columns, BatchNorm affine values and running statistics, LSTM gate
rows / recurrent columns / both biases, and the head Linear's input columns —
never by a same-length prefix.

These tests re-derive the top-(1-amount) L1 keep-sets independently from the
original weights and assert that each pruned tensor equals the original tensor
indexed by those sets. The channel-importance ordering is deliberately
increasing, so every keep-set is non-contiguous and a same-length *prefix* slice
(the previous implementation's `arange(len(keep))` / `[:, :n_keep]` behaviour)
cannot pass by accident.

The last test compares the whole pruned model with a *mask-equivalent* copy of
the original (dropped channels and hidden units neutralised), which only matches
if every slice, including the LSTM gate ordering, is aligned.
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
from torch import nn

from models import compress as C
from models.cnn_lstm import DeepConvLSTM

N_CHANNELS = 9
N_CLASSES = 6
T = 128
HIDDEN_DIM = 16
AMOUNT = 0.3
CONV_NAMES = ("conv.0", "conv.4", "conv.8")
CONV_BN_PAIRS = (("conv.0", "conv.1"), ("conv.4", "conv.5"), ("conv.8", "conv.9"))


def _keep_by_l1(norms: torch.Tensor, amount: float) -> torch.Tensor:
    """Top-(1-amount) indices by L1 norm, ascending — the pruning criterion."""
    n_keep = max(1, int(norms.numel() * (1.0 - amount)))
    return torch.topk(norms, n_keep).indices.sort().values


def _conv_keep(model: nn.Module, conv_name: str, amount: float) -> torch.Tensor:
    conv = dict(model.named_modules())[conv_name]
    return _keep_by_l1(conv.weight.detach().abs().sum(dim=(1, 2)), amount)


def _hidden_keep(model: nn.Module, amount: float) -> torch.Tensor:
    """Kept LSTM hidden units, re-derived from the original recurrent weights."""
    lstm = model.lstm
    hidden = lstm.hidden_size
    importance = torch.zeros(hidden)
    for layer in range(lstm.num_layers):
        weight_hh = getattr(lstm, f"weight_hh_l{layer}").detach()
        importance += weight_hh.abs().view(4, hidden, hidden).sum(dim=(0, 2))
    return _keep_by_l1(importance, amount)


def _complement(n: int, keep: torch.Tensor) -> torch.Tensor:
    kept = set(keep.tolist())
    return torch.tensor([i for i in range(n) if i not in kept], dtype=torch.long)


def _build_non_prefix_model() -> DeepConvLSTM:
    """DeepConvLSTM whose per-channel importance grows with the channel index.

    The kept channels are therefore the *last* few of every layer, so each keep
    set is non-contiguous. BatchNorm affine values and running statistics are
    made channel-specific so a prefix copy would be visible, while staying
    close enough to identity that activations survive the ReLUs (a collapsed
    all-zero signal would make the comparisons below vacuous).
    """
    torch.manual_seed(0)
    model = DeepConvLSTM(N_CHANNELS, N_CLASSES, hidden_dim=HIDDEN_DIM,
                         n_lstm_layers=2, dropout=0.3)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Conv1d):
                scale = torch.arange(1, module.out_channels + 1, dtype=torch.float32)
                module.weight.copy_(scale.view(-1, 1, 1).expand_as(module.weight) * 1e-3)
                module.bias.zero_()
            elif isinstance(module, nn.BatchNorm1d):
                idx = torch.arange(module.num_features, dtype=torch.float32)
                module.weight.copy_(1.0 + 0.01 * idx)
                module.bias.copy_(0.01 * idx)
                module.running_mean.copy_(0.001 * idx)
                module.running_var.copy_(1.0 + 0.01 * idx)
    return model


class DeepConvLSTMIndexAlignmentTests(unittest.TestCase):
    """Every pruned tensor must equal the original indexed by the keep-sets."""

    def setUp(self) -> None:
        self.model = _build_non_prefix_model()
        self.original = dict(self.model.named_modules())
        self.pruned = C.apply_pruning(self.model, AMOUNT, "structured")
        self.pruned_modules = dict(self.pruned.named_modules())
        self.keep = {name: _conv_keep(self.model, name, AMOUNT) for name in CONV_NAMES}
        self.hidden_keep = _hidden_keep(self.model, AMOUNT)

    def test_setup_uses_non_prefix_keep_sets(self):
        # A prefix keep-set would hide exactly the bug this module guards against.
        for name, keep in self.keep.items():
            with self.subTest(conv=name):
                self.assertLess(len(keep), self.original[name].out_channels)
                self.assertFalse(torch.equal(keep, torch.arange(len(keep))))
        self.assertLess(len(self.hidden_keep), HIDDEN_DIM)
        self.assertFalse(
            torch.equal(self.hidden_keep, torch.arange(len(self.hidden_keep)))
        )

    def test_conv_weights_use_exact_producer_indices(self):
        producer_keep = None
        for name in CONV_NAMES:
            keep = self.keep[name]
            original, pruned = self.original[name], self.pruned_modules[name]
            expected = original.weight.detach()[keep]
            with self.subTest(conv=name):
                self.assertEqual(pruned.out_channels, len(keep))
                if producer_keep is None:
                    self.assertEqual(pruned.in_channels, original.in_channels)
                else:
                    self.assertEqual(pruned.in_channels, len(producer_keep))
                    expected = expected[:, producer_keep]
                self.assertTrue(torch.equal(pruned.weight.detach(), expected))
                self.assertTrue(
                    torch.equal(pruned.bias.detach(), original.bias.detach()[keep])
                )
            producer_keep = keep

    def test_batchnorm_follows_its_producer_indices(self):
        for conv_name, bn_name in CONV_BN_PAIRS:
            keep = self.keep[conv_name]
            original, pruned = self.original[bn_name], self.pruned_modules[bn_name]
            with self.subTest(bn=bn_name):
                self.assertEqual(pruned.num_features, len(keep))
                for attr in ("weight", "bias", "running_mean", "running_var"):
                    self.assertTrue(
                        torch.equal(getattr(pruned, attr).detach(),
                                    getattr(original, attr).detach()[keep]),
                        msg=f"{bn_name}.{attr} is not indexed by the kept channels",
                    )

    def test_lstm_gates_recurrent_columns_and_biases_follow_hidden_keep(self):
        keep = self.hidden_keep
        original = self.original["lstm"]
        pruned = self.pruned_modules["lstm"]
        rows = torch.cat([keep + gate * original.hidden_size for gate in range(4)])
        input_keep = self.keep["conv.8"]

        self.assertEqual(pruned.hidden_size, len(keep))
        self.assertEqual(pruned.input_size, len(input_keep))
        self.assertEqual(pruned.num_layers, original.num_layers)

        for layer in range(original.num_layers):
            layer_keep = input_keep if layer == 0 else keep
            with self.subTest(layer=layer):
                self.assertTrue(torch.equal(
                    getattr(pruned, f"weight_ih_l{layer}").detach(),
                    getattr(original, f"weight_ih_l{layer}").detach()[rows][:, layer_keep],
                ))
                self.assertTrue(torch.equal(
                    getattr(pruned, f"weight_hh_l{layer}").detach(),
                    getattr(original, f"weight_hh_l{layer}").detach()[rows][:, keep],
                ))
                self.assertTrue(torch.equal(
                    getattr(pruned, f"bias_ih_l{layer}").detach(),
                    getattr(original, f"bias_ih_l{layer}").detach()[rows],
                ))
                self.assertTrue(torch.equal(
                    getattr(pruned, f"bias_hh_l{layer}").detach(),
                    getattr(original, f"bias_hh_l{layer}").detach()[rows],
                ))

    def test_head_linear_takes_exact_hidden_columns(self):
        keep = self.hidden_keep
        original = self.original["classifier.0"]
        pruned = self.pruned_modules["classifier.0"]

        self.assertEqual(pruned.in_features, len(keep))
        self.assertTrue(
            torch.equal(pruned.weight.detach(), original.weight.detach()[:, keep])
        )
        self.assertTrue(torch.equal(pruned.bias.detach(), original.bias.detach()))

        # The second head Linear consumes the first one's output, so it is untouched.
        self.assertTrue(torch.equal(
            self.pruned_modules["classifier.3"].weight.detach(),
            self.original["classifier.3"].weight.detach(),
        ))


class DeepConvLSTMMaskEquivalenceTests(unittest.TestCase):
    """The pruned model must equal the original with dropped units neutralised."""

    def test_pruned_model_matches_mask_equivalent_original(self):
        model = _build_non_prefix_model()
        pruned = C.apply_pruning(model, AMOUNT, "structured")

        keep = {name: _conv_keep(model, name, AMOUNT) for name in CONV_NAMES}
        hidden_keep = _hidden_keep(model, AMOUNT)

        masked = copy.deepcopy(model)
        masked_modules = dict(masked.named_modules())
        with torch.no_grad():
            # Dropped conv channels carry exactly zero forwards: zero their
            # weights/bias and the following BatchNorm's gamma/beta.
            for conv_name, bn_name in CONV_BN_PAIRS:
                conv, bn = masked_modules[conv_name], masked_modules[bn_name]
                drop = _complement(conv.out_channels, keep[conv_name])
                conv.weight[drop] = 0.0
                conv.bias[drop] = 0.0
                bn.weight[drop] = 0.0
                bn.bias[drop] = 0.0

            lstm = masked_modules["lstm"]
            drop_hidden = _complement(lstm.hidden_size, hidden_keep)
            for layer in range(lstm.num_layers):
                # Dropped previous hidden units must not feed the kept ones ...
                getattr(lstm, f"weight_hh_l{layer}")[:, drop_hidden] = 0.0
                if layer > 0:
                    # ... nor may their own outputs feed the next layer.
                    getattr(lstm, f"weight_ih_l{layer}")[:, drop_hidden] = 0.0
            # Dropped final hidden units must not reach the classifier.
            masked_modules["classifier.0"].weight[:, drop_hidden] = 0.0

        masked.eval()
        pruned.eval()
        x = torch.randn(4, N_CHANNELS, T, generator=torch.Generator().manual_seed(7))
        with torch.no_grad():
            pruned_features = pruned.conv(x)
            masked_features = masked.conv(x)[:, keep["conv.8"]]
            pruned_logits = pruned(x)
            masked_logits = masked(x)

        # A collapsed (all-zero) signal would make the equalities below vacuous.
        self.assertGreater(pruned_features.abs().max().item(), 1e-3)
        self.assertGreater(pruned_logits.abs().max().item(), 1e-3)

        self.assertTrue(
            torch.allclose(pruned_features, masked_features, atol=1e-5, rtol=1e-4),
            msg="pruned conv stack differs from the mask-equivalent original",
        )
        self.assertTrue(
            torch.allclose(pruned_logits, masked_logits, atol=1e-5, rtol=1e-4),
            msg="pruned logits differ from the mask-equivalent original",
        )


if __name__ == "__main__":
    unittest.main()
