"""`scripts/export_onnx.py` reads `prune`/`quantize` from the config as defaults.

`configs/lightweight.yaml` (paper B6) ships `prune: 0.3` and `quantize: true`.
Those values must be the script's defaults, with an explicit CLI flag winning.
No dataset or ONNX export is touched here.
"""

from __future__ import annotations

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import yaml

from scripts.export_onnx import parse_args, resolve_compression_options

LIGHTWEIGHT_CONFIG = os.path.join(REPO_ROOT, "configs", "lightweight.yaml")


def _load_lightweight() -> dict:
    with open(LIGHTWEIGHT_CONFIG) as f:
        return yaml.safe_load(f)


class ResolveCompressionOptionsTests(unittest.TestCase):
    def test_config_values_are_the_defaults(self):
        config = _load_lightweight()
        self.assertEqual(config.get("prune"), 0.3)
        self.assertIs(config.get("quantize"), True)

        prune, quantize = resolve_compression_options(config, None, False, None)

        self.assertEqual(prune, 0.3)
        self.assertIs(quantize, True)

    def test_empty_config_means_no_compression(self):
        self.assertEqual(resolve_compression_options({}, None, False, None),
                         (None, False))

    def test_cli_prune_overrides_config(self):
        prune, quantize = resolve_compression_options(
            _load_lightweight(), 0.5, False, None
        )
        self.assertEqual(prune, 0.5)
        self.assertIs(quantize, True)

    def test_no_prune_disables_config_pruning(self):
        prune, quantize = resolve_compression_options(
            _load_lightweight(), None, True, None
        )
        self.assertIsNone(prune)
        self.assertIs(quantize, True)

    def test_cli_quantize_flag_overrides_config(self):
        prune, quantize = resolve_compression_options(
            {"prune": 0.25, "quantize": False}, None, False, True
        )
        self.assertEqual(prune, 0.25)
        self.assertIs(quantize, True)

    def test_cli_no_quantize_overrides_config(self):
        prune, quantize = resolve_compression_options(
            _load_lightweight(), None, False, False
        )
        self.assertEqual(prune, 0.3)
        self.assertIs(quantize, False)


class ParseArgsDefaultsTests(unittest.TestCase):
    def _args(self, extra=()):
        return parse_args(
            ["--config", LIGHTWEIGHT_CONFIG, "--checkpoint", "ckpt.pt",
             "--out", "model.onnx", *extra]
        )

    def test_unset_flags_are_none_so_config_wins(self):
        args = self._args()
        self.assertIsNone(args.prune)
        self.assertIsNone(args.quantize)
        self.assertFalse(args.no_prune)

    def test_explicit_flags_are_recorded(self):
        args = self._args(["--prune", "0.4", "--quantize"])
        self.assertEqual(args.prune, 0.4)
        self.assertIs(args.quantize, True)

    def test_explicit_disable_flags_are_recorded(self):
        args = self._args(["--no-prune", "--no-quantize"])
        self.assertTrue(args.no_prune)
        self.assertIs(args.quantize, False)


if __name__ == "__main__":
    unittest.main()
