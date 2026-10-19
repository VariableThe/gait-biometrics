"""Checkpoints load safely: rich dicts and bare state_dicts work, pickles don't.

`engine/evaluate.py` and `scripts/export_onnx.py` used to unpickle with
``weights_only=False``, so a checkpoint could execute arbitrary code on load.
Both now use `models.common.load_checkpoint`, which pins ``weights_only=True``.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from models.common import load_checkpoint, save_checkpoint


def _tiny_model() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(),
                               torch.nn.Linear(3, 2))


class LoadCheckpointTests(unittest.TestCase):
    def test_rich_checkpoint_from_save_checkpoint_loads(self):
        model = _tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "best.pt")
            save_checkpoint(path, model, optimizer, 3,
                            {"train_loss": 0.5, "selection_metric": "train_loss"})

            state = load_checkpoint(path)

        expected = model.state_dict()
        self.assertEqual(set(state), set(expected))
        for key, value in expected.items():
            self.assertTrue(torch.equal(value, state[key]), msg=key)

    def test_bare_state_dict_loads(self):
        model = _tiny_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bare.pt")
            torch.save(model.state_dict(), path)

            state = load_checkpoint(path)

        self.assertEqual(set(state), set(model.state_dict()))

    def test_non_dict_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "list.pt")
            torch.save([1, 2, 3], path)
            with self.assertRaises(ValueError):
                load_checkpoint(path)

    def test_non_dict_model_state_dict_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "weird.pt")
            torch.save({"model_state_dict": "not-a-state-dict"}, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path)

    def test_malicious_pickle_is_rejected_and_writes_no_marker(self):
        class Evil:
            """Pickling this runs the reducer, which would write `marker`."""

            def __init__(self, marker_path: str) -> None:
                self.marker_path = marker_path

            def __reduce__(self):
                return (open, (self.marker_path, "w"))

        with tempfile.TemporaryDirectory() as tmp:
            marker = os.path.join(tmp, "pwned")
            path = os.path.join(tmp, "evil.pt")
            torch.save(Evil(marker), path)

            with self.assertRaises(Exception):
                load_checkpoint(path)

            self.assertFalse(os.path.exists(marker))

    def test_evaluate_delegate_loads_safely(self):
        from engine.evaluate import _load_checkpoint

        model = _tiny_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bare.pt")
            torch.save(model.state_dict(), path)
            state = _load_checkpoint(path)

        self.assertEqual(set(state), set(model.state_dict()))


class NoUnsafeCheckpointLoadTests(unittest.TestCase):
    """Regression guard: no call site may re-enable unsafe unpickling."""

    def test_no_source_file_disables_weights_only(self):
        for rel in ("engine/evaluate.py", "engine/train.py",
                    "engine/train_siamese.py", "scripts/export_onnx.py",
                    "models/common.py"):
            with self.subTest(file=rel):
                with open(os.path.join(REPO_ROOT, rel)) as f:
                    self.assertNotIn("weights_only=False", f.read())


if __name__ == "__main__":
    unittest.main()
