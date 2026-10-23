"""Device, loader and actual small pair-training runtime checks."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from data.adapter import WindowData
from engine.train import _device, _make_loader, _runtime
from engine.train_siamese import _embed_windows, train_siamese
from models.siamese import EmbeddingNet


def data():
    return WindowData(
        torch.randn(24, 3, 16),
        torch.zeros(24, dtype=torch.long),
        torch.arange(6).repeat_interleave(4),
        ["walking"],
        list(range(6)),
        20.0,
    )


class RuntimeTests(unittest.TestCase):
    def test_cpu_auto_and_unavailable_cuda(self):
        self.assertEqual(_device({}).type, "cpu")
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(_device({"device": "auto"}).type, "cpu")
            with self.assertRaisesRegex(ValueError, "unavailable"):
                _device({"device": "cuda"})

    def test_invalid_training_budgets_fail_early(self):
        for config, epochs, subset in (
            ({}, 0, None),
            ({"batch_size": 0}, 1, None),
            ({"num_workers": -1}, 1, None),
            ({}, 1, 0),
        ):
            with self.assertRaises(ValueError):
                _runtime(config, epochs, subset)

    def test_loader_shuffling_is_seeded_and_host_data_stays_on_cpu(self):
        source = data()
        config = {"batch_size": 3, "seed": 7, "num_workers": 0}
        a = torch.cat(
            [
                batch[2]
                for batch in _make_loader(source, config, True, torch.device("cpu"))
            ]
        )
        b = torch.cat(
            [
                batch[2]
                for batch in _make_loader(source, config, True, torch.device("cpu"))
            ]
        )
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(source.X.device.type, "cpu")
        self.assertFalse(torch.equal(a, source.y_subject))

    def test_real_triplet_and_contrastive_loops_save_safe_checkpoints(self):
        for loss in ("triplet", "contrastive"):
            config = {
                "source": "har",
                "data_root": "custom_data",
                "val_subjects": 2,
                "activity_ids": [0],
                "verification_pairing": "same_activity",
                "val_pairs": 40,
                "seed": 3,
                "batch_size": 8,
                "torch_threads": 1,
                "device": "cpu",
                "model_args": {"embed_dim": 8, "dropout": 0.0},
            }
            with (
                tempfile.TemporaryDirectory() as folder,
                mock.patch(
                    "engine.train_siamese.load_windows", return_value=data()
                ) as load,
            ):
                metrics = train_siamese(config, 1, None, loss, folder)
                checkpoint = torch.load(
                    os.path.join(folder, "best.pt"),
                    map_location="cpu",
                    weights_only=True,
                )
            self.assertEqual(load.call_args.kwargs["data_root"], "custom_data")
            self.assertEqual(load.call_args.args[1], "train")
            self.assertEqual(checkpoint["metrics"]["protocol"], metrics["protocol"])
            self.assertEqual(metrics["selection_metric"], "val_eer")
            self.assertTrue(0 <= metrics["per_epoch"][0]["val_eer"] <= 1)
            self.assertGreater(metrics["per_epoch"][0]["elapsed_seconds"], 0)
            self.assertTrue(
                set(metrics["protocol"]["train_subject_ids"]).isdisjoint(
                    metrics["protocol"]["val_subject_ids"]
                )
            )

    def test_embedding_batches_cover_every_window_on_cpu(self):
        torch.set_num_threads(1)
        model = EmbeddingNet(3, embed_dim=8, dropout=0)
        source = data()
        embedded = _embed_windows(model, source, 5, 0)
        self.assertEqual(embedded.shape, (24, 8))
        with torch.inference_mode():
            expected = model(source.X).numpy()
        self.assertTrue(
            torch.allclose(
                torch.from_numpy(embedded), torch.from_numpy(expected), atol=1e-6
            )
        )


if __name__ == "__main__":
    unittest.main()
