"""Training preflight and archive repair contracts; synthetic inputs only."""

from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from data.adapter import WindowData
from engine import train as activity
from engine import train_siamese as siamese
from scripts import setup_datasets as setup


def windows(activities=None) -> WindowData:
    subjects = torch.arange(6).repeat_interleave(4)
    return WindowData(
        torch.randn(24, 3, 16, generator=torch.Generator().manual_seed(19)),
        torch.zeros(24, dtype=torch.long)
        if activities is None
        else torch.tensor(activities, dtype=torch.long),
        subjects,
        ["walk", "sit"],
        list(range(6)),
        20.0,
    )


def zip_bytes(members) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def bundled_archive(root: Path, har=None, wisdm=None) -> Path:
    # Keep this fixture independent of production's completeness manifest.
    if har is None:
        har = {
            f"{setup.HAR_TOP_DIR}/{name}": b"fixture\n"
            for name in ("README.txt", "features.txt", "activity_labels.txt")
        }
        for split in ("train", "test"):
            for prefix in ("X", "y", "subject"):
                har[f"{setup.HAR_TOP_DIR}/{split}/{prefix}_{split}.txt"] = b"1\n"
            for sensor in ("body_acc", "body_gyro", "total_acc"):
                for axis in "xyz":
                    name = f"{sensor}_{axis}_{split}.txt"
                    har[f"{setup.HAR_TOP_DIR}/{split}/Inertial Signals/{name}"] = b"0\n"
    archive = root / "bundled.zip"
    archive.write_bytes(
        zip_bytes(
            {
                setup.HAR_OUTER_MEMBER: zip_bytes(
                    {f"{setup.HAR_TOP_DIR}.zip": zip_bytes(har)}
                ),
                setup.WISDM_MEMBER: zip_bytes(
                    {"train.csv": b"train\n", "test.csv": b"test\n"}
                    if wisdm is None
                    else wisdm
                ),
            }
        )
    )
    return archive


class IntegerValidationTests(unittest.TestCase):
    def test_activity_ids_reject_lossy_values_and_keep_original_labels(self):
        data = windows([0, 1] * 12)
        for bad in ([0.9], [True], ["0"], [np.nan], [], 1):
            with (
                self.subTest(ids=bad),
                self.assertRaisesRegex(ValueError, "activity_ids"),
            ):
                activity._filter_activities(data, bad)
        selected = activity._filter_activities(data, [np.int64(1)])
        self.assertEqual(selected.y_activity.tolist(), [1] * 12)
        self.assertEqual(selected.activity_names, ["walk", "sit"])

    def test_integer_runtime_fields_reject_fractions_before_rng_mutation(self):
        cases = [
            ({}, 1.5, None, "epochs"),
            ({}, True, None, "epochs"),
            ({}, 1, 2.5, "subset"),
        ]
        for field in (
            "batch_size",
            "torch_threads",
            "num_workers",
            "prefetch_factor",
            "seed",
            "window_size",
            "stride",
            "val_subjects",
        ):
            cases.append(({field: 1.5}, 1, None, field))
            cases.append(({field: True}, 1, None, field))
        cases.extend(
            [
                ({"prefetch_factor": 0}, 1, None, "prefetch_factor"),
                ({"seed": -1}, 1, None, "seed"),
                ({"seed": 2**32}, 1, None, "seed"),
            ]
        )
        for config, epochs, subset, name in cases:
            with (
                self.subTest(config=config, epochs=epochs, subset=subset),
                mock.patch.object(activity, "seed_everything") as seed,
                mock.patch.object(torch, "set_num_threads") as threads,
                self.assertRaisesRegex(ValueError, name),
            ):
                activity._runtime(config, epochs, subset)  # ty: ignore[invalid-argument-type] - deliberate invalid inputs
            seed.assert_not_called()
            threads.assert_not_called()

    def test_loader_and_split_helpers_do_not_bypass_integer_checks(self):
        data = windows()
        for config in ({"batch_size": 1.9}, {"num_workers": 0.9}, {"seed": 2.5}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                activity._make_loader(data, config, False, torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "val_subjects"):
            activity._split_validation(data, 2.5, 0)  # ty: ignore[invalid-argument-type]
        with self.assertRaisesRegex(ValueError, "subset"):
            activity._stratified_subset(data, 5.5, 0)  # ty: ignore[invalid-argument-type]
        with self.assertRaisesRegex(ValueError, "stride"):
            activity._chronological_validation(data, {"stride": 4.9})

    def test_window_dimensions_and_sampling_rate_are_validated(self):
        for config in (
            {"window_size": 16.9},
            {"model_args": {"n_channels": 3.9}},
            {"model_args": {"n_classes": 2.9}},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                activity._validate_windows(windows(), config)
        for rate in (0.0, -20.0, np.nan, np.inf, True, "20", [20]):
            data = windows()
            data.sampling_rate = rate  # ty: ignore[invalid-assignment] - exercise corrupted metadata
            with (
                self.subTest(rate=rate),
                self.assertRaisesRegex(ValueError, "sampling_rate"),
            ):
                activity._validate_windows(data, {})
        activity._validate_windows(windows(), {"window_size": 16})

    def test_triplet_count_is_exact_and_rejects_lossy_budgets(self):
        for bad in (1.5, True, "2", np.nan, np.inf, 0, -1):
            with (
                self.subTest(multiplier=bad),
                self.assertRaisesRegex(ValueError, "triplets_per_epoch"),
            ):
                siamese._n_triplets({"triplets_per_epoch": bad}, 24)
        for bad in (24.9, True, 0):
            with (
                self.subTest(samples=bad),
                self.assertRaisesRegex(ValueError, "n_samples"),
            ):
                siamese._n_triplets({}, bad)  # ty: ignore[invalid-argument-type]
        self.assertEqual(
            siamese._n_triplets({"triplets_per_epoch": np.int64(3)}, 24), 72
        )


class TrainingPreflightTests(unittest.TestCase):
    def config(self):
        return {
            "source": "har",
            "model": "cnn_baseline",
            "seed": 3,
            "torch_threads": 1,
            "batch_size": 8,
            "val_subjects": 2,
            "val_pairs": 12,
            "model_args": {"embed_dim": 8, "dropout": 0.0},
        }

    def test_static_invalid_settings_fail_before_data_or_model_load(self):
        shared = [
            {"batch_size": 2.5},
            {"activity_ids": [0.5]},
            {"stride": 64.9},
            {"val_subjects": 2.5},
            {"lr": np.nan},
            {"weight_decay": np.inf},
        ]
        for module in (activity, siamese):
            extras = shared + (
                [{"class_weight": "unsupported"}]
                if module is activity
                else [
                    {"triplets_per_epoch": 1.9},
                    {"val_pairs": 2.9},
                    {"val_pairs": 1},
                    {"verification_pairing": "unknown"},
                    {"model_args": {"embed_dim": 8.9}},
                    {"model_args": {"margin": np.nan}},
                    {"model_args": {"margin": -1}},
                    {"model_args": {"dropout": np.inf}},
                ]
            )
            for extra in extras:
                with (
                    self.subTest(module=module.__name__, extra=extra),
                    tempfile.TemporaryDirectory() as folder,
                    mock.patch.object(
                        module, "load_windows", return_value=windows()
                    ) as load,
                    mock.patch.object(torch.optim, "Adam") as optimizer,
                    self.assertRaises(ValueError),
                ):
                    if module is activity:
                        activity.train({**self.config(), **extra}, 1, None, folder)
                    else:
                        siamese.train_siamese(
                            {**self.config(), **extra}, 1, None, "triplet", folder
                        )
                load.assert_not_called()
                optimizer.assert_not_called()

    def test_fractional_yaml_epochs_are_not_coerced_by_either_cli(self):
        for module in (activity, siamese):
            with (
                self.subTest(module=module.__name__),
                tempfile.TemporaryDirectory() as folder,
            ):
                config = Path(folder) / "config.yaml"
                config.write_text(
                    "source: har\nmodel: cnn_baseline\nepochs: 1.9\n"
                    f"out_dir: {folder}\n"
                )
                with (
                    mock.patch.object(module, "load_windows") as load,
                    self.assertRaisesRegex(ValueError, "epochs"),
                ):
                    module.main(["--config", str(config)])
                load.assert_not_called()

    def test_same_activity_feasibility_fails_before_model_or_optimizer(self):
        # Both subjects have positives globally, but either no activity is shared
        # or every shared activity has singleton identities (no genuine pair).
        for labels in ([s % 2 for s in range(6) for _ in range(4)], [0, 1, 0, 1] * 6):
            data = windows(labels)
            if labels == [0, 1, 0, 1] * 6:
                # Give each validation subject one window per activity.
                data = activity._take(
                    data, torch.arange(24).reshape(6, 4)[:, :2].flatten()
                )
            with (
                self.subTest(labels=labels),
                tempfile.TemporaryDirectory() as folder,
                mock.patch.object(siamese, "load_windows", return_value=data),
                mock.patch.object(siamese, "EmbeddingNet") as model,
                mock.patch.object(torch.optim, "Adam") as optimizer,
                self.assertRaisesRegex(ValueError, "genuine/impostor|shared activity"),
            ):
                siamese.train_siamese(
                    {**self.config(), "verification_pairing": "same_activity"},
                    1,
                    None,
                    "triplet",
                    folder,
                )
            model.assert_not_called()
            optimizer.assert_not_called()

    def test_valid_same_activity_and_legacy_training_remain_usable(self):
        for extra, loss, metric in (
            ({"verification_pairing": "same_activity"}, "triplet", "val_eer"),
            ({"val_subjects": 0}, "contrastive", "train_loss"),
        ):
            with (
                self.subTest(extra=extra),
                tempfile.TemporaryDirectory() as folder,
                mock.patch.object(
                    siamese, "load_windows", return_value=windows()
                ) as load,
            ):
                metrics = siamese.train_siamese(
                    {**self.config(), **extra}, 1, None, loss, folder
                )
                saved = torch.load(
                    Path(folder) / "best.pt", map_location="cpu", weights_only=True
                )
            self.assertEqual(load.call_args.args[1], "train")
            self.assertEqual(metrics["selection_metric"], metric)
            self.assertEqual(saved["epoch"], 1)
            self.assertTrue(np.isfinite(metrics["final_loss"]))

    def test_unrestricted_validation_accepts_subjects_with_disjoint_activities(self):
        data = windows([s % 2 for s in range(6) for _ in range(4)])
        with (
            tempfile.TemporaryDirectory() as folder,
            mock.patch.object(siamese, "load_windows", return_value=data),
        ):
            metrics = siamese.train_siamese(self.config(), 1, None, "triplet", folder)
        self.assertEqual(metrics["selection_metric"], "val_eer")
        self.assertEqual(metrics["protocol"]["verification_pairing"], "unrestricted")


class DatasetRepairTests(unittest.TestCase):
    def test_readme_only_har_is_repaired_without_changing_existing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = bundled_archive(root)
            datasets = root / "data"
            marker = datasets / "uci_har" / setup.HAR_TOP_DIR / "README.txt"
            marker.parent.mkdir(parents=True)
            marker.write_bytes(b"preserve existing README")
            with (
                mock.patch.object(setup, "ARCHIVE", str(archive)),
                mock.patch.object(setup, "DATASETS_ROOT", str(datasets)),
            ):
                self.assertTrue(setup.setup_har())
                self.assertEqual(marker.read_bytes(), b"preserve existing README")
                signal = marker.parent / "test/Inertial Signals/body_gyro_z_test.txt"
                self.assertEqual(signal.read_bytes(), b"0\n")
                signal.unlink()
                self.assertTrue(setup.setup_har())
                self.assertTrue(signal.is_file())
                marker.unlink()
                self.assertTrue(setup.setup_har())
                with mock.patch.object(
                    zipfile,
                    "ZipFile",
                    side_effect=AssertionError("complete dataset must skip archive"),
                ):
                    self.assertFalse(setup.setup_har())

    def test_wisdm_repairs_empty_files_and_preserves_existing_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = bundled_archive(root)
            datasets = root / "data"
            dest = datasets / "wisdm"
            dest.mkdir(parents=True)
            (dest / "train.csv").write_bytes(b"keep train")
            (dest / "test.csv").touch()
            with (
                mock.patch.object(setup, "ARCHIVE", str(archive)),
                mock.patch.object(setup, "DATASETS_ROOT", str(datasets)),
            ):
                self.assertTrue(setup.setup_wisdm())
                self.assertEqual((dest / "train.csv").read_bytes(), b"keep train")
                self.assertEqual((dest / "test.csv").read_bytes(), b"test\n")
                self.assertFalse(setup.setup_wisdm())

    def test_incomplete_archive_rejected_before_extracting_any_files(self):
        for dataset in ("har", "wisdm"):
            with (
                self.subTest(dataset=dataset),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                archive = bundled_archive(
                    root,
                    har={f"{setup.HAR_TOP_DIR}/README.txt": b"partial"},
                    wisdm={"train.csv": b"partial"},
                )
                dest = root / "data"
                with (
                    mock.patch.object(setup, "ARCHIVE", str(archive)),
                    mock.patch.object(setup, "DATASETS_ROOT", str(dest)),
                    self.assertRaisesRegex(ValueError, "required"),
                ):
                    (setup.setup_har if dataset == "har" else setup.setup_wisdm)()
                self.assertFalse(dest.exists())

    def test_complete_symlinked_dataset_cannot_bypass_safety_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside"
            outside.mkdir()
            (outside / "train.csv").write_bytes(b"preserve")
            (outside / "test.csv").write_bytes(b"preserve")
            datasets = root / "data"
            datasets.mkdir()
            (datasets / "wisdm").symlink_to(outside, target_is_directory=True)
            with (
                mock.patch.object(setup, "DATASETS_ROOT", str(datasets)),
                self.assertRaisesRegex(ValueError, "symlink"),
            ):
                setup.setup_wisdm()
            self.assertEqual((outside / "train.csv").read_bytes(), b"preserve")

    def test_zip_preflight_and_atomic_write_preserve_existing_data_on_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "first.txt"
            target.write_bytes(b"preserve")
            with self.assertRaises(ValueError):
                setup._extract_zipfile_to(
                    zip_bytes({"first.txt": b"replace", "../escape": b"bad"}), str(root)
                )
            self.assertEqual(target.read_bytes(), b"preserve")
            # ZIP_STORED keeps bytes visible; corrupt the stored payload so CRC
            # verification fails after the destination has been selected.
            corrupt = zip_bytes({"first.txt": b"payload-for-corruption"}).replace(
                b"payload-for-corruption", b"PAYLOAD-FOR-CORRUPTION"
            )
            with self.assertRaises(zipfile.BadZipFile):
                setup._extract_zipfile_to(corrupt, str(root))
            self.assertEqual(target.read_bytes(), b"preserve")
            self.assertEqual(list(root.iterdir()), [target])

    def test_hardlinked_destination_cannot_overwrite_another_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "original"
            original.write_bytes(b"preserve")
            dest = root / "dest"
            dest.mkdir()
            (dest / "target").hardlink_to(original)
            setup._extract_zipfile_to(zip_bytes({"target": b"new"}), str(dest))
            self.assertEqual(original.read_bytes(), b"preserve")
            self.assertEqual((dest / "target").read_bytes(), b"new")


if __name__ == "__main__":
    unittest.main()
