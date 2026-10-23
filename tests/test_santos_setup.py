"""Integrity and destination safety for optional cross-day data acquisition."""

import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.setup_santos_dataset import prepare_santos


class SantosSetupTests(unittest.TestCase):
    def test_verified_cache_is_reused_without_network_or_extraction(self):
        raw = b"tiny trusted fixture; no extraction is requested"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "santos/santos-v7.zip"
            target.parent.mkdir()
            target.write_bytes(raw)
            with (
                patch("scripts.setup_santos_dataset.SIZE", len(raw)),
                patch("scripts.setup_santos_dataset.MD5", hashlib.md5(raw).hexdigest()),
                patch(
                    "scripts.setup_santos_dataset.SHA256",
                    hashlib.sha256(raw).hexdigest(),
                ),
                patch("urllib.request.urlopen") as network,
            ):
                result = prepare_santos(root, verify_only=True)
            network.assert_not_called()
            self.assertFalse(result["signal_values_parsed"])
            self.assertFalse(result["extracted"])
            self.assertTrue((target.parent / "source_provenance.json").is_file())

    def test_corrupt_download_is_never_promoted(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("urllib.request.urlopen", return_value=io.BytesIO(b"wrong archive")),
        ):
            with self.assertRaisesRegex(ValueError, "checksum"):
                prepare_santos(Path(temp))
            self.assertFalse((Path(temp) / "santos/santos-v7.zip").exists())
            self.assertFalse((Path(temp) / "santos/santos-v7.zip.part").exists())

    def test_existing_corruption_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "santos/santos-v7.zip"
            target.parent.mkdir()
            target.write_bytes(b"untrusted existing file")
            with self.assertRaisesRegex(ValueError, "checksum"):
                prepare_santos(root)
            self.assertEqual(target.read_bytes(), b"untrusted existing file")

    def test_leaf_directory_part_and_metadata_symlinks_are_rejected(self):
        for location in (
            "santos",
            "santos/santos-v7.zip",
            "santos/santos-v7.zip.part",
            "santos/source_provenance.json",
        ):
            with self.subTest(location=location), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "root"
                outside = Path(temp) / "outside"
                root.mkdir()
                outside.mkdir()
                sentinel = outside / "sentinel"
                sentinel.write_bytes(b"preserve")
                link = root / location
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(
                    outside if location == "santos" else sentinel,
                    target_is_directory=location == "santos",
                )
                with (
                    patch("urllib.request.urlopen") as network,
                    self.assertRaisesRegex(ValueError, "symlink"),
                ):
                    prepare_santos(root)
                network.assert_not_called()
                self.assertEqual(sentinel.read_bytes(), b"preserve")

    def test_verify_only_never_attempts_a_missing_download(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("urllib.request.urlopen") as network,
        ):
            with self.assertRaises(FileNotFoundError):
                prepare_santos(Path(temp), verify_only=True)
            network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
