"""Public setup rejects corrupt source archives and destination symlinks."""

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from scripts.setup_public_datasets import prepare_har, prepare_wisdm


class PublicDatasetTests(unittest.TestCase):
    def test_unrelated_zip_is_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "unrelated.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("hello.txt", "not HAR")
            with self.assertRaisesRegex(ValueError, "checksum"):
                prepare_har(root / "data", archive)
            self.assertFalse((root / "data").exists())

    def test_wisdm_leaf_and_directory_links_cannot_escape_data_root(self):
        for directory_link in (False, True):
            with (
                self.subTest(directory_link=directory_link),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                outside = root / "outside"
                outside.mkdir()
                victim = outside / "readme.txt"
                victim.write_text("preserve")
                data = root / "data"
                data.mkdir()
                if directory_link:
                    (data / "wisdm").symlink_to(outside, target_is_directory=True)
                else:
                    (data / "wisdm").mkdir()
                    (data / "wisdm/readme.txt").symlink_to(victim)
                with mock.patch(
                    "scripts.setup_public_datasets.sha256",
                    return_value="e573791269aab4a629721cf04b6031b7cbf8e14261cd2bc86e65887a2bc592d7",
                ):
                    with self.assertRaisesRegex(ValueError, "symlink"):
                        prepare_wisdm(data, root / "unused.tar.gz")
                self.assertEqual(victim.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
