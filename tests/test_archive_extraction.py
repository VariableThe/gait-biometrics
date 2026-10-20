"""Archive extraction must not escape its destination.

`scripts/setup_datasets.py` unpacks nested WISDM/HAR zips. A malicious or
malformed member named ``../x``, an absolute path, a Windows-style path or a
symlink would otherwise write outside the destination (zip-slip). Normal
members and the archive's macOS metadata junk must behave as before.
"""

from __future__ import annotations

import io
import os
import stat
import sys
import tempfile
import unittest
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.setup_datasets import _extract_zipfile_to, _safe_member_target  # noqa: E402


def _zip_bytes(members: list[tuple]) -> bytes:
    """Build an in-memory zip.

    Each member is ``(name, data)`` or ``(name, data, external_attr)``.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for member in members:
            name, data = member[0], member[1]
            info = zipfile.ZipInfo(name)
            if len(member) > 2:
                info.external_attr = member[2]
            else:
                info.external_attr = (stat.S_IFREG | 0o644) << 16
            zf.writestr(info, data)
    return buf.getvalue()


def _symlink_attr() -> int:
    return (stat.S_IFLNK | 0o777) << 16


class SafeMemberTargetTests(unittest.TestCase):
    def test_nested_relative_member_resolves_under_dest(self):
        with tempfile.TemporaryDirectory() as dest:
            target = _safe_member_target(dest, "a/b/c.txt")
            self.assertEqual(target, os.path.join(dest, "a", "b", "c.txt"))

    def test_rejects_traversal_and_absolute_and_drive_forms(self):
        with tempfile.TemporaryDirectory() as dest:
            for name in (
                "../escape.txt",
                "a/../../escape.txt",
                "..\\escape.txt",
                "/etc/passwd",
                "C:\\windows\\system32\\x.txt",
                "\\\\server\\share\\x.txt",
            ):
                with self.subTest(name=name):
                    with self.assertRaises(ValueError):
                        _safe_member_target(dest, name)

    def test_rejects_directory_entry(self):
        with tempfile.TemporaryDirectory() as dest:
            with self.assertRaises(ValueError):
                _safe_member_target(dest, "dir/")


class ExtractZipfileToTests(unittest.TestCase):
    def test_normal_archive_extracts_and_filters_metadata(self):
        payload = _zip_bytes([
            ("train/data.txt", b"hello"),
            ("nested/deep/file.bin", b"\x00\x01"),
            ("dir/", b""),
            ("__MACOSX/._data.txt", b"junk"),
            ("._data.txt", b"junk"),
            (".DS_Store", b"junk"),
        ])
        with tempfile.TemporaryDirectory() as root:
            dest = os.path.join(root, "dest")
            os.makedirs(dest)

            written = _extract_zipfile_to(payload, dest, skip_prefixes=("__MACOSX/",))

            self.assertEqual(
                sorted(os.path.relpath(p, dest) for p in written),
                [os.path.join("nested", "deep", "file.bin"),
                 os.path.join("train", "data.txt")],
            )
            for rel, content in (("train/data.txt", b"hello"),
                                 ("nested/deep/file.bin", b"\x00\x01")):
                with open(os.path.join(dest, rel), "rb") as f:
                    self.assertEqual(f.read(), content)
            self.assertFalse(os.path.exists(os.path.join(dest, "__MACOSX")))
            self.assertFalse(os.path.exists(os.path.join(dest, "._data.txt")))
            self.assertFalse(os.path.exists(os.path.join(dest, ".DS_Store")))

    def test_rejects_parent_traversal_and_writes_nothing_outside(self):
        with tempfile.TemporaryDirectory() as root:
            dest = os.path.join(root, "dest")
            os.makedirs(dest)
            payload = _zip_bytes([("../pwned.txt", b"escaped")])

            with self.assertRaises(ValueError):
                _extract_zipfile_to(payload, dest)

            self.assertFalse(os.path.exists(os.path.join(root, "pwned.txt")))

    def test_rejects_absolute_member(self):
        with tempfile.TemporaryDirectory() as root:
            dest = os.path.join(root, "dest")
            os.makedirs(dest)
            absolute = os.path.join(root, "absolute.txt")
            payload = _zip_bytes([(absolute, b"escaped")])

            with self.assertRaises(ValueError):
                _extract_zipfile_to(payload, dest)

            self.assertFalse(os.path.exists(absolute))

    def test_rejects_symlink_member(self):
        with tempfile.TemporaryDirectory() as root:
            dest = os.path.join(root, "dest")
            os.makedirs(dest)
            payload = _zip_bytes([("link.txt", "/etc/passwd", _symlink_attr())])

            with self.assertRaises(ValueError):
                _extract_zipfile_to(payload, dest)

            self.assertFalse(os.path.exists(os.path.join(dest, "link.txt")))

    def test_rejects_member_crossing_preexisting_symlink_out_of_dest(self):
        """`dest/escape -> outside` must not turn `escape/pwned.txt` into a write
        outside dest (a lexical abspath check cannot see this)."""
        with tempfile.TemporaryDirectory() as root:
            dest = os.path.join(root, "dest")
            outside = os.path.join(root, "outside")
            os.makedirs(dest)
            os.makedirs(outside)
            os.symlink(outside, os.path.join(dest, "escape"),
                       target_is_directory=True)
            payload = _zip_bytes([("escape/pwned.txt", b"escaped")])

            with self.assertRaises(ValueError):
                _extract_zipfile_to(payload, dest)

            self.assertFalse(os.path.exists(os.path.join(outside, "pwned.txt")))
            self.assertFalse(os.path.exists(os.path.join(dest, "escape", "pwned.txt")))

    def test_rejects_member_crossing_symlink_component_even_within_dest(self):
        with tempfile.TemporaryDirectory() as root:
            dest = os.path.join(root, "dest")
            inner = os.path.join(dest, "inner")
            os.makedirs(inner)
            os.symlink(inner, os.path.join(dest, "link"), target_is_directory=True)
            payload = _zip_bytes([("link/pwned.txt", b"escaped")])

            with self.assertRaises(ValueError):
                _extract_zipfile_to(payload, dest)

            self.assertFalse(os.path.exists(os.path.join(inner, "pwned.txt")))

    def test_existing_leaf_symlink_is_not_followed(self):
        """A pre-existing symlink exactly at the member's final path."""
        with tempfile.TemporaryDirectory() as root:
            dest = os.path.join(root, "dest")
            outside = os.path.join(root, "outside")
            os.makedirs(dest)
            os.makedirs(outside)
            victim = os.path.join(outside, "victim.txt")
            with open(victim, "w") as f:
                f.write("original")
            os.symlink(victim, os.path.join(dest, "data.txt"))
            payload = _zip_bytes([("data.txt", b"overwritten")])

            with self.assertRaises(ValueError):
                _extract_zipfile_to(payload, dest)

            with open(victim) as f:
                self.assertEqual(f.read(), "original")

    def test_dest_may_itself_be_a_symlink(self):
        with tempfile.TemporaryDirectory() as root:
            real = os.path.join(root, "real")
            os.makedirs(real)
            linked = os.path.join(root, "dest-link")
            os.symlink(real, linked, target_is_directory=True)

            written = _extract_zipfile_to(_zip_bytes([("a/b.txt", b"ok")]), linked)

            self.assertEqual(len(written), 1)
            with open(os.path.join(real, "a", "b.txt"), "rb") as f:
                self.assertEqual(f.read(), b"ok")
            # Traversal relative to the symlinked dest is still rejected.
            with self.assertRaises(ValueError):
                _extract_zipfile_to(_zip_bytes([("../x.txt", b"x")]), linked)
            self.assertFalse(os.path.exists(os.path.join(root, "x.txt")))


if __name__ == "__main__":
    unittest.main()
