"""Synthetic ZIP tests: no official dataset or held-out signals are accessed."""

import hashlib
import json
import stat
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile, ZipInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.santos import (
    SANTOS_ARCHIVE_PREFIX,
    SANTOS_CSV_HEADER,
    fingerprint_santos_members,
    load_santos_windows,
)


def member(subject=1, day=1, trial=1):
    return (
        f"{SANTOS_ARCHIVE_PREFIX}processed_data/user{subject:02d}/"
        f"capture_user{subject:02d}_day{day}_{trial:04d}_imu_walk.csv"
    )


def csv_bytes(frames, *, header=None):
    columns = SANTOS_CSV_HEADER if header is None else header
    rows = [",".join(columns)]
    rows.extend(
        f"{frame},{i + 0.25},{i + 9.81},{-i - 0.5},0.1,0.2,0.3"
        for i, frame in enumerate(frames)
    )
    return ("\n".join(rows) + "\n").encode()


class SantosWindowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "synthetic.zip"

    def archive(self, entries):
        with ZipFile(self.path, "w") as archive:
            for name, content in entries:
                archive.writestr(name, content)
        return self.path

    def load(self, subjects=(1,), **kwargs):
        return load_santos_windows(self.path, subjects, **kwargs)

    def test_exact_arrays_real_ids_units_frames_and_member_hashes(self):
        original = csv_bytes(range(1395, 1402))
        other = csv_bytes(range(-3, 1))
        self.archive([(member(3, 2, 10), other), (member(), original)])
        result = self.load([3, 1], window_size=3, stride=3)
        self.assertEqual(result["X"].shape, (3, 3, 3))
        self.assertEqual(result["X"].dtype, np.float32)
        for name in (
            "subjects",
            "days",
            "trials",
            "starts",
            "frames",
            "recording_indices",
        ):
            self.assertEqual(result[name].dtype, np.int64)
        np.testing.assert_array_equal(result["subjects"], [1, 1, 3])
        np.testing.assert_array_equal(result["days"], [1, 1, 2])
        np.testing.assert_array_equal(result["trials"], [1, 1, 10])
        np.testing.assert_array_equal(result["starts"], [0, 3, 0])
        np.testing.assert_array_equal(
            result["frames"], [[1395, 1396, 1397], [1398, 1399, 1400], [-3, -2, -1]]
        )
        np.testing.assert_array_equal(result["recording_indices"], [0, 0, 1])
        np.testing.assert_allclose(result["X"][0, :, 0], [0.25, 9.81, -0.5], rtol=1e-6)
        metadata = result["metadata"]
        json.dumps(metadata, allow_nan=False)
        self.assertEqual(
            metadata["recordings"][0]["sha256"], hashlib.sha256(original).hexdigest()
        )
        self.assertEqual(metadata["recordings"][0]["member"], member())
        self.assertEqual(metadata["recordings"][0]["tail_rows"], 1)
        self.assertEqual(metadata["recordings"][0]["source_rows"], 7)
        self.assertFalse(metadata["subject_coverage"][0]["eligible_both_days"])

    def test_unselected_payloads_are_never_opened_or_hashed(self):
        self.archive(
            [
                (member(), csv_bytes(range(4))),
                (member(6), b"unknown header\nnot signal values\n"),
                (f"{SANTOS_ARCHIVE_PREFIX}raw_data/user01/raw.csv", b"do not read"),
                (member().replace("_imu_walk.csv", "_nexus_walk.csv"), b"do not read"),
            ]
        )
        opened = []
        original_open = ZipFile.open

        def guarded_open(archive, name, *args, **kwargs):
            self.assertIsInstance(name, ZipInfo)
            opened.append(name.filename)
            self.assertEqual(name.filename, member())
            return original_open(archive, name, *args, **kwargs)

        with patch.object(ZipFile, "open", guarded_open):
            result = self.load(window_size=2, stride=2)
            manifest = fingerprint_santos_members(self.path, [1])
        self.assertEqual(opened, [member(), member()])
        self.assertEqual(
            manifest["recordings"][0]["sha256"],
            result["metadata"]["recordings"][0]["sha256"],
        )
        self.assertEqual(
            manifest["members_sha256"], result["metadata"]["members_sha256"]
        )
        json.dumps(manifest, allow_nan=False)

    def test_every_bad_row_breaks_continuity_and_retains_original_offsets(self):
        bad_rows = [
            "",
            "2,nan,0,0,0,0,0",
            "2,inf,0,0,0,0,0",
            "2,1e100,0,0,0,0,0",
            "2,0,0,0,0,0",
            "2,0,0,0,0,0,0,0",
            "2.0,0,0,0,0,0,0",
            "missing,0,0,0,0,0,0",
            "9223372036854775808,0,0,0,0,0,0",
            "2,word,0,0,0,0,0",
            "2,0,0,0,nan,0,0",
            "2,0,0,0,0,inf,0",
            "2,0,0,0,0,0,1e100",
        ]
        for bad in bad_rows:
            with self.subTest(bad=bad):
                payload = (
                    ",".join(SANTOS_CSV_HEADER)
                    + f"\n0,1,2,3,0,0,0\n1,2,3,4,0,0,0\n{bad}\n2,3,4,5,0,0,0\n3,4,5,6,0,0,0\n"
                ).encode()
                self.archive([(member(), payload)])
                result = self.load(window_size=3, stride=1)
                self.assertEqual(result["X"].shape, (0, 3, 3))
                self.assertEqual(result["frames"].shape, (0, 3))
                result = self.load(window_size=2, stride=1)
                np.testing.assert_array_equal(result["starts"], [0, 3])
                np.testing.assert_array_equal(result["frames"], [[0, 1], [2, 3]])
                recording = result["metadata"]["recordings"][0]
                self.assertEqual(recording["invalid_row_offsets"], [2])
                self.assertEqual(recording["source_rows"], 5)

    def test_gaps_duplicates_resets_and_large_exact_frames_form_new_segments(self):
        frames = [10, 11, 15, 16, 16, 17, -4, -3, 2**53 + 1, 2**53 + 2]
        self.archive([(member(), csv_bytes(frames))])
        self.assertEqual(len(self.load(window_size=3, stride=1)["X"]), 0)
        result = self.load(window_size=2, stride=2)
        np.testing.assert_array_equal(result["starts"], [0, 2, 4, 6, 8])
        np.testing.assert_array_equal(
            result["frames"], np.asarray(frames).reshape(-1, 2)
        )
        breaks = result["metadata"]["recordings"][0]["continuity_breaks"]
        self.assertEqual(
            [b["reason"] for b in breaks], ["gap", "duplicate", "reset", "gap"]
        )

    def test_int64_edges_do_not_wrap_and_fractional_or_float_frames_are_invalid(self):
        self.archive(
            [(member(), csv_bytes([2**63 - 2, 2**63 - 1, -(2**63), -(2**63) + 1]))]
        )
        result = self.load(window_size=2, stride=2)
        np.testing.assert_array_equal(
            result["frames"], [[2**63 - 2, 2**63 - 1], [-(2**63), -(2**63) + 1]]
        )
        self.assertEqual(
            result["metadata"]["recordings"][0]["continuity_breaks"][0]["reason"],
            "reset",
        )

    def test_trials_days_and_subjects_never_join_and_empty_coverage_is_explicit(self):
        self.archive(
            [
                (member(1, 1, 1), csv_bytes([0, 1])),
                (member(1, 1, 2), csv_bytes([2, 3])),
                (member(1, 2, 1), csv_bytes([4, 5])),
                (member(3, 1, 1), csv_bytes([6, 7])),
            ]
        )
        result = self.load([1, 3], window_size=3, stride=1)
        self.assertEqual(result["X"].shape, (0, 3, 3))
        self.assertEqual(len(result["metadata"]["recordings"]), 4)
        self.assertEqual(
            result["metadata"]["exclusions"]["zero_window_recordings"], [0, 1, 2, 3]
        )
        result = self.load([1, 3], window_size=2, stride=2)
        self.assertTrue(result["metadata"]["subject_coverage"][0]["eligible_both_days"])
        self.assertFalse(
            result["metadata"]["subject_coverage"][1]["eligible_both_days"]
        )
        self.assertEqual(
            len(result["metadata"]["exclusions"]["missing_recordings"]), 36
        )

    def test_stride_restarts_in_each_segment_and_reports_uncovered_rows(self):
        self.archive(
            [(member(), csv_bytes([0, 1, 2, 3, 4, 5, 20, 21, 22, 23, 24, 25]))]
        )
        result = self.load(window_size=2, stride=4)
        np.testing.assert_array_equal(result["starts"], [0, 4, 6, 10])
        self.assertEqual(result["metadata"]["recordings"][0]["covered_rows"], 8)
        self.assertEqual(result["metadata"]["recordings"][0]["stride_gap_rows"], 4)

    def test_unknown_headers_channel_changes_and_encoding_are_rejected(self):
        for payload in [
            csv_bytes([0, 1], header=("Frame", "Acc x", "Acc y", "Acc z")),
            b"frame,acc x,acc y,acc z\n0,1,2,3\n",
            b"frame,acc x,acc y,acc z,gyro x\n0,1,2,3,4\n",
            b"acc x,acc y,acc z\n1,2,3\n",
            b"\xff\xfeinvalid",
            b"",
            ",".join(SANTOS_CSV_HEADER).encode() + b'\n"unclosed',
        ]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.archive([(member(), payload)])
                self.load(window_size=2)

    def test_missing_and_invalid_subjects_and_window_parameters_fail(self):
        self.archive([(member(), csv_bytes([0, 1]))])
        for subjects in [[], [2], [True], [1.0], ["user01"], [1, 1], None]:
            with self.subTest(subjects=subjects), self.assertRaises(ValueError):
                self.load(subjects)
        with self.assertRaisesRegex(ValueError, "Missing requested subject"):
            self.load([3])
        for kwargs in [
            {"window_size": 0},
            {"stride": 0},
            {"window_size": 2.0},
            {"stride": True},
        ]:
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(ValueError, "positive integer"),
            ):
                self.load(**kwargs)

    def test_member_identity_day_trial_case_and_prefix_validation(self):
        valid = member()
        for bad in [
            valid.replace("capture_user01", "capture_user03"),
            valid.replace("day1", "day3"),
            valid.replace("_0001_", "_0000_"),
            valid.replace("_0001_", "_0011_"),
            valid.replace("user01/", "User01/"),
            valid.replace("processed_data", "Processed_Data"),
            valid.replace("_imu_walk.csv", "_imu_walk.CSV"),
            "other/" + valid,
            member(2),
        ]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.archive([(bad, csv_bytes([0, 1]))])
                self.load(window_size=2)

    def test_unsafe_unselected_paths_and_symlinks_fail_before_payload_access(self):
        bad_paths = [
            "../outside",
            "/absolute",
            "C:/absolute",
            "a//b",
            "a/./b",
            "a/../b",
            "a\\b",
            "a/\x01b",
        ]
        for bad in bad_paths:
            with self.subTest(bad=bad):
                self.archive([(member(), csv_bytes([0, 1])), (bad, b"not read")])
                with (
                    patch.object(
                        ZipFile, "open", side_effect=AssertionError("No payload access")
                    ),
                    self.assertRaisesRegex(ValueError, "Unsafe ZIP member"),
                ):
                    self.load(window_size=2)
        symlink = ZipInfo("symlink")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        self.archive([(member(), csv_bytes([0, 1])), (symlink, b"target")])
        with self.assertRaisesRegex(ValueError, "Nonregular"):
            self.load(window_size=2)

    def test_duplicate_members_fail_and_fingerprints_change_with_original_bytes(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.archive([(member(), csv_bytes([0, 1])), (member(), csv_bytes([0, 1]))])
        with self.assertRaisesRegex(ValueError, "Duplicate ZIP member"):
            self.load(window_size=2)
        self.archive([(member(), csv_bytes([0, 1]))])
        before = fingerprint_santos_members(self.path, [1])
        self.archive([(member(), csv_bytes([0, 1]) + b"\n")])
        after = fingerprint_santos_members(self.path, [1])
        self.assertNotEqual(before["members_sha256"], after["members_sha256"])


if __name__ == "__main__":
    unittest.main()
