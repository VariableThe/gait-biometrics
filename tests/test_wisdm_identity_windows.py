"""Synthetic adversarial records test identity provenance, not model accuracy."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.adapter import (
    WISDM_V1_ACTIVITY_NAMES,
    _read_wisdm_v1,
    _window_wisdm,
    _wisdm_subject_split,
    build_loader,
    load_windows,
)


class WisdmIdentityWindowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "wisdm").mkdir()

    def raw(self, records):
        path = self.root / "wisdm" / "WISDM_ar_v1.1_raw.txt"
        path.write_text(records)
        return path

    def test_real_identity_mapping_is_shared_and_source_splits_are_disjoint(self):
        ids = [2, 4, 7, 11, 33]
        records = []
        for subject in reversed(ids):
            for activity in ["Walking", "Jogging"]:
                for j in range(8):
                    records.append(
                        f"{subject},{activity},{1_000_000_000 + j * 50_000_000},{subject * 100 + j},0,1;\n"
                    )
        self.raw("".join(records))
        train = load_windows("wisdm_v1", "train", 4, 2, str(self.root))
        test = load_windows("wisdm_v1", "test", 4, 2, str(self.root))
        self.assertEqual(train.subject_ids, ids)
        self.assertEqual(test.subject_ids, ids)
        self.assertEqual(train.activity_names, WISDM_V1_ACTIVITY_NAMES)
        self.assertEqual(train.X.shape, (24, 3, 4))
        self.assertEqual(test.X.shape, (6, 3, 4))
        self.assertTrue(
            set(train.y_subject.tolist()).isdisjoint(test.y_subject.tolist())
        )
        for data in [train, test]:
            self.assertEqual(data.X.dtype, torch.float32)
            self.assertEqual(data.y_subject.dtype, torch.int64)
            for x, label in zip(data.X, data.y_subject):
                self.assertEqual(int(x[0, 0]) // 100, ids[int(label)])
                self.assertTrue(torch.equal(x[0, 1:] - x[0, :-1], torch.ones(3)))
        self.assertEqual(len(next(iter(build_loader(train, 3, False)))), 3)

    def test_subject_split_is_deterministic_and_independent_of_row_order(self):
        a, b = _wisdm_subject_split([33, 1, 2, 9, 4, 1])
        c, d = _wisdm_subject_split([1, 2, 4, 9, 33])
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)
        self.assertFalse(set(a) & set(b))
        with self.assertRaisesRegex(ValueError, "at least two"):
            _wisdm_subject_split([1, 1])

    def test_parser_preserves_integer_nanoseconds_and_breaks_at_invalid_record(self):
        base = 9_007_199_254_740_992
        path = self.raw(
            f"1,Walking,{base},1,2,3;1,Walking,{base + 1},2,3,4;\n"
            f"1,Walking,{base + 2},nan,0,0;\n1,Walking,{base + 3},3,4,5;\n"
        )
        with self.assertWarnsRegex(RuntimeWarning, "excluded 1"):
            df = _read_wisdm_v1(path)
        self.assertEqual(int(df.timestamp.iloc[1] - df.timestamp.iloc[0]), 1)
        self.assertEqual(
            df.attrs["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
        )
        self.assertEqual(df.attrs["invalid_records"], 1)
        data = _window_wisdm(df, 3, 1, WISDM_V1_ACTIVITY_NAMES, np.array([1]), 1e9)
        self.assertEqual(data.X.shape, (0, 3, 3))

    def test_missing_or_foreign_identity_and_unknown_activity_are_rejected(self):
        for record, message in [
            (" ,Walking,100,1,2,3;", "subject ID"),
            ("1600,Walking,100,1,2,3;", "subject ID"),
            ("1,A,100,1,2,3;", "activity"),
        ]:
            with (
                self.subTest(record=record),
                self.assertRaisesRegex(ValueError, message),
            ):
                _read_wisdm_v1(self.raw(record))

    def test_raw_cache_invalidates_on_file_change(self):
        path = self.raw("1,Walking,100,1,2,3;")
        first = _read_wisdm_v1(path)
        self.assertIs(first, _read_wisdm_v1(path))
        path.write_text("1,Walking,100,1,2,3;1,Walking,101,2,3,4;")
        self.assertEqual(len(_read_wisdm_v1(path)), 2)

    def test_legacy_sessions_activities_resets_duplicates_and_gaps_do_not_mix(self):
        records = []
        # Six two-sample runs; a three-sample window must never survive.
        for times, activity, session in [
            ([0, 0.05], 0, 0),
            ([0.1, 0.15], 1, 0),
            ([0.2, 0.25], 1, 1),
            ([0, 0.05], 1, 1),
            ([0.05, 0.1], 1, 1),
            ([3, 3.05], 1, 1),
        ]:
            records.extend(
                [(i, 0, 0, t, activity, session) for i, t in enumerate(times)]
            )
        df = pd.DataFrame(
            records,
            columns=["acc_x", "acc_y", "acc_z", "timestamp", "Activity", "session"],
        )
        df.to_csv(self.root / "wisdm" / "train.csv", index=False)
        data = load_windows("wisdm", "train", 3, 1, str(self.root))
        self.assertEqual(data.X.shape, (0, 3, 3))
        self.assertIsNone(data.y_subject)
        self.assertEqual(data.subject_ids, [])
        data = load_windows("wisdm", "train", 2, 1, str(self.root))
        self.assertEqual(len(data.X), 6)
        self.assertEqual(len(next(iter(build_loader(data, 2, False)))), 2)

    def test_removed_source_rows_cannot_stitch_same_subject_runs(self):
        df = pd.DataFrame(
            {
                "acc_x": [1, 2, 3, 4],
                "acc_y": [0] * 4,
                "acc_z": [0] * 4,
                "timestamp": [0, 0.05, 0.2, 0.25],
                "Activity": [0] * 4,
                "subject": [1] * 4,
                "_source_row": [0, 1, 4, 5],
            }
        )
        data = _window_wisdm(df, 3, 1, WISDM_V1_ACTIVITY_NAMES, np.array([1]))
        self.assertEqual(len(data.X), 0)

    def test_invalid_parameters_and_missing_raw_source_fail_explicitly(self):
        for size, stride in [(0, 1), (3, 0), (-1, 2)]:
            with self.assertRaisesRegex(ValueError, "positive"):
                load_windows("wisdm", "train", size, stride, str(self.root))
        with self.assertRaisesRegex(FileNotFoundError, "no recoverable subject"):
            load_windows("wisdm_v1", "train", 3, 1, str(self.root))


if __name__ == "__main__":
    unittest.main()
