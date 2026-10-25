"""Synthetic-only contract tests; no public evidence or signal files are read."""

from __future__ import annotations

import argparse
import builtins
import copy
import gzip
import hashlib
import importlib.util
import io
import json
import mmap
import os
import resource
import signal
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any
from unittest import mock

from scripts import santos_support_inventory as inventory

SYNTHETIC_REFERENCE = {
    "historical_windows": 87,
    "continuity_breaks": 1,
    "historical_tail_rows": 1441,
}


def encoded(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def hashed(value: Any) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def synthetic_record(
    subject: int, day: int, trial: int, lengths: list[int]
) -> dict[str, Any]:
    segments, breaks = [], []
    row, frame = 0, 1000
    for index, length in enumerate(lengths):
        if index:
            breaks.append(
                {
                    "row": row,
                    "previous_frame": frame - 11,
                    "frame": frame,
                    "reason": "gap",
                }
            )
        segments.append(
            {
                "start_row": row,
                "stop_row": row + length,
                "frame_start": frame,
                "frame_end": frame + length - 1,
                "rows": length,
                "windows": length // 256,
                "covered_rows": length // 256 * 256,
                "tail_rows": length % 256,
                "stride_gap_rows": 0,
            }
        )
        row += length
        frame += length + 10
    record = {
        "subject": subject,
        "day": day,
        "trial": trial,
        "member": f"A multi-sensor human gait dataset/processed_data/user{subject:02d}/capture_user{subject:02d}_day{day}_{trial:04d}_imu_walk.csv",
        "sha256": hashlib.sha256(
            f"synthetic-metadata-{subject}-{day}-{trial}".encode()
        ).hexdigest(),
        "uncompressed_bytes": 123,
        "recording_key": [7, subject, day, trial, "MCU", "processed_data"],
        "header": list(inventory.HEADER),
        "source_rows": row,
        "valid_rows": row,
        "invalid_rows": 0,
        "invalid_row_offsets": [],
        "continuity_breaks": breaks,
        "segments": segments,
    }
    for name in ("windows", "covered_rows", "tail_rows", "stride_gap_rows"):
        record[name] = sum(segment[name] for segment in segments)
    return record


def reseal_members(development: dict[str, Any]) -> None:
    source = development["source_coverage"]
    records = source["recordings"]
    fields = (
        "subject",
        "day",
        "trial",
        "member",
        "sha256",
        "uncompressed_bytes",
        "recording_key",
    )
    digest = hashed([{key: row[key] for key in fields} for row in records])
    source["members_sha256"] = digest
    development["data"]["members_sha256"] = digest
    development["data"]["members"] = [
        {key: row[key] for key in ("member", "subject", "day", "trial", "sha256")}
        for row in records
    ]


def fixture_graph(ids: list[int], trials: tuple[int, ...] = (1,)) -> dict[str, Any]:
    subjects = sorted(ids)
    pairs = [
        (probe, trial, claim)
        for probe in subjects
        for trial in trials
        for claim in subjects
    ]
    return {
        "labels": [int(probe == claim) for probe, _, claim in pairs],
        "claim_subject_ids": [claim for _, _, claim in pairs],
        "probe_subject_ids": [probe for probe, _, _ in pairs],
        "probe_trial_ids": [trial for _, trial, _ in pairs],
        "enrollment_trials": {str(subject): list(trials) for subject in subjects},
        "probe_trials": {str(subject): list(trials) for subject in subjects},
        "n_genuine": len(subjects) * len(trials),
        "n_impostor": len(subjects) * (len(subjects) - 1) * len(trials),
    }


def fixture(*, full: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    order = list(inventory.HISTORICAL_ORDER)
    partitions = (
        ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11), (12, 13, 14)),
        ((0, 5, 10), (1, 6, 11), (2, 7, 12), (3, 8, 13), (4, 9, 14)),
        ((0, 6, 12), (1, 7, 13), (2, 8, 14), (3, 9, 10), (4, 5, 11)),
        ((0, 7, 14), (1, 8, 10), (2, 9, 11), (3, 5, 12), (4, 6, 13)),
    )
    folds: list[dict[str, Any]] = []
    for partition in partitions:
        triples = [[order[i] for i in group] for group in partition]
        for position, validation in enumerate(triples):
            inner = []
            for offset in range(1, 5):
                held = triples[(position + offset) % 5]
                inner.append(
                    {
                        "held_ids": held,
                        "train_ids": [s for s in order if s not in validation + held],
                    }
                )
            folds.append(
                {
                    "fold": len(folds),
                    "validation_ids": validation,
                    "refit_ids": [s for s in order if s not in validation],
                    "inner": inner,
                }
            )
    old = {
        "plan": {
            "protocol": "santos_dev_canonical_orientation_v1",
            "payload_allowlist": order,
            "window": {"samples": 256, "stride": 256, "sampling_rate_hz": 100},
            "folds": folds,
        },
        "python": "synthetic",
        "versions": {},
        "sources": {"synthetic-source.py": "a" * 64},
    }
    records = [
        synthetic_record(
            s,
            day,
            1,
            [400, 400]
            if (s, day) == (1, 2)
            else [257]
            if (s, day) == (19, 2)
            else [800],
        )
        for s in inventory.DEV_IDS
        for day in (1, 2)
    ]
    records += [synthetic_record(1, 1, 2, [255]), synthetic_record(1, 2, 2, [1])]
    if full:
        # All 300 slots; 328 segments, 588 windows, 28 gaps and 33,810 tail rows.
        # This scales graph/support serialization without accessing public metadata.
        records = []
        segment_index = 0
        domain = [
            (s, d, t) for s in inventory.DEV_IDS for d in (1, 2) for t in range(1, 11)
        ]
        for index, (subject, day, trial) in enumerate(domain):
            base = [256, 256] if index < 28 else [512] if index < 288 else [256]
            lengths = []
            for length in base:
                lengths.append(length + (104 if segment_index < 26 else 103))
                segment_index += 1
            records.append(synthetic_record(subject, day, trial, lengths))
    records.sort(key=lambda r: (r["subject"], r["day"], r["trial"]))
    present = {(r["subject"], r["day"], r["trial"]) for r in records}
    source = {
        "subject_ids": list(inventory.DEV_IDS),
        "dataset_doi": "10.6084/m9.figshare.14727231.v7",
        "article_id": 14727231,
        "version": 7,
        "file_id": 35882624,
        "sampling_rate_hz": 100,
        "window_size": 256,
        "stride": 256,
        "source_header": list(inventory.HEADER),
        "channels": inventory.HEADER[1:],
        "channel_names": inventory.HEADER[1:],
        "channel_units": ["m/s^2"] * 3 + ["degrees/sec"] * 3,
        "archive_path": "/forbidden/never-open-this.zip",
        "recordings": records,
        "exclusions": {
            "missing_recordings": [
                {"subject": s, "day": d, "trial": t}
                for s in inventory.DEV_IDS
                for d in (1, 2)
                for t in range(1, 11)
                if (s, d, t) not in present
            ],
            "invalid_rows": 0,
            "continuity_breaks": sum(len(r["continuity_breaks"]) for r in records),
            "tail_rows": sum(r["tail_rows"] for r in records),
            "stride_gap_rows": 0,
            "zero_window_recordings": [
                i for i, r in enumerate(records) if r["windows"] == 0
            ],
        },
        "subject_coverage": [],
    }
    for subject in inventory.DEV_IDS:
        days = {
            str(day): {
                "present_trials": [
                    r["trial"]
                    for r in records
                    if r["subject"] == subject and r["day"] == day
                ],
                "eligible_trials": [
                    r["trial"]
                    for r in records
                    if r["subject"] == subject and r["day"] == day and r["windows"] > 0
                ],
                "windows": sum(
                    r["windows"]
                    for r in records
                    if r["subject"] == subject and r["day"] == day
                ),
            }
            for day in (1, 2)
        }
        source["subject_coverage"].append(
            {"subject": subject, "days": days, "eligible_both_days": True}
        )
    graph_coverage = {}
    graph_trials = tuple(range(1, 11)) if full else (1,)
    for fold in folds:
        inner = [
            fixture_graph(owner["held_ids"], graph_trials) for owner in fold["inner"]
        ]
        pool: dict[str, Any] = {
            name: [value for graph in inner for value in graph[name]]
            for name in inventory.CLAIM_COLUMNS
        }
        pool.update(
            n_genuine=sum(graph["n_genuine"] for graph in inner),
            n_impostor=sum(graph["n_impostor"] for graph in inner),
            directed_pairs=sorted(
                {
                    f"{c}<-{p}"
                    for graph in inner
                    for c, p in zip(
                        graph["claim_subject_ids"],
                        graph["probe_subject_ids"],
                        strict=True,
                    )
                    if c != p
                }
            ),
        )
        graph_coverage[str(fold["fold"])] = {
            "inner": inner,
            "pool": pool,
            "validation": fixture_graph(fold["validation_ids"], graph_trials),
            "cohort_inner": [
                fixture_graph(o["train_ids"], graph_trials) for o in fold["inner"]
            ],
            "cohort_refit": fixture_graph(fold["refit_ids"], graph_trials),
        }
    row_keys = [
        (r["subject"], r["day"], r["trial"], start)
        for r in records
        for s in r["segments"]
        for start in range(s["start_row"], s["stop_row"] - 255, 256)
    ]
    row_digest = hashlib.sha256(encoded(["int64", [len(row_keys), 4]]))
    for key in row_keys:
        row_digest.update(struct.pack("<4q", *key))
    development = {
        "source_coverage": source,
        "data": {"data_sha256": "b" * 64},
        "coverage": graph_coverage,
        "registration_sha256": hashlib.sha256(encoded(old) + b"\n").hexdigest(),
        "geometry": {
            "clean": {
                "row_key_columns": ["subject", "day", "trial", "start"],
                "rows": len(row_keys),
                "row_keys_sha256": row_digest.hexdigest(),
            }
        },
    }
    reseal_members(development)
    return old, development


class NumericContract(unittest.TestCase):
    def test_canonical_bytes_and_fingerprints_keep_the_existing_json_encoding(
        self,
    ) -> None:
        cases = (
            (None, b"null"),
            (False, b"false"),
            (True, b"true"),
            (1, b"1"),
            (1.0, b"1.0"),
            (0.0, b"0.0"),
            (-0.0, b"-0.0"),
            (2**100, b"1267650600228229401496703205376"),
            (1e-7, b"1e-07"),
            (
                {"z": [True, None, 1.0, -0.0], "a": {"\u00e9": 'line\n"'}},
                b'{"a":{"\\u00e9":"line\\n\\""},"z":[true,null,1.0,-0.0]}',
            ),
        )
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(inventory.canonical(value), expected)
                self.assertEqual(
                    inventory.fingerprint(value), hashlib.sha256(expected).hexdigest()
                )

    def test_direct_equivalence_is_exact_and_ignores_only_object_key_order(
        self,
    ) -> None:
        values = [None, False, 0, 0.0, -0.0, True, 1, 1.0, 2**63 - 1, 2**100, "1"]
        for actual in values:
            for expected in values:
                # An independent byte oracle, rather than Python's coercing equality.
                left, right = {"x": [actual], "a": {}}, {"a": {}, "x": [expected]}
                with self.subTest(actual=repr(actual), expected=repr(expected)):
                    if encoded(left) == encoded(right):
                        inventory.same(left, right, "numeric equivalence")
                    else:
                        with self.assertRaisesRegex(ValueError, "numeric equivalence"):
                            inventory.same(left, right, "numeric equivalence")
        for actual, expected in (([1, 2], [2, 1]), ({"a": 1}, {"b": 1}), ([], {})):
            with (
                self.subTest(actual=actual, expected=expected),
                self.assertRaises(ValueError),
            ):
                inventory.same(actual, expected, "structure")

    def test_hashing_explicitly_checks_the_decoded_roundtrip(self) -> None:
        for original, changed in (
            (1, 1.0),
            (False, 0),
            (-0.0, 0.0),
            (2**100, 2**100 - 1),
        ):
            with (
                self.subTest(original=repr(original)),
                mock.patch.object(
                    inventory.json, "loads", return_value=changed
                ) as decode,
                self.assertRaisesRegex(ValueError, "type-preserving roundtrip"),
            ):
                inventory.fingerprint(original)
            decode.assert_called_once()

    def test_equivalence_validates_both_exact_json_domains(self) -> None:
        class IntSubclass(int):
            pass

        class StrSubclass(str):
            pass

        class ListSubclass(list):
            pass

        invalid = (
            {1: "key"},
            {StrSubclass("x"): 1},
            IntSubclass(1),
            ListSubclass([1]),
            (1,),
            b"bytes",
            {"x": float("inf")},
            {"x": float("nan")},
        )
        for value in invalid:
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError):
                    inventory.canonical(value)
                for actual, expected in ((value, {}), ({}, value)):
                    with self.assertRaises(ValueError):
                        inventory.same(actual, expected, "invalid domain")

    def test_json_hash_preserves_types_signed_zero_and_large_integers(self) -> None:
        values = [0, False, 0.0, -0.0, 2**63 - 1, 2**63 - 2]
        self.assertEqual(
            len({inventory.fingerprint(value) for value in values}), len(values)
        )
        self.assertEqual(
            inventory.fingerprint({"x": [1, 2]}), inventory.fingerprint({"x": [1, 2]})
        )
        for value in (
            {1: "coerced key"},
            {"x": {False: 1}},
            (1, 2),
            {"x": float("nan")},
            float("inf"),
        ):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                inventory.fingerprint(value)

    def test_exact_integer_and_ambiguous_json_rejections(self) -> None:
        for value in (True, False, 256.0, -0.0, "256"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                inventory.integer(value, "test")
        for payload in (
            b'{"x":1,"x":2}',
            b'{"x":NaN}',
            b'{"x":Infinity}',
            b'{"x":1e999}',
            b'{"x":-0}',
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                inventory.decode_metadata(payload, False)
        value, _ = inventory.decode_metadata(
            b'{"x":9223372036854775807,"z":-0.0}', False
        )
        self.assertEqual(value["x"], 2**63 - 1)
        self.assertEqual(
            inventory.canonical(value), b'{"x":9223372036854775807,"z":-0.0}'
        )

    def test_nonadaptive_exact_length_summary(self) -> None:
        self.assertEqual(
            inventory.length_summary([1, 800])["median"],
            {"numerator": 801, "denominator": 2},
        )
        self.assertEqual(
            inventory.length_summary([768, 1, 256])["median"],
            {"numerator": 256, "denominator": 1},
        )
        self.assertIsNone(inventory.length_summary([])["minimum"])
        invalid_lengths: tuple[Any, ...] = (True, False, 1.0, -0.0, 0, -1, "1")
        for value in invalid_lengths:
            with self.subTest(value=value), self.assertRaises(ValueError):
                inventory.length_summary([value])


class SupportContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.old, cls.dev = fixture()
        with mock.patch.object(inventory, "KNOWN_RECONCILIATION", SYNTHETIC_REFERENCE):
            cls.result = inventory.derive_inventory(cls.old, cls.dev)

    def test_400_plus_400_cannot_become_one_768_block(self) -> None:
        trials = {
            (r["subject"], r["day"], r["trial"]): r for r in self.result["trials"]
        }
        split, whole = trials[1, 2, 1], trials[3, 2, 1]
        self.assertEqual(split["valid_rows"], whole["valid_rows"])
        self.assertTrue(split["fragmented"])
        self.assertTrue(whole["unbroken"])
        self.assertEqual(split["support"]["fixed768"]["unit_count"], 0)
        self.assertEqual(
            whole["support"]["fixed768"],
            {
                "unit_count": 1,
                "unique_original_row_coverage": 768,
                "unused_valid_rows": 32,
            },
        )
        self.assertEqual(split["support"]["historical256"]["unit_count"], 2)
        self.assertEqual(split["support"]["complete_segment"]["unit_count"], 2)

    def test_complete_segments_recover_tails_without_claiming_features(self) -> None:
        tiny = next(
            s
            for s in self.result["segments"]
            if (s["subject"], s["day"], s["trial"]) == (1, 2, 2)
        )
        self.assertEqual(tiny["length"], 1)
        self.assertEqual(tiny["support"]["historical256"]["unused_valid_rows"], 1)
        self.assertEqual(
            tiny["support"]["complete_segment"]["unique_original_row_coverage"], 1
        )
        self.assertEqual(tiny["support"]["complete_segment"]["unit_count"], 1)
        self.assertFalse(tiny["length_eligibility"]["at_least_256_rows"])
        self.assertEqual(
            tiny["length_eligibility"]["extraction_or_trainability"], "unassessed"
        )
        self.assertEqual(
            self.result["totals"]["complete_segment"]["unused_valid_rows"], 0
        )
        self.assertEqual(
            self.result["subject19_day2_trial1"]["support"]["historical256"][
                "unused_valid_rows"
            ],
            1,
        )

    def test_no_graph_expansion_and_original_failures_remain_ordered(self) -> None:
        self.assertEqual(
            [
                (r["subject"], r["day"], r["trial"])
                for r in self.result["source_present_absent_original_graph"]
            ],
            [(1, 1, 2), (1, 2, 2)],
        )
        self.assertEqual(self.result["original_graphs"], self.dev["coverage"])
        self.assertEqual(self.result["original_folds"], self.old["plan"]["folds"])
        self.assertEqual(len(self.result["trials"]), 300)
        missing = next(
            r
            for r in self.result["trials"]
            if (r["subject"], r["day"], r["trial"]) == (1, 2, 3)
        )
        self.assertEqual(missing["status"], "missing_as_frozen_metadata")
        self.assertTrue(all(missing["zero_units"].values()))
        found = False
        for fold in self.result["folds"]:
            support = fold["original_graph_support"]["validation"]["fixed768"]
            graph = self.dev["coverage"][str(fold["fold"])]["validation"]
            flags = support["any_original_enrollment_trial_and_probe_raw_support"]
            self.assertEqual(len(flags), len(graph["labels"]))
            self.assertEqual(support["original_n_genuine"], 3)
            self.assertEqual(support["original_n_impostor"], 6)
            for i, probe in enumerate(graph["probe_subject_ids"]):
                if probe in (1, 19):
                    self.assertFalse(flags[i])
                    found = True
        self.assertTrue(found)

    def test_all_15_same_20_folds_and_train_owner_gates(self) -> None:
        self.assertEqual(self.result["dev_ids"], list(inventory.DEV_IDS))
        self.assertEqual(len(self.result["folds"]), 20)
        for fold in self.result["folds"]:
            self.assertEqual(len(fold["inner_train_owners"]), 4)
            for owner in fold["inner_train_owners"] + [fold["refit_train_owner"]]:
                self.assertIn(len(owner["train_ids"]), (9, 12))
                for definition in inventory.DEFINITIONS:
                    entry = owner["definitions"][definition]
                    self.assertEqual(
                        [p["subject"] for p in entry["per_identity"]],
                        owner["train_ids"],
                    )
                    for p in entry["per_identity"]:
                        self.assertEqual(
                            p["unit_count"], sum(p["day_unit_counts"].values())
                        )
                        self.assertEqual(
                            p["proposed_min_two_units"], p["unit_count"] >= 2
                        )
                    self.assertEqual(
                        entry["support"]["unit_count"],
                        sum(p["unit_count"] for p in entry["per_identity"]),
                    )
        self.assertFalse(self.result["flags"]["fixed768"]["all_identity_support"])
        self.assertTrue(self.result["flags"]["fixed768"]["some_identity_support"])
        self.assertFalse(
            self.result["flags"]["fixed768"]["complete_historical_trial_support"]
        )

    def test_exact_historical_tile_binding_and_row_hash_link(self) -> None:
        windows = self.result["original_windows"]
        self.assertIn("verified historical", self.result["historical_row_key_link"])
        for segment in self.result["segments"]:
            for block in segment["support"]["fixed768"]["blocks"]:
                matching = [windows[i] for i in block["historical_window_indices"]]
                self.assertEqual(len(matching), 3)
                self.assertEqual(
                    [w["rows"][0] for w in matching],
                    [block["rows"][0] + n * 256 for n in range(3)],
                )
                self.assertEqual(matching[-1]["rows"][1], block["rows"][1])
        self.assertEqual(
            self.result["metadata_reconciliation"]["observed"], SYNTHETIC_REFERENCE
        )
        self.assertFalse(
            self.result["integrity_limits"]["member_payload_hashes_verified"]
        )

    def test_input_objects_remain_unchanged(self) -> None:
        old, development = fixture()
        before = encoded([old, development])
        with mock.patch.object(inventory, "KNOWN_RECONCILIATION", SYNTHETIC_REFERENCE):
            inventory.derive_inventory(old, development)
        self.assertEqual(encoded([old, development]), before)

    def test_empty_enrollment_has_no_any_or_all_raw_support(self) -> None:
        graph = fixture_graph([1])
        graph["enrollment_trials"]["1"] = []
        lookup = {(1, 2, 1): {"support": {"historical256": {"unit_count": 1}}}}
        support = inventory._claim_support(graph, lookup, "historical256")
        self.assertEqual(
            support["any_original_enrollment_trial_and_probe_raw_support"], [False]
        )
        self.assertEqual(
            support["all_original_enrollment_trials_and_probe_raw_support"], [False]
        )
        # This helper guard does not admit such a graph to the registered protocol.
        with self.assertRaisesRegex(ValueError, "both-day trial metadata"):
            inventory._graph_projection([1], {(1, 1): [], (1, 2): [1]})


class MalformedMetadata(unittest.TestCase):
    def check_mutation(self, mutate: Any, pattern: str = "STOP") -> None:
        old, development = fixture()
        mutate(old, development)
        with (
            mock.patch.object(inventory, "KNOWN_RECONCILIATION", SYNTHETIC_REFERENCE),
            self.assertRaisesRegex(ValueError, pattern),
        ):
            inventory.derive_inventory(old, development)

    def test_cohort_folds_and_order_are_exact(self) -> None:
        mutations = [
            lambda r, d: d["source_coverage"]["subject_ids"].pop(),
            lambda r, d: r["plan"]["payload_allowlist"].reverse(),
            lambda r, d: r["plan"]["folds"].pop(),
            lambda r, d: r["plan"]["folds"][0]["inner"][0]["train_ids"].reverse(),
            lambda r, d: r["plan"]["folds"][0].__setitem__("fold", False),
            lambda r, d: d["coverage"]["0"]["validation"][
                "claim_subject_ids"
            ].reverse(),
            lambda r, d: d["coverage"]["0"]["pool"].__setitem__("n_genuine", 12.0),
            lambda r, d: d["coverage"].pop("0"),
            lambda r, d: d.pop("source_coverage"),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                self.check_mutation(mutate)

    def test_conflicting_paths_ids_days_trials_duplicates_and_hashes(self) -> None:
        paths = [
            "../user01.csv",
            "A multi-sensor human gait dataset/processed_data/user03/capture_user01_day1_0001_imu_walk.csv",
            "A multi-sensor human gait dataset/processed_data/user01/capture_user01_day3_0001_imu_walk.csv",
            "A multi-sensor human gait dataset/processed_data/user01/capture_user01_day1_0011_imu_walk.csv",
        ]
        for path in paths:
            with self.subTest(path=path):
                self.check_mutation(
                    lambda r, d, p=path: d["source_coverage"]["recordings"][
                        0
                    ].__setitem__("member", p)
                )
        for field, value in (
            ("subject", True),
            ("day", 2),
            ("trial", 2),
            ("sha256", "A" * 64),
            ("uncompressed_bytes", 123.0),
        ):
            with self.subTest(field=field):
                self.check_mutation(
                    lambda r, d, f=field, v=value: d["source_coverage"]["recordings"][
                        0
                    ].__setitem__(f, v)
                )
        self.check_mutation(
            lambda r, d: d["source_coverage"]["recordings"].insert(
                0, copy.deepcopy(d["source_coverage"]["recordings"][0])
            ),
            "duplicate",
        )
        self.check_mutation(
            lambda r, d: d["source_coverage"].__setitem__("members_sha256", "c" * 64),
            "member manifest hash",
        )
        self.check_mutation(
            lambda r, d: d["data"]["members"].reverse(), "ordered member projection"
        )

    def test_frame_interval_break_and_count_malformations(self) -> None:
        for field, value in (
            ("frame_start", 1000.0),
            ("frame_end", 2**63),
            ("start_row", -1),
            ("stop_row", 799),
            ("rows", True),
            ("windows", 4),
            ("covered_rows", 767),
            ("tail_rows", -0.0),
        ):
            with self.subTest(field=field):
                self.check_mutation(
                    lambda r, d, f=field, v=value: d["source_coverage"]["recordings"][
                        0
                    ]["segments"][0].__setitem__(f, v)
                )
        for field, value in (
            ("source_rows", 801),
            ("invalid_row_offsets", [0]),
            ("windows", 3.0),
            ("valid_rows", 799),
        ):
            with self.subTest(field=field):
                self.check_mutation(
                    lambda r, d, f=field, v=value: d["source_coverage"]["recordings"][
                        0
                    ].__setitem__(f, v)
                )

        def wrong_break(_r: Any, d: Any) -> None:
            split = next(
                x for x in d["source_coverage"]["recordings"] if x["continuity_breaks"]
            )
            split["continuity_breaks"][0]["reason"] = "reset"

        self.check_mutation(wrong_break, "continuity-break")
        self.check_mutation(
            lambda r, d: d["geometry"]["clean"].__setitem__(
                "row_keys_sha256", "f" * 64
            ),
            "row-key hash",
        )
        self.check_mutation(
            lambda r, d: d["source_coverage"]["exclusions"]["missing_recordings"].pop(),
            "source exclusions",
        )

    def test_invalid_rows_partition_and_signed_int64_endpoints(self) -> None:
        record = synthetic_record(1, 1, 1, [1, 1])
        record.update(
            source_rows=3,
            valid_rows=2,
            invalid_rows=1,
            invalid_row_offsets=[1],
            continuity_breaks=[],
        )
        first, second = record["segments"]
        first.update(frame_start=-(2**63), frame_end=-(2**63))
        second.update(
            start_row=2, stop_row=3, frame_start=2**63 - 1, frame_end=2**63 - 1
        )
        inventory.validate_recording(record)
        record["invalid_row_offsets"] = [1, 1]
        with self.assertRaises(ValueError):
            inventory.validate_recording(record)

    def test_consistent_counts_do_not_excuse_overlaps_or_unexplained_splits(
        self,
    ) -> None:
        record = synthetic_record(1, 1, 1, [400, 400])
        second = record["segments"][1]
        second.update(start_row=399, stop_row=799)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            inventory.validate_recording(record)
        record = synthetic_record(1, 1, 1, [400, 400])
        second = record["segments"][1]
        second.update(frame_start=1400, frame_end=1799)
        record["continuity_breaks"] = []
        with self.assertRaisesRegex(ValueError, "unexplained split"):
            inventory.validate_recording(record)

    def test_reconciliation_is_a_hard_stop_not_an_observation_only(self) -> None:
        old, development = fixture()
        with (
            mock.patch.object(
                inventory,
                "KNOWN_RECONCILIATION",
                {**SYNTHETIC_REFERENCE, "historical_windows": 88},
            ),
            self.assertRaisesRegex(ValueError, "reconciliation reference"),
        ):
            inventory.derive_inventory(old, development)


class AttemptContract(unittest.TestCase):
    def setUp(self) -> None:
        # Behavior/schema tests share the scientific suite's interpreter. The real
        # irreversible limiter is exercised only by fresh full-cohort workers.
        self.limiter = self.enterContext(mock.patch.object(inventory, "enforce_limits"))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.old, self.dev = fixture()
        self.registration_input = self.base / "historical.json.gz"
        self.development_input = self.base / "development.json.gz"
        self.registration_input.write_bytes(
            gzip.compress(encoded(self.old) + b"\n", mtime=0)
        )
        self.development_input.write_bytes(
            gzip.compress(encoded(self.dev) + b"\n", mtime=0)
        )
        self.args = argparse.Namespace(
            out_root=self.base / "attempt",
            historical_registration_json=self.registration_input,
            development_metadata_json=self.development_input,
        )
        self.reference = mock.patch.object(
            inventory, "KNOWN_RECONCILIATION", SYNTHETIC_REFERENCE
        )
        self.reference.start()
        self.addCleanup(self.reference.stop)

    def read_json(self, name: str) -> dict[str, Any]:
        return json.loads((self.args.out_root / name).read_bytes())

    def test_run_and_verify_call_the_production_limiter(self) -> None:
        inventory.run(self.args)
        self.limiter.assert_called_once_with()
        self.limiter.reset_mock()
        inventory.verify(self.args)
        self.limiter.assert_called_once_with()

    def test_freeze_before_parse_and_derive_then_saved_only_verify(self) -> None:
        real_decode, real_derive = inventory.decode_metadata, inventory.derive_inventory
        checked = []

        def checked_decode(payload: bytes, compressed: bool) -> Any:
            frozen = self.read_json("registration.json")
            self.assertEqual(frozen["sources"], inventory.source_hashes())
            self.assertEqual(frozen["runtime"], inventory.runtime())
            self.assertEqual(frozen["plan"]["flags"], inventory.plan()["flags"])
            self.assertEqual(
                frozen["inputs"]["development_metadata"]["sha256"],
                hashlib.sha256(self.development_input.read_bytes()).hexdigest(),
            )
            checked.append("decode")
            return real_decode(payload, compressed)

        def checked_derive(old: Any, development: Any) -> Any:
            self.assertTrue((self.args.out_root / "attempt.json").exists())
            self.assertTrue((self.args.out_root / "registration.json").exists())
            checked.append("derive")
            return real_derive(old, development)

        with (
            mock.patch.object(inventory, "decode_metadata", side_effect=checked_decode),
            mock.patch.object(
                inventory, "derive_inventory", side_effect=checked_derive
            ),
        ):
            inventory.run(self.args)
        self.assertEqual(checked, ["decode", "decode", "derive"])
        self.registration_input.unlink()
        self.development_input.unlink()
        self.assertEqual(inventory.verify(self.args)["status"], "verified")

    def test_import_fit_archive_and_unrelated_read_forbidden_spies(self) -> None:
        real_import = builtins.__import__
        forbidden = ("numpy", "torch", "data", "models", "engine", "zipfile", "tarfile")

        def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.split(".")[0] in forbidden or (
                name.startswith("scripts.")
                and not name.endswith("santos_support_inventory")
            ):
                raise AssertionError("forbidden import: " + name)
            return real_import(name, *args, **kwargs)

        allowed = {inventory.ROOT / name for name in inventory.SOURCES} | {
            self.registration_input,
            self.development_input,
        }
        real_open = Path.open

        def guarded_open(path: Path, *args: Any, **kwargs: Any) -> Any:
            if path not in allowed and path.parent != self.args.out_root:
                raise AssertionError("forbidden file access: " + str(path))
            return real_open(path, *args, **kwargs)

        # Execute the source in a fresh module namespace while imports are guarded.
        spec = importlib.util.spec_from_file_location(
            "inventory_import_spy", inventory.__file__
        )
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        with mock.patch("builtins.__import__", side_effect=guarded_import):
            assert spec.loader is not None
            spec.loader.exec_module(module)
            with (
                mock.patch.object(Path, "open", guarded_open),
                mock.patch(
                    "builtins.open",
                    side_effect=AssertionError("unexpected builtins.open"),
                ),
            ):
                inventory.run(self.args)
                inventory.verify(self.args)

    def test_missing_fact_fails_consumes_attempt_and_disallows_retry(self) -> None:
        self.dev.pop("coverage")
        self.development_input.write_bytes(gzip.compress(encoded(self.dev), mtime=0))
        with self.assertRaisesRegex(ValueError, "missing necessary metadata fact"):
            inventory.run(self.args)
        self.assertTrue((self.args.out_root / "registration.json").exists())
        self.assertEqual(
            self.read_json("diagnostics.json")["status"], "failed_consumed"
        )
        before = {p.name: p.read_bytes() for p in self.args.out_root.iterdir()}
        with self.assertRaises(FileExistsError):
            inventory.run(self.args)
        self.assertEqual(
            before, {p.name: p.read_bytes() for p in self.args.out_root.iterdir()}
        )
        with self.assertRaisesRegex(ValueError, "failed attempt consumed"):
            inventory.verify(self.args)

    def test_historical_registration_byte_link_refuses_before_deriving(self) -> None:
        self.registration_input.write_bytes(
            gzip.compress(encoded(self.old) + b"\n\n", mtime=0)
        )
        with mock.patch.object(inventory, "derive_inventory") as derive:
            with self.assertRaisesRegex(ValueError, "uncompressed byte hash link"):
                inventory.run(self.args)
            derive.assert_not_called()

    def test_source_runtime_or_flag_changes_refuse_before_deriving(self) -> None:
        inventory.run(self.args)
        changed_runtime = {**inventory.runtime(), "python": "changed"}
        changed_sources = {**inventory.source_hashes(), inventory.SOURCES[0]: "e" * 64}
        changed_plan = copy.deepcopy(inventory.plan())
        changed_plan["flags"]["train_gate"] = "changed"
        for name, value in (
            ("runtime", changed_runtime),
            ("source_hashes", changed_sources),
            ("plan", changed_plan),
        ):
            with (
                self.subTest(name=name),
                mock.patch.object(inventory, name, return_value=value),
                mock.patch.object(inventory, "derive_inventory") as derive,
            ):
                with self.assertRaises(ValueError):
                    inventory.verify(self.args)
                derive.assert_not_called()

    def test_input_tampering_and_result_tampering_are_refused(self) -> None:
        inventory.run(self.args)
        result = self.read_json("result.json")
        result["inventory"]["flags"]["fixed768"]["all_identity_support"] = True
        (self.args.out_root / "result.json").write_bytes(encoded(result))
        with mock.patch.object(inventory, "derive_inventory") as derive:
            with self.assertRaisesRegex(ValueError, "result byte hash"):
                inventory.verify(self.args)
            derive.assert_not_called()
        # Even updating the diagnostic hash cannot make a changed result replay.
        diagnostics = self.read_json("diagnostics.json")
        diagnostics["result_sha256"] = hashlib.sha256(encoded(result)).hexdigest()
        (self.args.out_root / "diagnostics.json").write_bytes(encoded(diagnostics))
        with self.assertRaisesRegex(ValueError, "deterministic result replay"):
            inventory.verify(self.args)
        frozen = self.read_json("registration.json")
        snapshot = (
            self.args.out_root / frozen["inputs"]["development_metadata"]["snapshot"]
        )
        snapshot.write_bytes(snapshot.read_bytes() + b"tampered")
        with mock.patch.object(inventory, "derive_inventory") as derive:
            with self.assertRaisesRegex(ValueError, "exact byte count"):
                inventory.verify(self.args)
            derive.assert_not_called()

    def test_result_schema_and_binding_types_fail_before_loading_or_deriving(
        self,
    ) -> None:
        inventory.run(self.args)
        original = self.read_json("result.json")
        diagnostics = self.read_json("diagnostics.json")
        mutations = (
            lambda r: r.__setitem__("unregistered_field", {"raw_member_reads": 1}),
            lambda r: r.__setitem__("unregistered_field", None),
            lambda r: r.pop("inputs_sha256"),
            lambda r: r.__setitem__("registration_sha256", [r["registration_sha256"]]),
            lambda r: r.__setitem__("inputs_sha256", False),
            lambda r: r.__setitem__("inventory", []),
        )
        for index, mutate in enumerate(mutations):
            result = copy.deepcopy(original)
            mutate(result)
            payload = encoded(result)
            (self.args.out_root / "result.json").write_bytes(payload)
            diagnostics["result_sha256"] = hashlib.sha256(payload).hexdigest()
            (self.args.out_root / "diagnostics.json").write_bytes(encoded(diagnostics))
            with (
                self.subTest(index=index),
                mock.patch.object(inventory, "_load_bound") as load,
                mock.patch.object(inventory, "derive_inventory") as derive,
            ):
                with self.assertRaises(ValueError):
                    inventory.verify(self.args)
                load.assert_not_called()
                derive.assert_not_called()

    def test_entire_replay_rejects_rehashed_nested_type_changes(self) -> None:
        inventory.run(self.args)
        original = self.read_json("result.json")
        diagnostics = self.read_json("diagnostics.json")
        mutations = (
            lambda r: r["inventory"]["integrity_limits"].__setitem__(
                "raw_member_reads", False
            ),
            lambda r: r["inventory"]["integrity_limits"].__setitem__(
                "raw_member_reads", 0.0
            ),
            lambda r: r["inventory"]["integrity_limits"].__setitem__(
                "raw_member_reads", -0.0
            ),
            lambda r: r["inventory"]["flags"]["historical256"].__setitem__(
                "all_identity_support", 1
            ),
        )
        for index, mutate in enumerate(mutations):
            result = copy.deepcopy(original)
            mutate(result)
            payload = encoded(result)
            (self.args.out_root / "result.json").write_bytes(payload)
            diagnostics["result_sha256"] = hashlib.sha256(payload).hexdigest()
            (self.args.out_root / "diagnostics.json").write_bytes(encoded(diagnostics))
            with (
                self.subTest(index=index),
                self.assertRaisesRegex(ValueError, "deterministic result replay"),
            ):
                inventory.verify(self.args)

    def test_missing_input_and_archive_suffix_have_no_fallback(self) -> None:
        self.development_input.unlink()
        with self.assertRaisesRegex(ValueError, "regular nonsymlink file"):
            inventory.run(self.args)
        self.assertEqual(
            self.read_json("diagnostics.json")["status"], "failed_consumed"
        )
        self.assertTrue(self.read_json("registration.json")["preflight_incomplete"])
        self.args.out_root = self.base / "second-attempt"
        self.args.development_metadata_json = self.base / "no-access.zip"
        with (
            mock.patch.object(
                Path, "open", side_effect=AssertionError("archive must not be opened")
            ),
            self.assertRaises(ValueError),
        ):
            inventory.read_file(self.args.development_metadata_json, metadata=True)

    def test_cli_synthetic_run_verify_and_forbidden_arguments(self) -> None:
        # main() also installs a process-wide handler; keep this parser test local.
        self.enterContext(mock.patch.object(inventory.signal, "signal"))
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(
                inventory.main(
                    [
                        "--stage",
                        "run",
                        "--out-root",
                        str(self.args.out_root),
                        "--historical-registration-json",
                        str(self.registration_input),
                        "--development-metadata-json",
                        str(self.development_input),
                    ]
                ),
                0,
            )
            self.assertEqual(
                inventory.main(
                    [
                        "--stage",
                        "verify",
                        "--out-root",
                        str(self.args.out_root),
                    ]
                ),
                0,
            )
        self.assertIn('"status":"verified"', output.getvalue())
        for extra in (
            ["--archive-path", "/forbidden.zip"],
            ["--development-metadata-json", str(self.development_input)],
        ):
            with (
                self.subTest(extra=extra),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                inventory.main(
                    ["--stage", "verify", "--out-root", str(self.args.out_root), *extra]
                )
            self.assertEqual(error.exception.code, 2)

    def test_existing_empty_output_root_is_not_a_fresh_attempt(self) -> None:
        self.args.out_root.mkdir()
        with self.assertRaises(FileExistsError):
            inventory.run(self.args)
        self.assertEqual(list(self.args.out_root.iterdir()), [])


class LimitContract(unittest.TestCase):
    def test_administrative_limits_only_reduce_and_bind_one_cpu(self) -> None:
        with (
            mock.patch.object(os, "sched_getaffinity", return_value={2, 4}),
            mock.patch.object(os, "sched_setaffinity") as affinity,
            mock.patch.object(resource, "getrlimit", return_value=(30, 40)),
            mock.patch.object(resource, "setrlimit") as limits,
            mock.patch.dict(os.environ),
        ):
            inventory.enforce_limits()
            affinity.assert_called_once_with(0, {2})
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "1")
            self.assertEqual(
                limits.call_args_list,
                [
                    mock.call(resource.RLIMIT_CPU, (30, 40)),
                    mock.call(resource.RLIMIT_AS, (30, 40)),
                ],
            )


class FullCohortContract(unittest.TestCase):
    """Real ceilings belong to fresh workers, never the shared test interpreter."""

    def prepare_fixture(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.old, self.dev = fixture(full=True)
        self.registration_input = self.base / "synthetic-registration.json.gz"
        self.development_input = self.base / "synthetic-development.json.gz"
        self.registration_input.write_bytes(
            gzip.compress(encoded(self.old) + b"\n", mtime=0)
        )
        self.development_input.write_bytes(
            gzip.compress(encoded(self.dev) + b"\n", mtime=0)
        )
        self.args = argparse.Namespace(
            out_root=self.base / "synthetic-attempt",
            historical_registration_json=self.registration_input,
            development_metadata_json=self.development_input,
        )

    def assert_real_limits(self) -> None:
        self.assertEqual(resource.getrlimit(resource.RLIMIT_CPU), (59, 60))
        self.assertEqual(
            resource.getrlimit(resource.RLIMIT_AS), (512 * 1024 * 1024,) * 2
        )
        self.assertEqual(len(os.sched_getaffinity(0)), 1)
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            self.assertEqual(os.environ[name], "1")

    def assert_full_result(self) -> None:
        result = json.loads((self.args.out_root / "result.json").read_bytes())
        self.assertEqual(
            set(result), {"registration_sha256", "inputs_sha256", "inventory"}
        )
        observed = result["inventory"]
        self.assertEqual(
            observed["metadata_reconciliation"]["observed"],
            {
                "historical_windows": 588,
                "continuity_breaks": 28,
                "historical_tail_rows": 33810,
            },
        )
        self.assertEqual(len(observed["original_recording_order"]), 300)
        self.assertEqual(len(observed["trials"]), 300)
        self.assertEqual(len(observed["segments"]), 328)
        self.assertEqual(len(observed["original_windows"]), 588)
        self.assertEqual(len(observed["folds"]), 20)
        self.assertEqual(observed["original_graphs"], self.dev["coverage"])
        self.assertEqual(observed["original_folds"], self.old["plan"]["folds"])
        for row in observed["trials"]:
            self.assertEqual(row["status"], "source_present")
            self.assertTrue(row["in_original_trial_graph"])
        for graphs in observed["original_graphs"].values():
            for graph in (
                graphs["inner"]
                + graphs["cohort_inner"]
                + [graphs["validation"], graphs["cohort_refit"]]
            ):
                for trials in graph["enrollment_trials"].values():
                    self.assertEqual(trials, list(range(1, 11)))
                for trials in graph["probe_trials"].values():
                    self.assertEqual(trials, list(range(1, 11)))

    def test_full_300_run_and_saved_only_replay_with_real_limits(self) -> None:
        report = isolated_worker("api")
        self.assertEqual(report["status"], "verified")

    def test_fresh_process_cli_full_300_run_and_replay_with_real_limits(self) -> None:
        report = isolated_worker("cli")
        self.assertEqual(report["status"], "verified")

    def full_api_worker(self) -> None:
        self.assert_real_limits()
        self.assertEqual(inventory.run(self.args)["status"], "complete")
        self.assert_real_limits()
        self.assert_full_result()
        self.registration_input.unlink()
        self.development_input.unlink()
        self.assertEqual(
            inventory.verify(argparse.Namespace(out_root=self.args.out_root))["status"],
            "verified",
        )
        self.assert_real_limits()

    def full_cli_worker(self) -> None:
        self.assert_real_limits()
        # Each CLI starts a fresh interpreter, inherits these real limits, and enforces
        # the same limits itself. The replay gets no original input paths or files.
        command = [sys.executable, str(Path(inventory.__file__).resolve())]
        common = ["--out-root", str(self.args.out_root)]
        run = subprocess.run(
            [
                *command,
                "--stage",
                "run",
                *common,
                "--historical-registration-json",
                str(self.registration_input),
                "--development-metadata-json",
                str(self.development_input),
            ],
            capture_output=True,
            text=True,
            timeout=65,
            check=False,
        )
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(json.loads(run.stdout)["status"], "complete")
        self.assert_full_result()
        self.registration_input.unlink()
        self.development_input.unlink()
        replay = subprocess.run(
            [*command, "--stage", "verify", *common],
            capture_output=True,
            text=True,
            timeout=65,
            check=False,
        )
        self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
        self.assertEqual(json.loads(replay.stdout)["status"], "verified")
        self.assert_real_limits()


def process_state() -> tuple[Any, ...]:
    """Snapshot actual runner state, including every environment entry, privately."""
    return (
        resource.getrlimit(resource.RLIMIT_CPU),
        resource.getrlimit(resource.RLIMIT_AS),
        os.sched_getaffinity(0),
        dict(os.environ),
        signal.getsignal(signal.SIGXCPU),
    )


def isolated_worker(mode: str) -> dict[str, Any]:
    before = process_state()
    # exec starts without a scientific parent's imported modules or address mappings.
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv.pop(1), run_name='__main__')",
            str(Path(__file__).resolve()),
            "--isolated-worker",
            mode,
        ],
        cwd=inventory.ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if process_state() != before:
        raise AssertionError("isolated worker changed the shared runner state")
    if completed.returncode:
        raise AssertionError(
            f"{mode} worker exited {completed.returncode}\n"
            + completed.stdout
            + completed.stderr
        )
    return json.loads(completed.stdout)


def contract_suite() -> unittest.TestSuite:
    # Exclude the integration wrapper itself so its nested suite cannot recurse.
    return unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromTestCase(case)
        for case in (
            NumericContract,
            SupportContract,
            MalformedMetadata,
            AttemptContract,
            LimitContract,
            FullCohortContract,
        )
    )


class ParentStateResult(unittest.TestResult):
    def startTest(self, test: unittest.TestCase) -> None:
        self.before = process_state()
        super().startTest(test)

    def stopTest(self, test: unittest.TestCase) -> None:
        if process_state() != self.before:
            try:
                raise AssertionError(
                    "test leaked runner limits/affinity/environment/handler"
                )
            except AssertionError:
                self.addFailure(test, sys.exc_info())
        super().stopTest(test)


class ProcessIsolationContract(unittest.TestCase):
    def test_large_parent_state_survives_the_entire_contract_suite(self) -> None:
        report = isolated_worker("large-parent")
        self.assertEqual(report["tests_run"], contract_suite().countTestCases())
        self.assertEqual(report["reserved_bytes"], 768 * 1024 * 1024)
        self.assertTrue(report["parent_unchanged"])


def worker_main(mode: str) -> None:
    if mode == "large-parent":
        before = process_state()
        reserved_bytes = 768 * 1024 * 1024
        # Model a scientific interpreter exceeding the inventory's virtual-memory
        # ceiling without importing dependencies or committing 768 MiB of RAM.
        with mmap.mmap(-1, reserved_bytes) as scientific_mapping:
            scientific_mapping[0] = 1
            result = ParentStateResult()
            contract_suite().run(result)
            if not result.wasSuccessful():
                raise AssertionError(
                    "contract suite failed inside the large parent\n"
                    + "\n".join(
                        message for _, message in result.errors + result.failures
                    )
                )
            if process_state() != before:
                raise AssertionError("contract suite changed the large parent state")
            # An unrelated later test must still be able to allocate memory.
            with mmap.mmap(-1, 16 * 1024 * 1024) as subsequent_mapping:
                subsequent_mapping[-1] = scientific_mapping[0]
        print(
            json.dumps(
                {
                    "tests_run": result.testsRun,
                    "reserved_bytes": reserved_bytes,
                    "parent_unchanged": True,
                }
            )
        )
        return
    case = FullCohortContract()
    try:
        # Only this new interpreter receives the irreversible real OS ceilings.
        inventory.enforce_limits()
        case.assert_real_limits()
        case.prepare_fixture()
        if mode == "api":
            case.full_api_worker()
        elif mode == "cli":
            case.full_cli_worker()
        else:
            raise ValueError("unknown isolated worker mode")
        print(json.dumps({"status": "verified", "mode": mode}))
    finally:
        case.doCleanups()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--isolated-worker":
        worker_main(sys.argv[2])
    else:
        unittest.main()
