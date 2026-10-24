"""Synthetic JSONL integration and trust-boundary regression checks."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from engine.continuous_evaluation import AuthenticationStream
from scripts.stream_authentication import main, run_stream


def score(timestamp, **overrides):
    return {
        "type": "score",
        "timestamp": timestamp,
        "score": 1,
        "walking_probability": 1,
        "quality_ok": True,
        "recording_id": "synthetic",
        **overrides,
    }


class StreamAuthenticationTests(unittest.TestCase):
    def run_lines(self, lines, **options):
        source = io.StringIO("".join(line + "\n" for line in lines))
        output = io.StringIO()
        status = run_stream(AuthenticationStream(0.5, **options), source, output)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        for index, row in enumerate(rows, start=1):
            self.assertEqual(row["line_number"], index)
            self.assertEqual(row["authenticated"], row["decision"] == "accept")
            json.dumps(row, allow_nan=False)
        return status, rows

    def test_score_silence_expiry_and_new_recording_cannot_release_challenge(self):
        status, rows = self.run_lines(
            [
                json.dumps(score(2, window_start=0)),
                json.dumps(score(4, window_start=2)),
                '{"type":"advance","timestamp":13.9}',
                '{"type":"advance","timestamp":14}',
                json.dumps(score(16, window_start=14, recording_id="new")),
            ]
        )
        self.assertEqual(status, 0)
        self.assertEqual(
            [row["decision"] for row in rows],
            ["defer", "accept", "defer", "challenge", "challenge"],
        )
        self.assertEqual(rows[3]["reason"], "evidence_gap")
        self.assertTrue(rows[3]["challenge_started"])
        self.assertFalse(rows[4]["challenge_started"])
        self.assertFalse(rows[4]["evidence_added"])

    def test_explicit_fallback_is_latched_and_eof_does_not_advance_time(self):
        status, rows = self.run_lines(
            [
                json.dumps(score(0)),
                json.dumps(score(1)),
                '{"type":"fallback","timestamp":2}',
                json.dumps(score(3)),
            ]
        )
        self.assertEqual(status, 0)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[2]["reason"], "fallback_requested")
        self.assertEqual(rows[3]["reason"], "fallback_requested")
        self.assertEqual(rows[3]["timestamp"], 3)

    def test_reset_and_unknown_events_are_rejected_even_before_challenge(self):
        for kind in ("reset", "recover", "fallback_success", "update", "other"):
            with self.subTest(kind=kind):
                status, rows = self.run_lines(
                    [
                        json.dumps({"type": kind, "timestamp": 0}),
                        json.dumps(score(1)),
                        json.dumps(score(2)),
                    ]
                )
                self.assertEqual(status, 2)
                self.assertEqual(rows[0]["event_error"], "unsupported_event_type")
                self.assertTrue(all(row["challenge_required"] for row in rows))
                self.assertTrue(all(not row["evidence_added"] for row in rows))
                self.assertTrue(rows[-1]["input_valid"])

    def test_reset_after_challenge_cannot_clear_clock_or_latch(self):
        status, rows = self.run_lines(
            [
                '{"type":"fallback","timestamp":10}',
                '{"type":"reset","timestamp":11}',
                json.dumps(score(1)),
                json.dumps(score(12)),
            ]
        )
        self.assertEqual(status, 2)
        self.assertEqual(rows[2]["event_error"], "nonmonotonic_timestamp")
        self.assertIsNone(rows[2]["timestamp"])
        self.assertTrue(all(row["challenge_required"] for row in rows))
        self.assertFalse(rows[3]["evidence_added"])

    def test_malformed_json_and_nonfinite_values_fail_closed_without_nan_output(self):
        bad_lines = [
            "",
            "  ",
            "{",
            "[]",
            "null",
            "true",
            "1",
            '"text"',
            '{"type":"advance","timestamp":NaN}',
            '{"type":"advance","timestamp":Infinity}',
            '{"type":"advance","timestamp":-Infinity}',
            '{"type":"advance","timestamp":1e999}',
            '{"type":"advance","timestamp":0,"timestamp":1}',
            '{"type":"advance","type":"fallback","timestamp":0}',
            '{"type":"advance","timestamp":0} {"type":"advance","timestamp":1}',
            '{"type":"advance","timestamp":0,}',
            "\udcff",
        ]
        for line in bad_lines:
            with self.subTest(line=repr(line)):
                status, rows = self.run_lines(
                    [line, json.dumps(score(2)), json.dumps(score(3))]
                )
                self.assertEqual(status, 2)
                self.assertFalse(rows[0]["input_valid"])
                self.assertTrue(all(row["challenge_required"] for row in rows))
                self.assertEqual(sum(row["challenge_started"] for row in rows), 1)

    def test_gate_fields_are_explicit_and_types_are_strict(self):
        invalid = []
        for key in (
            "type",
            "timestamp",
            "score",
            "walking_probability",
            "quality_ok",
            "recording_id",
        ):
            event = score(2)
            del event[key]
            invalid.append(event)
        invalid.extend(
            score(2) | {key: value}
            for key, values in (
                ("timestamp", (True, -1, "2", None)),
                ("score", (True, "1", [], float("inf"), float("nan"), 10**1000)),
                ("walking_probability", (True, None, -0.1, 1.1, "1")),
                ("quality_ok", (1, 0, None, "true")),
                ("recording_id", (True, 1.0, [], {}, 2**53, "\ud800")),
                ("window_start", (True, -1, 2, 3, "0")),
            )
            for value in values
        )
        invalid.extend(
            [score(2, extra=1), {"type": "advance", "timestamp": 2, "score": 1}]
        )
        for event in invalid:
            with self.subTest(event=repr(event)[:100]):
                status, rows = self.run_lines([json.dumps(event), json.dumps(score(3))])
                self.assertEqual(status, 2)
                self.assertFalse(rows[0]["input_valid"])
                self.assertEqual(rows[-1]["decision"], "challenge")

    def test_null_score_and_failed_gates_abstain_instead_of_negative_votes(self):
        for overrides, reason in (
            ({"score": None}, "missing_score"),
            ({"quality_ok": False}, "quality_gate"),
            ({"walking_probability": 0}, "not_walking"),
        ):
            with self.subTest(reason=reason):
                status, rows = self.run_lines(
                    [
                        json.dumps(score(0)),
                        json.dumps(score(1)),
                        json.dumps(score(2, **overrides)),
                        '{"type":"advance","timestamp":11}',
                    ]
                )
                self.assertEqual(status, 0)
                self.assertEqual(rows[2]["decision"], "defer")
                self.assertEqual(rows[2]["reason"], reason)
                self.assertFalse(rows[2]["evidence_added"])
                self.assertEqual(rows[2]["reject_count"], 0)
                self.assertEqual(rows[3]["reason"], "evidence_gap")

    def test_global_clock_is_strict_across_all_event_types_and_errors(self):
        for second in (
            score(2, recording_id="new"),
            {"type": "advance", "timestamp": 2},
            {"type": "fallback", "timestamp": 1},
        ):
            with self.subTest(second=second):
                status, rows = self.run_lines(
                    [json.dumps(score(2)), json.dumps(second), json.dumps(score(3))]
                )
                self.assertEqual(status, 2)
                self.assertEqual(rows[1]["event_error"], "nonmonotonic_timestamp")
                self.assertIsNone(rows[1]["timestamp"])
                self.assertEqual(rows[2]["decision"], "challenge")
        stream = AuthenticationStream(0.5)
        stream.process_event(score(2, unknown=True))
        self.assertEqual(stream.last_timestamp, 2)
        duplicate = stream.process_event(score(2))
        self.assertEqual(duplicate["event_error"], "nonmonotonic_timestamp")

    def test_window_support_checks_gated_overlap_and_staleness(self):
        for first, second, expected in (
            (
                score(2, window_start=0),
                score(3, window_start=1),
                "overlapping_evidence",
            ),
            (
                score(2, window_start=0, quality_ok=False),
                score(3, window_start=1),
                "overlapping_evidence",
            ),
            (score(2, window_start=0), score(3), "missing_window_start"),
            (score(2), score(30, window_start=0), "stale_window"),
        ):
            with self.subTest(expected=expected):
                status, rows = self.run_lines([json.dumps(first), json.dumps(second)])
                self.assertEqual(status, 2)
                self.assertEqual(rows[1]["event_error"], expected)
                self.assertTrue(rows[1]["challenge_required"])

    def test_empty_input_is_empty_and_output_is_flushed_per_event(self):
        status, rows = self.run_lines([])
        self.assertEqual((status, rows), (0, []))
        output = Mock()
        status = run_stream(
            AuthenticationStream(0.5), io.StringIO(json.dumps(score(0)) + "\n"), output
        )
        self.assertEqual(status, 0)
        output.write.assert_called_once()
        output.flush.assert_called_once()

    def test_cli_requires_explicit_valid_threshold_and_policy_configuration(self):
        for args in (
            [],
            ["--threshold", "nan"],
            ["--threshold", "inf"],
            ["--threshold", "0", "--history-size", "0"],
        ):
            with (
                self.subTest(args=args),
                patch("sys.stderr", new=io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                main(args)
            self.assertEqual(error.exception.code, 2)

    def test_cli_module_end_to_end_uses_only_standard_library(self):
        command = [
            sys.executable,
            "-S",
            "-m",
            "scripts.stream_authentication",
            "--threshold",
            "0.5",
        ]
        env = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            CUDA_VISIBLE_DEVICES="",
        )
        result = subprocess.run(
            command,
            input="\n".join(
                [
                    json.dumps(score(0)),
                    json.dumps(score(1)),
                    '{"type":"advance","timestamp":11}',
                ]
            )
            + "\n",
            text=True,
            capture_output=True,
            check=False,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(
            [row["decision"] for row in rows], ["defer", "accept", "challenge"]
        )


if __name__ == "__main__":
    unittest.main()
