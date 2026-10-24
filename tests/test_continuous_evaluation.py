"""Exact, synthetic timed replay; no datasets, model, fitting, or GPU."""

from __future__ import annotations

import json
import unittest
from typing import Any

from engine.continuous_evaluation import (
    AuthenticationStream,
    ObservedInterval,
    Takeover,
    evaluate_stream,
)


def score(timestamp, value=1, **overrides):
    return {
        "type": "score",
        "timestamp": timestamp,
        "score": value,
        "walking_probability": 1,
        "quality_ok": True,
        "recording_id": "synthetic",
        **overrides,
    }


def tick(timestamp, kind="advance"):
    return {"type": kind, "timestamp": timestamp}


class ContinuousEvaluationTests(unittest.TestCase):
    def replay(self, events, intervals, attacks=(), **options):
        result = evaluate_stream(
            events,
            threshold=0.5,
            observed_intervals=intervals,
            takeovers=attacks,
            **options,
        )
        json.dumps(result, allow_nan=False)
        return result

    def test_owner_rate_uses_only_observed_sparse_intervals_and_new_challenges(self):
        events = [
            tick(0),
            score(1),
            tick(3601, "fallback"),
            score(3602, recording_id="new"),
            tick(4200),
        ]
        result = self.replay(
            events,
            [ObservedInterval(0, 600, "owner"), ObservedInterval(3600, 4200, "owner")],
            max_evidence_age_seconds=10000,
            max_gap_seconds=10000,
        )
        metrics = result["metrics"]
        self.assertTrue(metrics["valid"])
        self.assertEqual(metrics["observed_seconds"]["owner"], 1200)
        self.assertEqual(metrics["owner_observed_hours"], 1 / 3)
        self.assertEqual(metrics["owner_false_challenges"], 1)
        self.assertEqual(metrics["owner_false_challenges_per_observed_hour"], 3)
        self.assertEqual(metrics["challenge_episode_count"], 1)
        self.assertEqual(metrics["challenge_output_count"], 3)
        self.assertFalse(result["decisions"][-2]["evidence_added"])
        self.assertEqual(result["decisions"][-1]["identity"], "unobserved")

    def test_invalid_truth_and_abstention_are_not_impostor_or_owner_errors(self):
        result = self.replay(
            [
                tick(0),
                score(1, quality_ok=False),
                tick(3),
                tick(5, "fallback"),
                tick(6),
            ],
            [
                ObservedInterval(0, 2, "owner"),
                ObservedInterval(2, 4, "invalid"),
                ObservedInterval(4, 6, "impostor"),
            ],
        )
        metrics = result["metrics"]
        self.assertTrue(metrics["valid"])
        self.assertEqual(
            metrics["observed_seconds"], {"owner": 2, "impostor": 2, "invalid": 2}
        )
        self.assertEqual(
            metrics["event_counts_by_identity"],
            {"owner": 2, "impostor": 1, "invalid": 1, "unobserved": 1},
        )
        self.assertEqual(metrics["challenge_counts_by_identity"]["impostor"], 1)
        self.assertEqual(metrics["owner_false_challenges_per_observed_hour"], 0)
        self.assertEqual(result["decisions"][1]["reason"], "quality_gate")

    def test_undetected_attack_is_censored_not_zero_latency(self):
        result = self.replay(
            [
                tick(0),
                score(1, window_start=0),
                score(3, window_start=2),
                tick(4),
                score(5, window_start=4),
                score(7, 0, window_start=6),
                tick(10),
            ],
            [
                ObservedInterval(0, 2, "owner"),
                ObservedInterval(2, 4, "impostor"),
                ObservedInterval(4, 6, "owner"),
                ObservedInterval(6, 10, "impostor"),
            ],
            [Takeover("undetected", 2, 4), Takeover("detected", 6, 10)],
            history_size=1,
            required_accepts=1,
            required_rejects=1,
        )
        metrics = result["metrics"]
        undetected, detected = metrics["takeovers"]
        self.assertEqual(undetected["status"], "right_censored")
        self.assertIsNone(undetected["time_to_challenge_seconds"])
        self.assertEqual(undetected["censor_time_seconds"], 2)
        self.assertEqual(detected["status"], "detected")
        self.assertEqual(detected["time_to_challenge_seconds"], 1)
        self.assertIsNone(detected["censor_time_seconds"])
        self.assertEqual(metrics["detected_takeovers"], 1)
        self.assertEqual(metrics["right_censored_takeovers"], 1)
        self.assertEqual(metrics["mean_detected_time_to_challenge_seconds"], 1)

    def test_all_censored_and_no_owner_exposure_return_null_not_zero(self):
        result = self.replay(
            [tick(0), score(1), tick(2)],
            [ObservedInterval(0, 2, "impostor")],
            [Takeover("never-detected", 0, 2)],
        )
        metrics = result["metrics"]
        self.assertEqual(metrics["right_censored_takeovers"], 1)
        self.assertIsNone(metrics["mean_detected_time_to_challenge_seconds"])
        self.assertIsNone(metrics["owner_false_challenges_per_observed_hour"])

    def test_already_challenged_takeover_is_excluded_from_latency(self):
        metrics = self.replay(
            [tick(0), tick(1, "fallback"), score(3, 0), tick(4)],
            [ObservedInterval(0, 2, "owner"), ObservedInterval(2, 4, "impostor")],
            [Takeover("not-at-risk", 2, 4)],
        )["metrics"]
        self.assertEqual(metrics["takeovers"][0]["status"], "already_challenged")
        self.assertIsNone(metrics["takeovers"][0]["time_to_challenge_seconds"])
        self.assertIsNone(metrics["takeovers"][0]["censor_time_seconds"])
        self.assertEqual(metrics["already_challenged_takeovers"], 1)
        self.assertEqual(metrics["right_censored_takeovers"], 0)

    def test_half_open_follow_up_includes_onset_and_excludes_end(self):
        for challenge_time, expected, latency in (
            (2, "detected", 0),
            (4, "right_censored", None),
        ):
            with self.subTest(challenge_time=challenge_time):
                events = [tick(0), tick(challenge_time, "fallback")]
                if challenge_time < 4:
                    events.append(tick(4))
                outcome = self.replay(
                    events,
                    [
                        ObservedInterval(0, 2, "owner"),
                        ObservedInterval(2, 4, "impostor"),
                    ],
                    [Takeover("boundary", 2, 4)],
                )["metrics"]["takeovers"][0]
                self.assertEqual(outcome["status"], expected)
                self.assertEqual(outcome["time_to_challenge_seconds"], latency)

    def test_malformed_replay_is_challenged_and_metric_claims_are_invalid(self):
        result = self.replay(
            [tick(0), '{"type":"score","score":NaN}', tick(3), score(4), tick(5)],
            [ObservedInterval(0, 2, "owner"), ObservedInterval(2, 5, "impostor")],
            [Takeover("unknown-timing", 2, 5)],
        )
        metrics = result["metrics"]
        self.assertFalse(metrics["valid"])
        self.assertEqual(metrics["invalid_event_count"], 1)
        self.assertEqual(metrics["challenge_counts_by_identity"]["invalid"], 1)
        self.assertIsNone(metrics["owner_false_challenges_per_observed_hour"])
        self.assertEqual(metrics["takeovers"][0]["status"], "invalid_stream")
        self.assertEqual(metrics["invalid_takeovers"], 1)
        self.assertIsNone(metrics["mean_detected_time_to_challenge_seconds"])
        self.assertTrue(
            all(row["challenge_required"] for row in result["decisions"][1:])
        )

    def test_overlap_and_omitted_support_fail_closed_in_shared_policy(self):
        for later in (score(3, window_start=1), score(3)):
            with self.subTest(later=later):
                stream = AuthenticationStream(0.5)
                stream.process_event(score(2, window_start=0))
                failed = stream.process_event(later)
                self.assertFalse(failed["input_valid"])
                self.assertTrue(failed["challenge_required"])
                self.assertFalse(failed["evidence_added"])
                still = stream.process_event(
                    score(4, window_start=3, recording_id="new")
                )
                self.assertEqual(still["decision"], "challenge")
                self.assertFalse(still["challenge_started"])

    def test_metadata_is_sorted_but_overlapping_global_exposure_is_rejected(self):
        result = self.replay(
            [tick(0), tick(4)],
            [ObservedInterval(2, 4, "owner"), ObservedInterval(0, 2, "owner")],
        )
        self.assertEqual(result["metrics"]["observed_seconds"]["owner"], 4)
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.replay(
                [tick(0), tick(4)],
                [ObservedInterval(0, 3, "owner"), ObservedInterval(2, 4, "impostor")],
            )

    def test_takeover_must_end_at_observation_gap_and_have_impostor_truth(self):
        for intervals in (
            [ObservedInterval(0, 4, "owner")],
            [ObservedInterval(0, 1, "impostor"), ObservedInterval(2, 4, "impostor")],
            [ObservedInterval(0, 1, "impostor"), ObservedInterval(1, 4, "invalid")],
        ):
            with (
                self.subTest(intervals=intervals),
                self.assertRaisesRegex(ValueError, "continuous observed impostor"),
            ):
                self.replay([tick(0), tick(4)], intervals, [Takeover("gap", 0, 4)])
        metrics = self.replay(
            [tick(0), tick(4)],
            [ObservedInterval(0, 1, "impostor"), ObservedInterval(2, 4, "impostor")],
            [Takeover("censored-at-gap", 0, 1)],
        )["metrics"]
        self.assertEqual(metrics["takeovers"][0]["censor_time_seconds"], 1)

    def test_duplicate_or_overlapping_takeovers_are_rejected(self):
        for attacks in (
            [Takeover("same", 0, 1), Takeover("same", 2, 3)],
            [Takeover("one", 0, 2), Takeover("two", 1, 3)],
        ):
            with self.subTest(attacks=attacks), self.assertRaises(ValueError):
                self.replay(
                    [tick(0), tick(4)], [ObservedInterval(0, 4, "impostor")], attacks
                )

    def test_clock_must_bracket_exposure_and_no_terminal_tick_is_invented(self):
        for events, intervals in (
            ([], [ObservedInterval(0, 1, "owner")]),
            ([tick(1), tick(2)], [ObservedInterval(0, 2, "owner")]),
            ([tick(0), tick(1)], [ObservedInterval(0, 2, "owner")]),
        ):
            with (
                self.subTest(events=events),
                self.assertRaisesRegex(ValueError, "bracket"),
            ):
                self.replay(events, intervals)
        empty = self.replay([], [])["metrics"]
        self.assertEqual(empty["event_count"], 0)
        self.assertIsNone(empty["owner_false_challenges_per_observed_hour"])

    def test_interval_and_takeover_validation_rejects_bad_time_and_labels(self):
        invalid_times: tuple[Any, ...] = (
            True,
            -1,
            "1",
            float("nan"),
            float("inf"),
            10**1000,
        )
        for bad in invalid_times:
            with self.subTest(bad=repr(bad)[:30]), self.assertRaises(ValueError):
                ObservedInterval(bad, 2, "owner")
            with self.assertRaises(ValueError):
                Takeover("attack", 0, bad)
        for start, end, identity in (
            (1, 1, "owner"),
            (2, 1, "owner"),
            (0, 1, "unknown"),
        ):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                ObservedInterval(start, end, identity)
        with self.assertRaises(ValueError):
            Takeover("", 0, 1)
        with self.assertRaises(ValueError):
            Takeover("zero", 1, 1)

    def test_replay_is_deterministic_and_truth_cannot_change_policy(self):
        events = [tick(0), score(1), score(2), score(3, 0), score(4, 0), tick(5)]
        owner = self.replay(events, [ObservedInterval(0, 5, "owner")])
        again = self.replay(events, [ObservedInterval(0, 5, "owner")])
        impostor = self.replay(events, [ObservedInterval(0, 5, "impostor")])
        self.assertEqual(owner, again)
        for left, right in zip(owner["decisions"], impostor["decisions"], strict=True):
            self.assertEqual(left["decision"], right["decision"])
            self.assertEqual(left["reason"], right["reason"])
        self.assertEqual(owner["metrics"]["owner_false_challenges"], 1)
        self.assertEqual(impostor["metrics"]["owner_false_challenges"], 0)


if __name__ == "__main__":
    unittest.main()
