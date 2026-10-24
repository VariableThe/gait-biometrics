"""Synthetic policy transitions only: no datasets, model, fitting, or GPU."""

from __future__ import annotations

import json
import unittest

from engine.continuous_auth import ContinuousAuthenticator


class ContinuousAuthenticationTests(unittest.TestCase):
    def assert_decision(self, result, decision, reason=None):
        self.assertEqual(result["decision"], decision)
        self.assertEqual(result["authenticated"], decision == "accept")
        self.assertEqual(result["challenge_required"], decision == "challenge")
        self.assertEqual(
            result["evidence_count"], result["accept_count"] + result["reject_count"]
        )
        if reason is not None:
            self.assertEqual(result["reason"], reason)
        json.dumps(result, allow_nan=False)
        return result

    def test_accept_requires_quorum_and_includes_threshold_ties(self):
        auth = ContinuousAuthenticator(-0.3)
        first = self.assert_decision(
            auth.update(-0.3, 0), "defer", "insufficient_evidence"
        )
        self.assertEqual(first["accept_count"], 1)
        accepted = self.assert_decision(
            auth.update(-0.3, 1, walking_probability=0.8), "accept", "accept_quorum"
        )
        self.assertEqual(accepted["accept_count"], 2)
        self.assertEqual(accepted["evidence_age"], 0)
        self.assertEqual(accepted["oldest_evidence_age"], 1)

    def test_reject_quorum_latches_and_cannot_be_overridden_by_good_scores(self):
        auth = ContinuousAuthenticator(0.5)
        self.assert_decision(auth.update(0.4, 0), "defer", "contradictory_evidence")
        rejected = self.assert_decision(
            auth.update(0.4, 1), "challenge", "reject_quorum"
        )
        self.assertEqual(rejected["reject_count"], 2)
        for time in (2, 3, 4):
            result = self.assert_decision(
                auth.update(1, time, recording_id=f"new-{time}"),
                "challenge",
                "reject_quorum",
            )
            self.assertFalse(result["evidence_added"])

    def test_contradictory_history_never_accepts_and_ages_out_by_count(self):
        auth = ContinuousAuthenticator(0.5)
        self.assert_decision(auth.update(1, 0), "defer")
        self.assert_decision(auth.update(1, 1), "accept")
        for time, score in ((2, 0), (3, 1), (4, 1)):
            result = self.assert_decision(
                auth.update(score, time), "defer", "contradictory_evidence"
            )
            self.assertLessEqual(result["evidence_count"], 3)
        result = self.assert_decision(auth.update(1, 5), "accept")
        self.assertEqual(result["accept_count"], 3)
        self.assertEqual(result["reject_count"], 0)

    def test_rejection_dominates_when_both_configured_quorums_are_present(self):
        auth = ContinuousAuthenticator(0.5, required_accepts=1, required_rejects=1)
        self.assert_decision(auth.update(1, 0), "accept")
        result = self.assert_decision(auth.update(0, 1), "challenge", "reject_quorum")
        self.assertEqual(result["accept_count"], 1)
        self.assertEqual(result["reject_count"], 1)

    def test_history_is_bounded_on_long_synthetic_stream(self):
        auth = ContinuousAuthenticator(0)
        for time in range(1000):
            result = self.assert_decision(
                auth.update(1, time), "defer" if not time else "accept"
            )
            self.assertLessEqual(result["evidence_count"], 3)
        self.assertEqual(result["oldest_evidence_age"], 2)

    def test_gate_or_missing_score_abstains_without_refreshing_evidence(self):
        for kwargs, reason in (
            ({"score": None}, "missing_score"),
            ({"score": 1, "quality_ok": False}, "quality_gate"),
            ({"score": 1, "walking_probability": 0.79}, "not_walking"),
        ):
            with self.subTest(reason=reason):
                auth = ContinuousAuthenticator(0.5)
                auth.update(1, 0)
                auth.update(1, 1)
                for time in range(2, 11):
                    result = self.assert_decision(
                        auth.update(timestamp=time, **kwargs), "defer", reason
                    )
                    self.assertEqual(result["evidence_count"], 2)
                    self.assertEqual(result["evidence_age"], time - 1)
                    self.assertFalse(result["evidence_added"])
                self.assert_decision(
                    auth.update(timestamp=11, **kwargs), "challenge", "evidence_gap"
                )

    def test_never_walking_eventually_requires_fallback(self):
        auth = ContinuousAuthenticator(0.5)
        for time in range(10):
            result = self.assert_decision(
                auth.update(1, time, walking_probability=0), "defer", "not_walking"
            )
            self.assertEqual(result["evidence_count"], 0)
            self.assertIsNone(result["evidence_age"])
        self.assert_decision(
            auth.update(1, 10, walking_probability=0), "challenge", "evidence_gap"
        )

    def test_silence_expires_via_advance_at_exact_deadline(self):
        auth = ContinuousAuthenticator(0)
        auth.update(1, 0)
        auth.update(1, 1)
        result = self.assert_decision(auth.advance(10.99), "defer", "awaiting_evidence")
        self.assertFalse(result["evidence_added"])
        self.assert_decision(auth.advance(11), "challenge", "evidence_gap")

    def test_good_score_after_gap_cannot_resurrect_old_consensus(self):
        auth = ContinuousAuthenticator(0)
        auth.update(1, 0)
        auth.update(1, 1)
        result = self.assert_decision(auth.update(1, 11), "challenge", "evidence_gap")
        self.assertFalse(result["evidence_added"])

    def test_old_votes_expire_before_the_current_score_is_counted(self):
        auth = ContinuousAuthenticator(
            0, max_evidence_age_seconds=4, max_gap_seconds=10
        )
        auth.update(1, 0)
        self.assert_decision(auth.update(1, 1), "accept")
        result = self.assert_decision(auth.update(1, 4), "accept")
        self.assertEqual(result["accept_count"], 2)
        self.assertEqual(result["oldest_evidence_age"], 3)
        result = self.assert_decision(auth.update(1, 5), "accept")
        self.assertEqual(result["accept_count"], 2)
        self.assertEqual(result["oldest_evidence_age"], 1)

    def test_latest_reject_cannot_keep_old_authentication_alive(self):
        auth = ContinuousAuthenticator(
            0,
            history_size=100,
            required_rejects=100,
            max_evidence_age_seconds=5,
            max_gap_seconds=10,
        )
        auth.update(1, 0)
        auth.update(1, 1)
        for time in range(2, 6):
            self.assert_decision(
                auth.update(-1, time), "defer", "contradictory_evidence"
            )
        self.assert_decision(auth.update(1, 6), "challenge", "authentication_expired")

    def test_no_consensus_times_out_even_with_recent_gated_scores(self):
        auth = ContinuousAuthenticator(
            0,
            history_size=10,
            required_accepts=10,
            required_rejects=10,
            max_evidence_age_seconds=4,
            max_gap_seconds=10,
        )
        for time in range(4):
            self.assert_decision(auth.update(1, time), "defer", "insufficient_evidence")
        self.assert_decision(auth.update(1, 4), "challenge", "consensus_timeout")

    def test_recording_boundary_clears_votes_including_none_boundaries(self):
        auth = ContinuousAuthenticator(0)
        auth.update(1, 0, recording_id="a")
        self.assert_decision(auth.update(1, 1, recording_id="a"), "accept")
        for time, recording in ((2, "b"), (3, "a"), (4, None), (5, "a")):
            result = self.assert_decision(
                auth.update(1, time, recording_id=recording),
                "defer",
                "insufficient_evidence",
            )
            self.assertTrue(result["recording_changed"])
            self.assertEqual(result["evidence_count"], 1)
        self.assert_decision(auth.update(1, 6, recording_id="a"), "accept")

    def test_repeated_recording_boundaries_do_not_postpone_timeout(self):
        auth = ContinuousAuthenticator(0, max_evidence_age_seconds=4)
        for time in range(4):
            self.assert_decision(auth.update(1, time, recording_id=str(time)), "defer")
        self.assert_decision(
            auth.update(1, 4, recording_id="new"), "challenge", "consensus_timeout"
        )

    def test_boundaries_do_not_allow_clock_rewind_or_clear_challenge(self):
        auth = ContinuousAuthenticator(0)
        auth.update(1, 2, recording_id="a")
        self.assert_decision(
            auth.update(1, 1, recording_id="b"), "challenge", "nonmonotonic_timestamp"
        )
        self.assert_decision(
            auth.update(1, 3, recording_id="c"), "challenge", "nonmonotonic_timestamp"
        )

    def test_duplicate_and_reversed_calls_never_reuse_an_accept(self):
        for operation, time in (
            ("update", 1),
            ("update", 0),
            ("advance", 1),
            ("advance", 0),
        ):
            with self.subTest(operation=operation, time=time):
                auth = ContinuousAuthenticator(0)
                auth.update(1, 0)
                auth.update(1, 1)
                result = (
                    auth.update(1, time)
                    if operation == "update"
                    else auth.advance(time)
                )
                self.assert_decision(result, "challenge", "nonmonotonic_timestamp")
                self.assert_decision(
                    auth.update(1, 2), "challenge", "nonmonotonic_timestamp"
                )

    def test_invalid_runtime_numeric_inputs_fail_closed_and_are_json_safe(self):
        invalid = (float("nan"), float("inf"), -float("inf"), True, "1", [], 10**1000)
        for field in ("timestamp", "score", "walking_probability", "window_start"):
            for value in invalid:
                with self.subTest(field=field, value=repr(value)[:30]):
                    auth = ContinuousAuthenticator(0)
                    auth.update(1, 0)
                    auth.update(1, 1)
                    kwargs = {"score": 1, "timestamp": 2, field: value}
                    result = self.assert_decision(auth.update(**kwargs), "challenge")
                    self.assertFalse(result["evidence_added"])

    def test_invalid_walking_range_quality_and_identifiers_fail_closed(self):
        for field, values in (
            ("walking_probability", (-0.1, 1.1, None)),
            ("quality_ok", (None, 0, 1, "true", [])),
            ("recording_id", (True, 1.0, [], {}, 2**53, 10**5000)),
            ("timestamp", (-1, None)),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    kwargs = {"score": 1, "timestamp": 2, field: value}
                    self.assert_decision(
                        ContinuousAuthenticator(0).update(**kwargs), "challenge"
                    )

    def test_touching_half_open_intervals_are_independent_and_causal(self):
        auth = ContinuousAuthenticator(0)
        first = self.assert_decision(auth.update(1, 2, window_start=0), "defer")
        self.assertEqual(first["evidence_age"], 2)
        result = self.assert_decision(auth.update(1, 4, window_start=2), "accept")
        self.assertEqual(result["oldest_evidence_age"], 4)
        self.assertEqual(result["evidence_age"], 2)

    def test_overlapping_windows_cannot_form_a_quorum(self):
        auth = ContinuousAuthenticator(0)
        auth.update(1, 2, window_start=0)
        result = self.assert_decision(
            auth.update(1, 3, window_start=1), "challenge", "overlapping_evidence"
        )
        self.assertEqual(result["evidence_count"], 1)
        self.assertFalse(result["evidence_added"])

    def test_overlap_watermark_survives_bounded_history_eviction(self):
        auth = ContinuousAuthenticator(
            0, history_size=1, required_accepts=1, required_rejects=1
        )
        for end in (2, 4, 6):
            self.assert_decision(auth.update(1, end, window_start=end - 2), "accept")
        self.assert_decision(
            auth.update(1, 7, window_start=3), "challenge", "overlapping_evidence"
        )

    def test_gated_or_unscored_windows_still_block_overlapping_reuse(self):
        for kwargs in ({"score": None}, {"score": 1, "quality_ok": False}):
            with self.subTest(kwargs=kwargs):
                auth = ContinuousAuthenticator(0)
                self.assert_decision(
                    auth.update(timestamp=2, window_start=0, **kwargs), "defer"
                )
                self.assert_decision(
                    auth.update(1, 3, window_start=1),
                    "challenge",
                    "overlapping_evidence",
                )

    def test_interval_metadata_cannot_be_silently_dropped(self):
        auth = ContinuousAuthenticator(0)
        auth.update(1, 2, window_start=0)
        self.assert_decision(auth.update(1, 3), "challenge", "missing_window_start")

    def test_explicit_windows_must_be_positive_causal_and_fresh(self):
        for start, reason in (
            (-1, "invalid_window_start"),
            (2, "invalid_window_start"),
            (3, "invalid_window_start"),
        ):
            with self.subTest(start=start):
                self.assert_decision(
                    ContinuousAuthenticator(0).update(1, 2, window_start=start),
                    "challenge",
                    reason,
                )
        self.assert_decision(
            ContinuousAuthenticator(0).update(1, 30, window_start=0),
            "challenge",
            "stale_window",
        )

    def test_expiry_uses_oldest_raw_support_not_only_window_end(self):
        auth = ContinuousAuthenticator(0, max_evidence_age_seconds=5)
        auth.update(1, 4, window_start=0)
        result = self.assert_decision(
            auth.update(1, 5, window_start=4), "defer", "insufficient_evidence"
        )
        self.assertEqual(result["evidence_count"], 1)
        self.assert_decision(auth.update(1, 6, window_start=5), "accept")

    def test_advance_does_not_consume_a_window_or_create_evidence(self):
        auth = ContinuousAuthenticator(0)
        auth.update(1, 2, window_start=0)
        self.assert_decision(auth.advance(3), "defer", "awaiting_evidence")
        self.assert_decision(auth.update(1, 4, window_start=2), "accept")

    def test_reset_releases_latch_but_requires_new_independent_consensus(self):
        auth = ContinuousAuthenticator(0)
        auth.update(-1, 10, recording_id="a")
        auth.update(-1, 11, recording_id="a")
        self.assertIsNone(auth.reset())
        result = self.assert_decision(
            auth.update(1, 2, recording_id="b", window_start=0), "defer"
        )
        self.assertEqual(result["accept_count"], 1)
        self.assertFalse(result["recording_changed"])
        self.assert_decision(
            auth.update(1, 4, recording_id="b", window_start=2), "accept"
        )

    def test_runtime_can_explicitly_require_fallback(self):
        auth = ContinuousAuthenticator(0)
        auth.update(1, 0)
        auth.update(1, 1)
        result = self.assert_decision(
            auth.request_fallback(2), "challenge", "fallback_requested"
        )
        self.assertFalse(result["evidence_added"])
        self.assert_decision(auth.update(1, 3), "challenge", "fallback_requested")
        self.assert_decision(
            auth.request_fallback(4), "challenge", "fallback_requested"
        )

    def test_fallback_request_does_not_erase_a_prior_failure_reason(self):
        auth = ContinuousAuthenticator(0)
        auth.update(-1, 0)
        auth.update(-1, 1)
        self.assert_decision(auth.request_fallback(2), "challenge", "reject_quorum")
        self.assert_decision(
            ContinuousAuthenticator(0).request_fallback(float("nan")),
            "challenge",
            "invalid_timestamp",
        )

    def test_constructor_rejects_invalid_configuration(self):
        configurations = [
            {"threshold": value}
            for value in (None, True, "0", float("nan"), float("inf"), 10**1000)
        ]
        configurations += [
            {field: value}
            for field in ("history_size", "required_accepts", "required_rejects")
            for value in (0, -1, 1.5, True, "2")
        ]
        configurations += [
            {field: value}
            for field in ("max_evidence_age_seconds", "max_gap_seconds")
            for value in (0, -1, True, float("nan"), float("inf"))
        ]
        configurations += [
            {"walking_threshold": value}
            for value in (-0.1, 1.1, True, float("nan"), float("inf"))
        ]
        configurations += [
            {"required_accepts": 4},
            {"required_rejects": 4},
            {"history_size": 10**1000},
        ]
        for kwargs in configurations:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ContinuousAuthenticator(**({"threshold": 0} | kwargs))


if __name__ == "__main__":
    unittest.main()
