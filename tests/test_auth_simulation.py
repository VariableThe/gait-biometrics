"""Synthetic stress replay only: no models, real data, fitting, or artifacts."""

from __future__ import annotations

import json
import math
import random
import unittest
from dataclasses import replace
from itertools import pairwise
from typing import Any
from unittest.mock import patch

from engine.auth_simulation import (
    DEFAULT_POLICIES,
    SCENARIOS,
    Delivery,
    Episode,
    ScenarioConfig,
    ScoreWindow,
    build_scenario,
    replay_episode,
    run_grid,
)
from engine.continuous_evaluation import (
    AuthenticationStream,
    ObservedInterval,
    Takeover,
)


def window(index=0, *, source="raw-owner", **overrides):
    parameters: dict[str, Any] = {
        "source_recording": source,
        "window_id": f"window-{index}",
        "raw_start": index * 2,
        "raw_end": index * 2 + 2,
        "score": 1,
        **overrides,
    }
    return ScoreWindow(**parameters)


def episode(deliveries=(), *, onset=6, end=16, intervals=None, timer=1):
    return Episode(
        "manual",
        19,
        0,
        end,
        Takeover("handoff", onset, end),
        tuple(deliveries),
        tuple(
            intervals
            or (
                ObservedInterval(0, onset, "owner"),
                ObservedInterval(onset, end, "impostor"),
            )
        ),
        timer,
    )


class AuthSimulationTests(unittest.TestCase):
    def replay(self, planned, policy=DEFAULT_POLICIES[1]):
        result = replay_episode(planned, policy, include_trace=True)
        json.dumps(result, allow_nan=False)
        self.assertTrue(result["metrics"]["valid"])
        times = [event["timestamp"] for event in result["events"]]
        self.assertTrue(all(right > left for left, right in pairwise(times)))
        self.assertEqual(len(times), len(set(times)))
        return result

    def test_abrupt_takeover_matches_independent_closed_form_reference(self):
        # Owner evidence precedes handoff; R fresh consecutive negatives require
        # exactly R * duration after onset, before any deadline for these settings.
        for duration in (0.5, 2, 3):
            for policy in DEFAULT_POLICIES:
                with self.subTest(duration=duration, policy=policy.name):
                    planned = build_scenario(
                        ScenarioConfig(
                            window_seconds=duration, timer_seconds=duration / 2
                        )
                    )
                    result = self.replay(planned, policy)
                    outcome = result["metrics"]["takeovers"][0]
                    self.assertTrue(result["warmup_had_accept"])
                    self.assertEqual(result["metrics"]["owner_false_challenges"], 0)
                    self.assertEqual(outcome["status"], "detected")
                    self.assertEqual(
                        outcome["time_to_challenge_seconds"],
                        policy.required_rejects * duration,
                    )
                    self.assertEqual(
                        result["first_challenge"]["reason"], "reject_quorum"
                    )

    def test_bursts_match_window_pattern_reference_without_independent_trials(self):
        config = ScenarioConfig(
            scenario="bursty_acceptance", accept_burst_windows=4, reject_burst_windows=2
        )
        planned = build_scenario(config)
        for policy in DEFAULT_POLICIES:
            result = self.replay(planned, policy)
            attack_windows = result["score_windows"][config.owner_windows :]
            self.assertEqual(
                [int(row["window"]["score"] >= 0.5) for row in attack_windows],
                [1, 1, 1, 1, 0, 0, 1, 1],
            )
            expected = (4 + policy.required_rejects) * config.window_seconds
            self.assertEqual(
                result["metrics"]["takeovers"][0]["time_to_challenge_seconds"], expected
            )

    def test_silence_timer_matches_independent_deadline_and_rounding(self):
        planned = build_scenario(ScenarioConfig(scenario="silence", timer_seconds=3))
        result = self.replay(planned)
        last_score = planned.deliveries[-1].delivered_at
        deadline = last_score + DEFAULT_POLICIES[1].max_gap_seconds
        first_tick = math.ceil(deadline / planned.timer_seconds) * planned.timer_seconds
        self.assertEqual(result["first_challenge"]["timestamp"], first_tick)
        self.assertEqual(result["first_challenge"]["reason"], "evidence_gap")
        self.assertEqual(
            result["metrics"]["takeovers"][0]["time_to_challenge_seconds"],
            first_tick - planned.takeover.onset,
        )
        self.assertTrue(
            all(
                event["type"] == "advance"
                for event in result["events"]
                if event["timestamp"] >= planned.takeover.onset
            )
        )
        # Known simulated presence during silence is declared separately from raw data.
        self.assertEqual(
            result["metrics"]["observed_seconds"]["impostor"],
            planned.end - planned.takeover.onset,
        )

    def test_identity_handoff_does_not_inject_extra_policy_clock_calls(self):
        earlier = self.replay(episode(onset=6, end=20, timer=5))
        later = self.replay(episode(onset=12, end=20, timer=5))
        self.assertEqual(
            [row["timestamp"] for row in earlier["events"]], [0, 5, 10, 15, 20]
        )
        self.assertEqual(earlier["events"], later["events"])
        self.assertEqual(earlier["first_challenge"]["timestamp"], 10)
        self.assertEqual(later["first_challenge"]["timestamp"], 10)
        self.assertEqual(earlier["metrics"]["takeovers"][0]["status"], "detected")
        self.assertEqual(
            later["metrics"]["takeovers"][0]["status"], "already_challenged"
        )

    def test_exact_gap_score_cannot_refresh_expired_authentication(self):
        planned = episode(
            (Delivery(window(), 0, 2), Delivery(window(1), 10, 12)), onset=8, end=14
        )
        policy = replace(DEFAULT_POLICIES[0], max_gap_seconds=10)
        result = self.replay(planned, policy)
        at_deadline = next(row for row in result["decisions"] if row["timestamp"] == 12)
        self.assertEqual(at_deadline["event_type"], "score")
        self.assertEqual(at_deadline["reason"], "evidence_gap")
        self.assertFalse(at_deadline["evidence_added"])
        self.assertEqual(result["first_challenge"]["timestamp"], 12)

    def test_owner_challenge_stays_latched_through_handoff_and_recording_changes(self):
        planned = build_scenario(ScenarioConfig(scenario="owner_challenge"))
        result = self.replay(planned)
        self.assertEqual(result["metrics"]["owner_false_challenges"], 1)
        self.assertEqual(result["eligible_takeovers"], 0)
        outcome = result["metrics"]["takeovers"][0]
        self.assertEqual(outcome["status"], "already_challenged")
        self.assertIsNone(outcome["time_to_challenge_seconds"])
        self.assertIsNone(outcome["censor_time_seconds"])
        after = [
            row
            for row in result["decisions"]
            if row["timestamp"] >= result["first_challenge"]["timestamp"]
        ]
        self.assertEqual(sum(row["challenge_started"] for row in after), 1)
        self.assertTrue(all(row["challenge_required"] for row in after))
        self.assertFalse(any(row["evidence_added"] for row in after[1:]))

    def test_recording_resets_do_not_reset_clock_or_acceptance_timeout(self):
        planned = build_scenario(
            ScenarioConfig(scenario="recording_resets", attacker_windows=18)
        )
        result = self.replay(planned)
        self.assertEqual(result["support_rejection_count"], 0)
        # Last warmup acceptance is t=12. Per-window recording changes prevent
        # two retained rejects, but cannot postpone its 30-second expiry at t=42.
        self.assertEqual(result["first_challenge"]["timestamp"], 42)
        self.assertEqual(result["first_challenge"]["reason"], "authentication_expired")
        self.assertEqual(
            result["metrics"]["takeovers"][0]["time_to_challenge_seconds"], 28
        )
        self.assertTrue(any(row["recording_changed"] for row in result["decisions"]))

    def test_short_recording_reset_follow_up_is_right_censored_not_zero(self):
        planned = build_scenario(ScenarioConfig(scenario="recording_resets"))
        result = self.replay(planned)
        self.assertIsNone(result["first_challenge"])
        outcome = result["metrics"]["takeovers"][0]
        self.assertEqual(outcome["status"], "right_censored")
        self.assertIsNone(outcome["time_to_challenge_seconds"])
        self.assertEqual(
            outcome["censor_time_seconds"], planned.end - planned.takeover.onset
        )
        self.assertIsNone(result["metrics"]["mean_detected_time_to_challenge_seconds"])

    def test_challenge_exactly_at_follow_up_end_is_censored(self):
        planned = episode((Delivery(window(), 0, 2),), onset=6, end=12)
        result = self.replay(planned, DEFAULT_POLICIES[0])
        self.assertEqual(result["first_challenge"]["timestamp"], 12)
        self.assertEqual(result["first_challenge"]["identity"], "unobserved")
        self.assertEqual(result["metrics"]["takeovers"][0]["status"], "right_censored")
        self.assertEqual(result["metrics"]["takeovers"][0]["censor_time_seconds"], 6)

    def test_challenge_at_onset_belongs_to_impostor_and_has_zero_latency(self):
        planned = episode((Delivery(window(), 0, 2),), onset=12, end=16)
        result = self.replay(planned, DEFAULT_POLICIES[0])
        self.assertEqual(result["metrics"]["owner_false_challenges"], 0)
        self.assertEqual(result["first_challenge"]["identity"], "impostor")
        self.assertEqual(
            result["metrics"]["takeovers"][0]["time_to_challenge_seconds"], 0
        )

    def test_observation_gap_censors_without_inventing_sparse_exposure(self):
        planned = episode(
            (Delivery(window(), 0, 2),),
            end=20,
            intervals=(
                ObservedInterval(0, 2, "owner"),
                ObservedInterval(6, 8, "impostor"),
                ObservedInterval(15, 17, "impostor"),
            ),
        )
        # First challenge t=12 is beyond the first observation gap at t=8.
        planned = replace(planned, takeover=Takeover("handoff", 6, 8))
        result = self.replay(planned, DEFAULT_POLICIES[0])
        self.assertEqual(
            result["metrics"]["observed_seconds"],
            {"owner": 2, "impostor": 4, "invalid": 0},
        )
        outcome = result["metrics"]["takeovers"][0]
        self.assertEqual(outcome["status"], "right_censored")
        self.assertEqual(outcome["censor_time_seconds"], 2)
        with self.assertRaisesRegex(ValueError, "continuous observed impostor"):
            replay_episode(replace(planned, takeover=Takeover("bridge-gap", 6, 17)))

    def test_replay_and_renamed_overlap_rejected_across_presented_recording_ids(self):
        for name, error in (
            ("replay", "source_window_replayed"),
            ("overlap", "source_support_reused_or_reordered"),
        ):
            with self.subTest(scenario=name):
                result = self.replay(build_scenario(ScenarioConfig(scenario=name)))
                bad = result["score_windows"][6]
                self.assertEqual(bad["support_error"], error)
                self.assertEqual(bad["recording_id"], "reset-0")
                self.assertEqual(bad["event_type"], "fallback")
                self.assertEqual(result["first_challenge"]["source_rejection"], error)
                self.assertEqual(result["first_challenge"]["trigger"], "source_guard")
                decision = next(
                    row
                    for row in result["decisions"]
                    if row["timestamp"] == bad["delivered_at"]
                )
                self.assertFalse(decision["evidence_added"])

    def test_source_ids_cannot_reuse_different_support_under_same_window_id(self):
        original = window()
        renamed = replace(original, raw_start=4, raw_end=6)
        planned = episode((Delivery(original, 0, 2), Delivery(renamed, 2, 4, "new")))
        result = self.replay(planned)
        self.assertEqual(
            result["score_windows"][1]["support_error"], "source_window_replayed"
        )

    def test_gated_or_missing_score_support_still_cannot_be_reused(self):
        for gates in (
            {"quality_ok": False},
            {"walking_probability": 0},
            {"score": None},
        ):
            with self.subTest(gates=gates):
                raw = window(**gates)
                planned = episode(
                    (
                        Delivery(raw, 0, 2),
                        Delivery(
                            replace(raw, window_id="alias", score=1), 2, 4, "changed"
                        ),
                    )
                )
                result = self.replay(planned)
                self.assertEqual(
                    result["score_windows"][1]["support_error"],
                    "source_support_reused_or_reordered",
                )
                self.assertFalse(result["decisions"][2]["evidence_added"])

    def test_global_overlap_rejected_even_for_distinct_sources_and_recording_ids(self):
        planned = episode(
            (
                Delivery(window(), 0, 2),
                Delivery(window(source="other"), 1, 3, "changed"),
            )
        )
        result = self.replay(planned)
        self.assertEqual(
            result["score_windows"][1]["support_error"], "global_support_overlap"
        )
        self.assertEqual(result["first_challenge"]["reason"], "fallback_requested")

    def test_touching_raw_and_global_support_is_valid(self):
        planned = episode((Delivery(window(), 0, 2), Delivery(window(1), 2, 4)))
        result = self.replay(planned)
        self.assertEqual(result["support_rejection_count"], 0)
        self.assertTrue(result["warmup_had_accept"])

    def test_delayed_raw_age_includes_duration_and_latency(self):
        planned = build_scenario(
            ScenarioConfig(scenario="delayed_delivery", delay_seconds=4)
        )
        result = self.replay(planned)
        audit = result["score_windows"][6]
        self.assertEqual(audit["duration_seconds"], 2)
        self.assertEqual(audit["delivery_delay_seconds"], 4)
        self.assertEqual(audit["raw_age_seconds"], 6)
        decision = next(
            row
            for row in result["decisions"]
            if row["timestamp"] == audit["delivered_at"]
        )
        self.assertEqual(decision["evidence_age"], 6)
        self.assertEqual(result["support_rejection_count"], 0)

    def test_stale_raw_support_equality_rejected_and_silence_already_enforced(self):
        planned = build_scenario(
            ScenarioConfig(scenario="delayed_delivery", delay_seconds=28)
        )
        result = self.replay(planned)
        audit = result["score_windows"][6]
        self.assertEqual(audit["raw_age_seconds"], 30)
        self.assertEqual(audit["support_error"], "stale_raw_support")
        self.assertEqual(audit["event_type"], "fallback")
        self.assertEqual(result["first_challenge"]["timestamp"], 22)
        self.assertLess(result["first_challenge"]["timestamp"], audit["delivered_at"])
        self.assertEqual(result["first_challenge"]["trigger"], "policy")

    def test_conservative_delayed_support_cannot_overlap_next_capture(self):
        planned = episode(
            (Delivery(window(), 0, 4), Delivery(window(1), 2, 5)), onset=8
        )
        result = self.replay(planned)
        self.assertEqual(
            result["score_windows"][1]["support_error"], "global_support_overlap"
        )

    def test_raw_window_must_not_straddle_handoff(self):
        planned = episode((Delivery(window(), 5, 7),), onset=6)
        result = self.replay(planned)
        self.assertEqual(
            result["score_windows"][0]["support_error"], "support_crosses_handoff"
        )

    def test_blackout_gates_do_not_add_votes_or_hide_known_owner_exposure(self):
        for name, reason in (
            ("quality_blackout", "quality_gate"),
            ("walking_blackout", "not_walking"),
        ):
            result = self.replay(build_scenario(ScenarioConfig(scenario=name)))
            failed = next(row for row in result["decisions"] if row["timestamp"] == 16)
            self.assertEqual(failed["reason"], reason)
            self.assertFalse(failed["evidence_added"])
            self.assertEqual(result["first_challenge"]["timestamp"], 22)
            self.assertEqual(result["first_challenge"]["reason"], "evidence_gap")

    def test_owner_blackout_challenge_is_counted_before_handoff(self):
        planned = episode(
            (
                Delivery(window(index, quality_ok=False), index * 2, index * 2 + 2)
                for index in range(5)
            ),
            onset=12,
        )
        result = self.replay(planned)
        self.assertFalse(result["warmup_had_accept"])
        self.assertEqual(result["metrics"]["observed_seconds"]["owner"], 12)
        self.assertEqual(result["metrics"]["owner_false_challenges"], 1)
        self.assertEqual(
            result["metrics"]["takeovers"][0]["status"], "already_challenged"
        )

    def test_unaccepted_warmup_is_reported_without_silently_excluding_episode(self):
        result = self.replay(
            build_scenario(ScenarioConfig(owner_windows=1)), DEFAULT_POLICIES[2]
        )
        self.assertFalse(result["warmup_had_accept"])
        self.assertEqual(result["eligible_takeovers"], 1)

    def test_replay_matches_direct_shared_stream_without_new_policy_logic(self):
        planned = build_scenario(ScenarioConfig(scenario="bursty_acceptance"))
        result = self.replay(planned)
        stream = AuthenticationStream(
            DEFAULT_POLICIES[1].threshold, **DEFAULT_POLICIES[1].options()
        )
        expected = [stream.process_event(event) for event in result["events"]]
        for actual, reference in zip(result["decisions"], expected, strict=True):
            self.assertEqual(
                {key: value for key, value in actual.items() if key != "identity"},
                reference,
            )

    def test_grid_uses_fresh_stream_and_never_resets_within_episode(self):
        planned = build_scenario(ScenarioConfig())
        from engine.continuous_auth import ContinuousAuthenticator

        original_reset = ContinuousAuthenticator.reset
        calls = []

        def initial_reset(instance):
            # Constructor calls reset once; no later call is authorized here.
            self.assertFalse(any(prior is instance for prior in calls))
            calls.append(instance)
            original_reset(instance)

        with patch.object(ContinuousAuthenticator, "reset", initial_reset):
            first = run_grid([planned])
        self.assertEqual(len(calls), len(DEFAULT_POLICIES))
        again = run_grid([planned])
        self.assertEqual(first, again)
        reverse = run_grid([planned], reversed(DEFAULT_POLICIES))
        self.assertEqual(first["comparisons"], list(reversed(reverse["comparisons"])))

    def test_grid_aggregates_exposure_and_conditional_latency_by_hand(self):
        plans = [
            build_scenario(ScenarioConfig(scenario=name))
            for name in ("abrupt_takeover", "recording_resets", "owner_challenge")
        ]
        report = run_grid(plans, [DEFAULT_POLICIES[1]])
        summary = report["comparisons"][0]["summary"]
        self.assertEqual(summary["owner_observed_seconds"], 42)
        self.assertEqual(summary["impostor_observed_seconds"], 54)
        self.assertEqual(summary["owner_false_challenges"], 1)
        self.assertAlmostEqual(
            summary["owner_false_challenges_per_observed_hour"], 3600 / 42
        )
        self.assertEqual(summary["already_challenged_takeovers"], 1)
        self.assertEqual(summary["eligible_takeovers"], 2)
        self.assertEqual(summary["detected_takeovers"], 1)
        self.assertEqual(summary["right_censored_takeovers"], 1)
        self.assertEqual(summary["detected_fraction_of_eligible"], 0.5)
        self.assertEqual(summary["mean_detected_time_to_challenge_seconds"], 4)
        self.assertEqual(summary["seeds"], [7, 7, 7])

    def test_determinism_full_default_grid_and_rng_isolation(self):
        state = random.getstate()
        plans = [build_scenario(ScenarioConfig(scenario=name)) for name in SCENARIOS]
        self.assertEqual(random.getstate(), state)
        result = run_grid(plans)
        self.assertEqual(result, run_grid(plans))
        json.dumps(result, allow_nan=False)
        self.assertEqual(len(result["comparisons"]), 3)
        self.assertTrue(
            all(
                not cell["summary"]["invalid_takeovers"]
                for cell in result["comparisons"]
            )
        )
        changed = build_scenario(ScenarioConfig(seed=8))
        self.assertNotEqual(
            plans[0].deliveries[0].window.score, changed.deliveries[0].window.score
        )
        self.assertEqual(
            plans[0].deliveries[0].window.score_origin, "synthetic_injected"
        )

    def test_hostile_window_inputs_are_rejected_deterministically(self):
        bad_numbers: tuple[Any, ...] = (
            True,
            "2",
            float("nan"),
            float("inf"),
            10**1000,
            None,
            [],
            {},
        )
        for field in ("raw_start", "raw_end", "walking_probability", "score"):
            for bad in bad_numbers:
                if field == "score" and bad is None:
                    continue
                with (
                    self.subTest(field=field, bad=repr(bad)[:20]),
                    self.assertRaises(ValueError),
                ):
                    window(**{field: bad})
        for changes in (
            {"quality_ok": 1},
            {"raw_end": 0},
            {"raw_start": -1},
            {"walking_probability": 1.1},
            {"source_recording": ""},
            {"source_recording": "\ud800"},
            {"window_id": []},
            {"score_origin": "real_sensor"},
        ):
            with self.subTest(changes=repr(changes)), self.assertRaises(ValueError):
                replace(window(), **changes)
        self.assertEqual(window(score=-100).score, -100)
        self.assertEqual(window(score=None).score, None)
        self.assertEqual(
            window(score_origin="external_unverified").score_origin,
            "external_unverified",
        )

    def test_bad_delivery_order_and_future_support_are_not_silently_repaired(self):
        for capture, available in (
            (0, 1),
            (-1, 2),
            (True, 2),
            (0, float("nan")),
            (1e308, 1e308),
        ):
            with self.subTest(capture=capture), self.assertRaises(ValueError):
                Delivery(window(), capture, available)
        for ordered in (
            (Delivery(window(1), 2, 4), Delivery(window(), 0, 2)),
            (Delivery(window(), 0, 2), Delivery(window(source="another"), 0, 2)),
        ):
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                episode(ordered)

    def test_invalid_configs_and_unbounded_timer_work_are_rejected(self):
        invalid_configs: tuple[dict[str, Any], ...] = (
            {"seed": True},
            {"seed": -1},
            {"seed": 2**53},
            {"owner_windows": 0},
            {"attacker_windows": 1.5},
            {"timer_seconds": 0},
            {"window_seconds": float("inf")},
            {"delay_seconds": -1},
            {"scenario": "unknown"},
            {"accept_burst_windows": 0},
            {"blackout_windows": False},
            {"score_jitter": float("nan")},
        )
        for changes in invalid_configs:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ScenarioConfig(**changes)
        with self.assertRaisesRegex(ValueError, "budget"):
            build_scenario(ScenarioConfig(timer_seconds=1e-9))
        with self.assertRaises(ValueError):
            replace(DEFAULT_POLICIES[1], required_accepts=4)
        with self.assertRaises(ValueError):
            run_grid([])
        with self.assertRaisesRegex(ValueError, "unique"):
            run_grid([build_scenario()] * 2)
        with self.assertRaisesRegex(ValueError, "unique"):
            run_grid([build_scenario()], [DEFAULT_POLICIES[1]] * 2)

    def test_fractional_capture_support_remains_exactly_nonoverlapping(self):
        for duration in (0.1, 0.2, 0.3, 1 / 3, 0.7, 1.1):
            for name in SCENARIOS:
                with self.subTest(duration=duration, scenario=name):
                    result = self.replay(
                        build_scenario(
                            ScenarioConfig(scenario=name, window_seconds=duration)
                        )
                    )
                    rejected = [
                        row for row in result["score_windows"] if row["support_error"]
                    ]
                    self.assertEqual(len(rejected), int(name in ("replay", "overlap")))


if __name__ == "__main__":
    unittest.main()
