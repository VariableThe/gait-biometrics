"""Deterministic synthetic-splice stress replay; no sensors, models, or fitting.

Source support is checked before events reach the existing AuthenticationStream
via evaluate_stream. Rejected support requests fallback, never a fabricated vote.
All exposure is explicitly declared simulation truth, not inferred recording time.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from math import fsum, isfinite
from random import Random
from typing import Any

from engine.continuous_evaluation import (
    AuthenticationStream,
    ObservedInterval,
    Takeover,
    evaluate_stream,
)

SCENARIOS = (
    "abrupt_takeover",
    "bursty_acceptance",
    "walking_blackout",
    "quality_blackout",
    "silence",
    "delayed_delivery",
    "replay",
    "overlap",
    "recording_resets",
    "owner_challenge",
)
_MAX_EVENTS = 100_000


def _number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")  # noqa: TRY004 -- uniform validation contract
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
        )
    return result


def _integer(value: object, name: str, maximum: int = 2**53 - 1) -> None:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be a nonnegative integer <= {maximum}")


def _text(value: object, name: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty UTF-8 string")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ValueError(f"{name} must be a nonempty UTF-8 string") from error


@dataclass(frozen=True)
class ScoreWindow:
    """A source-bound score, with half-open support in its source's raw clock.

    ``duration_seconds`` is derived, so contradictory durations cannot be given.
    ``external_unverified`` is an adapter label, not a sensor/model attestation.
    Raw coordinates and IDs must refer to canonical recordings, even under attack.
    """

    source_recording: str
    window_id: str
    raw_start: float
    raw_end: float
    score: float | None
    walking_probability: float = 1
    quality_ok: bool = True
    score_origin: str = "synthetic_injected"

    def __post_init__(self) -> None:
        _text(self.source_recording, "source_recording")
        _text(self.window_id, "window_id")
        for name in ("raw_start", "raw_end", "walking_probability"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if self.raw_end <= self.raw_start:
            raise ValueError("raw support must have positive duration")
        if self.walking_probability > 1:
            raise ValueError("walking_probability must be in [0, 1]")
        if type(self.quality_ok) is not bool:
            raise ValueError("quality_ok must be boolean")
        if self.score is not None:
            # Similarity scores can be negative; unlike times they are unbounded.
            if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
                raise ValueError("score must be finite or None")
            try:
                value = float(self.score)
            except (OverflowError, ValueError) as error:
                raise ValueError("score must be finite or None") from error
            if not isfinite(value):
                raise ValueError("score must be finite or None")
            object.__setattr__(self, "score", value)
        if self.score_origin not in ("synthetic_injected", "external_unverified"):
            raise ValueError("unsupported score_origin")

    @property
    def duration_seconds(self) -> float:
        return self.raw_end - self.raw_start


@dataclass(frozen=True)
class Delivery:
    """Map source support to capture time and later score availability.

    The policy receives [capture_start, delivered_at), conservatively including
    delivery delay. ``recording_id`` is the presented ID; it cannot erase source
    provenance or global support watermarks in this harness.
    """

    window: ScoreWindow
    capture_start: float
    delivered_at: float
    recording_id: str = "synthetic-device"

    def __post_init__(self) -> None:
        if not isinstance(self.window, ScoreWindow):
            raise ValueError("window must be a ScoreWindow")  # noqa: TRY004 -- uniform validation contract
        _text(self.recording_id, "recording_id")
        for name in ("capture_start", "delivered_at"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if (
            not isfinite(self.capture_end)
            or self.capture_end <= self.capture_start
            or self.delivered_at < self.capture_end
        ):
            raise ValueError("delivery must follow representable complete raw capture")

    @property
    def capture_end(self) -> float:
        return self.capture_start + self.window.duration_seconds

    @property
    def raw_age_seconds(self) -> float:
        return self.delivered_at - self.capture_start


@dataclass(frozen=True)
class Episode:
    """One owner warmup -> handoff; no reset or fallback recovery inside it.

    Explicit observed intervals may have gaps. Takeover follow-up must stop at
    the first gap; evaluate_stream enforces continuous observed impostor truth.
    Delivery order must already be strictly increasing; it is never repaired.
    """

    episode_id: str
    seed: int
    start: float
    end: float
    takeover: Takeover
    deliveries: tuple[Delivery, ...]
    observed_intervals: tuple[ObservedInterval, ...]
    timer_seconds: float = 1
    scenario_config: ScenarioConfig | None = None

    def __post_init__(self) -> None:
        _text(self.episode_id, "episode_id")
        _integer(self.seed, "seed")
        for name in ("start", "end", "timer_seconds"):
            object.__setattr__(
                self,
                name,
                _number(getattr(self, name), name, positive=name == "timer_seconds"),
            )
        if not isinstance(self.takeover, Takeover):
            raise ValueError("takeover must be a Takeover")  # noqa: TRY004 -- uniform validation contract
        if not self.start < self.takeover.onset < self.takeover.end <= self.end:
            raise ValueError(
                "episode must have positive owner warmup and takeover follow-up"
            )
        object.__setattr__(self, "deliveries", tuple(self.deliveries))
        object.__setattr__(self, "observed_intervals", tuple(self.observed_intervals))
        previous = self.start
        for delivery in self.deliveries:
            if not isinstance(delivery, Delivery):
                raise ValueError("deliveries must contain Delivery values")  # noqa: TRY004 -- uniform validation contract
            if (
                delivery.capture_start < self.start
                or not previous < delivery.delivered_at <= self.end
            ):
                raise ValueError(
                    "deliveries require strictly increasing global times within episode"
                )
            previous = delivery.delivered_at
        has_owner = False
        for interval in self.observed_intervals:
            if not isinstance(interval, ObservedInterval):
                raise ValueError(  # noqa: TRY004 -- uniform validation contract
                    "observed_intervals must contain ObservedInterval values"
                )
            if interval.start < self.start or interval.end > self.end:
                raise ValueError("observed intervals must lie inside episode")
            if interval.identity == "owner":
                has_owner = True
                if interval.end > self.takeover.onset:
                    raise ValueError("owner observation must precede handoff")
            if interval.identity == "impostor" and interval.start < self.takeover.onset:
                raise ValueError("impostor observation must follow handoff")
        if not has_owner:
            raise ValueError("explicit observed owner warmup is required")
        span = self.end - self.start
        if span / self.timer_seconds > _MAX_EVENTS - len(self.deliveries) - 3:
            raise ValueError("episode exceeds timer/event budget")
        if self.start + self.timer_seconds <= self.start:
            raise ValueError("timer cadence is not representable on this clock")


@dataclass(frozen=True)
class PolicySpec:
    """Illustrative fixed policy settings; threshold is never calibrated here."""

    name: str = "2-of-3-veto"
    threshold: float = 0.5
    history_size: int = 3
    required_accepts: int = 2
    required_rejects: int = 2
    walking_threshold: float = 0.8
    max_evidence_age_seconds: float = 30
    max_gap_seconds: float = 10

    def __post_init__(self) -> None:
        _text(self.name, "policy name")
        AuthenticationStream(self.threshold, **self.options())

    def options(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in ("name", "threshold")
        }


DEFAULT_POLICIES = (
    PolicySpec("1-of-1-veto", history_size=1, required_accepts=1, required_rejects=1),
    PolicySpec(),
    PolicySpec("3-of-5-veto", history_size=5, required_accepts=3, required_rejects=2),
)


def _events(
    episode: Episode, policy: PolicySpec
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: dict[float, dict[str, Any]] = {}
    # Multiply a tick index rather than accumulating floating point drift.
    count = int((episode.end - episode.start) / episode.timer_seconds)
    previous = episode.start
    for index in range(1, count + 1):
        timestamp = episode.start + index * episode.timer_seconds
        if timestamp <= previous:
            raise ValueError("timer cadence is not strictly representable")
        if timestamp < episode.end:
            events[timestamp] = {"type": "advance", "timestamp": timestamp}
        previous = timestamp
    # Evaluation identity boundaries do not inject extra policy clock calls.
    for timestamp in (episode.start, episode.end):
        events[timestamp] = {"type": "advance", "timestamp": timestamp}
    seen = set()
    source_ends: dict[str, float] = {}
    last_end = episode.start
    audit = []
    for delivery in episode.deliveries:
        window = delivery.window
        key = (window.source_recording, window.window_id)
        error = None
        if key in seen:
            error = "source_window_replayed"
        elif window.raw_start < source_ends.get(window.source_recording, 0):
            error = "source_support_reused_or_reordered"
        elif delivery.capture_start < last_end:
            error = "global_support_overlap"
        elif delivery.capture_start < episode.takeover.onset < delivery.capture_end:
            error = "support_crosses_handoff"
        elif delivery.raw_age_seconds >= policy.max_evidence_age_seconds:
            error = "stale_raw_support"
        seen.add(key)
        source_ends[window.source_recording] = max(
            window.raw_end, source_ends.get(window.source_recording, 0)
        )
        last_end = delivery.delivered_at
        event: dict[str, Any] = {"type": "fallback", "timestamp": delivery.delivered_at}
        if error is None:
            event.update(
                type="score",
                score=window.score,
                walking_probability=window.walking_probability,
                quality_ok=window.quality_ok,
                recording_id=delivery.recording_id,
                window_start=delivery.capture_start,
            )
        # A score/fallback at a timer instant performs that clock check itself.
        events[delivery.delivered_at] = event
        audit.append(
            {
                **asdict(delivery),
                "duration_seconds": window.duration_seconds,
                "capture_end": delivery.capture_end,
                "raw_age_seconds": delivery.raw_age_seconds,
                "delivery_delay_seconds": delivery.delivered_at - delivery.capture_end,
                "support_error": error,
                "event_type": event["type"],
            }
        )
    return [events[time] for time in sorted(events)], audit


def replay_episode(
    episode: Episode,
    policy: PolicySpec = DEFAULT_POLICIES[1],
    *,
    include_trace: bool = False,
) -> dict[str, Any]:
    """Run one fresh stream, retaining every owner challenge and censor record.

    Support rejection requests fallback at availability. Thus a detected endpoint
    can be a harness provenance guard or timeout, not biometric discrimination.
    """
    events, audit = _events(episode, policy)
    result = evaluate_stream(
        events,
        threshold=policy.threshold,
        observed_intervals=episode.observed_intervals,
        takeovers=[episode.takeover],
        **policy.options(),
    )
    metrics = result["metrics"]
    decisions = result["decisions"]
    warmup = [row for row in decisions if row["timestamp"] < episode.takeover.onset]
    first = next((row for row in decisions if row["challenge_started"]), None)
    rejected = {
        row["delivered_at"]: row["support_error"]
        for row in audit
        if row["support_error"]
    }
    report = {
        "episode_id": episode.episode_id,
        "seed": episode.seed,
        "policy": asdict(policy),
        "start": episode.start,
        "end": episode.end,
        "timer_seconds": episode.timer_seconds,
        "scenario_config": asdict(episode.scenario_config)
        if episode.scenario_config is not None
        else None,
        "observed_intervals": [asdict(item) for item in episode.observed_intervals],
        "takeover": asdict(episode.takeover),
        "warmup_had_accept": any(row["authenticated"] for row in warmup),
        "last_pre_handoff_decision": warmup[-1]["decision"],
        "eligible_takeovers": metrics["detected_takeovers"]
        + metrics["right_censored_takeovers"],
        "support_rejection_count": len(rejected),
        "first_challenge": None
        if first is None
        else {
            "timestamp": first["timestamp"],
            "reason": first["reason"],
            "source_rejection": rejected.get(first["timestamp"]),
            "trigger": "source_guard"
            if first["reason"] == "fallback_requested"
            and first["timestamp"] in rejected
            else "policy",
            "identity": first["identity"],
        },
        "score_windows": audit,
        "metrics": metrics,
    }
    if include_trace:
        report.update(events=events, decisions=decisions)
    return report


@dataclass(frozen=True)
class ScenarioConfig:
    """Small artificial scenarios; seeded jitter is not empirical variability."""

    scenario: str = "abrupt_takeover"
    seed: int = 7
    owner_windows: int = 6
    attacker_windows: int = 8
    window_seconds: float = 2
    timer_seconds: float = 1
    delay_seconds: float = 4
    blackout_windows: int = 6
    accept_burst_windows: int = 3
    reject_burst_windows: int = 2
    score_jitter: float = 0.05

    def __post_init__(self) -> None:
        if self.scenario not in SCENARIOS:
            raise ValueError("unsupported scenario")
        _integer(self.seed, "seed")
        for name in (
            "owner_windows",
            "attacker_windows",
            "blackout_windows",
            "accept_burst_windows",
            "reject_burst_windows",
        ):
            _integer(getattr(self, name), name, 10_000)
            if name != "blackout_windows" and getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "window_seconds",
            "timer_seconds",
            "delay_seconds",
            "score_jitter",
        ):
            object.__setattr__(
                self,
                name,
                _number(
                    getattr(self, name),
                    name,
                    positive=name in ("window_seconds", "timer_seconds"),
                ),
            )


def build_scenario(config: ScenarioConfig | None = None) -> Episode:
    """Create explicitly synthetic windows and continuous assumed identity truth.

    Delayed capture uses a stretched cadence so conservative availability-ended
    support does not overlap. Silence has known simulated presence, no scores.
    """
    if config is None:
        config = ScenarioConfig()
    rng = Random(config.seed)
    duration = config.window_seconds
    delay = config.delay_seconds if config.scenario == "delayed_delivery" else 0
    deliveries = []

    def window(
        source: str, index: int, capture: float, value: float, **gates: Any
    ) -> ScoreWindow:
        return ScoreWindow(
            source,
            f"{source}-{index}",
            capture,
            capture + duration,
            value + rng.uniform(-config.score_jitter, config.score_jitter),
            **gates,
        )

    capture = 0.0
    for index in range(config.owner_windows):
        value = 0 if config.scenario == "owner_challenge" else 1
        candidate = window("synthetic-owner", index, capture, value)
        deliveries.append(
            Delivery(
                candidate,
                capture,
                capture + candidate.duration_seconds,
            )
        )
        capture = deliveries[-1].delivered_at
    onset = capture + duration
    capture = onset
    if config.scenario != "silence":
        for index in range(config.attacker_windows):
            value = 0
            if config.scenario == "bursty_acceptance":
                period = config.accept_burst_windows + config.reject_burst_windows
                value = int(index % period < config.accept_burst_windows)
            gates = {
                "walking_probability": 0
                if config.scenario == "walking_blackout"
                and index < config.blackout_windows
                else 1,
                "quality_ok": not (
                    config.scenario == "quality_blackout"
                    and index < config.blackout_windows
                ),
            }
            candidate = window("synthetic-impostor", index, capture, value, **gates)
            if index == 0 and config.scenario == "replay":
                candidate = deliveries[-1].window
            if index == 0 and config.scenario == "overlap":
                owner_last = deliveries[-1].window
                raw_start = owner_last.raw_start + owner_last.duration_seconds / 2
                candidate = ScoreWindow(
                    "synthetic-owner",
                    "renamed-overlap",
                    raw_start,
                    raw_start + duration,
                    1,
                )
            recording = (
                f"reset-{index}"
                if config.scenario in ("recording_resets", "overlap", "replay")
                else "synthetic-device"
            )
            deliveries.append(
                Delivery(
                    candidate,
                    capture,
                    capture + candidate.duration_seconds + delay,
                    recording,
                )
            )
            # Use the actual prior endpoint as the next capture boundary. This
            # preserves exact touching support for fractional-second durations.
            capture = deliveries[-1].delivered_at
    else:
        capture = onset + config.attacker_windows * duration
    end = capture + duration
    return Episode(
        f"{config.scenario}:seed-{config.seed}",
        config.seed,
        0,
        end,
        Takeover("synthetic-handoff", onset, end),
        tuple(deliveries),
        (ObservedInterval(0, onset, "owner"), ObservedInterval(onset, end, "impostor")),
        config.timer_seconds,
        config,
    )


def run_grid(
    episodes: Iterable[Episode],
    policies: Iterable[PolicySpec] = DEFAULT_POLICIES,
    *,
    include_trace: bool = False,
) -> dict[str, Any]:
    """Compare the same episodes with a fresh policy per episode/grid cell.

    Aggregates are descriptive counts, pooled exposure and detected-only means.
    Reusing sources/seeds across policies or episodes never implies independence.
    """
    episodes, policies = tuple(episodes), tuple(policies)
    if not episodes or not policies:
        raise ValueError("grid requires episodes and policies")
    if len({item.episode_id for item in episodes}) != len(episodes):
        raise ValueError("episode IDs must be unique")
    if len({item.name for item in policies}) != len(policies):
        raise ValueError("policy names must be unique")
    comparisons = []
    for policy in policies:
        reports = [
            replay_episode(episode, policy, include_trace=include_trace)
            for episode in episodes
        ]
        metrics = [report["metrics"] for report in reports]
        owner_seconds = fsum(item["observed_seconds"]["owner"] for item in metrics)
        owner_hours = owner_seconds / 3600
        owner_challenges = sum(item["owner_false_challenges"] for item in metrics)
        latencies = [
            attack["time_to_challenge_seconds"]
            for item in metrics
            for attack in item["takeovers"]
            if attack["status"] == "detected"
        ]
        totals = {
            key: sum(item[key] for item in metrics)
            for key in (
                "detected_takeovers",
                "right_censored_takeovers",
                "already_challenged_takeovers",
                "invalid_takeovers",
            )
        }
        eligible = totals["detected_takeovers"] + totals["right_censored_takeovers"]
        summary = {
            **totals,
            "episodes": len(reports),
            "seeds": [episode.seed for episode in episodes],
            "eligible_takeovers": eligible,
            "owner_observed_seconds": owner_seconds,
            "impostor_observed_seconds": fsum(
                item["observed_seconds"]["impostor"] for item in metrics
            ),
            "owner_false_challenges": owner_challenges,
            "owner_false_challenges_per_observed_hour": owner_challenges / owner_hours
            if owner_hours > 0 and all(item["valid"] for item in metrics)
            else None,
            "detected_fraction_of_eligible": totals["detected_takeovers"] / eligible
            if eligible
            else None,
            "mean_detected_time_to_challenge_seconds": fsum(
                value / len(latencies) for value in latencies
            )
            if latencies
            else None,
        }
        comparisons.append(
            {"policy": asdict(policy), "summary": summary, "episodes": reports}
        )
    return {
        "kind": "synthetic_splice_stress_replay",
        "policy_family": "bounded K-of-N with contradiction veto; not SPRT",
        "claims": "Descriptive synthetic replay only; no independence, empirical sensor accuracy, or daylong claims.",
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help="repeat for explicitly listed seeds (default: 7)",
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        action="append",
        help="repeat for selected scenarios (default: all)",
    )
    parser.add_argument(
        "--trace", action="store_true", help="include policy events and decisions"
    )
    args = parser.parse_args()
    try:
        episodes = [
            build_scenario(ScenarioConfig(scenario=name, seed=seed))
            for name in (args.scenario or SCENARIOS)
            for seed in (args.seed or [7])
        ]
        report = run_grid(episodes, include_trace=args.trace)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, allow_nan=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
