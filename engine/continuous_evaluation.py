"""Strict causal score replay and explicitly timed evaluation; standard library only.

No models, data, calibration, or fallback verification are performed here.
Observation metadata is trusted evaluation input, never an authentication event.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from math import fsum, isfinite
from typing import Any

from engine.continuous_auth import ContinuousAuthenticator

_IDENTITIES = ("owner", "impostor", "invalid")
_POLICY_INPUT_ERRORS = {
    "invalid_recording_id",
    "invalid_walking_probability",
    "invalid_quality_flag",
    "invalid_score",
    "invalid_window_start",
    "stale_window",
    "overlapping_evidence",
    "missing_window_start",
}


def _number(value: object, name: str) -> float:
    # Accept JSON numbers, not booleans, numeric strings, or arbitrary coercions.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid_{name}")  # noqa: TRY004 -- uniform validation contract
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"invalid_{name}") from error
    if not isfinite(normalized):
        raise ValueError(f"invalid_{name}")
    return normalized


def _time(value: object, name: str = "timestamp") -> float:
    normalized = _number(value, name)
    if normalized < 0:
        raise ValueError(f"invalid_{name}")
    return normalized


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError("nonfinite_json_number")


def _json_float(value: str) -> float:
    return _number(float(value), "json_number")


def strict_json_loads(line: str) -> Any:
    """Reject duplicate keys, NaN/Infinity, float overflow, and raw surrogates.

    A JSON document is not yet a valid stream event. Escaped surrogates in
    recording IDs are also rejected by the event schema's UTF-8 validation.
    """
    line.encode("utf-8", errors="strict")
    return json.loads(
        line,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
        parse_float=_json_float,
    )


class AuthenticationStream:
    """One authentication episode, with no untrusted reset/recovery operation.

    ``score`` events require explicit score, walking probability, quality flag,
    recording ID, and timestamp. ``window_start`` is optional, subject to the
    underlying policy's nonoverlap contract. ``advance`` and ``fallback`` only
    carry a timestamp. Any malformed event permanently challenges this stream.
    A trusted caller may start a *new instance* after verifying fallback.
    """

    def __init__(self, threshold: float, **policy_options: Any):
        self._policy = ContinuousAuthenticator(threshold, **policy_options)
        self._last_timestamp: float | None = None
        self._first_timestamp: float | None = None
        self._protocol_error: str | None = None
        self._challenged = False
        self._last_result: dict[str, Any] = {
            "decision": "defer",
            "reason": "awaiting_evidence",
            "authenticated": False,
            "challenge_required": False,
            "timestamp": None,
            "recording_id": None,
            "recording_changed": False,
            "evidence_added": False,
            "evidence_count": 0,
            "accept_count": 0,
            "reject_count": 0,
            "evidence_age": None,
            "oldest_evidence_age": None,
        }

    @property
    def first_timestamp(self) -> float | None:
        return self._first_timestamp

    @property
    def last_timestamp(self) -> float | None:
        return self._last_timestamp

    def _clock(self, event: object) -> float | None:
        if type(event) is not dict:
            return None
        try:
            timestamp = _time(event.get("timestamp"))
        except ValueError:
            return None
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            return None
        return timestamp

    def _validate(self, event: Any) -> dict[str, Any]:
        if type(event) is not dict:
            raise ValueError("event_must_be_object")
        kind = event.get("type")
        if type(kind) is not str or kind not in ("score", "advance", "fallback"):
            raise ValueError("unsupported_event_type")
        required = {"type", "timestamp"}
        optional: set[str] = set()
        if kind == "score":
            required |= {"score", "walking_probability", "quality_ok", "recording_id"}
            optional = {"window_start"}
        if required - event.keys():
            raise ValueError("missing_event_field")
        if event.keys() - required - optional:
            raise ValueError("unknown_event_field")
        timestamp = _time(event["timestamp"])
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("nonmonotonic_timestamp")
        normalized = dict(event, timestamp=timestamp)
        if kind != "score":
            return normalized
        if event["score"] is not None:
            normalized["score"] = _number(event["score"], "score")
        walking = _number(event["walking_probability"], "walking_probability")
        if not 0 <= walking <= 1:
            raise ValueError("invalid_walking_probability")
        normalized["walking_probability"] = walking
        if type(event["quality_ok"]) is not bool:
            raise ValueError("invalid_quality_flag")
        recording = event["recording_id"]
        if not (
            recording is None
            or type(recording) is str
            or (type(recording) is int and abs(recording) <= 2**53 - 1)
        ):
            raise ValueError("invalid_recording_id")
        if type(recording) is str:
            try:
                recording.encode("utf-8", errors="strict")
            except UnicodeError as error:
                raise ValueError("invalid_recording_id") from error
        if event.get("window_start") is not None:
            start = _time(event["window_start"], "window_start")
            if start >= timestamp:
                raise ValueError("invalid_window_start")
            if timestamp - start >= self._policy.max_evidence_age_seconds:
                raise ValueError("stale_window")
            normalized["window_start"] = start
        return normalized

    def _finish(
        self,
        result: dict[str, Any],
        *,
        event_type: str | None,
        error: str | None = None,
    ) -> dict[str, Any]:
        output = dict(result)
        output.update(
            event_type=event_type,
            input_valid=error is None,
            event_error=error,
            latched_error=self._protocol_error,
            policy_reason=result["reason"],
        )
        if self._protocol_error is not None:
            output.update(
                decision="challenge",
                reason="invalid_event",
                authenticated=False,
                challenge_required=True,
            )
        output["challenge_started"] = (
            output["challenge_required"] and not self._challenged
        )
        self._challenged = self._challenged or output["challenge_required"]
        return output

    def _invalid(self, error: str, event: object = None) -> dict[str, Any]:
        if self._protocol_error is None:
            self._protocol_error = error
        timestamp = self._clock(event)
        if timestamp is not None:
            self._remember_time(timestamp)
            self._last_result = self._policy.advance(timestamp)
            result = self._last_result
        else:
            # No invented time, clock rewind, or reuse of an earlier accept.
            result = dict(self._last_result)
            result.update(
                timestamp=None,
                evidence_added=False,
                recording_changed=False,
                evidence_age=None,
                oldest_evidence_age=None,
            )
        return self._finish(result, event_type=None, error=error)

    def _remember_time(self, timestamp: float) -> None:
        if self._first_timestamp is None:
            self._first_timestamp = timestamp
        self._last_timestamp = timestamp

    def process_event(self, event: object) -> dict[str, Any]:
        """Validate and consume one event; runtime errors return a latched challenge.

        A schema-invalid event with a valid later timestamp consumes that clock
        time via ``advance``. Missing/invalid/reversed times never move the clock.
        After a protocol failure, even valid scores only advance time.
        """
        try:
            normalized = self._validate(event)
        except ValueError as error:
            return self._invalid(str(error), event)
        timestamp = normalized["timestamp"]
        self._remember_time(timestamp)
        kind = normalized["type"]
        if self._protocol_error is not None or kind == "advance":
            result = self._policy.advance(timestamp)
        elif kind == "fallback":
            result = self._policy.request_fallback(timestamp)
        else:
            result = self._policy.update(
                normalized["score"],
                timestamp,
                walking_probability=normalized["walking_probability"],
                quality_ok=normalized["quality_ok"],
                recording_id=normalized["recording_id"],
                window_start=normalized.get("window_start"),
            )
        error = None
        if not self._challenged and result["reason"] in _POLICY_INPUT_ERRORS:
            error = result["reason"]
        self._last_result = result
        return self._finish(result, event_type=kind, error=error)

    def process_line(self, line: str) -> dict[str, Any]:
        """Consume one strict JSONL line, including malformed/blank lines."""
        try:
            event = strict_json_loads(line)
        except ValueError, UnicodeError, RecursionError, OverflowError:
            return self._invalid("invalid_json")
        return self.process_event(event)


@dataclass(frozen=True)
class ObservedInterval:
    """Trusted, truly observed identity exposure on half-open ``[start, end)``.

    ``invalid`` means identity/exposure is unusable for owner/impostor metrics.
    A failed walking/quality gate alone does not make known owner exposure invalid.
    """

    start: float
    end: float
    identity: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _time(self.start, "interval_start"))
        object.__setattr__(self, "end", _time(self.end, "interval_end"))
        if self.end <= self.start:
            raise ValueError("observed interval must have positive duration")
        if type(self.identity) is not str or self.identity not in _IDENTITIES:
            raise ValueError("identity must be owner, impostor, or invalid")


@dataclass(frozen=True)
class Takeover:
    """A declared takeover observed from ``onset`` until exclusive censor ``end``.

    The entire interval must have observed impostor exposure; end at the first
    observation gap. A challenge at ``end`` is outside this attack's follow-up.
    """

    attack_id: str
    onset: float
    end: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "onset", _time(self.onset, "takeover_onset"))
        object.__setattr__(self, "end", _time(self.end, "takeover_end"))
        if type(self.attack_id) is not str or not self.attack_id:
            raise ValueError("attack_id must be a nonempty string")
        if self.end <= self.onset:
            raise ValueError("takeover must have positive follow-up")


def _metadata(
    observed_intervals: Iterable[ObservedInterval], takeovers: Iterable[Takeover]
) -> tuple[list[ObservedInterval], list[Takeover]]:
    intervals = list(observed_intervals)
    attacks = list(takeovers)
    if any(not isinstance(item, ObservedInterval) for item in intervals):
        raise ValueError("observed_intervals must contain ObservedInterval values")
    if any(not isinstance(item, Takeover) for item in attacks):
        raise ValueError("takeovers must contain Takeover values")
    intervals.sort(key=lambda item: item.start)
    attacks.sort(key=lambda item: item.onset)
    if any(right.start < left.end for left, right in pairwise(intervals)):
        raise ValueError("observed intervals must not overlap on the global clock")
    if len({attack.attack_id for attack in attacks}) != len(attacks):
        raise ValueError("attack IDs must be unique")
    if any(right.onset < left.end for left, right in pairwise(attacks)):
        raise ValueError("takeovers must not overlap")
    for attack in attacks:
        covered_until = attack.onset
        for interval in intervals:
            if interval.end <= covered_until:
                continue
            if interval.start > covered_until or interval.identity != "impostor":
                break
            covered_until = interval.end
            if covered_until >= attack.end:
                break
        if covered_until < attack.end:
            raise ValueError("takeover requires continuous observed impostor exposure")
    return intervals, attacks


def evaluate_stream(
    events: Iterable[object],
    *,
    threshold: float,
    observed_intervals: Iterable[ObservedInterval],
    takeovers: Iterable[Takeover] = (),
    **policy_options: Any,
) -> dict[str, Any]:
    """Replay one episode causally and return decisions plus descriptive metrics.

    Each event is a parsed object, or a JSONL string parsed strictly. Labels never
    reach the policy. Explicit intervals, not recording spans or score spacing,
    determine exposure. Start/end clock observations must bracket all metadata;
    send an initial/terminal ``advance`` if necessary. No future clock is invented.

    Malformed streams remain challenged and are marked invalid for metric claims.
    Repeated latched outputs count as one challenge episode. Already-challenged
    takeovers are excluded; undetected eligible attacks are right-censored with
    null latency. The mean is explicitly conditional on detected attacks only.
    """
    intervals, attacks = _metadata(observed_intervals, takeovers)
    stream = AuthenticationStream(threshold, **policy_options)
    decisions = [
        stream.process_line(event)
        if isinstance(event, str)
        else stream.process_event(event)
        for event in events
    ]
    if intervals and (
        stream.first_timestamp is None
        or stream.last_timestamp is None
        or intervals[0].start < stream.first_timestamp
        or intervals[-1].end > stream.last_timestamp
    ):
        raise ValueError("stream clock must bracket all observed intervals")
    starts = [interval.start for interval in intervals]

    def identity_at(timestamp: float | None) -> str:
        if timestamp is None:
            return "invalid"
        index = bisect_right(starts, timestamp) - 1
        if index >= 0 and timestamp < intervals[index].end:
            return intervals[index].identity
        return "unobserved"

    identities = (*_IDENTITIES, "unobserved")
    event_counts = Counter({identity: 0 for identity in identities})
    challenge_counts = Counter(event_counts)
    first_challenge = None
    invalid_events = 0
    challenge_outputs = 0
    for decision in decisions:
        identity = identity_at(decision["timestamp"])
        decision["identity"] = identity
        event_counts[identity] += 1
        invalid_events += not decision["input_valid"]
        challenge_outputs += decision["challenge_required"]
        if decision["challenge_started"]:
            challenge_counts[identity] += 1
            first_challenge = decision["timestamp"]
    exposure = {
        identity: fsum(
            interval.end - interval.start
            for interval in intervals
            if interval.identity == identity
        )
        for identity in _IDENTITIES
    }
    if not all(isfinite(value) for value in exposure.values()):
        raise ValueError("exposure exceeds finite supported range")
    valid = invalid_events == 0
    owner_hours = exposure["owner"] / 3600
    owner_rate = (
        challenge_counts["owner"] / owner_hours if valid and owner_hours > 0 else None
    )
    if owner_rate is not None and not isfinite(owner_rate):
        raise ValueError("owner false-challenge rate exceeds finite supported range")
    outcomes = []
    detected_times = []
    for attack in attacks:
        duration = attack.end - attack.onset
        latency = None
        censor_time = None
        if not valid:
            status = "invalid_stream"
        elif first_challenge is not None and first_challenge < attack.onset:
            status = "already_challenged"
        elif first_challenge is not None and first_challenge < attack.end:
            status = "detected"
            latency = first_challenge - attack.onset
            detected_times.append(latency)
        else:
            status = "right_censored"
            censor_time = duration
        outcomes.append(
            {
                "attack_id": attack.attack_id,
                "onset": attack.onset,
                "end": attack.end,
                "status": status,
                "follow_up_seconds": duration,
                "time_to_challenge_seconds": latency,
                "censor_time_seconds": censor_time,
            }
        )
    status_counts = Counter(item["status"] for item in outcomes)
    metrics = {
        "valid": valid,
        "event_count": len(decisions),
        "invalid_event_count": invalid_events,
        "event_counts_by_identity": dict(event_counts),
        "observed_seconds": exposure,
        "owner_observed_hours": owner_hours,
        "owner_false_challenges": challenge_counts["owner"],
        "owner_false_challenges_per_observed_hour": owner_rate,
        "challenge_counts_by_identity": dict(challenge_counts),
        "challenge_episode_count": sum(challenge_counts.values()),
        "challenge_output_count": challenge_outputs,
        "takeovers": outcomes,
        "detected_takeovers": status_counts["detected"],
        "right_censored_takeovers": status_counts["right_censored"],
        "already_challenged_takeovers": status_counts["already_challenged"],
        "invalid_takeovers": status_counts["invalid_stream"],
        "mean_detected_time_to_challenge_seconds": (
            fsum(value / len(detected_times) for value in detected_times)
            if detected_times
            else None
        ),
    }
    return {"decisions": decisions, "metrics": metrics}
