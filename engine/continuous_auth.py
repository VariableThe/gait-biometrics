"""Causal, bounded score-level authentication policy; standard library only.

No model, calibration, data loading, enrollment, or first-unlock behavior lives
here. See docs/continuous_auth.md for clock and nonoverlap requirements.
"""

from __future__ import annotations

from collections import deque
from math import isfinite
from numbers import Integral, Real
from sys import maxsize


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        value = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return value if isfinite(value) else None


class ContinuousAuthenticator:
    """Require fresh, gated, noncontradictory K-of-N score evidence.

    ``score >= threshold`` votes to accept. Any negative vote in retained
    history prevents acceptance, and ``required_rejects`` negatives require
    explicit fallback. Challenges latch until ``reset()``; neither a new
    recording nor subsequent high scores can bypass fallback.

    All calls share a strictly increasing finite nonnegative seconds clock.
    Optional ``window_start`` describes the half-open interval [start, time).
    After supplying it, keep supplying it for scored updates in that recording.
    Otherwise the caller must guarantee that raw evidence never overlaps.
    """

    def __init__(
        self,
        threshold: float,
        *,
        history_size: int = 3,
        required_accepts: int = 2,
        required_rejects: int = 2,
        walking_threshold: float = 0.8,
        max_evidence_age_seconds: float = 30,
        max_gap_seconds: float = 10,
    ):
        threshold_value = _finite(threshold)
        walking_value = _finite(walking_threshold)
        age_value = _finite(max_evidence_age_seconds)
        gap_value = _finite(max_gap_seconds)
        if threshold_value is None:
            raise ValueError("threshold must be a finite real number")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or value < 1
            for value in (history_size, required_accepts, required_rejects)
        ):
            raise ValueError(
                "history size and vote requirements must be positive integers"
            )
        if required_accepts > history_size or required_rejects > history_size:
            raise ValueError("vote requirements must not exceed history_size")
        if history_size > maxsize:
            raise ValueError("history_size exceeds the supported deque size")
        if walking_value is None or not 0 <= walking_value <= 1:
            raise ValueError("walking_threshold must be finite and in [0, 1]")
        if age_value is None or age_value <= 0 or gap_value is None or gap_value <= 0:
            raise ValueError("evidence age and gap limits must be finite and positive")
        self.threshold = threshold_value
        self.history_size = int(history_size)
        self.required_accepts = int(required_accepts)
        self.required_rejects = int(required_rejects)
        self.walking_threshold = walking_value
        self.max_evidence_age_seconds = age_value
        self.max_gap_seconds = gap_value
        self.reset()

    def reset(self) -> None:
        """Start a new episode; caller must authorize any challenge recovery.

        Clears history, the challenge latch, recording identity, and clock.
        This method does not verify that an external challenge succeeded.
        """
        self._history = deque(maxlen=self.history_size)
        self._last_timestamp = None
        self._started_at = None
        self._last_evidence_at = None
        self._last_accept_at = None
        self._challenge_reason = None
        self._recording_seen = False
        self._recording_id = None
        self._last_window_end = None
        self._requires_window_start = False

    def _challenge(self, reason):
        if self._challenge_reason is None:
            self._challenge_reason = reason

    def _tick(self, timestamp):
        now = _finite(timestamp)
        if now is None or now < 0:
            self._challenge("invalid_timestamp")
            return None
        if self._last_timestamp is not None and now <= self._last_timestamp:
            self._challenge("nonmonotonic_timestamp")
            return None
        self._last_timestamp = now
        if self._started_at is None:
            self._started_at = now
        while (
            self._history and now - self._history[0][2] >= self.max_evidence_age_seconds
        ):
            self._history.popleft()
        # Evaluate deadlines BEFORE considering incoming evidence. A score at
        # or after a deadline cannot revive an expired authentication episode.
        accept_anchor = (
            self._started_at if self._last_accept_at is None else self._last_accept_at
        )
        evidence_anchor = (
            self._started_at
            if self._last_evidence_at is None
            else self._last_evidence_at
        )
        if now - accept_anchor >= self.max_evidence_age_seconds:
            self._challenge(
                "consensus_timeout"
                if self._last_accept_at is None
                else "authentication_expired"
            )
        elif now - evidence_anchor >= self.max_gap_seconds:
            self._challenge("evidence_gap")
        return now

    def _result(self, now, reason, *, added=False, changed=False):
        accepts = sum(vote for _, vote, _ in self._history)
        challenge = self._challenge_reason is not None
        reason = self._challenge_reason if challenge else reason
        decision = (
            "challenge"
            if challenge
            else "accept"
            if reason == "accept_quorum"
            else "defer"
        )
        return {
            "decision": decision,
            "reason": reason,
            "authenticated": decision == "accept",
            "challenge_required": challenge,
            "timestamp": now,
            "recording_id": self._recording_id,
            "recording_changed": changed,
            "evidence_added": added,
            "evidence_count": len(self._history),
            "accept_count": accepts,
            "reject_count": len(self._history) - accepts,
            "evidence_age": (
                now - self._history[-1][2]
                if now is not None and self._history
                else None
            ),
            "oldest_evidence_age": (
                now - self._history[0][2] if now is not None and self._history else None
            ),
        }

    def update(
        self,
        score: float | None,
        timestamp: float,
        walking_probability: float = 1,
        quality_ok: bool = True,
        recording_id: str | int | None = None,
        *,
        window_start: float | None = None,
    ) -> dict:
        """Consume one observation and return a JSON-safe decision dictionary.

        Malformed runtime input fails closed into a latched challenge rather
        than raising. Missing scores and failed gates abstain without updating
        evidence or acceptance freshness. Recording changes clear votes and
        interval tracking, but never extend deadlines or clear a challenge.
        """
        now = self._tick(timestamp)
        if now is None or self._challenge_reason is not None:
            return self._result(now, "awaiting_evidence")
        if not (
            recording_id is None
            or type(recording_id) is str
            or (type(recording_id) is int and abs(recording_id) <= 2**53 - 1)
        ):
            self._challenge("invalid_recording_id")
            return self._result(now, "invalid_recording_id")
        changed = self._recording_seen and recording_id != self._recording_id
        if changed:
            self._history.clear()
            self._last_window_end = None
            self._requires_window_start = False
        self._recording_seen = True
        self._recording_id = recording_id
        walking = _finite(walking_probability)
        value = None if score is None else _finite(score)
        support_start = now
        reason = None
        if walking is None or not 0 <= walking <= 1:
            reason = "invalid_walking_probability"
        elif type(quality_ok) is not bool:
            reason = "invalid_quality_flag"
        elif score is not None and value is None:
            reason = "invalid_score"
        elif window_start is not None:
            start = _finite(window_start)
            if start is None or start < 0 or start >= now:
                reason = "invalid_window_start"
            elif now - start >= self.max_evidence_age_seconds:
                reason = "stale_window"
            elif self._last_window_end is not None and start < self._last_window_end:
                reason = "overlapping_evidence"
            else:
                self._requires_window_start = True
                support_start = start
        elif self._requires_window_start and score is not None:
            reason = "missing_window_start"
        if reason is not None:
            self._challenge(reason)
            return self._result(now, reason, changed=changed)
        # Track presented windows even when a gate abstains; overlap must not
        # disappear merely because a previous window had no usable score.
        self._last_window_end = now
        if not quality_ok:
            return self._result(now, "quality_gate", changed=changed)
        if walking < self.walking_threshold:
            return self._result(now, "not_walking", changed=changed)
        if value is None:
            return self._result(now, "missing_score", changed=changed)
        self._history.append((now, value >= self.threshold, support_start))
        self._last_evidence_at = now
        accepts = sum(vote for _, vote, _ in self._history)
        rejects = len(self._history) - accepts
        if rejects >= self.required_rejects:
            self._challenge("reject_quorum")
            reason = "reject_quorum"
        elif rejects:
            reason = "contradictory_evidence"
        elif accepts >= self.required_accepts:
            self._last_accept_at = now
            reason = "accept_quorum"
        else:
            reason = "insufficient_evidence"
        return self._result(now, reason, added=True, changed=changed)

    def advance(self, timestamp: float) -> dict:
        """Advance the shared clock without a score, enforcing all deadlines.

        This call never returns ``accept`` and never refreshes evidence. Call
        it during sensor silence; real elapsed time is not observed implicitly.
        """
        now = self._tick(timestamp)
        return self._result(now, "awaiting_evidence")

    def request_fallback(self, timestamp: float) -> dict:
        """Explicitly latch a challenge, for example after a runtime failure.

        An existing challenge retains its original reason. Like every other
        observation, this request must advance the shared monotonic clock.
        """
        now = self._tick(timestamp)
        self._challenge("fallback_requested")
        return self._result(now, "fallback_requested")
