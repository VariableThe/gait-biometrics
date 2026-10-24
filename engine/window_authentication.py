"""Research-only, source-bound completed-window authentication adapter.

No model loading, sensor I/O, enrollment fitting, or threshold learning occurs
here. The trusted host supplies frozen artifacts and owns fallback verification.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any

import numpy as np

from engine.continuous_evaluation import AuthenticationStream
from models.walking_gate import SourceContract, WalkingGate, canonical, require_contract

SCORE_PROTOCOL = "single_completed_window_fixed_template_cosine_minus_one_v1"


class _WindowInputError(ValueError):
    """Validation reason safe to expose, unlike arbitrary encoder exceptions."""


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise _WindowInputError(f"invalid_{name}")
    try:
        value = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise _WindowInputError(f"invalid_{name}") from error
    if not isfinite(value):
        raise _WindowInputError(f"invalid_{name}")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise _WindowInputError(f"invalid_{name}")
    value.encode("utf-8", errors="strict")
    return value


def _fingerprint(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise _WindowInputError("fingerprints must be lowercase SHA256 hex strings")


def _unit_vector(value):
    vector = np.asarray(value)
    if vector.ndim != 1 or not vector.size or vector.dtype.kind not in "fiu":
        raise _WindowInputError("finite nonempty real numeric vector required")
    with np.errstate(over="ignore", invalid="ignore"):
        vector = np.array(vector, dtype=np.float64, copy=True)
    if not np.isfinite(vector).all():
        raise _WindowInputError("finite embedding required")
    scale = np.max(np.abs(vector))
    if scale == 0:
        raise _WindowInputError("nonzero embedding required")
    vector /= scale  # Avoid overflow/underflow in the norm of finite inputs.
    return vector / np.linalg.norm(vector)


@dataclass(frozen=True)
class EnrollmentTemplate:
    """One claimed identity's immutable, normalized, numeric fixed template.

    ``vector`` is copied into a tuple, so changing the caller's array cannot
    change enrollment. Fingerprints bind numeric state, not its provenance.
    """

    claimed_identity: str
    source_contract: SourceContract
    encoder_fingerprint: str
    vector: Any

    def __post_init__(self):
        _text(self.claimed_identity, "claimed_identity")
        if not isinstance(self.source_contract, SourceContract):
            raise TypeError("explicit enrollment SourceContract required")
        _fingerprint(self.encoder_fingerprint)
        object.__setattr__(self, "vector", tuple(map(float, _unit_vector(self.vector))))

    @property
    def fingerprint(self) -> str:
        state = {
            "schema": "fixed_window_enrollment_v1",
            "claimed_identity": self.claimed_identity,
            "source_contract": self.source_contract.to_dict(),
            "encoder_fingerprint": self.encoder_fingerprint,
            "vector": self.vector,
        }
        return hashlib.sha256(canonical(state)).hexdigest()


@dataclass(frozen=True)
class WindowCalibration:
    """Explicit external frozen calibration for this exact window protocol.

    All fields are required. Existing trial-score calibration dictionaries are
    deliberately not accepted. These declarations cannot prove study quality.
    """

    source_contract: SourceContract
    encoder_fingerprint: str
    template_fingerprint: str
    threshold: float
    score_protocol: str
    frozen: bool
    calibration_id: str

    def __post_init__(self):
        if not isinstance(self.source_contract, SourceContract):
            raise TypeError("explicit calibration SourceContract required")
        _fingerprint(self.encoder_fingerprint)
        _fingerprint(self.template_fingerprint)
        if self.score_protocol != SCORE_PROTOCOL or self.frozen is not True:
            raise ValueError("explicit frozen single-window score protocol required")
        _text(self.calibration_id, "calibration_id")
        object.__setattr__(self, "threshold", _number(self.threshold, "threshold"))


class WindowAuthenticator:
    """One source, encoder, claimed template, calibration, and policy episode.

    The encoder consumes numeric ``(1,C,T)`` and returns numeric ``(1,D)``.
    It must already be frozen and in inference mode; wrap framework-specific
    conversion/no-grad behavior in the trusted callable. Gate scores are
    logistic outputs, not calibrated walking probabilities.

    Runtime failures latch fallback. There is no reset: after independently
    verifying fallback, the trusted host may construct a new instance.
    """

    def __init__(
        self,
        *,
        encoder: Callable,
        encoder_fingerprint: str,
        walking_gate: WalkingGate,
        source_contract: SourceContract,
        enrollment: EnrollmentTemplate,
        calibration: WindowCalibration,
        **policy_options: Any,
    ):
        if not callable(encoder):
            raise TypeError("encoder callable required")
        _fingerprint(encoder_fingerprint)
        if not isinstance(walking_gate, WalkingGate):
            raise TypeError("actual WalkingGate required")
        if not isinstance(enrollment, EnrollmentTemplate) or not isinstance(
            calibration, WindowCalibration
        ):
            raise TypeError("EnrollmentTemplate and WindowCalibration required")
        require_contract(source_contract, walking_gate.contract)
        require_contract(enrollment.source_contract, source_contract)
        require_contract(calibration.source_contract, source_contract)
        if (
            encoder_fingerprint != enrollment.encoder_fingerprint
            or encoder_fingerprint != calibration.encoder_fingerprint
            or enrollment.fingerprint != calibration.template_fingerprint
        ):
            raise ValueError("encoder/template calibration binding mismatch")
        if "walking_threshold" in policy_options:
            raise ValueError("walking_threshold is fixed by walking_gate.threshold")
        self._encoder = encoder
        # Snapshot mutable NumPy buffers/provenance held by the caller's gate.
        self._gate = WalkingGate.from_dict(walking_gate.to_dict())
        self._contract = source_contract
        self._template = enrollment.vector
        self._stream = AuthenticationStream(
            calibration.threshold,
            walking_threshold=self._gate.threshold,
            **policy_options,
        )
        self._fault: str | None = None
        self._challenged = False

    def _finish(self, result, *, score=None, walking_score=None, error=None):
        output = dict(result)
        output.update(
            score=score,
            walking_score=walking_score,
            score_protocol=SCORE_PROTOCOL,
            adapter_error=self._fault,
        )
        if self._fault is not None:
            output.update(
                decision="challenge",
                reason="adapter_failure",
                authenticated=False,
                challenge_required=True,
            )
        if error is not None:
            output.update(input_valid=False, event_error=error)
        self._challenged = self._challenged or output["challenge_required"]
        return output

    def _fail(self, timestamp, error):
        if self._fault is None:
            self._fault = error
        return self._finish(
            self._stream.process_event({"type": "fallback", "timestamp": timestamp}),
            error=error,
        )

    def update(
        self,
        window,
        timestamp,
        *,
        window_start=None,
        contract=None,
        quality_ok=None,
        recording_id=None,
    ) -> dict:
        """Consume exactly one completed raw window at its availability time.

        Every update must explicitly supply the exact source contract, raw
        start, and Python-bool quality flag. Support is conservatively declared
        as [window_start, timestamp), including inference/delivery delay.
        Even gated windows consume support without refreshing consensus.
        """
        if self._challenged:
            return self.advance(timestamp)
        try:
            now = _number(timestamp, "timestamp")
            if now < 0:
                raise _WindowInputError("invalid_timestamp")
            last = self._stream.last_timestamp
            if last is not None and now <= last:
                raise _WindowInputError("nonmonotonic_timestamp")
            if not isinstance(contract, SourceContract) or contract != self._contract:
                raise _WindowInputError("incompatible_source_contract")
            start = _number(window_start, "window_start")
            if start < 0 or start >= now:
                raise _WindowInputError("invalid_window_start")
            # Compare endpoints directly: subtraction can round an exact
            # first-to-last span upward at a nonzero clock origin.
            last_sample = (
                start + (self._contract.window_length - 1) / self._contract.sample_rate
            )
            if now <= last_sample:
                raise _WindowInputError("incomplete_window_interval")
            if type(quality_ok) is not bool:
                raise _WindowInputError("invalid_quality_flag")
            raw = np.asarray(window)
            if (
                raw.shape
                != (1, len(self._contract.channels), self._contract.window_length)
                or raw.dtype.kind not in "fiu"
                or not np.isfinite(raw).all()
            ):
                raise _WindowInputError("invalid_raw_window")
            # Copy so the gate and encoder see the same completed input.
            raw = np.array(raw, copy=True)
            walking = self._gate.predict_proba(raw, contract=contract)
            if walking.shape != (1,):
                raise _WindowInputError("invalid_walking_score")
            walking_score = _number(walking[0], "walking_score")
            if not 0 <= walking_score <= 1:
                raise _WindowInputError("invalid_walking_score")
            score = None
            if quality_ok and walking_score >= self._gate.threshold:
                embedding = np.asarray(self._encoder(raw))
                if embedding.shape != (1, len(self._template)):
                    raise _WindowInputError("invalid_embedding_shape")
                unit = _unit_vector(embedding[0])
                score = float(np.clip(unit @ self._template, -1.0, 1.0) - 1.0)
        except Exception as error:  # noqa: BLE001 -- inference failures must latch fallback
            # Include framework/callable inference failures, without returning
            # exception text that could contain sensitive application details.
            reason = (
                str(error)
                if isinstance(error, _WindowInputError)
                else "inference_failure"
            )
            return self._fail(timestamp, reason)
        return self._finish(
            self._stream.process_event(
                {
                    "type": "score",
                    "timestamp": now,
                    "window_start": start,
                    "score": score,
                    "walking_probability": walking_score,
                    "quality_ok": quality_ok,
                    "recording_id": recording_id,
                }
            ),
            score=score,
            walking_score=walking_score,
        )

    def advance(self, timestamp) -> dict:
        """Tick the same global clock during silence; never renew acceptance."""
        return self._finish(
            self._stream.process_event({"type": "advance", "timestamp": timestamp})
        )

    def request_fallback(self, timestamp) -> dict:
        """Latch a challenge; only a trusted host can authorize a new instance."""
        return self._finish(
            self._stream.process_event({"type": "fallback", "timestamp": timestamp})
        )
