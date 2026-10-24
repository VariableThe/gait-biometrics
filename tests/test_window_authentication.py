"""Synthetic-only model -> gate -> fixed-template -> temporal-policy checks."""

from dataclasses import FrozenInstanceError, replace
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest

from engine.window_authentication import (
    SCORE_PROTOCOL,
    EnrollmentTemplate,
    WindowAuthenticator,
    WindowCalibration,
)
from models.walking_gate import SourceContract, WalkingGate

ENCODER_FINGERPRINT = "a" * 64


@pytest.fixture(scope="module")
def fitted():
    contract = SourceContract(
        "synthetic_adapter_only", 20, ("x", "y", "z"), ("m/s^2", "m/s^2", "m/s^2"), 64
    )
    rng = np.random.default_rng(81)
    time = np.arange(contract.window_length) / contract.sample_rate
    windows = rng.normal(0, 0.005, (24, 3, contract.window_length))
    windows[:, 2] += 9.81
    labels = np.arange(len(windows)) % 2
    for row in np.flatnonzero(labels == 1):
        phase = row / 10
        windows[row, 0] += 0.5 * np.sin(2 * np.pi * 1.8 * time + phase)
        windows[row, 2] += 1.5 * np.cos(2 * np.pi * 1.8 * time + phase)
    gate = WalkingGate.fit(
        windows,
        labels,
        activity_names=["Sitting", "Walking"],
        subjects=np.ones(len(windows), dtype=np.int64),
        fit_subject_ids=[1],
        forbidden_subject_ids=[2],
        contract=contract,
    )
    return gate, windows[1:2], windows[0:1]


def bindings(contract, *, vector=(1, 0), threshold=-0.5):
    enrollment = EnrollmentTemplate(
        "synthetic-owner", contract, ENCODER_FINGERPRINT, vector
    )
    calibration = WindowCalibration(
        source_contract=contract,
        encoder_fingerprint=ENCODER_FINGERPRINT,
        template_fingerprint=enrollment.fingerprint,
        threshold=threshold,
        score_protocol=SCORE_PROTOCOL,
        frozen=True,
        calibration_id="synthetic-only-no-biometric-calibration",
    )
    return enrollment, calibration


def make_auth(gate, encoder=None, *, threshold=-0.5, **policy):
    enrollment, calibration = bindings(gate.contract, threshold=threshold)
    return WindowAuthenticator(
        encoder=encoder if encoder is not None else lambda raw: np.array([[1, 0]]),
        encoder_fingerprint=ENCODER_FINGERPRINT,
        walking_gate=gate,
        source_contract=gate.contract,
        enrollment=enrollment,
        calibration=calibration,
        **policy,
    )


def update(auth, gate, window, timestamp=4, start=0, **overrides):
    return auth.update(
        window,
        timestamp,
        **{
            "window_start": start,
            "contract": gate.contract,
            "quality_ok": True,
            "recording_id": "synthetic-a",
            **overrides,
        },
    )


def assert_challenge(result):
    assert result["decision"] == "challenge"
    assert result["challenge_required"] and not result["authenticated"]
    assert not result["evidence_added"]


@pytest.mark.parametrize("strictly_after", [False, True])
def test_exclusive_endpoint_must_follow_last_included_sample(fitted, strictly_after):
    gate, walking, _ = fitted
    encoder = Mock(return_value=np.array([[1, 0]]))
    auth = make_auth(gate, encoder)
    last_sample = (gate.contract.window_length - 1) / gate.contract.sample_rate
    endpoint = (
        float(np.nextafter(last_sample, np.inf)) if strictly_after else last_sample
    )
    result = update(auth, gate, walking, endpoint, 0)
    if strictly_after:
        assert result["decision"] == "defer" and result["evidence_added"]
        assert encoder.call_count == 1
    else:
        assert_challenge(result)
        assert result["adapter_error"] == "incomplete_window_interval"
        assert encoder.call_count == 0


def test_shared_last_sample_cannot_supply_the_second_accept_vote(fitted):
    gate, walking, _ = fitted
    auth = make_auth(gate)
    last_sample = (gate.contract.window_length - 1) / gate.contract.sample_rate
    available = float(np.nextafter(last_sample, np.inf))
    assert update(auth, gate, walking, available, 0)["decision"] == "defer"
    # A second window starting at the first window's last included sample
    # overlaps the first half-open support, even though it has enough duration.
    result = update(auth, gate, walking, 2 * last_sample + 1, last_sample)
    assert_challenge(result)
    assert result["reason"] == "overlapping_evidence"


def test_nonzero_clock_origin_does_not_round_a_shared_sample_outside_support(fitted):
    gate, walking, _ = fitted
    # Only the timing guard is under test; reuse a numeric fixture without a
    # new fit or any assertion that weights transfer to this synthetic shape.
    contract = replace(gate.contract, sample_rate=16, window_length=33)
    gate = replace(gate, contract=contract)
    window = walking[:, :, :33]
    assert 4.4 - 2.4 > 2.0  # Subtraction would incorrectly pass a strict guard.
    assert 4.4 == 2.4 + 2.0
    encoder = Mock(return_value=np.array([[1, 0]]))
    with patch.object(WalkingGate, "predict_proba", return_value=np.array([0.99])):
        auth = make_auth(gate, encoder)
        result = update(auth, gate, window, 4.4, 2.4)
        assert_challenge(result)
        assert result["adapter_error"] == "incomplete_window_interval"
        assert encoder.call_count == 0
        assert_challenge(update(auth, gate, window, 6.5, 4.4))

        auth = make_auth(gate)
        available = float(np.nextafter(4.4, np.inf))
        assert update(auth, gate, window, available, 2.4)["decision"] == "defer"
        assert_challenge(update(auth, gate, window, 6.5, 4.4))


def test_enrollment_copies_normalizes_and_binds_immutable_numeric_state(fitted):
    gate, _, _ = fitted
    vector = np.array([3.0, 4.0])
    enrollment, calibration = bindings(gate.contract, vector=vector)
    fingerprint = enrollment.fingerprint
    assert enrollment.vector == pytest.approx((0.6, 0.8))
    vector[:] = 0
    assert enrollment.fingerprint == fingerprint == calibration.template_fingerprint
    with pytest.raises(FrozenInstanceError):
        setattr(enrollment, "vector", (1, 0))  # noqa: B010 -- exercise frozen assignment
    assert replace(enrollment, claimed_identity="other").fingerprint != fingerprint
    assert replace(enrollment, vector=(1, 0)).fingerprint != fingerprint
    assert replace(enrollment, encoder_fingerprint="b" * 64).fingerprint != fingerprint


@pytest.mark.parametrize(
    "vector",
    [
        (0, 0),
        (float("nan"), 1),
        (float("inf"), 1),
        (True, False),
        ("1", "2"),
        (1j, 2j),
        (),
        [[1, 0]],
    ],
)
def test_bad_enrollment_fails_at_initialization(fitted, vector):
    with pytest.raises(ValueError):
        bindings(fitted[0].contract, vector=vector)


@pytest.mark.parametrize(
    "embedding,expected", [((3, 4), -0.4), ((0, 5), -1), ((-4, 0), -2)]
)
def test_hand_calculated_single_window_cosines_and_nct_nd_callable(
    fitted, embedding, expected
):
    gate, walking, _ = fitted
    encoder = Mock(return_value=np.array([embedding]))
    auth = make_auth(gate, encoder, threshold=-2, required_accepts=1)
    result = update(auth, gate, walking)
    assert result["score"] == pytest.approx(expected)
    assert result["score_protocol"] == SCORE_PROTOCOL
    assert result["authenticated"] and result["evidence_count"] == 1
    np.testing.assert_array_equal(encoder.call_args.args[0], walking)
    assert encoder.call_args.args[0].shape == (1, 3, 64)


def test_gate_operating_point_equality_is_used_explicitly_by_policy(fitted):
    gate, _, sitting = fitted
    walking_score = float(gate.predict_proba(sitting, contract=gate.contract)[0])
    assert walking_score < 0.8
    gate = gate.with_threshold(walking_score)
    encoder = Mock(return_value=np.array([[1, 0]]))
    auth = make_auth(gate, encoder, threshold=0, required_accepts=1)
    result = update(auth, gate, sitting)
    assert result["walking_score"] == walking_score
    assert result["score"] == 0
    assert result["authenticated"]  # Both score and walking equality pass.
    assert encoder.call_count == 1
    with pytest.raises(ValueError, match="walking_threshold"):
        make_auth(gate, walking_threshold=0.8)


def test_walking_gate_runs_first_and_encoder_skips_quality_and_nonwalking(fitted):
    gate, walking, sitting = fitted
    encoder = Mock(side_effect=AssertionError("encoder must not run"))
    auth = make_auth(gate, encoder)
    events = []
    predict = WalkingGate.predict_proba

    def gate_call(self, raw, **kwargs):
        events.append("walking_gate")
        return predict(self, raw, **kwargs)

    with patch.object(WalkingGate, "predict_proba", gate_call):
        quality = update(auth, gate, walking, quality_ok=False)
        nonwalking = update(auth, gate, sitting, timestamp=8, start=4)
    assert events == ["walking_gate", "walking_gate"]
    assert quality["reason"] == "quality_gate"
    assert nonwalking["reason"] == "not_walking"
    assert quality["score"] is None and nonwalking["score"] is None
    assert not quality["evidence_added"] and not nonwalking["evidence_added"]
    assert encoder.call_count == 0


def test_gates_consume_support_and_do_not_refresh_gap_or_acceptance(fitted):
    gate, walking, sitting = fitted
    auth = make_auth(gate)
    assert update(auth, gate, walking)["decision"] == "defer"
    assert update(auth, gate, walking, 8, 4)["authenticated"]
    gated = update(auth, gate, sitting, 12, 8)
    assert gated["reason"] == "not_walking" and gated["evidence_count"] == 2
    assert gated["evidence_age"] == 8  # Based on raw support start, not delivery.
    assert_challenge(auth.advance(18))  # Gap from last eligible availability at 8.
    auth = make_auth(gate)
    update(auth, gate, walking, quality_ok=False)
    overlap = update(auth, gate, walking, 8, 3.5)
    assert_challenge(overlap)
    assert overlap["reason"] == "overlapping_evidence"


@pytest.mark.parametrize(
    "delta",
    [
        {"source": "other"},
        {"sample_rate": 25},
        {"channels": ("z", "y", "x")},
        {"units": ("wisdm_native_10_per_g",) * 3},
        {"window_length": 65},
    ],
)
def test_source_contract_guard_on_initialization_and_each_update(fitted, delta):
    gate, walking, _ = fitted
    other = replace(gate.contract, **delta)
    enrollment, calibration = bindings(other)
    with pytest.raises(ValueError, match="contract"):
        WindowAuthenticator(
            encoder=lambda raw: [[1, 0]],
            encoder_fingerprint=ENCODER_FINGERPRINT,
            walking_gate=gate,
            source_contract=other,
            enrollment=enrollment,
            calibration=calibration,
        )
    encoder = Mock(return_value=np.array([[1, 0]]))
    auth = make_auth(gate, encoder, required_accepts=1)
    assert update(auth, gate, walking)["authenticated"]
    assert_challenge(update(auth, gate, walking, 8, 4, contract=other))
    assert_challenge(update(auth, gate, walking, 12, 8))
    assert encoder.call_count == 1


def test_calibration_requires_explicit_frozen_window_protocol_and_exact_binding(fitted):
    gate, _, _ = fitted
    enrollment, calibration = bindings(gate.contract)
    for delta in (
        {"frozen": False},
        {"frozen": 1},
        {"score_protocol": "cross_day_trial_cosine"},
        {"threshold": float("nan")},
        {"threshold": True},
        {"encoder_fingerprint": "unbound"},
        {"calibration_id": ""},
    ):
        with pytest.raises(ValueError):
            replace(calibration, **delta)
    arguments: dict[str, Any] = {
        "encoder": lambda raw: [[1, 0]],
        "encoder_fingerprint": ENCODER_FINGERPRINT,
        "walking_gate": gate,
        "source_contract": gate.contract,
        "enrollment": enrollment,
        "calibration": calibration,
    }
    for wrong in (
        replace(calibration, template_fingerprint="b" * 64),
        replace(calibration, encoder_fingerprint="b" * 64),
        replace(calibration, source_contract=replace(gate.contract, source="other")),
        {"protocol": "cross_day_secure_enrollment_v1", "threshold": -0.5},
    ):
        wrong_arguments: dict[str, Any] = {**arguments, "calibration": wrong}
        with pytest.raises((TypeError, ValueError)):
            WindowAuthenticator(**wrong_arguments)
    wrong_arguments = {**arguments, "encoder_fingerprint": "b" * 64}
    with pytest.raises(ValueError, match="binding"):
        WindowAuthenticator(**wrong_arguments)
    wrong_arguments = {**arguments, "enrollment": replace(enrollment, vector=(0, 1))}
    with pytest.raises(ValueError, match="binding"):
        WindowAuthenticator(**wrong_arguments)


@pytest.mark.parametrize(
    "kind", ["nan", "inf", "string", "bool", "complex", "ct", "batch", "length"]
)
def test_invalid_raw_window_fails_closed_after_acceptance(fitted, kind):
    gate, walking, _ = fitted
    encoder = Mock(return_value=np.array([[1, 0]]))
    auth = make_auth(gate, encoder, required_accepts=1)
    assert update(auth, gate, walking)["authenticated"]
    bad = walking.copy()
    if kind in ("nan", "inf"):
        bad[0, 0, 0] = float(kind)
    else:
        bad = {
            "string": walking.astype(str),
            "bool": walking.astype(bool),
            "complex": walking.astype(complex),
            "ct": walking[0],
            "batch": np.repeat(walking, 2, axis=0),
            "length": walking[:, :, :-1],
        }[kind]
    assert_challenge(update(auth, gate, bad, 8, 4))
    assert_challenge(update(auth, gate, walking, 12, 8))
    assert encoder.call_count == 1


@pytest.mark.parametrize(
    "embedding",
    [
        [[0, 0]],
        [[float("nan"), 1]],
        [[float("inf"), 1]],
        [[1, 0, 0]],
        [1, 0],
        [[1, 0], [1, 0]],
        [["1", "0"]],
        [[True, False]],
        [[1j, 0]],
    ],
)
def test_invalid_embeddings_latch_fallback_and_cannot_reuse_cached_accept(
    fitted, embedding
):
    gate, walking, _ = fitted
    encoder = Mock(side_effect=[np.array([[1, 0]]), np.asarray(embedding)])
    auth = make_auth(gate, encoder, required_accepts=1)
    assert update(auth, gate, walking)["authenticated"]
    assert_challenge(update(auth, gate, walking, 8, 4))
    assert_challenge(update(auth, gate, walking, 12, 8))
    assert encoder.call_count == 2


def test_inference_failures_and_nonfinite_gate_outputs_latch(fitted):
    gate, walking, _ = fitted
    for failure in (RuntimeError("synthetic failure"), ValueError("synthetic failure")):
        auth = make_auth(gate, Mock(side_effect=failure))
        assert_challenge(update(auth, gate, walking))
        assert_challenge(auth.advance(5))
    for output in (np.array([np.nan]), np.array([1.1]), np.array([0.9, 0.9])):
        encoder = Mock(return_value=np.array([[1, 0]]))
        auth = make_auth(gate, encoder)
        with patch.object(WalkingGate, "predict_proba", return_value=output):
            assert_challenge(update(auth, gate, walking))
        assert encoder.call_count == 0


def test_large_and_tiny_finite_embeddings_normalize_without_overflow(fitted):
    gate, walking, _ = fitted
    for scale in (1e308, 1e-320):
        auth = make_auth(
            gate,
            lambda raw, scale=scale: np.array([[scale, scale]]),
            required_accepts=1,
        )
        assert update(auth, gate, walking)["score"] == pytest.approx(2**-0.5 - 1)


@pytest.mark.parametrize(
    "timestamp,start",
    [(4, 4), (4, -1), (4, None), (4, 1), (float("nan"), 0), (True, 0), ("4", 0)],
)
def test_invalid_clock_and_incomplete_support_fail_closed(fitted, timestamp, start):
    gate, walking, _ = fitted
    auth = make_auth(gate)
    assert_challenge(update(auth, gate, walking, timestamp, start))
    assert_challenge(auth.advance(9))


def test_stale_delay_global_clock_and_new_recording_cannot_release_fallback(fitted):
    gate, walking, _ = fitted
    for timestamp in (4, 3):
        auth = make_auth(gate, required_accepts=1)
        assert update(auth, gate, walking)["authenticated"]
        assert_challenge(update(auth, gate, walking, timestamp, 0, recording_id="new"))
        assert_challenge(update(auth, gate, walking, 8, 4))
    auth = make_auth(gate)
    stale = update(auth, gate, walking, 30, 0)
    assert_challenge(stale)
    assert stale["event_error"] == "stale_window"
    auth = make_auth(gate)
    assert auth.advance(0)["decision"] == "defer"
    assert_challenge(
        update(auth, gate, walking, 10, 6)
    )  # Deadline precedes new evidence.


def test_inference_delay_interval_overlap_and_silence_require_fallback(fitted):
    gate, walking, _ = fitted
    auth = make_auth(gate)
    update(auth, gate, walking, 5, 0)  # Last sensor sample was nominally at 3.15.
    delayed_overlap = update(auth, gate, walking, 9, 4)
    assert delayed_overlap["reason"] == "overlapping_evidence"
    assert_challenge(delayed_overlap)
    auth = make_auth(
        gate, required_accepts=1, max_evidence_age_seconds=12, max_gap_seconds=20
    )
    assert update(auth, gate, walking)["authenticated"]
    assert auth.advance(15.9)["decision"] == "defer"
    assert_challenge(auth.advance(16))


def test_explicit_fallback_has_no_reset_and_new_instance_requires_host(fitted):
    gate, walking, _ = fitted
    auth = make_auth(gate)
    assert_challenge(auth.request_fallback(0))
    assert_challenge(update(auth, gate, walking))
    assert not hasattr(auth, "reset")
    assert not hasattr(auth, "recover")
    assert update(make_auth(gate), gate, walking)["decision"] == "defer"


@pytest.mark.parametrize(
    "overrides",
    [
        {"contract": None},
        {"quality_ok": None},
        {"quality_ok": 1},
        {"recording_id": True},
    ],
)
def test_required_runtime_metadata_cannot_be_omitted_or_coerced(fitted, overrides):
    gate, walking, _ = fitted
    auth = make_auth(gate, required_accepts=1)
    assert update(auth, gate, walking)["authenticated"]
    assert_challenge(update(auth, gate, walking, 8, 4, **overrides))


def test_gate_snapshot_is_not_changed_by_callers_mutable_buffers(fitted):
    gate, walking, _ = fitted
    external = WalkingGate.from_dict(gate.to_dict())
    auth = make_auth(external, required_accepts=1)
    external.coefficients.flags.writeable = True
    external.coefficients[:] = 0
    external.fit_provenance["fit_calls"] = 7
    assert update(auth, gate, walking)["authenticated"]
