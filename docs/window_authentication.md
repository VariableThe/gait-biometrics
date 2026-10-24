# Source-bound completed-window authentication

`engine.window_authentication` connects a trusted encoder callable and the actual
`models.walking_gate.WalkingGate` to the existing bounded
[continuous policy](continuous_auth.md), through `AuthenticationStream`. One
`WindowAuthenticator` represents one claimed enrolled identity and one episode.
This is research software: it performs no phone/sensor I/O, model loading,
enrollment acquisition, threshold learning, or fallback verification. Synthetic
tests establish software behavior, with no biometric accuracy or deployment claim.

The trusted host supplies three bindings:

- `EnrollmentTemplate(claimed_identity, source_contract, encoder_fingerprint,
  vector)` copies a finite, nonzero, real numeric `(D,)` vector, normalizes it,
  and stores an immutable tuple. Its SHA256 fingerprint covers that normalized
  vector, identity, exact source contract, and encoder fingerprint. The template
  stays fixed; updates never enroll, adapt, or change the claimed identity.
- `WindowCalibration(source_contract, encoder_fingerprint, template_fingerprint,
  threshold, score_protocol, frozen, calibration_id)` requires every field
  explicitly. `frozen` must be Python `True` and `score_protocol` must equal
  `SCORE_PROTOCOL`. The threshold must be a finite real number; a finite value
  above zero represents reject-all on this score scale. No threshold is inferred
  from a gate, checkpoint, existing study, or enrollment vector.
- `WindowAuthenticator` receives the encoder callable, its SHA256 fingerprint,
  source contract, actual fitted/loaded walking gate, enrollment, calibration,
  and optional temporal-policy settings. All source, encoder, and template
  bindings must match at initialization. Bad bindings fail immediately. The
  adapter snapshots the gate's numeric state and uses `walking_gate.threshold`
  explicitly as the policy walking threshold; a separate override is rejected.

Fingerprints must be lowercase 64-character SHA256 hex strings. The host must
identify the complete frozen encoder/preprocessing state honestly; the adapter
cannot derive or verify a fingerprint for an arbitrary callable. Hashes detect
state mismatches, not malicious replacement or provenance fraud. External
calibration declarations cannot establish source representativeness, independent
enrollment/calibration/probes, adequate sample size, or study quality.

## Exact score and calibration protocol

`SCORE_PROTOCOL` is
`single_completed_window_fixed_template_cosine_minus_one_v1`. Each eligible raw
window is independently encoded to one `(1,D)` vector, normalized, and scored
against the one fixed normalized enrollment template:

```text
score = clip(dot(unit(window_embedding), fixed_unit_template), -1, 1) - 1
```

The score lies in `[-2,0]`; larger means more similar and `score >= threshold`
votes positive. Scaling before normalization avoids overflow for large finite
embeddings. There is no trial aggregation, prototype maximum, cohort score,
future-window normalization, or adaptive template/threshold. For template
`(1,0)` and embedding `(3,4)`, cosine is `0.6` and score is `-0.4`.

The host must provide externally frozen calibration for **this single-window
protocol**, exact source, encoder, and fixed enrollment template. A Santos
trial-score artifact is not a `WindowCalibration` and is rejected. Its aggregated
trial threshold is incompatible even when its arithmetic also uses cosine minus
one. Relabeling old trial metadata as the required protocol does not make it
valid calibration; the host remains responsible for its truth. No compatible
trained encoder, calibration bundle, or usable threshold is asserted to exist.

## Trusted integration

This function illustrates the API using artifacts supplied by a trusted host;
it creates no placeholder model or threshold:

```python
from engine.window_authentication import (
    SCORE_PROTOCOL, EnrollmentTemplate, WindowAuthenticator, WindowCalibration,
)

def start_episode(*, encoder, encoder_sha256, gate, contract, owner_id,
                  template_vector, frozen_threshold, calibration_reference):
    enrollment = EnrollmentTemplate(
        owner_id, contract, encoder_sha256, template_vector,
    )
    calibration = WindowCalibration(
        source_contract=contract,
        encoder_fingerprint=encoder_sha256,
        template_fingerprint=enrollment.fingerprint,
        threshold=frozen_threshold,
        score_protocol=SCORE_PROTOCOL,
        frozen=True,
        calibration_id=calibration_reference,
    )
    return WindowAuthenticator(
        encoder=encoder,
        encoder_fingerprint=encoder_sha256,
        walking_gate=gate,
        source_contract=contract,
        enrollment=enrollment,
        calibration=calibration,
    )

# The trusted host subsequently calls:
# result = episode.update(
#     completed_window, availability_timestamp,
#     window_start=raw_window_start, contract=contract,
#     quality_ok=trusted_quality_flag, recording_id=stable_recording_id,
# )
# result = episode.advance(later_timestamp)          # Timer during silence.
# result = episode.request_fallback(later_timestamp) # Runtime failure, etc.
```

The callable accepts a real numeric NumPy `(1,C,T)` array and returns a real
numeric array-like `(1,D)` result. A simple NumPy callable suffices. A trusted
framework wrapper must handle tensor/device/dtype conversion, frozen inference
mode, gradient suppression, and conversion back to numeric array-like output.
The adapter does not call training/evaluation methods or fingerprint weights.
Embedding dimension must exactly match enrollment, and every embedding must be
finite and nonzero. Numeric strings, booleans, object and complex arrays fail.

Every update explicitly supplies a `SourceContract` equal to the initialized
contract, including source, sampling rate, ordered channels, units, window
length, and acceleration convention. The actual gate must share it. The current
gate supports three total-acceleration channels with explicit supported units;
there is no resampling, channel selection, unit conversion, or implicit transfer
from a phone contract to a six-axis Santos leg contract. Matching declarations
cannot prove acquisition provenance or robustness to a new placement/session.

Raw input must have exactly `(1,C,T)` shape and contain finite real numbers.
The gate additionally enforces its acceleration bounds. After validation, the
walking gate runs first, including on quality-abstained windows. Its outputs are
logistic scores, **not calibrated probabilities**, despite the existing policy's
`walking_probability` parameter name. If quality is Python `False` or the gate
score is below the gate threshold, the encoder is skipped and the policy receives
a missing identity score. These windows consume their declared intervals but add
no votes and refresh neither evidence nor acceptance freshness. Quality must be
an explicit Python boolean. Threshold equality passes both gates and scoring.

## One clock, conservative support, and fallback

All updates, `advance`, and `request_fallback` share the same trusted finite,
nonnegative, strictly increasing seconds clock. `timestamp` means **availability
to the policy**, after inference/delivery, and `window_start` means raw acquisition
start. The declared half-open interval is `[window_start, timestamp)`. It must be
positive, strictly exceed the `(T-1)/fs` first-to-last nominal sample span, and be
strictly shorter than the policy's maximum evidence age. The host must ensure a completed,
contiguous window and causal timestamps; the adapter cannot inspect sensor time
or verify actual elapsed inference from two caller-supplied numbers.

The endpoint must be **after** the last included sample because it is exclusive.
Using that sample's timestamp as the endpoint would allow it to reappear as the
next window's first sample and count toward two votes. A conventional acquisition
support `[start, start + T/fs)` satisfies this requirement; actual availability
must also include inference/delivery delay.

Inference/delivery delay therefore counts against freshness and nonoverlap.
Within a recording the next raw start must be at least the previous window's
**availability** time, including a previous gated window. This is deliberately
stricter than checking actual sensor overlap: disjoint raw windows can still be
rejected when capture of the second began before delivery of the first. For
example, 64 samples at 20 Hz starting at 0 have nominal last sample at 3.15; if
the result is available at 5, a second raw window beginning at 4 overlaps the
declared `[0,5)` support. Do not remove delay by relabeling old raw evidence.
Policy deadlines are evaluated before a new score can refresh evidence.

`advance` on a host timer enforces silence deadlines and never returns acceptance
or renews earlier acceptance. There is no background thread or implicit clock.
The host must act on each result, never retain a cached acceptance across a defer,
failure, process/transport exception, or challenge. Stable recording IDs follow
the existing policy: changing recording clears votes and support tracking while
preserving the global clock, deadlines, and fallback latch.

Runtime input, contract, numeric, and inference failures return a latched
challenge. Output retains stream/policy fields and adds `score`, `walking_score`,
`score_protocol`, and `adapter_error`. Adapter failures use
`reason="adapter_failure"`; stream/policy failures retain their existing reasons.
A failing update never returns the previous acceptance. Later updates only advance
the clock after challenge and cannot invoke the encoder or release fallback.
Neither `reset`, recovery, nor fallback-success input exists. Only the trusted
host may create a new instance **after independently verified fallback**.

`tests/test_window_authentication.py` uses a callable test encoder, an actual
walking gate fitted only to synthetic signals, hand-computed cosines, binding and
source guards, gate abstentions, malformed inputs, inference failures, interval
overlap, delay, clock faults, and silence. No real data or frozen study artifact
is opened or modified to exercise this adapter.
