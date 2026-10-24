# External score stream and causal evaluation

`python -m scripts.stream_authentication` runs one instance of the bounded policy
in [continuous_auth.md](continuous_auth.md). It uses only the Python standard
library. Scores, walking probabilities, quality decisions, enrollment, and
execution/verification of fallback are external inputs. It loads no model,
dataset, checkpoint, study artifact, CALIB, or TEST data.

The threshold is required explicitly and scores must be higher-is-more-similar.
Choose the model, gate inputs, threshold, and entire temporal policy using
authorized source-domain development/calibration evidence for the intended
single-window stream. A threshold calibrated on aggregated Santos trials cannot
silently become a threshold for individual windows. These tools and synthetic
tests establish software behavior, with no phone-deployment or biometric accuracy
claim. Artificially concatenated recordings must be labelled **synthetic splices**.

## Strict JSONL CLI

Run from the repository root, for example:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  uv run --no-project --python ../.venv/bin/python python \
  -m scripts.stream_authentication --threshold 0.5 <<'JSONL'
{"type":"advance","timestamp":0}
{"type":"score","timestamp":2,"score":0.8,"walking_probability":0.95,"quality_ok":true,"recording_id":"synthetic-a","window_start":0}
{"type":"score","timestamp":4,"score":0.9,"walking_probability":0.95,"quality_ok":true,"recording_id":"synthetic-a","window_start":2}
{"type":"advance","timestamp":14}
{"type":"score","timestamp":16,"score":0.9,"walking_probability":0.95,"quality_ok":true,"recording_id":"synthetic-b","window_start":14}
JSONL
```

The decisions are `defer`, `defer`, `accept`, `challenge`, `challenge`.
The exact ten-second evidence gap at 14 latches fallback; changing recordings
or subsequently supplying high scores cannot release it. `--help` lists the
history/quorum, walking threshold, evidence age, and gap options. Defaults match
`ContinuousAuthenticator`; only the score threshold has no default.

Every line must contain exactly one JSON object with one of these schemas:

```json
{"type":"score","timestamp":2,"score":0.8,"walking_probability":0.95,"quality_ok":true,"recording_id":"a","window_start":0}
{"type":"advance","timestamp":3}
{"type":"fallback","timestamp":4}
```

For `score`, all shown fields except `window_start` are required. `score` may be
null to report missing inference. Both gates must be supplied explicitly;
walking probability is finite and in `[0,1]`, and quality is a JSON boolean.
An unsuccessful gate or null score abstains without adding a negative vote or
refreshing evidence. The recording ID is a string, a JSON-safe integer in
`[-(2**53-1), 2**53-1]`, or null. Boolean IDs and numeric strings used as numbers
are invalid. Numbers must be finite; boolean values never qualify as numbers.

`window_start` may be omitted or null until explicit support metadata is used.
Then subsequent scored updates in that recording must provide it. Raw support
is half-open `[window_start, timestamp)` and must be fresh and disjoint from
prior updates, including gated/missing-score updates. `advance` does not consume
raw support. Without metadata, the producer must guarantee causal, disjoint,
fresh evidence. Honest recording IDs and declared support cannot prove raw
sensor provenance. The timestamp is policy availability time; include inference
delay conservatively in support as described in the policy documentation.

All event types share a finite, nonnegative, strictly increasing global seconds
clock. Recording changes never restart it. Send `advance` events on a timer
during silence: this CLI has no background clock, and EOF creates no tick.
Choose timer cadence to bound fallback latency. `fallback` explicitly requires
a challenge, for example after an inference failure. Neither an earlier accept
nor a later defer authorizes indefinite use of cached acceptance.

Unknown/extra/missing fields, blank lines, nonobjects, duplicate keys,
NaN/Infinity/overflow, malformed JSON, invalid gate/interval values, and clock
duplicates/reversals fail closed. Each input line emits and flushes one JSON-safe
decision. Output adds `line_number`, `event_type`, `input_valid`, `event_error`,
`latched_error`, `policy_reason`, and `challenge_started` to the policy fields.
`challenge_started` is true only on the first transition to challenge. A protocol
fault sets `reason="invalid_event"`; the underlying reason remains available as
`policy_reason`. Policy-detected overlap or missing support retains its specific
policy reason and sets `input_valid=false` on the offending event.

After a protocol fault, valid later scores only advance the clock and remain
challenged. A malformed object with a valid later timestamp consumes that time;
an unparseable or invalid/duplicate/reversed timestamp is output as null and
never rewinds or invents time. Malformed input exits with status 2 after the
remaining lines are processed; valid policy challenges exit 0. Empty input emits
nothing. The host must handle process/transport failures as unavailable
authentication and enforce `authenticated` and `challenge_required` per result.

There is **no reset, recovery, or fallback-success event**. Such an event itself
requires a challenge. `AuthenticationStream` also exposes no reset method.
Only a trusted host that independently verifies fallback may create a new
instance/episode. Untrusted stream input must never control that host lifecycle.

## Timed evaluation API

`engine.continuous_evaluation.evaluate_stream` replays parsed event objects or
strict JSONL strings through exactly the same stream/policy and returns
`{"decisions": [...], "metrics": {...}}`. It accepts the required threshold and
the same policy keyword options. Trusted labels are separate from authentication
events and cannot influence a decision:

```python
from engine.continuous_evaluation import ObservedInterval, Takeover, evaluate_stream

result = evaluate_stream(
    [
        {"type": "advance", "timestamp": 0},
        {"type": "fallback", "timestamp": 7},
        {"type": "advance", "timestamp": 10},
    ],
    threshold=0.5,  # illustrative only
    observed_intervals=[
        ObservedInterval(0, 2, "owner"),
        ObservedInterval(2, 4, "impostor"),
        ObservedInterval(4, 6, "owner"),
        ObservedInterval(6, 10, "impostor"),
    ],
    takeovers=[Takeover("undetected", 2, 4), Takeover("detected", 6, 10)],
)
assert result["metrics"]["takeovers"][0]["time_to_challenge_seconds"] is None
assert result["metrics"]["takeovers"][0]["censor_time_seconds"] == 2
assert result["metrics"]["mean_detected_time_to_challenge_seconds"] == 1
```

This toy fallback demonstrates the accounting, not inferred biometric detection.
Any first challenge, including silence timeout or explicitly requested fallback,
is the policy endpoint; the metric does not infer that an impostor caused it.

Observation intervals use half-open `[start,end)` on the same global clock and
have identity `owner`, `impostor`, or `invalid`. They may be supplied unordered,
but must not overlap, even across recording IDs. Supply **only truly observed
exposure**, such as the disjoint intervals actually observed; a sparse recording's
first-to-last score span is not an observation interval by default. Walking,
quality, and missing-score abstentions still count as owner exposure when the
owner's presence is independently known. `invalid` means unusable identity or
exposure truth, and is excluded from owner/impostor exposure. Gaps are unobserved.
The replay clock must bracket every interval; provide initial/terminal `advance`
events rather than letting the evaluator invent monitoring before/after replay.

Owner false challenges count **transitions** to challenge during owner exposure.
The rate is this count divided by the sum of observed owner interval lengths
in hours. Repeated latched outputs contribute no extra challenges. Decisions and
challenge counts distinguish owner, impostor, invalid, and unobserved identity;
malformed input is separately counted via `invalid_event_count`. The rate is null
when owner exposure is absent or the stream is malformed. A challenge without a
usable timestamp has invalid attribution. Boundary events at an interval's end
are outside that interval; a touching next interval owns the boundary.

Each `Takeover(attack_id,onset,end)` requires positive, continuous observed
impostor exposure throughout `[onset,end)`. End follow-up at the first observation
gap; do not bridge an unobserved gap or use a later challenge as earlier
detection. IDs must be unique and attack intervals must not overlap. Outcomes are:

- `detected`: a new challenge in `[onset,end)`, with latency `challenge-onset`.
- `right_censored`: no challenge in that interval; latency is null and
  `censor_time_seconds=end-onset`. A challenge exactly at end is outside follow-up.
- `already_challenged`: fallback was latched before onset; exclude this attack
  from detection latency and censor counts. It was not an active unlocked attack.
- `invalid_stream`: malformed input makes metric claims invalid; latency and
  censor time are null. Diagnostic decisions/exposure/counts remain available.

`mean_detected_time_to_challenge_seconds` averages detected attacks only; it is
null when none were detected. It is a conditional descriptive mean, not an
overall latency estimate. Censored attacks are never counted as zero. Retain the
individual censor records for any later justified survival analysis.

One instance is one episode and can latch at most once. The owner rate is
descriptive for the supplied observation, including observed time while latched;
it cannot estimate recurrent daylong challenge burden without independently
verified recovery episodes and representative timed streams. Replay stores its
decision trace for audit; only the live policy/CLI has bounded history memory.

## Historical studies

These are new opt-in files; frozen study scripts, manifests, and artifacts remain
untouched. If other current source fixes change registered source hashes, replay
historical studies in a separate checkout of the recorded historical source
commit/snapshot and verify its bytes against the original hashes. Preserve those
registrations and results; do not rewrite hashes or bypass audits. No reserved
real-data replay was performed to implement or test this stream layer.
