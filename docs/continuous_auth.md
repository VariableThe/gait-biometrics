# Causal score-level continuous authentication

`engine.continuous_auth.ContinuousAuthenticator` consumes higher-is-more-similar
scores. It uses only the Python standard library, loads no data, and imports no
model or Torch. This is a deterministic research policy, with no claim of real
accuracy, daylong error rates, security certification, or phone deployment.
Enrollment, first unlock, walking/quality estimation, threshold calibration,
model inference, and execution of a challenge belong to the caller.

```python
from engine.continuous_auth import ContinuousAuthenticator

auth = ContinuousAuthenticator(
    threshold=-0.3,  # illustrative only; independently calibrate the real policy
    history_size=3,
    required_accepts=2,
    required_rejects=2,
    walking_threshold=0.8,
    max_evidence_age_seconds=30,
    max_gap_seconds=10,
)

first = auth.update(-0.2, 2, recording_id="session-a", window_start=0)
assert first["decision"] == "defer"
second = auth.update(-0.1, 4, recording_id="session-a", window_start=2)
assert second["decision"] == "accept"
silent = auth.advance(14)
assert silent["decision"] == "challenge"  # exact ten-second evidence gap
assert auth.update(-0.1, 16, recording_id="session-a", window_start=14)["decision"] == "challenge"

# Only after the caller authorizes a new episode, e.g. a successful fallback:
auth.reset()
```

## API and output

The constructor signature above lists all options and defaults. The threshold
must be a finite real number. History size and both quorums must be positive
integers; each quorum must fit within the history. The walking threshold is in
`[0, 1]`; both time limits are finite and strictly positive. Invalid constructor
arguments raise `ValueError`. Boolean values do not qualify as numeric inputs.

```python
update(score, timestamp, walking_probability=1, quality_ok=True,
       recording_id=None, *, window_start=None) -> dict
advance(timestamp) -> dict
request_fallback(timestamp) -> dict
reset() -> None
```

Every returned dictionary has these JSON-native fields and can be serialized
with `json.dumps(result, allow_nan=False)`:

- `decision`: `"accept"`, `"defer"`, or `"challenge"`.
- `reason`: a stable machine-readable explanation, listed below.
- `authenticated`: true **only** when this call returns `accept`.
- `challenge_required`: true when fallback has latched.
- `timestamp`: the finite normalized seconds value, or null for an invalid,
  duplicate, or reversed timestamp. An invalid timestamp never rewinds the clock.
- `recording_id`: the last validated ID, a string, a JSON-safe integer with
  absolute value at most `2**53 - 1`, or null. Boolean IDs are invalid.
- `recording_changed`: whether this update changed a previously seen ID.
- `evidence_added`: whether this call appended a valid gated vote.
- `evidence_count`, `accept_count`, `reject_count`: retained history counts.
- `evidence_age`, `oldest_evidence_age`: seconds since the newest and oldest
  retained votes' raw support starts. Without interval metadata, use the score
  timestamp as that vote's support start. Both fields are null when no votes
  remain or this call has an invalid timestamp. Diagnostic counts can remain
  populated after a challenge; they never authorize recovery.

No authorization lease is implicit in an earlier `accept`. A subsequent gate
failure, missing score, contradictory history, or `advance` returns `defer` with
`authenticated=false`. The caller must enforce this result and the eventual
challenge rather than cache an old acceptance indefinitely. `defer` means no
positive authentication decision on this observation; it does not mean a
successful fallback. `request_fallback` allows runtime/inference failures or an
external policy to require a challenge immediately. This module does not run
the challenge or attest that the user passed it.

## Evidence and decision rules

1. Check time and expire history/deadlines **before** considering this score.
2. Changing `recording_id`, including a change to or from null, clears votes and
   interval tracking. The current observation can begin the new history. The
   first supplied ID is initialization, not a change. A change never rewinds
   time, extends a deadline, or clears a latched challenge. Keep an explicit,
   stable ID per recording; an omitted ID is the actual null ID, not shorthand
   for the previous ID.
3. Validate input and window metadata. Malformed runtime input requires a
   challenge; it cannot reuse earlier acceptance.
4. If quality is false, walking probability is below its threshold, or the score
   is missing (`None`), abstain without adding a vote or refreshing timers.
   Walking equality passes. Quality must be a Python `bool`; probability must
   be finite and in `[0, 1]`.
5. A finite score `>= threshold` votes positive, including equality; a score
   below it votes negative. Keep at most the newest `history_size` eligible
   votes. Gates do not occupy history slots or manufacture negative votes.
6. At least `required_rejects` retained negative votes latch `challenge`, even
   if the positive quorum is also present. Otherwise **any** retained negative
   vote prevents acceptance (`contradictory_evidence`). Acceptance requires at
   least `required_accepts` positive votes and zero retained negatives. Fewer
   positive votes defer. A lone contradiction can leave by count eviction or
   age expiry; it cannot be ignored by a positive majority.

This is a conservative bounded K-of-N policy, with an additional contradiction
veto. It is not a score average, a fitted likelihood accumulator, or evidence
that raw windows are statistically independent. Memory use is `O(history_size)`;
each update's count/prune work is also bounded by that configured size.

## Exact clock, freshness, and interval invariants

All calls other than `reset` share one finite, nonnegative, **strictly increasing**
clock in seconds. Use a trusted monotonic clock, or one coherent replay clock;
never mix Unix time, per-recording relative time, and process monotonic time.
Duplicate or reversed times latch `nonmonotonic_timestamp`. Advance to a later
time when requesting fallback after an update. `reset` clears the clock for a
new episode; changing recording ID does not. Time is observed only on calls:
the runtime must call `advance` on a timer during sensor silence, with a cadence
appropriate to its required fallback latency. There is no background thread.

A vote is discarded when `now - support_start >= max_evidence_age_seconds`.
Equality is expired. With metadata, `support_start` is the raw interval start;
without it, the score timestamp is used and raw freshness is a caller guarantee.
Every interval must satisfy `0 <= window_start < timestamp`, and its duration
must be strictly less than the maximum evidence age. This prevents a newly
delivered score for an ancient or excessively long window from being counted
as fresh merely because delivery is recent.

The timestamp is the window end in the policy clock and the time this score is
available to this policy. No input after that timestamp may contribute to it.
For delayed inference, the caller must conservatively include delivery delay
in the declared interval (end it at availability), or request fallback for a
stale result. Do not relabel delayed raw evidence as freshly captured data.

Optional `window_start` describes half-open support `[window_start, timestamp)`.
Within one recording it must be at least the preceding **update** timestamp;
touching intervals are allowed, overlapping or replayed intervals challenge.
The watermark includes gated and missing-score updates and survives vote
eviction. `advance` does not consume a raw interval. Once interval metadata is
supplied, subsequent scored updates in that recording must supply it; omission
challenges. A missing-score update may omit it but still advances the watermark.
If metadata was previously absent, the first explicit interval must still
start at or after the preceding update. A recording boundary clears this
watermark because old and new recording evidence are never combined.

Without `window_start`, the policy cannot detect overlapping or replayed raw
samples with different timestamps. The caller **must** submit only disjoint,
causally produced, fresh raw evidence and honest recording IDs. In particular,
overlapping sliding windows, duplicated streams, reused embeddings, or multiple
scores of the same raw samples must not become separate votes. Interval checks
verify declared support, not sample provenance or sensor authenticity.

Two independent fallback deadlines prevent indefinite abstention:

- The evidence gap is measured from the last valid gated vote, or the first
  observed timestamp if there has been no vote. At a gap **greater than or equal
  to** `max_gap_seconds`, latch `evidence_gap`. Missing scores, gates, heartbeats,
  and recording changes do not refresh it.
- Time without acceptance is measured from the last `accept`, or the first
  observed timestamp before any acceptance. At **greater than or equal to**
  `max_evidence_age_seconds`, latch `authentication_expired` (previously
  accepted) or `consensus_timeout` (never accepted). Even a stream of fresh,
  contradictory votes or continual recording changes cannot postpone it.

When both deadlines fire on the same call, the acceptance-age reason takes
precedence. Incoming scores are considered only if neither deadline has fired.
Thus a fresh high score at an expired deadline cannot resurrect old history.
Once challenged, all methods retain the first challenge reason until `reset`.
Only the caller can authorize a new episode after its explicit fallback.

Ordinary defer reasons are `insufficient_evidence`, `contradictory_evidence`,
`quality_gate`, `not_walking`, `missing_score`, and `awaiting_evidence`.
Acceptance uses `accept_quorum`. Other challenge reasons are `reject_quorum`,
`fallback_requested`, `invalid_timestamp`, `nonmonotonic_timestamp`,
`invalid_recording_id`, `invalid_walking_probability`, `invalid_quality_flag`,
`invalid_score`, `invalid_window_start`, `stale_window`, `overlapping_evidence`,
and `missing_window_start`.

## Evaluation and replay limits

The tests exercise synthetic score transitions, clock failures, silence, gates,
recording changes, contradiction, freshness, overlap, and explicit fallback.
They make no claim about biometric error rates. No metric helper infers owner
hours or takeover events from trial scores: the runtime must supply observed
owner exposure, separate challenge episodes from repeated latched outputs,
and define takeover onset, detection, observation gaps, and right-censoring.
Label any concatenated owner/impostor trials as **synthetic splices**; do not
extrapolate short-trial rates into daylong false-challenge claims.

This module is opt-in and does not alter frozen study artifacts. If later core
fixes change source hashes recorded in a historical study, replay that study
from its recorded historical source commit in a separate checkout. Preserve
the original manifest and hashes rather than rewriting them to describe current
source. Policy evaluation needs separately calibrated thresholds and held-out,
properly timed streams; adding this policy does not establish their accuracy.
