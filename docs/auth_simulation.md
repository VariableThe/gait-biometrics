# Deterministic authentication stress replay

`engine.auth_simulation` adds standard-library-only **synthetic-splice** replay
around the current `AuthenticationStream` / `evaluate_stream`. It loads no
recordings, artifacts, models, calibration files, or learned parameters and fits
nothing. Existing authentication and evaluation code is unchanged. The policy
is bounded **K-of-N with a contradiction veto**, not SPRT or a likelihood-ratio
test. This harness tests policy behavior, timing, and accounting; its injected
scores are not scores measured from real sensors.

## Reproduce a small comparison

From `gait-biometrics`, using the existing interpreter without dependency sync:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
  uv run --no-project --python ../.venv/bin/python python -m engine.auth_simulation \
  --scenario abrupt_takeover --scenario recording_resets --seed 7
```

Omit `--scenario` for ten scenarios, one seed, and three fixed policies: 30
episodes, not a parameter search. Repeat `--seed` for explicitly chosen seeds.
`--trace` includes every event and policy decision. Output is deterministic JSON
with sorted keys and finite numbers. Redirect stdout to a new report location
when needed. Repeating a scenario/seed or a policy name is rejected rather than
counting the same episode twice. The CLI also runs as the module API; no separate
script or dependency installation is needed.

```python
from engine.auth_simulation import (
    DEFAULT_POLICIES, ScenarioConfig, build_scenario, replay_episode, run_grid,
)

config = ScenarioConfig(scenario="bursty_acceptance", seed=7,
                        accept_burst_windows=4, reject_burst_windows=2)
episode = build_scenario(config)
one = replay_episode(episode, DEFAULT_POLICIES[1], include_trace=True)
comparison = run_grid([episode])
assert one["metrics"]["takeovers"][0]["time_to_challenge_seconds"] == 12
```

The fixed grid uses `(history_size, required_accepts, required_rejects)` of
`(1,1,1)`, `(3,2,2)`, and `(5,3,2)`, all with threshold `0.5`, walking threshold
`0.8`, evidence age `30 s`, and gap deadline `10 s`. These are illustrative
settings. API callers can pass a short sequence of `PolicySpec` values to
`run_grid`; it does not choose a winner or tune a threshold on these episodes.

## Scenarios and explicit assumptions

`ScenarioConfig` defaults to six owner windows, eight attacker windows, `2 s`
windows, a `1 s` timer, seed `7`, and score jitter of `0.05`. Owner scores are
injected around `1`, negative impostor scores around `0`; the jitter uses a local
`random.Random(seed)`, leaving the global random state unchanged. It usually
does not change binary decisions at threshold `0.5`. Seeds identify reproducible
artificial perturbations, not independently sampled people or recordings.

| Scenario | Construction |
| --- | --- |
| `abrupt_takeover` | High owner scores, then consecutive low attacker scores. |
| `bursty_acceptance` | Entire blocks of high attacker scores (default three), then low scores (default two), repeated. The correlation is deliberately imposed. |
| `walking_blackout` | First six attacker windows fail the walking gate. |
| `quality_blackout` | First six attacker windows fail the quality gate. |
| `silence` | No attacker deliveries; timer events continue through declared follow-up. |
| `delayed_delivery` | Attacker deliveries delayed by configurable `delay_seconds` (default four); duration plus delay determines raw age. |
| `replay` | Present the last owner window again under a changed policy recording ID. |
| `overlap` | Rename a window overlapping the owner's raw support and change the presented recording ID. |
| `recording_resets` | Change the presented recording ID on every fresh attacker window. This clears votes but cannot restart deadlines. |
| `owner_challenge` | Low scores during owner warmup; retain the owner false challenge and the already-challenged handoff. |

Warmup ends one window duration after the last owner delivery, before attacker
capture starts. Follow-up ends one duration after the last attacker delivery.
Synthetic identity presence is **explicitly assumed** throughout the declared
owner and attacker intervals, including gate failure and silence. This is what
permits timeout attribution in these scenarios; it is not evidence that a real
sensor observed someone during a gap. Delayed capture cadence is stretched to
avoid overlapping the conservative availability-ended windows. Real pipelined
overlapping deliveries cannot simply be substituted as independent votes.

## Source-bound API and rejection

`ScoreWindow(source_recording, window_id, raw_start, raw_end, score,
walking_probability=1, quality_ok=True, score_origin="synthetic_injected")`
binds a score to half-open support in a canonical source recording. Its
`duration_seconds` property is `raw_end - raw_start`, with no independent duration
field to contradict the support. Missing scores use `None`. Similarity scores
may be negative; all numeric inputs must be finite, and booleans are not numbers.

`Delivery(window, capture_start, delivered_at, recording_id="synthetic-device")`
maps that raw duration onto the episode clock. Capture must finish before or at
delivery; `capture_end = capture_start + duration`. Audit records include raw
coordinates, IDs, duration, capture and delivery times, gates, score origin,
`raw_age_seconds = delivered_at - capture_start`, and
`delivery_delay_seconds = delivered_at - capture_end`.

`Episode(episode_id, seed, start, end, takeover, deliveries, observed_intervals,
timer_seconds=1)` declares one owner warmup and one attacker handoff. Supply
`Takeover` and `ObservedInterval` from `engine.continuous_evaluation` directly.
Custom intervals may be disjoint; they are never inferred from first/last score
times or stretched across an observation gap. They must bracket no time outside
the episode; the takeover requires continuous impostor exposure until its end.
There must be observed owner warmup before onset. `scenario_config` in generated
episodes/reports records the generator parameters; custom episodes can omit it.

Within each episode, the source adapter rejects:

- Repeated `(source_recording, window_id)`, including relabeled support.
- Raw support that overlaps or moves backward within a canonical source recording,
  even with a new window ID or a changed presented `recording_id`.
- Global declared support starting before the previous delivery, even across
  distinct recordings. Gates and missing scores consume this watermark too.
- Raw capture spanning the owner/attacker boundary.
- Raw age **greater than or equal to** the configured maximum evidence age.

Rejected support becomes a `fallback` event at availability, with its reason
retained in `score_windows[].support_error`. It never becomes a new score vote.
Thus intentional source attacks remain valid evaluation episodes: fallback is
the observable endpoint. `first_challenge.trigger` distinguishes a provenance
guard fallback from a policy endpoint; `source_rejection` also records a guard
failure coincident with an earlier-priority policy deadline. A guard endpoint
does not demonstrate that the biometric score detected an attacker. Constructor
errors raise `ValueError` and do not produce a partial metric report.

Source IDs, support coordinates, and capture mappings are trusted adapter input.
Renaming the canonical source recording itself can conceal raw reuse; these
checks cannot authenticate bytes or defeat a malicious provenance provider.
Reuse across separate episodes is allowed and never establishes independence.

## Clock, lifecycle, and metrics

`evaluate_stream` creates a **fresh** `AuthenticationStream` per grid cell. No
reset or verified recovery occurs within an episode. Warmup owner challenges
remain latched through handoff and subsequent recording changes. No episode is
discarded for failing to accept during warmup: `warmup_had_accept` and
`last_pre_handoff_decision` expose that condition. “Eligible” means unchallenged
at onset under the evaluator, not proof that a deployment granted access.

Deliveries must already have strictly increasing global availability times;
reversed/duplicate times are rejected, not sorted or nudged. Timer `advance`
events cover the episode from initial to terminal time, including silence.
Identity handoff metadata does not inject extra policy clock calls. A
score/fallback at the same timer instant replaces that tick because
it checks deadlines itself. The policy sees `window_start=capture_start` and
`timestamp=delivered_at`, including delay in raw freshness. Timers never consume
raw support, refresh acceptance, or fabricate a score. There is no background
thread. Challenge latency depends on the specified timer cadence and delivery
schedule; it is not automatically the mathematical deadline. An event budget
rejects excessively fine timers before replay.

Reports retain owner false-challenge **transitions**, observed owner/impostor
seconds, individual takeover outcomes and follow-up lengths, right-censor times,
already-challenged attacks, eligible attacks, seeds, source audit, and optionally
the full decision trace. Summaries pool exposure and counts rather than average
per-episode rates. Conditional detection latency averages **detected attacks
only**; censored attacks have null latency, never zero. Follow-up is half-open:
a challenge exactly at onset has zero latency; a challenge exactly at end is
outside follow-up. A challenge after the first observation gap cannot detect the
earlier attack. The evaluator's diagnostic invalid-stream outcomes remain
distinct from both censoring and detection.

The per-observed-hour rate is a descriptive normalization of explicitly declared
short exposure, including time while latched. It is not a recurrent or daylong
false-challenge estimate. A detected fraction in this deliberately selected
scenario set is not an empirical FAR/FRR or deployment detection probability.
No confidence bounds assume independent windows, subjects, sources, or seeds.

## Verification and integration

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
  uv run --no-project --python ../.venv/bin/python python -m pytest -q \
  tests/test_auth_simulation.py tests/test_continuous_auth.py \
  tests/test_continuous_evaluation.py tests/test_stream_authentication.py
```

Tests include independently calculated quorum latencies, burst patterns, silence
deadlines and timer rounding, half-open censor boundaries, owner blackout,
already-challenged handoff, delayed raw age, source replay/overlap, latching across
recording changes, deterministic hostile inputs, and exact exposure pooling.
A separate parity check verifies that emitted events use the existing stream's
decisions. Every cell starts a new policy, with no reset after construction.

To use precomputed current-data scores later, an authorized adapter can construct
`ScoreWindow(..., score_origin="external_unverified")` and explicit `Delivery`
and `Episode` objects. This label is not verification of a model, threshold,
subject split, score freshness, raw support, or independence. Preserve canonical
recording IDs and sample-derived support; do not supply overlapping sliding
windows, reused embeddings, or per-recording zero clocks as new evidence. Supply
independently known observation intervals and end takeover follow-up at gaps.
An artificial owner/impostor splice remains synthetic even when its individual
scores came from existing data. This implementation performs no such data replay
and changes no registered study scripts, manifests, hashes, or artifacts.
