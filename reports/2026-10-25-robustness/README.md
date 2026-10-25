# Covariance baselines, sensor stresses and causal replay

The next paper-guided comparison completed **300 CPU covariance fits** and
replayed all **660 score sets**. Neither new representation passes the registered
clean-data gates. Retain the existing representation and policy C. The new
diagnostics expose substantial sensor sensitivity and quantify policy timing;
they do not improve or replace the published TEST operating point.

## Clean DEV result: no promotion

| Representation | Mean FAR | Mean FRR | Folds FAR >2% | People no worse than control | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Existing spectrum + fusion, 362D | 0.8362% | 15.1264% | 3/20 | 15/15 | Retain |
| Full covariance + fusion, 494D | 1.2500% | 18.2989% | 1/20 | 10/15 | Reject |
| Matrix-log covariance + fusion, 494D | 1.0862% | 17.1322% | 3/20 | 9/15 | Reject |

Twenty overlapping outer DEV folds each use four nine-person inner fits and a
twelve-person refit per arm. All three arms retain automatic-shrinkage WCCN, raw
scores, the equal mean/two-medoid enrollment blend and pooled threshold policy C.
Source-derived coverage registers before feature extraction/fitting. Nomination
uses clean outcomes only, exact fractions of counts and all four safety/usability
gates. Full covariance versus log covariance isolates the logarithmic transform;
comparison with the spectrum control also changes regularization/parameterization.
The control exactly reproduces the prior policy-C fractions `97/11600` FAR and
`329/2175` FRR. These are **equal-fold mean fractions**, not pooled claim counts.

The matrix-log mathematical baseline adapts
[Barachant et al., ESANN2012](https://www.esann.org/sites/default/files/proceedings/legacy/es2012-130.pdf),
sections 2.1/2.3. Its original task was EEG classification. This pipeline is neither
a reproduction of that experiment nor a pure Riemannian-distance classifier.
[The frozen protocol](../../docs/covariance_robustness_study.md) and
[feature definition](../../docs/lag_covariance.md) specify every transformation.

## Fixed-threshold sensor sensitivity

These are synthetic **probe-only** changes to the same DEV recordings. Day1
enrollment and clean-calibrated thresholds remain fixed. Stress outcomes never
participate in nomination or recalibration.

| Condition | Control FAR / FRR | Full covariance FAR / FRR | Log covariance FAR / FRR |
| --- | ---: | ---: | ---: |
| Clean | 0.836% / 15.126% | 1.250% / 18.299% | 1.086% / 17.132% |
| Common x rotation 30° | 2.187% / 85.190% | 1.520% / 88.046% | 1.767% / 90.063% |
| Common x rotation 90° | 0.083% / 99.500% | 1.000% / 99.167% | 1.667% / 98.667% |
| Both-triad gain 0.9 | 0.000% / 41.764% | 0.500% / 39.454% | 0.333% / 34.770% |
| Both-triad gain 1.1 | 0.000% / 37.828% | 0.167% / 40.149% | 0.000% / 34.805% |
| Gaussian noise 1% | 0.000% / 63.540% | 0.250% / 55.293% | 0.083% / 54.149% |
| Gaussian noise 5% | 0.000% / 100.000% | 0.000% / 100.000% | 0.000% / 100.000% |

Noise amplitude is the stated fraction of triad dynamic RMS from refit TRAIN
identities only; original window keys and seed 20261025 determine each draw.
Acceleration and angular velocity retain separate native units. Proper rotations
apply consistently to both vectors. The full fusion descriptors include axis
features, so adding invariant blocks never made the complete system invariant.
The very low FAR under heavy noise comes with complete rejection of genuine
claims; it does not represent improved authentication.

[Um et al.2017](https://arxiv.org/abs/1706.00527), section3.2, motivates rotation,
scaling and noise sensitivities while warning that transformations need not
preserve labels. [Hoang et al.2015](https://arxiv.org/abs/1602.03199) studies gait
orientation instability. These engineering stress magnitudes are not measured
phone-noise or relocation distributions. The transforms are not MODA reproduction,
physical gait simulation, new subjects or evidence of spoof resistance.

## Timed synthetic takeover replay

Ten explicit scenarios, nine seeds and three fixed quorum policies produce
**270 replay cells**. Two invocations reproduced identical output bytes
(`25c7cb0fd82244c71d763e1e48a020c13524c5558c7e5c900f1f08f503f85c31`).
Seeds perturb artificial scores but do not change their threshold decisions in
these fixtures. They are not independent people, sessions or security trials.
No policy is selected from this deliberately constructed scenario set.

For the existing 2-of-3 contradiction-veto policy, with 2-second windows:

| Constructed scenario | Observed outcome |
| --- | --- |
| Abrupt low-scoring impostor | Challenge 4 seconds after handoff |
| Three high-scoring impostor windows, then two low | Challenge 10 seconds after handoff |
| Walking blackout, quality blackout or silence | Gap deadline challenges 8 seconds after handoff |
| Four-second delayed delivery | Challenge 12 seconds after handoff; raw age includes delivery delay |
| Replayed or overlapping owner support | Provenance guard requests fallback 2 seconds after handoff |
| Recording ID changed every attacker window | Censored at 18 seconds in the short fixture |
| Owner already challenged during warmup | Retained as owner false challenge and ineligible handoff |

Each policy retains all nine deliberately injected owner-challenge episodes.
The default policy detects 72 of 81 eligible short-fixture attacks and retains nine
censored recording-reset attacks. This fraction is a property of the chosen
fixtures, not deployment detection probability. A provenance-guard endpoint is
reported separately from a biometric/policy endpoint.

After observing short-follow-up censoring, an explicitly diagnostic extension
lengthened the same recording-reset scenario to 24 attacker windows, adding
**27 replay cells** without changing policy parameters. The 2-of-3 and 3-of-5
policies then challenge 28 seconds after handoff at `authentication_expired`;
the 1-of-1 policy challenges after 2 seconds. Recording changes clear votes but do
not extend the acceptance deadline. Current `defer` decisions do not authorize
continued access while waiting for a challenge.

All latencies refer to the declared synthetic schedule, timer cadence and injected
scores. Owner/impostor presence during silence is explicitly assumed simulation
truth. Unknown real observation gaps cannot be filled this way. Right-censored
attacks keep null latency, and reported average latency is detected-only.
No recurrent/daylong challenge rate or real takeover latency is established.
[The simulation guide](../../docs/auth_simulation.md) describes clocks, trusted
source support and reproducible CLI/API construction. The current bounded quorum
is not the sequential likelihood-ratio method in
[IDNet, section 6](https://arxiv.org/html/1606.03238); that remains a distinct future
baseline requiring window-level calibration.

## Verification and published evidence

Frozen experiment source: main repository commit
[`b5a014c`](https://github.com/VariableThe/gait-biometrics/commit/b5a014c).
The registration records source hashes, Python/NumPy/SciPy/scikit-learn versions,
the complete candidate grid, folds, numerical settings and DEV-only payload
allowlist before data materialization. The source trial graphs register before
features/fits. No original Santos CALIB or TEST payload is requested.

Strict verification passed: 300 model states and TRAIN assignments/statistics,
240 inner and 420 outer score replays, exact metadata/cache-derived thresholds,
unchanged threshold decisions and exact count-derived nomination. The synthetic
end-to-end test counts actual fit calls; focused checks reject corrupted model
assignments, hashes, scales and thresholds. Real feature/perturbation integration
and vanished-trial coverage are also persistent regressions.

Final full regression suites passed **996 main / 992 Student B tests**, each
with **1,545 passing subtests** and eight existing export/quantization warnings.
The frozen focused study suite passed 23 tests. Ruff/ty pass across
all new Python files. The frozen lock audit found no known vulnerabilities or
adverse project statuses in 87 packages. Independent static reviews found no
blocking study/simulation issue; simulation review also matched 192 episodes to
an independent reference. These checks establish implementation and accounting,
not the representativeness of this small reused cohort.

Publication contains lossless 660 score NPZs, all 20 threshold files, source/data
registrations, per-comparison counts, summary, numeric artifact inventory,
strict-verification record, full compressed simulation traces and validation logs.
`covariance/comparisons.json.gz` omits bulky scoring-template coordinates and
retains their hashes, plus the original full-result hash. The full 57MB
`results.json` and approximately450MB model archive remain in the local
`artifacts/covariance-robustness-v1` experiment root. Full numerical model replay
requires that root or a fresh registered run, not only this compact publication.
No historical registration or report was overwritten. The [independent evidence audit](audit.md) records the separate review scope and its limits.

The next useful representation tests should target the demonstrated orientation
and noise failures, with clean usability retained as a gate. Existing data can
support such exploratory comparisons and source-bound window replay. Fresh phone
sessions are still required to measure actual deployment generalization. The
published TEST point remains 1.5% FAR, above the intended 1% target; none of these
results establishes a state-of-the-art or maximum-attainable claim.
