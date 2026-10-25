# Canonical orientation: a nominated DEV operating-point improvement

The registered IDNet-inspired representation plus the existing lag-spectrum
block passes all four clean DEV nomination gates. It lowers mean FAR from
**0.8362% to 0.5000%**, with a small mean FRR improvement from **15.1264% to
14.7874%**. Under the prescribed constant 30°/90° coordinate rotations, its
errors remain the same as clean, while the raw-coordinate control deteriorates.
This is an exploratory nomination for the registered operating objective.
**Ranking diagnostics worsen**, gain/noise weaknesses remain, and no original
CALIB/TEST or deployment threshold changes.

## Clean results and exact nomination

| Representation | Mean FAR | Mean FRR | Folds FAR >2% | People no worse | Mean EER | Mean AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unchanged control362 | 0.8362% | 15.1264% | 3/20 | 15/15 | 4.7960% | 0.97438 |
| Canonical336 | 0.7557% | 18.2816% | 3/20 | 9/15 | 6.6293% | 0.95910 |
| Canonical362 | **0.5000%** | **14.7874%** | **2/20** | **10/15** | 5.8764% | 0.96240 |

Canonical336 fails the rejection and per-person gates. Canonical362 alone meets
mean FAR<=1%, no increase in folds above 2% FAR, strictly lower FRR and at least
10/15 people's mean FRR no worse. All decisions use exact fractions derived
from integer errors. Its mean FAR is `1/200`, FRR `2573/17400`; control is
`97/11600` and `329/2175`. These are **equal-fold mean fractions**, not pooled
independent claim counts. The clean FRR reduction is 0.3391 percentage points;
there is no significance or population uncertainty claim.

Across repeated clean validation appearances, the actual counts are 6/1,192
false accepts and 88/596 rejects for Canonical362, versus 10/1,192 and 90/596 for
control. The clean usability gain is just **two fewer rejects** in these dependent
appearances. These pooled counts are distinct from the equal-fold fractions above.
Five of the six canonical false accepts target identity 24; the aggregate FAR
does not establish equally strong protection for every claimant.

The operating-point improvement does not mean every ROC property improves:
mean EER rises 4.7960%→5.8764%, and mean AUC falls 0.97438→0.96240. The predeclared
objective uses fixed calibrated cutoffs and its four gates, not EER. A method
chosen for another operating objective may differ.

The comparison uses the same fifteen DEV identities, twenty balanced overlapping
outer folds, automatic identity-balanced WCCN, raw mean/two-medoid score blend
and policy C. Four nine-person inner models score four disjoint held triples;
pooled clean scores select empirical FAR<=1%. A twelve-person clean refit scores
the outer triple at that unchanged cutoff. The fixed budget completed 300 fits,
240 inner score sets and 420 outer comparisons. Original CALIB/TEST member
payloads remain unopened by this runner. Previously inspected TEST people cannot
supply independent confirmation through another evaluation.

## Constant-coordinate-rotation sensitivity

Day1 enrollment and clean-calibrated thresholds stay fixed. Synthetic changes
affect day2 probes only and never participate in nomination or recalibration.

| Condition | Control FAR / FRR | Canonical336 FAR / FRR | Canonical362 FAR / FRR |
| --- | ---: | ---: | ---: |
| Clean | 0.836% / 15.126% | 0.756% / 18.282% | 0.500% / 14.787% |
| Common x rotation 30° | 2.187% / 85.190% | 0.756% / 18.282% | 0.500% / 14.787% |
| Common x rotation 90° | 0.083% / 99.500% | 0.756% / 18.282% | 0.500% / 14.787% |
| Both-triad gain 0.9 | 0.000% / 41.764% | 0.167% / 52.718% | 0.000% / 45.207% |
| Both-triad gain 1.1 | 0.000% / 37.828% | 0.083% / 44.793% | 0.167% / 39.828% |
| Gaussian noise 1% | 0.000% / 63.540% | 0.000% / 55.615% | 0.250% / 51.787% |
| Gaussian noise 5% | 0.000% / 100.000% | 0.000% / 100.000% | 0.000% / 100.000% |

Both canonical arms retain the clean FAR/FRR under these specified rotations.
This tests a constant proper coordinate change shared by acceleration and
angular velocity. It does not simulate physical phone displacement, dynamically
rotating the device during the window, altered walking motion or independent
sensor-frame misalignment. The amplitude-gain cases are worse than control;
1% noise improves rejection but remains poor, and 5% noise rejects every genuine
claim. Low FAR paired with 100% FRR is unusable, not a security improvement.
Noise uses refit TRAIN-only dynamic triad RMS, original row keys and fixed
seed 20261025. Magnitudes are engineering probes, not measured phone noise.

## Individual usability remains uneven

Four people improve, six tie and five worsen relative to control. Person 9
remains completely rejected; the aggregate improvement does not fix that user.

| DEV ID | Control mean FRR | Canonical362 mean FRR |
| --- | ---: | ---: |
| 1 | 2.50% | 2.50% |
| 11 | 15.00% | 2.50% |
| 13 | 27.50% | 32.50% |
| 14 | 0.00% | 25.00% |
| 16 | 0.00% | 0.00% |
| 17 | 2.50% | 5.00% |
| 19 | 2.78% | 0.00% |
| 24 | 0.00% | 0.00% |
| 25 | 2.50% | 0.00% |
| 27 | 0.00% | 0.00% |
| 29 | 0.00% | 2.50% |
| 3 | 5.00% | 7.50% |
| 5 | 67.50% | 42.50% |
| 7 | 0.00% | 0.00% |
| 9 | 100.00% | 100.00% |

## Paper adaptation and numerical review

[IDNet section 3.3, equations 5–13](https://arxiv.org/html/1606.03238#S3.SS3)
motivates mean-acceleration gravity and horizontal-PCA frames. This adaptation
uses existing 256-sample windows in native units. Both physical triads share a
proper frame; both PCA signs receive equal feature weight **before metric
learning**. Canonical336 contains axis/norm208 plus invariant128; Canonical362
appends the original 26 raw-window lag-spectrum features. It adds no IDNet cycle
extraction, resampling, filtering, amplitude normalization, CNN or one-class SVM.
The original paper's accuracy is not transferred to this dataset or pipeline.

Pre-fit review found that exact axis-spectrum `argmax` could switch between equal
peaks after an otherwise equivariant rotation. A local canonical-only correction
selects the lowest frequency within 64 float64 eps of maximum power, matching
the existing invariant descriptor's convention. The shared historical extractor
and control remain unchanged. The corrected counterexample remained stable
across 512 additional proper rotations; the old argmax switched in 158 cases.
Tests retain the balanced-peak regression, sign-pair ordering, shared-triad
geometry, proper determinant, reconstruction and numerical-range checks.

Numerical validity does not prove physical gravity or walking. Pure noise can
pass the frame guards, and weak horizontal eigengaps can produce large frame
changes under noise. Invalid frames retain their source rows with NaNs; no
fallback axes or imputation create authentication evidence. Window-end feature
extraction uses only its current window, not later windows.

## Registration, replay and publication

Frozen source is main commit
[`108a559bc1a83355dc85335e9f640786846f894a`](https://github.com/VariableThe/gait-biometrics/commit/108a559bc1a83355dc85335e9f640786846f894a).
The [protocol](../../docs/canonical_orientation_study.md) records all 41 source
hashes, runtime versions, folds, geometry/peak guards, representations, scores,
threshold policy and conditions before DEV materialization. All source-derived
claim graphs and 141 geometry contexts register before features or fits. The
588 original clean DEV windows and every required perturbed context retain
identifiable frames; invalid context-window count is zero. This is a conditioning
check, not a learned quality gate or 588 independent people/sessions.

Strict verification passed: all 300 numeric model states, 141 geometry contexts,
240 inner/420 outer score sets, exact metadata, TRAIN assignments/statistics,
cache-derived thresholds/counts and exact nomination. Numerical score tolerance
is 1e-6 with identical fresh/cache decisions at every frozen threshold. It fits
nothing. End-to-end tests count actual tiny TRAIN fits and use distinct arm/stress
features; malformed assignments, numeric artifacts, graphs, thresholds and
selection fail. Exact-selection regressions include unequal denominators.

Final full suites pass **1,047 main / 1,043 Student B tests**, each with **1,592
subtests** and eight pre-existing export/quantization warnings. The focused new
suite passes 51 tests plus 47 subtests. All new Python files pass Ruff and ty.
No new dependencies or GPU training are required. The preceding frozen-lock
audit remains published in the robustness report; the lock did not change.

Publication includes 660 lossless score caches, 141 geometry NPZs, 20 threshold
files, registrations, compact comparison counts, summary and 1,121 numeric
artifact fingerprints. Large model matrices and full scoring-template metadata
remain in the original local `artifacts/canonical-orientation-v1` root.
`evidence/comparisons.json.gz` retains hashes for omitted scoring metadata and
the full original result hash. Full numerical model replay needs that local
archive or a newly registered reproduction. The Student B copy is the same
experiment, not an independent replication. No prior source/artifact manifests
were rewritten to match this new method.

The [independent evidence review](audit.md) checks all 1,121 artifacts and exact
selection separately from the root's numerical replay. Its cached-coverage
reconstruction preserves 28 source continuity breaks, 33,810 tail rows and the
zero-window subject 19/day 2/trial 1 recording. Archive authenticity remains bound
to the caller's previously verified official release; see the
[original cross-day authenticity evidence](../2026-10-23-cross-day/README.md).
Member hashes and cache consistency do not independently re-establish raw truth.

This is a useful DEV nomination and a reduction in the measured constant-frame
rotation sensitivity. Fresh repeated phone sessions, actual device timing and
single-window calibration remain necessary for live authentication claims.
The previously inspected TEST result is still 1.5% FAR, not this 0.5% DEV mean.
