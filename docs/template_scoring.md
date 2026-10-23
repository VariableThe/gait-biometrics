# Enrollment-only cross-day template scoring

`engine.template_scoring` is a NumPy-only, opt-in scorer. It loads no data,
imports no Torch, fits no representation and selects no decision threshold.
Use authentic subject/day/trial metadata from an independently validated
adapter. The scorer cannot establish identity authenticity, raw-sample
nonoverlap, session provenance or threshold calibration provenance itself.

```python
from engine.template_scoring import score_trials, paired_subject_diagnostics

labels, scores, metadata = score_trials(
    embeddings, subjects, days, trials,
    template_method="mean", probe_method="mean",
)
```

`embeddings` is a finite, nonempty real `(N,D)` matrix; each of the other arrays
has `N` integer entries. Subject IDs are nonnegative int64-compatible authentic
labels, days are 1/2 and trials are 1..10. Window vectors are unit-normalized,
even if already supplied as unit vectors. For each subject/day/trial, take the
arithmetic mean of its unit windows and normalize that mean. A window norm
below `1e-12` fails explicitly. A trial mean norm below `1e-12` is excluded and
reported rather than assigned an arbitrary direction.

Day1 eligible trials enroll, with each trial receiving equal weight regardless
of its number of windows. Every eligible day2 trial is a separate probe.
A subject needs at least one eligible enrollment trial and one eligible probe;
at least two subjects must survive. A cancelling mean enrollment template is
excluded for `mean`. If too few identities survive, `CoverageError`, a
`ValueError` subclass, retains the JSON-safe coverage report in `.coverage`.
The medoid candidates can retain subjects whose arithmetic mean cancels, so
candidate coverage can differ; report that difference explicitly.

Only these three candidates exist; `probe_method` supports only `"mean"`:

- `mean`: normalize the equal-trial arithmetic mean of day1 trial vectors.
  This preserves `scripts.cross_day_verification.cross_day_scores` arithmetic
  and claim scores exactly for ordinary inputs. The old scorer takes requested
  IDs in caller order; this API always sorts observed IDs. Compare rows by
  identity/trial keys if the old caller supplied a different identity order.
- `robust`: select the day1 angular medoid, the trial minimizing the sum of
  angular distances to all other enrollment trials. Exact cost ties choose
  the smallest trial ID. The stored prototype is that trial's unit vector.
  This limits the influence of a distant trial relative to arithmetic averaging;
  the name is not a guarantee of improved errors or a robustness theorem.
- `multi`: minimize the total nearest angular distance over every pair of
  enrollment trials, using at most `10 choose 2 = 45` candidate pairs. Exact
  ties choose the lexicographically first trial pair. Store the two unit trial
  exemplars and use their **maximum** cosine to a probe. One enrollment trial,
  or directions agreeing within `1e-12` radians, gives one prototype. Enrollment
  membership ties choose the first prototype. There is no initialization,
  stochastic clustering, iteration, sklearn dependency or probe fitting.

For medoid distances the implementation uses
`2*arcsin(clip(norm(u-v)/2,0,1))`, equivalent to `arccos(u @ v)` for unit
vectors but exactly zero for identical stored vectors. Floating cost ties are
resolved only when exactly equal; nearly equal costs are not artificially tied.
Multi-template prototype weights are not fitted or used. Maximum similarity
offers additional opportunities for an impostor match, so a mean-template
threshold must never be reused as if it calibrated the multi-template scorer.

Scores are `clip(cosine,-1,1) - 1`, in `[-2,0]`; larger means more similar.
Accept iff `score >= threshold`, including ties. All directed claims within the
eligible cohort are scored once per probe. Output order is sorted probe
subject, then sorted probe trial, then sorted claim subject. Labels are
`np.int64` (genuine 1, impostor 0), scores are `np.float64`, and metadata contains
only JSON-native values, with no NaN or Infinity.

The stable metadata layout is:

```text
protocol, template_method, probe_method, score, score_order
claim_subject_ids, probe_subject_ids, probe_trial_ids  # score-aligned lists
coverage:
  requested_subject_ids, eligible_subject_ids
  n_enrollment_identities, n_probe_identities
  n_enrollment_trials, n_probe_trials, n_windows
  n_genuine, n_impostor, far_resolution
  per_subject: [{subject, days: {"1"/"2": {
    n_windows, eligible_trials, missing_trials}}}]
  exclusions: [{subject, reason, optional day/trial/windows}]
templates: [{subject, enrollment_trial_ids, prototype_vectors,
  prototype_trial_ids, trial_prototype_indices, n_prototypes}]
calibration: {threshold_calibrated: false, required, candidate: {
  template_method, probe_method}}
fit_population, multi_fusion  # multi_fusion is null outside multi
```

`prototype_trial_ids` is empty for an arithmetic-mean prototype, which is not
an observed trial; otherwise it lists selected exemplars. The indices in
`trial_prototype_indices` align with `enrollment_trial_ids`. These diagnostics
contain biometric vectors and should be handled like other experiment artifacts.
`requested_subject_ids` means the sorted IDs present in the supplied arrays;
an entirely absent requested ID cannot be inferred from this API.

Select the descriptor and template method jointly using the five development
folds only. Obtain a separate calibration-cohort threshold for each candidate
that is reported, freeze all choices, and evaluate the exploratory test cohort
once. Thresholds must also reflect the descriptor and protocol configuration;
metadata's template-method label alone is not complete calibration provenance.
Previously inspected HAR, WISDM and Santos test cohorts remain inspected.
A new output directory or new template scorer does not create a fresh blind
test. This module supplies no test-adaptive threshold or candidate-selection API.

## Paired subject diagnostics

```python
report = paired_subject_diagnostics(
    labels, baseline_scores, new_scores,
    metadata["claim_subject_ids"], metadata["probe_subject_ids"],
    thresholds={"baseline": baseline_calibrated, "new": new_calibrated},
)
```

Inputs must refer to exactly the same ordered claim graph. Compare score-aligned
claim/probe/trial keys and labels from both calls before invoking this helper;
it receives only one set of identity arrays and cannot detect a silently
permuted candidate score array. If candidate eligibility differs, disclose it
and explicitly build a common graph before a paired comparison. Comparing
array offsets from differently covered cohorts is invalid.

The returned `per_subject` list preserves each user's `genuine`,
`impostor_by_probe` (outgoing) and `impostor_by_claim` (incoming) decision counts.
Each block has `n`, baseline/new `accepted`, `rejected`, `acceptance_rate` and
`acceptance_difference` (new minus baseline). A missing impostor block has zero
counts and null rates, rather than a fabricated zero rate. Every graph identity
must have genuine decisions. The report also gives per-user genuine acceptance
differences, their equal-user macro mean and positive/negative/zero counts.
These can distinguish a pooled improvement from a change dominated by one
person or unequal trial counts. Negative differences are retained.

An exact sign-flip **reference** enumerates all `2**S` sign assignments of the
per-user genuine acceptance differences and counts those with absolute macro
mean at least as large as observed. It assumes independent subject summaries
and a null distribution invariant to every independent sign flip (for example,
independent symmetric differences). Zero mean alone is insufficient. Shared
training and calibration induce additional dependencies; this module does not
prove those assumptions. It reports an exploratory sensitivity calculation,
not a post-selection confirmatory p-value. With five nonzero differences the
smallest inclusive two-sided fraction is `2/32 = 0.0625`; zeros and ties can
make it coarser. All-zero differences return one. Default enumeration stops
above 16 subjects; `max_exact_subjects` may be an integer 1..20, and above that
limit the reference is explicitly absent, not randomly approximated.

The returned `exact_sign_flip` includes `status`, `n_assignments`,
`two_sided_tail_fraction`, `assumptions`, `scope` and `interpretation`.
No confidence interval, bootstrap interval or impostor p-value is computed.
Directed impostor edges share both subjects, as well as windows, trials and
templates. Aggregating incoming or outgoing edges by one endpoint does not
remove dependence through the other endpoint. Five people cannot support
deployment FAR assurance from hundreds of repeated comparisons.

## Primary research and adaptations

Primary sources inspected on 2026-10-24:

1. Uludag, Ross and Jain, **Biometric template selection and update: a case study
   in fingerprints** (2004), Section 2 (*Template selection*) and Section 5
   (*Discussion and future work*),
   [author manuscript](https://www.cse.msu.edu/~rossarun/pubs/RossTemplate_PR04.pdf),
   [publisher](https://doi.org/10.1016/j.patcog.2003.11.012). MDIST favors typical
   enrollment impressions; DEND represents within-user variability by selected
   impressions. Our angular one/two-medoid objectives adapt these purposes to
   unit gait embeddings. They do not reproduce DEND, the fingerprint matcher
   or template updating, and the paper supplies no gait robustness guarantee.
2. Roh, Kim and Kim, **Robust periocular authentication in HMD environments
   using multi-template fusion strategies** (2026), Section 3.1.2, Eq. (2),
   [official full text](https://onlinelibrary.wiley.com/doi/full/10.4218/etrij.2025-0459).
   Minimum template distance motivates our equivalent maximum cosine rule.
   Its discussion identifies the extra impostor-match opportunity. Our fixed
   two-medoid enrollment design differs from its image processing and template
   selection; its independent Gaussian score analysis does not apply to our
   correlated template comparisons. Reported accuracy is not an expected gain.
3. Winkler et al., **Permutation inference for the general linear model**
   (2014), *Permutations and exchangeability*, *Restricted exchangeability* and
   *Number of permutations*,
   [primary full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC4010955/).
   Its exchangeability/sign-symmetry requirements motivate the explicit
   assumptions on the genuine-subject reference. This is not a reproduction
   of its GLM method and does not justify sign flipping dependent impostor edges.

## Focused checks

```bash
TMPDIR=/home/sherlock/Code/github/VariableThe/artifacts/tmp \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --no-project --python ../.venv/bin/python python -m unittest discover \
  -s tests -p test_template_scoring.py -v
```

The synthetic tests forbid Torch imports and cover exact legacy arithmetic,
unit/equal-trial weighting, sorted directed coverage, probe isolation,
cancellation reports, medoid/prototype choices, deterministic ties, malformed
inputs, candidate-specific threshold ties and five-user sign enumeration.
They establish implementation invariants, not biometric performance.
