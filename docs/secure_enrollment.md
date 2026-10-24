# Exploratory enrollment score fusion

`engine.secure_enrollment` supplies five NumPy-only, trial-level candidates for
examining sensitivity to a single extreme template match. These are exploratory
adaptations of enrollment averaging, the existing two-medoid matcher, convex
score fusion, top-k aggregation and temperature logmeanexp. They supply no
accuracy, security or robustness guarantee. The module loads no dataset or
saved scores, imports no Torch and fits no representation or decision threshold.

```python
from engine.secure_enrollment import score_secure_trials

labels, scores, metadata = score_secure_trials(
    embeddings, subjects, days, trials, method="mean_multi",
)
```

`method` is a required keyword. Input validation and trial aggregation reuse
`engine.template_scoring`: `embeddings` is a finite, nonempty real numeric
`(N, D)` matrix, and the other arrays are one-dimensional integer arrays of
length `N`. Subject IDs are authentic nonnegative int64-compatible labels,
days are 1/2 and trials are 1..10. Boolean, fractional, complex and object
arrays are rejected where inappropriate. Every input window is validated,
including rows belonging to identities subsequently excluded from coverage.
Window norms below `1e-12` fail explicitly; finite large vectors use the shared
overflow-safe normalization. Inputs are never modified.

Each window is unit-normalized. Windows matching the **complete** authentic
`(subject, day, trial)` key are averaged, then that mean is unit-normalized.
Every eligible day1 trial receives one enrollment vector regardless of its
window count. Each eligible day2 trial is a separate probe. Day2 vectors never
select, weight or fit enrollment templates. Subject IDs are preserved, including
zero and large integers; trial numbers reused by other identities or days do
not merge groups. Callers remain responsible for authentic metadata and
independent data provenance.

For a probe `p`, let `e_i` be the original day1 unit trial vectors and let
`r_i = clip(p @ e_i, -1, 1) - 1`. All returned scores are on the original
cosine-minus-one scale `[-2, 0]`; larger means more similar.

- `mean`: the exact existing `score_trials(..., template_method="mean")`
  score, using the normalized equal-trial enrollment mean.
- `multi`: the exact existing `score_trials(..., template_method="multi")`
  score on the common graph described below. Its deterministic exhaustive
  two-angular-medoid selection, tie rules, one-prototype fallback and maximum
  cosine fusion are reused directly; none are reimplemented.
- `mean_multi`: `0.5 * mean_score + 0.5 * multi_score`. Both component scores
  already include clipping and the minus-one transform. Weights are fixed and
  are never learned from probes. At the scalar score level, an extreme change
  in only the multi component has half its effect on this fusion.
- `top2`: average the largest `min(2, n)` values among **all original** `r_i`.
  With one enrollment trial, use its score. Distinct trial IDs with identical
  directions remain separate trials; windows within a trial do not become
  extra templates. The rule reduces the contribution of one high match when
  at least two trials are present, but two high matches can still dominate.
- `lse_0.1`: with fixed temperature `t = 0.1` and `m = max(r_i)`, return
  `m + t * log(mean(exp((r_i - m) / t)))`. This is mathematically
  `t * log(mean(exp(clip(p @ e_i, -1, 1) / t))) - 1`. Subtracting the maximum
  keeps exponential arguments nonpositive. Averaging inside the logarithm
  avoids a template-count bonus for identical similarities. All original
  enrollment trials contribute; a high score can still dominate at this
  temperature. This is neither a maximum nor an unnormalized logsumexp.

The method names describe aggregation rules, not established biometric
improvements. No scores or empirical outcomes are used to choose these rules.
Each candidate needs its own independently calibrated threshold; shared score
units do not make a threshold transferable. Acceptance uses `score >= threshold`
when the caller supplies a frozen threshold. This API makes no decisions.

## Common claims and explicit coverage

Every method first obtains the **mean baseline's** eligibility, labels and
coverage from `score_trials`. A cancelled trial mean is excluded and reported;
an identity missing eligible enrollment or probes, or with a cancelled mean
enrollment template, is excluded and reported for **every** method. Partial
trial availability is allowed, and missing trial IDs remain in the report.
This common baseline rule is deliberately retained even where `multi`, `top2`
or `lse_0.1` could mathematically score a cancelled-mean identity. Exclusions
are visible in `coverage.exclusions`; all supplied subject IDs remain in
`coverage.requested_subject_ids` and `coverage.per_subject`. An entirely absent
identity cannot be inferred from this API.

At least two identities must survive the common rule. Otherwise every method
raises the existing `engine.template_scoring.CoverageError`, whose `.coverage`
contains the same full diagnostic report. There is no method-specific rescue,
silent identity relabeling or fallback to a different graph.

Every surviving identity claims against **every** surviving day2 probe,
including its own. Output order is sorted probe subject, probe trial, then
claim subject. `multi` scores are joined onto this graph by the full
`(claim subject, probe subject, probe trial)` key, so any identities retained
only by the legacy medoid scorer cannot shift comparison rows. Mean scores
match the existing mean scorer exactly; multi scores match its exact subset
on the common graph. If mean/multi eligibility already agrees, both arrays
match their respective original scorer outputs exactly.

Labels have dtype `np.int64`, scores have dtype `np.float64`, and metadata is
JSON-safe. These original claim metadata fields are identical across methods:
`claim_subject_ids`, `probe_subject_ids`, `probe_trial_ids`, `score_order` and
the entire `coverage` object, including exclusions and FAR resolution. Compare
these fields and labels before any paired analysis. Reordering windows within
a trial can change floating-point reduction rounding; numerical permutation
checks allow that precision, while claim keys and coverage remain exact.

Metadata retains the original layout and adds `method`, `base_protocol`,
`score_range` and `coverage_policy`, with protocol
`cross_day_secure_enrollment_v1`. `template_method`, `score` and
`calibration.candidate` describe the chosen rule. `templates` describes the
mean prototype for `mean`, medoids for `multi`/`mean_multi`, and every original
eligible day1 trial for `top2`/`lse_0.1`. Fusion additionally records
`mean_templates` and its fixed component weights. LSE records temperature 0.1.
Template vectors are experiment artifacts; day2 probes are never included in
them. No calibration is performed or claimed by the metadata.

## Focused synthetic checks

From `gait-biometrics`, using the existing shared environment without syncing
or installing dependencies:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
uv run --offline --no-project --python ../.venv/bin/python \
  python -B -m unittest discover -s tests -p test_secure_enrollment.py -v
```

Checks cover handcomputed scalar scores, exact existing mean/multi agreement,
shift-consistent fusion, exhaustive common graphs and exclusions, authentic
identity/trial joins, day2 isolation, permutation invariance, equal trial
weighting, one-trial fallback, duplicate enrollment trials, stable logmeanexp,
strict validation and finite extreme embeddings. Tests forbid Torch imports
and use only generated arrays; they establish implementation behavior without
making performance claims or reading real datasets or saved test results.
