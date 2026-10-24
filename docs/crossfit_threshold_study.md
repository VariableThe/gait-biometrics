# DEV-only pooled calibration comparison

This follow-up tests calibration rather than adding representation capacity.
It never loads the original CALIB or TEST numerical payloads. Previous Santos
TEST inspection and adaptive expansion after development inspection remain
disclosed. No published TEST threshold or result changes in this study.

The fixed representation is the existing 362-dimensional lagged-covariance
descriptor and automatic-shrinkage WCCN. Compare the previously selected equal
mean/two-template score blend and the two-template control. Twenty outer
validation triples follow the existing four balanced five-fold partitions.
For each outer triple, the other four triples each serve once as inner
calibration people, while the remaining nine people fit the metric. The next
triple in the partition reproduces the previous study's calibration role.
Also fit one metric on all twelve nonvalidation people. The budget is exactly
100 fresh CPU covariance fits, sharing metrics across the two scorers. No
historical models are reused and no neural/GPU training is authorized here.

Four policies are fixed before fitting:

- **A:** the prior empirical FAR≤1% rule on the next calibration triple,
  applied to its nine-person fitted metric.
- **B:** empirical FAR≤1% on all four inner held-triple score sets pooled,
  applied to the same nine-person fit as A.
- **C:** the same pooled threshold, applied to the twelve-person refit.
- **D:** an impostor order-statistic threshold with nominal independent-score
  parameters alpha=1%, delta=10%, applied to the twelve-person refit.

Pooling supplies up to 120 genuine and 240 impostor claims, but covers only 24 of the
132 possible directed person pairs among those twelve people. Score sets use
different fitted models. Validation people are excluded from every fit and
every threshold calculation. All policies score the same outer trial claims.
Original day1 enrollment/day2 probes and authentic labels are preserved. Exact
source-derived enrollment/probe trial IDs, directed claim graphs and counts are
registered before features or fitting. Source trials with no valid window are
declared through those graphs; they are never invented. Later scoring exclusions
or additional missing trials fail against that exact registered coverage.
Nominal rank feasibility is checked from the registered counts before fitting.

Promotion is deliberately stricter than a favorable average. A replacement for
the blend's policy A must have mean validation FAR≤1%, no increase in the
number of splits with FAR>2%, strictly lower mean FRR, and at least ten of
fifteen people's mean FRR no worse. Within passing policies, choose minimum FRR,
then FAR, then fixed policy order. Otherwise retain A. The two-template scorer
is a comparison, not an additional selectable representation. All rates used
for gates, strict gains and tie-breaking are exact fractions of raw integer
error counts and denominators; floating means are presentation only. Report complete
per-person rejection and directed-pair false-accept counts. The concentration
of A/blend errors in its two most affected pairs is a diagnostic gate for a
possible separately registered normalization study, not permission to add arms.

## What the mathematical references do and do not establish

Tong, Feng and Li's
[NP umbrella algorithm](https://arxiv.org/abs/1608.03109) chooses an order
statistic to limit a type-I violation probability under independent calibration
observations from a fixed scoring rule. The threshold diagnostic here uses
`binom.cdf(j,n,alpha)` and a strict boundary above the (j+1)-th largest impostor
score. At n=240, alpha=.01 and delta=.1, only j=0 qualifies. This is a nominal
reference calculation: these gait claims are clustered by person, pair and
trial, and pooled score sets use different fitted metrics. It gives no
population FAR guarantee here. The probability is over repeated calibration
samples under the model assumptions, not a posterior probability that a
particular fixed classifier's FAR exceeds alpha. The prior empirical grouped
ROC rule also uses genuine scores; its exact behavior is not identical to a
single fixed impostor order statistic or its expected quantile.

Barber, Candès, Ramdas and Tibshirani's
[jackknife+ and CV+ construction](https://arxiv.org/abs/1905.02928) combines
leave-out predictions and residual quantiles to construct predictive intervals
under exchangeability. Pooling scalar biometric scores and applying their
threshold to a refitted metric is not that construction, and inherits none of
its formal coverage guarantees. B and C are empirical score-transfer
hypotheses. These distinctions correct overly broad interpretations of the
research advisory that motivated this comparison.

## Registration and verification

Run from the model repository with a prepared environment and a fresh root:

```bash
uv run --no-project --python ../.venv/bin/python python -m scripts.crossfit_threshold_study \
  --archive-path ../datasets/santos/santos-v7.zip \
  --out-root ../artifacts/crossfit-thresholds-v2 --stage run
# Repeat the same command using --stage verify.
```

The archive audit reads compressed bytes and member inventory, not held-out
numerical CSV payloads. `run` loads only fifteen DEV people, then exclusively
registers the full plan, source hashes, archive audit and original DEV data/member
hashes and exact source graphs before feature extraction or fits. Dependency
versions and Python are recorded, and this study's tests and method documentation
are included in its source manifest. Every numeric metric state, pooled
score cache, outer score cache, threshold, claim graph and metric is retained.
`verify` checks the registered source/data/model/artifact fingerprints, replays
all inner score pools and all 160 outer comparisons, and recomputes exact cached
metrics and promotion decisions. Thresholds are recomputed exactly from frozen
pool caches, including the first inner segment used by A, after fresh scores
pass replay tolerance; pair-error counts likewise use original outer score
caches. Numeric score replay tolerates roundoff up to
absolute/relative 1e-6; labels, metadata and thresholds agree exactly. The final
validation file is exclusive; rerunning verification under an already completed
root raises instead of overwriting its evidence.

All averages and pair counts are descriptive. Outer splits, claims and person
pairs are dependent. More score rows do not create more independent people.
No confidence interval, posterior probability or significance claim is made,
and neither CALIB/TEST pooling nor twenty-person representation fitting is
performed. Any later promotion to final evaluation needs its own declared
protocol and an independent review of the concrete lock.

The first DEV-only attempt completed one hundred CPU covariance fits but was
excluded after review found that floating means could spuriously promote a tied
rejection count and that nominal trial coverage was not enforced. Its source,
registration and fitted states remain archived separately. Version two uses a
fresh root, exact arithmetic, pre-fit source graphs, cached threshold replay
and stronger regression tests. Historical cost is two hundred CPU covariance
fits; the v2 selectable budget remains one hundred. The original CALIB and
TEST cohorts were not opened in either attempt. No v1 result participates in
v2 selection.
