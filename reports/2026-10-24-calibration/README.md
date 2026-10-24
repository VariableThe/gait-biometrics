# DEV-only pooled calibration comparison

The pooled-threshold plus twelve-person-refit policy passed the registered
development promotion gate. Against the fixed blend's prior policy, mean FAR
fell from **0.917% to 0.836%**, and mean FRR from **19.276% to 15.126%**.
The number of splits exceeding 2% FAR stayed at three of twenty, and twelve
of fifteen people's mean FRR was no worse. These are descriptive results from
overlapping development splits, not new held-out or population estimates.
**No original CALIB or TEST numerical payload was opened in this study.**

The earlier five-person reused TEST result remains unchanged: 3/200 false
accepts and 0/50 genuine rejections at its original frozen threshold. This new
study nominates a calibration/refit policy; it neither changes that threshold
nor establishes a better TEST FAR.

## Fixed comparison and full outcomes

The representation remains the 362-dimensional lagged-covariance descriptor
with automatic-shrinkage WCCN. The prior equal mean/two-template score blend
is fixed; the two-template scorer is a comparison. Twenty outer validation
triples come from four balanced five-fold partitions of fifteen DEV people.
Within each outer split, each of the four other triples supplies held-out
calibration scores from a metric fitted on the remaining nine people. One
additional metric fits all twelve nonvalidation people. Every validation
person is excluded from all fits and threshold pools in that split.

- **A:** prior empirical FAR≤1% threshold on the next held triple, applied to
  its nine-person fitted metric.
- **B:** empirical FAR≤1% threshold pooled over four held triples, applied to
  the same nine-person fit as A.
- **C:** B's identical threshold, applied to the twelve-person refit.
- **D:** nominal independent-score order-statistic cutoff with alpha=1% and
  delta=10%, applied to the twelve-person refit. Its assumptions do not hold
  for the dependent pooled gait scores, so it is a reference policy only.

| Scorer / policy | Mean FAR | Mean FRR | Splits FAR>2% | Eligible replacement |
| --- | ---: | ---: | ---: | --- |
| Blend A | 0.917% | 19.276% | 3/20 | Reference |
| Blend B | 1.753% | 13.787% | 4/20 | No: FAR and bad-split gates |
| Blend C | 0.836% | 15.126% | 3/20 | **Yes: sole eligible policy** |
| Blend D | 0.667% | 21.506% | 3/20 | No: higher FRR |
| Two-template A | 3.178% | 23.684% | 5/20 | Comparison only |
| Two-template B | 1.670% | 17.161% | 3/20 | Comparison only |
| Two-template C | 1.256% | 18.678% | 4/20 | Comparison only |
| Two-template D | 0.250% | 31.736% | 1/20 | Comparison only |

Pooling alone did not satisfy the false-accept constraint. C changes both the
threshold source and the number of metric-fitting people relative to A; the
gain must not be attributed solely to a better tail estimate. No additional
representation, enrollment scorer or neural fit was selected here.

## Per-person limits

Five people improve, seven tie, and three worsen under C relative to A. The
three worsened mean rejection rates are identity13: 25%→27.5%, identity19:
0%→2.778%, and identity25: 0%→2.5%. Identity9 remains completely rejected
(100% mean FRR), and identity5 remains at 67.5% (from 70%). Passing the
aggregate gate therefore does not establish acceptable usability for everyone.
All exact per-person fractions and denominators are retained in the evidence.

The prior blend's eleven DEV false accepts have eight concentrated in two
directed person pairs. This triggers the separately declared diagnostic for
studying claimant normalization later. No normalization arm was added to this
registration, and those repeated decisions are not eleven independent people.

## Exact gate, coverage and reproducibility

The promotion rule was fixed before the v2 fits: blend FAR≤1%, no increase in
splits with FAR>2%, strictly lower mean FRR, and at least ten people's mean FRR
no worse than A. All comparisons use exact rational arithmetic from raw integer
error counts and denominators; floats are presentation only. A's exact mean
FAR/FRR are `11/1200` and `559/2900`; C's are `97/11600` and `329/2175`.
These are equal-split averages of rates, not pooled independent-trial ratios.

Exact source-derived enrollment and probe trial graphs are registered before
features or fitting. Some source trials have no eligible window. Inner score
pools therefore contain 119/238 or 120/240 genuine/impostor claims, with 24
directed person pairs. A later dropped trial, identity or template fails against
the registered graph. No missing trial is reconstructed. Original source/member
and array fingerprints, full graphs and numerical package versions accompany
the registration. The source manifest includes all 21 declared implementation,
method-documentation and test files.

Every saved metric state, score pool, outer score cache, threshold and promotion
decision passed the runner's verification. Fresh score replay allows
absolute/relative 1e-6 rounding differences. Exact thresholds are then recomputed
from the original frozen caches, including A's first inner segment; original
outer caches determine exact counts. This prevents numerical replay from
silently changing selection or operating decisions.

The first DEV-only attempt completed one hundred CPU covariance fits but was
discarded after review found floating-mean tie promotion and unenforced
nominal trial counts. Its source, registration, results and state inventory are
retained under `v1/` with an exclusion disposition. The fresh v2 uses one
hundred fits, for **two hundred historical CPU covariance fits** total. Only
v2 participates in selection; neither attempt opened original CALIB or TEST.
No new GPU training was performed.

`v2/` publishes original JSON bytes compressed losslessly with gzip and all
two hundred calibration-pool/outer score caches. `artifact_inventory.json`
records every original artifact's size/SHA256, publication status, and published
digest. The one hundred dense fitted states and feature cache remain in the
local experiment archive and are inventoried, rather than all committed.
The compact report alone is not a complete resumable replay root; use the
local states or run the frozen protocol anew. No raw sensor signals are
published. Follow [the full protocol](../../docs/crossfit_threshold_study.md)
and [tail diagnostics](../../docs/tail_calibration.md) for commands and formulas.

Both full suites pass: **656 tests +878 subtests** main and **652 +878** Student
B, with eight existing ONNX/quantization warnings. The 197 focused checks cover
exact ties and gate boundaries, count consistency, deterministic selection,
source-graph exclusions, real fit-call counts, no original held-out reads,
one-ULP replay, and tampered thresholds. Mirroring describes one experiment.

## Research assumptions and next scope

The order-statistic reference follows Tong, Feng and Li,
[*Neyman-Pearson classification algorithms and NP-ROC*](https://arxiv.org/abs/1608.03109).
Its repeated-sampling bound assumes independent calibration observations from
a fixed scoring rule. Here scores are dependent and come from different fitted
metrics. Pooling them and applying a cutoff to a refit is also not the
[jackknife+/CV+ predictive-interval construction](https://arxiv.org/abs/1905.02928)
and inherits none of its formal coverage guarantees. No posterior probability,
significance test, population confidence interval or certified FAR is claimed.

The next bounded research question is whether claimant normalization can reduce
the concentrated false accepts without worsening the difficult genuine users.
It needs a separate registered comparison. A final evaluation protocol would
also need to specify how out-of-fold scores transfer to a final fitted metric;
this DEV-only study does not silently pool original CALIB into training or
recalibrate previously inspected TEST. Fresh repeated-session phone data remain
necessary for stronger confirmation.

Dataset attribution: Santos, Wanderley, Tavares and Rocha,
[Santos gait dataset v7](https://doi.org/10.6084/m9.figshare.14727231.v7),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Published derived
artifacts add windowing, descriptors, fitted metrics and scoring/calibration;
the study is not a reproduction of the dataset authors' original analysis.
