# Claimant normalization: fixed exploratory follow-up

`engine.claimant_normalization.normalize_claimant_scores` scores the existing
equal mean/two-medoid blend on exactly the same directed claims. Raw scores
remain bit-identical to `score_secure_trials(..., method="mean_multi")`.

Z normalization uses every frozen TRAIN day2 trial probe against each claimant's
day1 enrollment templates, then subtracts the population mean and divides by
the population standard deviation, floored at 0.001. S normalization averages
that Z score with a T score: each target probe against frozen TRAIN day1 blend
templates. Cohort IDs must be disjoint from every target ID, dimensions must
match, and every cohort person must retain both days and every source trial.
Cancellation of even one TRAIN trial fails instead of silently shrinking the
normalization cohort. Cohort coverage and exclusions are retained in metadata.
Target probes never fit
enrollment templates or another target's normalization statistics. Standardized
scores are unbounded and require their own threshold.

The fixed DEV-only study uses the earlier 20 outer splits, four inner
nine-person fits and one twelve-person refit per split. The earlier policy C
is held fixed. Each arm pools four held-triple calibrations for an empirical
FAR<=1% threshold; its outer score uses the twelve-person refit and its cohort.
Exactly 100 covariance fits are shared by raw/Z/S arms. Actual source trial
coverage, including each inner/refit TRAIN trial inventory, is registered before
features or fits. Empirical policy C does not require nominal policy D's sample
size feasibility. No original CALIB/TEST payloads
are loaded. All arms, including failures to improve, are retained.

Promotion requires exact equal-fold mean FAR<=1%, no increase in folds above
2% FAR, strictly smaller mean FRR and at least 10/15 people no worse than raw.
Ties use exact FRR, then FAR, then registered arm order. Otherwise retain raw.
Registration pins the complete plan, source/tests/documentation hashes, runtime
versions, archive digest, source members, and claim graphs before any fit.
Verification replays all scores and metadata, checks every artifact digest and
metric/cohort training assignment, and derives thresholds and metrics from
the frozen numeric caches. Current Python/package versions must match the
registration. Score replay permits atol=rtol=1e-6, but also requires identical
fresh/cache accept decisions at every frozen threshold; tolerance cannot hide a
changed FAR/FRR. Metadata remains exact. It performs no fit. A fresh empty output root is
required. Failed attempts must be retained rather than overwritten.

```bash
uv run --extra cpu python -m scripts.claimant_normalization_study --stage run \
  --archive-path datasets/santos/santos-v7.zip --out-root artifacts/claimant-v1
uv run --extra cpu python -m scripts.claimant_normalization_study --stage verify \
  --archive-path datasets/santos/santos-v7.zip --out-root artifacts/claimant-v1
```

The cohort people also fitted the metric. Claims repeat people, folds overlap,
the candidate set followed earlier DEV inspection and original TEST was already
inspected. This supplies descriptive DEV evidence, no independent score,
population FAR, NP, CV+ or significance guarantee. No deployment threshold is
changed by a DEV nomination. Normalization definitions follow
[Mariéthoz and Bengio, Idiap RR 04-62](https://publications.idiap.ch/publications/show/1049).
