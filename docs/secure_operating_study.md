# Exploratory selection at calibrated operating points

`scripts.secure_operating_study` compares fifteen fixed candidates: the existing
362-dimensional six-axis fusion plus lagged-covariance descriptor, WCCN
shrinkage `auto`, `0.2`, or `0.5`, and each of the five enrollment scorers in
[secure enrollment](secure_enrollment.md). This is additional search motivated
by earlier development results. All Santos test identities have been inspected.
Neither a fresh root nor the registration makes its test evaluation independent.

The study reuses the preceding four balanced five-fold partitions of fifteen
development people. In every split, nine people fit the metric, the next triple
in the partition calibrates a threshold, and the remaining triple validates it.
Roles never overlap within a split. Each person validates four times,
calibrates four times, and trains twelve times. The fixed budget is sixty
covariance fits and three hundred candidate comparisons; neural training is
unnecessary for this study. Original five-person CALIB and five-person TEST
cohorts play no part in development selection.

For each candidate, the inner calibration threshold is the least restrictive
attainable threshold with empirical FAR at most 1%. A three-person calibration
cohort provides only sixty impostor claims, so this means zero observed false
accepts, and is an imprecise estimate of the tail. The threshold is applied
unchanged to different validation people. Among candidates whose equal-split
mean validation FAR is at most 1%, select lowest mean FRR, breaking ties by FAR,
EER, then registered order. If none qualifies, choose lowest mean FAR, then FRR,
then EER, and report `constraint_met: false`. Selection does not turn an
empirical constraint into a population guarantee.

The original `auto/multi` lagcov method is the matched control. It uses the same
nine training people, inner calibration people and validation claims. Existing
20-split results trained on twelve people; their averages are a separate
protocol and must not be presented as a matched improvement.

Run three separate stages from the model repository, with identical relative
archive spelling and an existing prepared environment:

```bash
uv run --no-project --python ../.venv/bin/python python -m scripts.secure_operating_study \
  --archive-path ../datasets/santos/santos-v7.zip \
  --out-root ../artifacts/secure-operating-v1 --stage develop
# Repeat the same command with --stage lock, then --stage evaluate.
```

`develop` exclusively registers the candidate list, role assignment, archive
audit, DEV member/data hashes, source hashes, and package versions before
features or fits. Every fold metric, calibration/validation score cache,
threshold, full metric and claim graph is retained. `lock` checks all source,
data, model and score fingerprints, replays all folds, and exactly recomputes
cached operating metrics before fitting the selected method and control on all
fifteen development people. Only then does it reserve a lock attempt and open
the original five-person CALIB cohort. Each method receives its own frozen
threshold at empirical CALIB FAR at most 1%.

`evaluate` rechecks development evidence and CALIB replay, exclusively reserves
the TEST attempt, and scores the selected method and control on identical
claims. Failed attempts remain consumed. Score replay tolerates absolute and
relative roundoff of 1e-6; original labels, metadata, thresholds and cached
metrics are checked exactly. Artifacts use numeric NPZ with pickle disabled.
Source hashing covers the explicit imported implementation list, not every
environment binary. Checksums provide integrity evidence, not an external
signed registration service.

All fold averages, worst-person rejection and five-person test comparisons are
descriptive. Overlapping partitions, repeated trials and directed claims are
dependent. Selection from the same development population can overfit its
criterion even with separate roles inside a split. Fresh populations and
recording conditions remain necessary to confirm an apparent gain.
