# Exploratory mentor backend follow-up

This is candidate expansion after earlier DEV inspection. All historical TEST
identities have already been inspected. Neither this registration nor its new
output root restores independent confirmation. No neural training, GPU work, or
new acquisition is part of this runner. The original multisensor study remains
frozen. Backend definitions and their research provenance belong to the separate
session-robust and cohort-normalization documentation; this document specifies
selection, replay, and limits of interpretation.

The public interface is `python -m scripts.mentor_backend_study --archive-path
PATH --out-root PATH --stage develop|lock|evaluate`. Execute each stage separately
in that order. The output root must initially be empty. Failed development and
lock runs require a new declared exploratory root. TEST attempts are consumed
even if loading, scoring, or saving subsequently fails. Relative archive paths
are preserved in loader metadata; use exactly the same path spelling at every
stage because whole-data fingerprints include that metadata. Artifact paths are
relative to the study root. No existing v2 registration or lock is required.

## Fixed candidates and comparisons

The fixed candidate order is `control`, `snorm`, `nuisance`, `lagcov`,
`snorm_nuisance`, `snorm_lagcov`. Selection considers exactly these six.
`control` uses the existing 336-dimensional six-channel axes-plus-invariant
fusion descriptor, WCCN with automatic shrinkage, and `multi` enrollment.
`lagcov` appends the registered 26 lagged covariance features, using lags
0, 5, 10, 20, for 362 dimensions. `nuisance` applies two train-derived nuisance
directions to the original 336 features before WCCN. The SNorm variants retain
their corresponding descriptor and metric, using standard-deviation floor
0.001. All six candidates use the same `multi` enrollment protocol and must
retain identical directed claim graphs within each validation population.

`oldbaseline` is a comparison only: the original 104 acceleration features,
WCCN automatic shrinkage, and `mean` enrollment. It is evaluated in every DEV
split, at CALIB, and at TEST for continuity with the published comparison. It
cannot win selection. Its claims must match the six candidates as well.

The 20 splits comprise four balanced five-fold partitions. The original first
partition reproduces `previous.development_folds()` exactly at folds 0–4. Each
additional partition uses the same `np.random.default_rng(2026)` generator to
permute all 15 `DEV_IDS`, chunk the permutation into five triples, and sort each
triple. If any triple duplicates an earlier accepted held-out subset, reject
the entire permutation and draw another. Accept three additional partitions,
requiring all 20 held-out subsets to be unique. Every DEV identity validates
exactly four times, once per partition. Each split trains on the other 12 DEV
identities, using both days, and validates on the three held-out DEV identities.
No held-out identity fits centering, scales, nuisance directions, covariance,
or a cohort. The registration records this resampling rule explicitly.

An initial DEV-only pilot used 20 unbalanced random held-out triples. Its
validation frequencies differed across people. That pilot and its source are
archived separately and discarded from selection to correct representation,
not to obtain a preferred outcome. The balanced schedule is fixed before its
fresh development run; no CALIB or TEST values were read by this follow-up
before this correction. All six methods, their numerical definitions, and the
equal-split mean-EER selection rule remain unchanged. Use a fresh output root;
the archived pilot cannot be reused under the revised registration and hashes.

SNorm uses only the fitted fold's TRAIN window embeddings, with their original
subject/day/trial arrays. It never invents days or trials or constructs a cohort
from evaluation people. Final cohorts contain all 15 DEV identities only. The
cohort module performs enrollment and probe normalization against those frozen
vectors; no CALIB or TEST observations update the cohort or metric.

Selection requires all 20 results for all six candidates and the baseline.
The winner minimizes mean split EER, weighting splits equally; exact ties use
the fixed candidate order. Every split retains EER, AUC, per-probe-person EER,
coverage, claim graph SHA-256, complete scoring metadata, numeric model state,
training cohort arrays, and cached labels/scores. Original-five results remain
identifiable by fold indices 0–4. No one-standard-error rule, significance test,
or independent-fold confidence interval is computed.

## Registration, replay, and one-shot evaluation

`develop` verifies official archive compressed bytes and its member inventory,
then loads DEV only. Before feature extraction or fitting it exclusively writes
`registration.json`: the full fixed plan and splits, implementation source
hashes, original DEV whole-data fingerprint, selected original member hashes,
and their aggregate. Compressed-byte archive hashing is not CSV value loading.
Feature caches are lossless numeric arrays; descriptors are computed once per
stage and reused. Fitted states and cohorts use NPZ with `allow_pickle=False`.

`lock` verifies plan, code, archive, DEV arrays/metadata/member hashes, and the
feature cache against freshly extracted descriptors. It verifies all 140 saved
fold models (20 × six candidates plus the baseline), checks their training IDs,
reconstructs original training cohort vectors from those states, replays the
validation scores, and exactly recomputes every saved metric from frozen score
caches. Replayed score/cohort values must agree within absolute and relative
tolerances of 1e-6; label order and metadata agree exactly. Cached metrics,
selection, and artifact digests agree exactly. This tolerates numeric replay
roundoff without changing the scores that determined selection.

Only after that complete verification does the runner fit the selected method,
control, and old baseline on all DEV, serialize their final states/cohorts, and
reserve a lock attempt. It then opens the five CALIB identities, once for this
lock attempt. Each distinct method gets its own least restrictive attainable
threshold with empirical CALIB FAR ≤ 1%. Acceptance uses `score >= threshold`,
and tied scores move together. CALIB results never choose a method or change
its parameters. If the winner is control, its identical role is stored once.
Calibration score artifacts, full metrics, data/member hashes, selected methods,
and final model/cohort artifact hashes are included in the self-hashed lock.

`evaluate` verifies the lock and all development evidence again without fitting,
then replays CALIB using final states, verifying its cached scores, metrics, and
thresholds. It exclusively creates `test_attempt.json` before the only TEST
payload loader. A preexisting attempt or final result prevents any retry under
that root. Winner, control, and baseline use their respective frozen thresholds
on the same claim graph. The exclusively created final result includes a content
hash, TEST data/member hashes, coverage, EER/AUC, FAR/FRR, per-person errors,
score artifact hashes, and complete scoring metadata. EER is descriptive and
does not choose the operating threshold. Final results are append-only under
normal runner execution; hashes are integrity evidence, not a signed external
registry or protection against deliberate filesystem rewriting.

## Interpretation and primary sources

Twenty overlapping balanced splits of 15 people do not create 20 independent
studies. Equal validation frequency improves representation; it does not supply
independent samples or justify a population confidence interval.
The additional candidates were motivated by earlier inspected development
results. Repeated selection can overfit a noisy model-selection criterion;
lower mean DEV EER is selection evidence and is not an unbiased performance
estimate. Cawley and Talbot demonstrate this distinction in *On Over-fitting in
Model Selection and Subsequent Selection Bias in Performance Evaluation*,
JMLR **11**(70), 2079–2107 (2010), especially Sections 2 and 5–6
([primary paper](https://www.jmlr.org/papers/v11/cawley10a.html)).

Varma and Simon show that tuning and evaluating with the same CV evidence can
produce optimistic estimates; their proposed nested procedure repeats model
selection inside each outer split. This runner does not claim that its repeated
DEV selection is nested validation: *Bias in error estimation when using
cross-validation for model selection*, BMC Bioinformatics **7**, 91 (2006),
doi:[10.1186/1471-2105-7-91](https://doi.org/10.1186/1471-2105-7-91).

Five CALIB people provide coarse empirical tail calibration, and five already
inspected TEST people provide limited descriptive comparisons. Repeated trials,
templates, and directed claims are dependent. A large claim count does not
increase the number of independent people. No apparent improvement here proves
deployment FAR or generalization to new populations, sessions, placements, or
acquisition conditions. The useful outcome is a reproducible backend comparison
and evidence to justify a future independently acquired evaluation.
