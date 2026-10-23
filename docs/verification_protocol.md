# Verification protocol and evidence limits

The evaluator measures **held-out-subject, within-recording discrimination** when a checkpoint records its training and validation subjects. It does not establish cross-session continuous authentication. Legacy checkpoints without this metadata are explicitly marked as having unverified training-subject separation.

## Running a fixed-threshold evaluation

From either model repository, using the shared environment (or replace its path with your isolated environment):

```bash
uv run --no-project --python ../.venv/bin/python python -m engine.evaluate \
  --config configs/verification_har.yaml \
  --checkpoint runs/verification_har/best.pt \
  --task verification --calibrate --calibration-target-far 0.01 \
  --out-dir runs/verification_har/evaluation
```

Use `configs/wisdm_verification.yaml` and the corresponding checkpoint/output paths for `source: wisdm_v1`. Set `data_root` to the extracted dataset directory in the run configuration. The example checkpoint path is a placeholder for the actual run, not an assertion that a model exists there. Both rich trainer checkpoints and bare embedding state dictionaries can be scored; calibration requires the rich checkpoint's `metrics.protocol`.

The evaluator loads only the checkpoint's recorded `val_subject_ids` from the training source, verifies they are disjoint from optimizer-training and test subjects, and embeds them with the frozen checkpoint. It chooses the least restrictive **attainable** validation threshold with empirical FAR at most the requested target, then applies that exact threshold to test scores. It never tunes it against test labels. Checkpoint selection and threshold calibration currently reuse validation subjects; a separate calibration cohort is preferable for future larger studies. If training used a window subset, calibration includes all available windows of those held-out validation subjects, and reports the actual count.

Source, activity filter, pairing mode, channel count, window size, stride and sampling rate must agree with recorded training metadata. Unsupported normalization metadata fails rather than silently applying a different transform. `data_root` may change when moving the same dataset to another machine. File identity and historical preprocessing cannot be proved from a legacy checkpoint.

## Scores, ties and operating points

A genuine label is 1; an impostor label is 0. Scores are `cosine_similarity - 1`, in [-2, 0], with higher scores more genuine. Acceptance uses **score >= threshold**, including ties. Nonfinite embeddings, zero-norm rows, malformed identity labels, invalid pair budgets and single-class score arrays fail explicitly. A finite float immediately above the largest score represents reject-all; JSON contains no NaN or Infinity.

The old merged evaluator swept tied observations individually. A tied group must instead enter the ROC together: otherwise its label order can manufacture discrimination. Student C's independent operating-point names are now available in both repositories. Thresholds are not linearly interpolated between ROC points because that does not reproduce interpolated acceptance rates, especially at ties. The official [scikit-learn ROC contract](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) specifies acceptance at score >= threshold and the initial infinite reject-all sentinel; its output arrays have matching lengths.

- `eer`: descriptive linear interpolation of the FAR/FRR crossing, potentially unattainable by any deterministic threshold.
- `eer_threshold`: closest **empirical** crossing, strictly diagnostic and test-derived. `far_at_eer_threshold` and `frr_at_eer_threshold` report its actual rates. It is not a deployable threshold and need not attain the interpolated EER.
- `frr_at_far_0.01`: minimum empirical FRR among thresholds with FAR **<=** 1%; `far_at_frr_0.01` is the converse constraint. These are descriptive test operating points, not calibrated decisions. Their meanings differ from PR3's interpolated target rates; `pairing_version: 2` and explicit semantics identify the new reports.
- `fixed_threshold`: actual test decision counts/rates at the frozen validation threshold; `calibration` separately records its validation provenance, target and observed rates.
- Legacy `far` and `frr` keys remain aliases of interpolated EER for old callers. Their adjacent metadata warns that they are not independent measurements. Prefer `fixed_threshold`.

## Pair sampling and activity controls

The positional API `verification_metrics(embeddings, subject_ids, n_pairs=10000, seed=42)` remains usable by the trainer. New keyword arguments are `activity_ids` (one real activity label per window) and `threshold` (an externally calibrated fixed value).

Sampling is deterministic for the same seed, row order, labels and pairing version. It draws `floor(n_pairs/2)` genuine and the remainder impostor pairs **with replacement**, uniformly over eligible subjects within each activity stratum. It never pairs a window with itself and records unordered pair reuse. `n_pairs` is the actual number scored; `n_unique_pairs`, separate genuine/impostor unique counts, window coverage, per-subject pair participation and per-activity budgets expose what was measured. Unique pairs are **not** statistically independent observations: they share windows and recordings. Do not treat 10,000 sampled comparisons as 10,000 independent users or use that number to claim a deployment FAR confidence bound.

The CLI uses `verification_pairing` from configuration, then checkpoint metadata, then `same_activity`. The low-level API without activity labels remains unrestricted. New walking configurations use zero-indexed `activity_ids: [0]`. With multiple activities, both members of every comparison share an activity, and genuine/impostor strata have matching budgets except the possible extra impostor pair for an odd request. Activities without both genuine and impostor coverage are reported as excluded; no viable stratum raises an actionable error. Activity labels are ground-truth evaluation controls, not an implemented online walking detector. All-activity results are broader inertial identity discrimination and must not be presented as walking-only gait performance.

Changed sampling, activities or score arithmetic can change measured EER. Compare models using this same evaluator, same windows, seed and pair budget; do not attribute an old-versus-new protocol difference to model learning. Test scores must not select models, hyperparameters, seeds or thresholds.

## Primary research and dataset checks

Sources below were inspected on 2026-10-23. These are focused protocol observations, not claims to reproduce the papers' systems or reported accuracy.

1. Dehghani, Glatard and Shihab, **Subject Cross Validation in Human Activity Recognition**, [arXiv:1904.02666v2](https://arxiv.org/abs/1904.02666v2), [full text](https://arxiv.org/pdf/1904.02666), Section 2.2, PDF pages 2–3: the authors explain within-subject temporal dependence and split training/testing by subject to remove that dependency across partitions. Their overlap findings concern HAR classifiers. Our inference is to preserve subject-disjoint representation learning and calibration; this does not remove overlap between two genuine windows inside the held-out test recording.
2. Cheung and Vhaduri, **Continuous Authentication of Wearable Device Users from Heart Rate, Gait, and Breathing Data**, [arXiv:2008.10779v1](https://arxiv.org/abs/2008.10779v1), [full text](https://arxiv.org/pdf/2008.10779), Sections I-C, II-E, III-A and IV: gait availability is conditional on walking; their system combines modalities by context. Their leave-one-out procedure holds out **instances**, not whole unseen subjects, and their limitations identify a three-subject feasibility study and a need for extended-period evaluation. Consequently its authentication figures cannot be compared directly to this repository's unseen-subject pair ROC. Our pragmatic fix is explicit walking filtering and activity-matched comparisons; an online activity gate and fallback authentication remain future work.
3. Lee et al., **Gait-Based Continuous Authentication Using a Novel Sensor Compensation Algorithm and Geometric Features Extracted From Wearable Sensors**, IEEE Access (2022), [DOI](https://doi.org/10.1109/ACCESS.2022.3221813), [author-institution full text](https://scholarworks.bwise.kr/hanyang/bitstream/2021.sw.hanyang/182135/1/Gait-Based_Continuous_Authentication_Using_a_Novel_Sensor_Compensation_Algorithm_and_Geometric_Features_Extracted_From_Wearable_Sensors.pdf), abstract, PDF page 1: 20 participants recorded across three days produced different same-day and cross-day error rates. This is evidence for measuring session shift, not a promised improvement for our CNN. The needed fix is authentic session/date/recording metadata plus separate enrollment and later-session probes.

The [official UCI HAR description](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) describes 30 participants, a participant-level 70/30 split, 50 Hz signals, 128-reading windows and 50% overlap. A real test-subject split therefore supports unseen-subject evaluation, but a random genuine pair can still share raw samples. `WindowData` carries no session identifiers or recording offsets, so this evaluator cannot exclude overlap or assert distinct sessions.

The [official WISDM v1.1 release page](https://www.cis.fordham.edu/wisdm/dataset.php) documents raw records and masked participant IDs. Identity is unavailable in the project's anonymous derived CSV, not inherently absent from WISDM. `wisdm_v1` must preserve the authentic raw IDs; timestamps, row order or synthetic counters must never be promoted to identity or verified day/session IDs.

The current Siamese code is a convolutional embedding baseline trained with metric loss. Its optional DAE decoder is not on the embedding forward path; these tests do not establish fidelity to the cited denoising-autoencoder paper.

## What remains unmeasured

There is no enrollment/probe separation by day, verified exclusion of overlapping genuine windows, temporal decision policy, takeover detection delay, spoof resistance test or device-energy measurement here. Report fixed-threshold errors on held-out subjects as preliminary discrimination evidence. A low observed validation FAR, even zero, is not a guaranteed test or deployment FAR. Next steps are independent session metadata, separated enrollment/calibration/probes, per-subject uncertainty analysis and an explicitly evaluated continuous decision policy.

`tests/test_verification_protocol.py` uses known score tables, an activity-only embedding control and real `EmbeddingNet` inference over synthetic windows. It verifies numerical and protocol behavior; its synthetic results are not dataset-backed biometric performance claims.
