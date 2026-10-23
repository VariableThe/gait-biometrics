# Training and validation protocol

`engine.train` loads only the original training split. By default it assigns
complete, real subjects to validation (`val_subjects`, otherwise 20% rounded
up), using sorted source IDs and `random.Random(seed).sample`. It chooses
`best.pt` by minimum unweighted validation activity cross entropy. Ties retain
the earlier checkpoint. Any optional training subset is taken **after** the
subject split, so changing the training budget cannot move validation subjects.
Balanced activity weights, when requested, use training labels only.

For multitask activity/identity models, identity targets are contiguous ranks
of the retained training subjects. Checkpoints record the original-ID mapping.
Validation subjects are unseen identities; validation scores only activity,
without treating their raw IDs as identity-head targets. The identity head is
an auxiliary training objective, not a measured unseen-identity classifier.

Legacy WISDM CSVs lack verified subject identities. Default subject-disjoint
training rejects these usefully. An explicit `validation_strategy: chronological`
permits activity-only training: retain ordered windows, hold out the last
`val_fraction` (default 0.2), and discard `ceil(window_size / stride)` windows
at the training boundary. This purges adjacent-window overlap but cannot prove
subject independence or recording independence when the source omitted IDs.
Multitask models require real identities and cannot use this fallback.

`engine.train_siamese` preserves the existing optional subject-disjoint
`val_subjects` carve. A usable validation split requires at least two subjects
and a genuine-pair-capable identity. Training requires the same minimum coverage.
Identity-preserving subsets reserve up to two windows for every identity,
failing explicitly if the requested budget cannot retain them. Triplet counts
are based on the actual retained training windows, multiplied by
`triplets_per_epoch`; validation windows never contribute to that budget.

New full-run configs are `validation_har.yaml`, `verification_har.yaml`, and
`wisdm_verification.yaml`. Both verification configs use only zero-indexed
walking activity `0`, four held-out training subjects, 20 epochs, and
`verification_pairing: same_activity`. Validation selects minimum EER using
fixed pair seed and count, with activity-stratified pairs when the evaluator
supports them. Unspecified legacy Siamese configs preserve unrestricted pairing
and `val_subjects: 0` training-loss selection for compatibility; these are not
the recommended experiment configs. The original test split is reserved for
final evaluation after selection.

`device: auto` selects CUDA when available and CPU otherwise; `--device` can
override this. Datasets stay on CPU. Pair indices are sampled from cached
identity groups and gathered in bounded batches; choosing an alternative
negative identity takes constant time. The embedding architecture, loss and
separate anchor/positive/negative forward passes are unchanged. No mining,
normalization, mixed precision, compilation, or accuracy gain is claimed by
this change. The training inputs and inference inputs keep their source units.

Checkpoints retain the existing tensor/primitive-only format and are readable
with `torch.load(..., weights_only=True)`. New protocol-aware checkpoints store
`metrics.protocol` including source, data root, train/validation subject IDs,
selected activities, channel count, window length, stride, sampling rate, and
normalization (`none`). Run JSON records the selected epoch, all epoch metrics,
device, total wall time, and git commit. New protocol runs also record per-epoch
wall time. These times include validation; compare equivalent devices, sources,
budgets and validation settings before claiming a speed improvement.

The official [UCI HAR dataset description](https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones)
describes 30 participants, a participant-based train/test partition, and
128-reading windows at 50 Hz with 50% overlap. This supports splitting subjects
before window subsampling. Walking is distinct from sitting and standing, so
mixing every activity is not a walking-only gait experiment.

The official [WISDM Activity Prediction dataset page](https://www.cis.fordham.edu/wisdm/dataset.php)
distinguishes walking from other activities and lists the v1.1 raw files. Use
the identity-preserving `wisdm_v1` source for verification, rather than inventing
identities for legacy CSV rows. The source adapter owns its native 20 Hz windowing
and authentic subject split.

Within-recording windows and pairs can overlap or recur. Validation EER and
final within-recording discrimination do not establish cross-session continuous
authentication, deployment FAR, or independent authentication trials. Calibrate
operating thresholds on validation only with the evaluator; report its final
held-out test metrics and protocol limitations separately.

Validation tests cover deterministic subject partitions, absence of test-split
loads, checkpoint selection when validation worsens despite improving train
loss, unseen multitask identities, chronological purging, identity coverage,
valid deterministic pairs/triplets, small-subject failures, CPU runtime, actual
tiny triplet/contrastive optimization, and safe checkpoint reloads. These are
correctness checks, not measured dataset accuracy results.
