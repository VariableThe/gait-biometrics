# Registered exploratory Santos multisensor study

`scripts/multisensor_study.py` registers `santos_multisensor_exploratory_v1`.
**Santos TEST has already been inspected: every new TEST result is exploratory.**
The explicit disclosure flag is `prior_test_inspection: true` in the registered
plan, lock, attempt and result. This frozen runner has no
`--prior-test-inspection` CLI argument; supplying that option fails argument
parsing. The similarly named option in `scripts/cross_day_verification.py`
belongs to that historical runner. A new output directory does not restore
independent confirmation, and the boolean does not record a detailed inspection
history.

The registered comparator is the historical **acceleration-only, axes plus
magnitudes, WCCN with automatic shrinkage, mean-template** method:
`descriptor_accel_axes_auto/mean`. The study refits and rescores this baseline
under the same protocol as the winner; old published numbers are not a matched
comparison.

The fixed identity roles are:

- DEV: `7,16,11,9,29,27,5,13,3,1,25,24,14,19,17`. Five folds hold out successive
  triples in this order; each fit uses the other twelve users.
- Independent CALIB users: `21,18,23,12,30`. They select operating thresholds,
  never the winning candidate or fitted representation.
- Previously inspected TEST users: `26,28,22,15,6`.

Input is authentic aligned acceleration xyz and gyroscope xyz, 100 Hz,
256-sample windows with stride 256. Acceleration candidates slice the first
three channels from the same six-axis data. Day 1 enrolls and day 2 supplies
trial probes. Window embeddings are normalized and averaged into normalized
trial vectors. Enrollment uses `mean` (normalized mean), `robust` (one angular
medoid), or `multi` (up to two angular medoids, maximum cosine score). Day 2
does not fit enrollment templates. Scores are clipped cosine minus one;
acceptance is `score >= threshold`.

There are exactly **30 candidates**:

- Six descriptor specifications, each with three templates: acceleration axes,
  acceleration+gyro axes, acceleration invariants, acceleration+gyro invariants,
  and acceleration+gyro axes/invariant fusion, all with WCCN shrinkage `auto`;
  the sixth is the same fusion with shrinkage `0.2`. This gives 18 candidates.
- Four neural families: `accel_supcon`, `accel_cross_day_supcon`,
  `accel_gyro_supcon`, `accel_gyro_cross_day_supcon`. Each family is a uniform
  two-seed ensemble, concatenating seed-42 and seed-43 embeddings and dividing
  by `sqrt(2)`. Three templates per four ensembles give 12 neural candidates.
  Individual seeds are diagnostics, not additional selectable candidates.

The neural configuration is fixed: multiscale encoder, embedding dimension 64,
width 32, dropout 0.1, statistics fusion, explicit physical vector groups,
magnitudes and training-only normalization. Training uses float32, two Torch
threads, AdamW (`lr=0.001`, `weight_decay=0.0001`), temperature 0.1, linear
training-identity CE weight 0.25, zero label smoothing, gradient clip 1,
no scheduler or augmentation, and the final epoch 40. Both objectives use the
registered two-day identity-balanced sampler; do not replace it with the
generic metric trainer's defaults. Fold preprocessing, metrics and neural
normalization fit only optimizer-training identities.

The **cross-host GPU training budget is exactly 40 CV runs × 40 epochs**:
`4 families × 5 folds × 2 seeds = 40`, totaling 1,600 CV run-epochs.
There are **exactly two additional 40-epoch final fits only if the selected
winner is neural** (1,680 run-epochs overall); a descriptor winner adds zero
neural fits. Template scoring and lock-time verification reuse checkpoints;
they do not authorize more training. This is one aggregate budget across all
hosts, not a budget per host. Assign each family to one host, preserve outputs,
and record any failure before considering a retry. The runner has no global
cross-host scheduler or checkpoint-resume stage.

Run the following from the project directory using the already prepared shared
environment and pinned Santos v7 archive. These are operator commands, not
evidence that this guide executed training. Change the interpreter path for a
different prepared host. Keep the relative archive argument identical across
hosts: the loader includes its spelling in the fingerprinted metadata.

```bash
cd /home/sherlock/Code/github/VariableThe/gait-biometrics
study_python='/home/sherlock/Code/github/VariableThe/.venv/bin/python'
study_archive='datasets/santos/santos-v7.zip'
study_root='artifacts/santos-multisensor-exploratory-v1'

uv run --no-project --python "$study_python" python -m scripts.multisensor_study \
  --stage register --archive-path "$study_archive" --out-root "$study_root"

uv run --no-project --python "$study_python" python -m scripts.multisensor_study \
  --stage train --family descriptors --device cpu \
  --archive-path "$study_archive" --out-root "$study_root"
uv run --no-project --python "$study_python" python -m scripts.multisensor_study \
  --stage train --family accel_supcon --device cuda \
  --archive-path "$study_archive" --out-root "$study_root"
uv run --no-project --python "$study_python" python -m scripts.multisensor_study \
  --stage train --family accel_cross_day_supcon --device cuda \
  --archive-path "$study_archive" --out-root "$study_root"
uv run --no-project --python "$study_python" python -m scripts.multisensor_study \
  --stage train --family accel_gyro_supcon --device cuda \
  --archive-path "$study_archive" --out-root "$study_root"
uv run --no-project --python "$study_python" python -m scripts.multisensor_study \
  --stage train --family accel_gyro_cross_day_supcon --device cuda \
  --archive-path "$study_archive" --out-root "$study_root"

uv run --no-project --python "$study_python" python -m scripts.multisensor_study \
  --stage lock --device cuda \
  --archive-path "$study_archive" --out-root "$study_root"

uv run --no-project --python "$study_python" python -m scripts.multisensor_study \
  --stage evaluate --device cuda \
  --archive-path "$study_archive" --out-root "$study_root"
```

Each of the five `train --family` calls internally completes all five folds;
there is no fold CLI argument. For distributed execution, register once, copy
the identical `registration.json` to each otherwise empty host study root,
run only assigned families, then gather the five complete family directories
under one root before `lock`. Preserve relative artifact paths and bytes.
Do not independently register a different plan or execute every family on every
host. The commands above show the full sequential workflow on one host.

`register` requires a nonexistent or completely empty study root. It freezes
the plan, explicit source-file SHA-256 map, official archive size/MD5/SHA-256
and member inventory, DEV arrays/metadata fingerprint and DEV member hashes
before fitting. Archive auditing hashes compressed bytes and inspects names,
including TEST inventory, without parsing TEST numerical payloads. Subsequent
stages check the registration, archive and source hashes; DEV values are
rechecked. The `SOURCES` tuple is the complete implementation-hash scope:
**documentation is excluded from this runner's source hashes**, including this
guide. It is not a hash of every repository file or the Python environment.
Use identical frozen source bytes, archive bytes and compatible prepared
environments on every host; retain environment/hardware provenance separately.

`lock` requires all 30 candidates and their complete five-fold artifacts. It
chooses minimum equal-fold mean EER, with ties resolved by fixed candidate order.
Before reading CALIB, it reloads fitted fold artifacts and recomputes every
candidate's EER, AUC, coverage and claim-graph hash. Each candidate's validation
graph must match the baseline in each fold. Only then does it fit the baseline
and selected winner on all DEV users, and load reserved CALIB users. Each method
gets its **own independently calibrated threshold** targeting empirical CALIB
FAR ≤1%; never transfer the baseline threshold to the winner. If winner and
baseline are identical, their fitted artifact is reused. `lock.json` freezes
selection, fitted artifacts, thresholds, calibration fingerprint and checksums
before TEST numerical loading.

`evaluate` verifies the lock, source/archive/artifact hashes and DEV/CALIB
fingerprints. It exclusively creates `exploratory_test_attempt.json` before
loading TEST; the reservation prevents a second evaluation even after failure.
Preserve it. The result and saved scores are
`exploratory_test_result.json` and `exploratory_test_scores.npz`. Only the winner
and registered baseline are compared. Their ordered claim identities, probe
identities, probe trial IDs and labels must match on TEST. Report an improvement
only with **matched VALIDATION and TEST evidence**, coverage/exclusions, and the
frozen per-method thresholds. A graph mismatch invalidates the comparison;
dropping difficult claims is not improvement evidence. Descriptive TEST EER
does not replace the CALIB threshold or authorize selection after TEST.

Five TEST users cannot establish deployment FAR, broad population performance
or strong confirmatory significance. Claims share users, templates and probes;
claim counts are not independent sample sizes. Report per-user errors and
paired subject diagnostics as descriptive, alongside pooled EER/AUC and frozen
threshold FAR/FRR. Attaining 1% FAR on five CALIB users does not guarantee 1% on
TEST. A favorable result remains exploratory because of the prior inspection.

The primary research scope is documented in the following method notes; links
are carried over from those notes without new web verification:

- [Multisensor features](multisensor_features.md):
  [IDNet](https://arxiv.org/html/1606.03238),
  [Kobayashi et al.](https://staff.aist.go.jp/takumi.kobayashi/publication/2011/ICASSP2011.pdf)
  and [Zhong and Deng](https://www.baesystems.com/en-us/dam/jcr:aa05526e-72e2-472c-be8c-fe7b1bebbefb/TS-2014-070-Sensor-Orientation-Invariant.pdf)
  motivate magnitudes, vector spectra and temporal inner products. The fixed
  summaries omit those systems' cycle/frame extraction, full correlation or
  dynamic-image representations and classifiers. Invariance covers constant
  orthogonal transforms per triad, not moving sensors or placement changes;
  retained raw axes and neural encoders are not fully rotation invariant.
- [Encoder methods](gait_encoder_methods.md):
  [InceptionTime](https://arxiv.org/abs/1909.04939) and
  [Squeeze-and-Excitation](https://arxiv.org/abs/1709.01507) inspire the compact
  multiscale backbone; kernels, normalization, pooling and objective differ.
  The temporal architecture described there is not a registered study family.
- [Metric-learning methods](metric_learning_methods.md):
  [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) motivates
  identity-positive averaging. Cross-day positive restrictions are a local
  adaptation; the generic trainer's triplet/hybrid options, scheduler and
  best-epoch selection are outside this frozen study.
- [Angular-margin methods](angular_margin_methods.md):
  [ArcFace](https://arxiv.org/abs/1801.07698v3) is an available loss-component
  adaptation. **ArcFace is not registered here**; the study uses a linear
  training-identity classifier.
- [Enrollment methods](enrollment_methods.md):
  [IDNet](https://arxiv.org/html/1606.03238v3) and
  [AuthentiSense](https://arxiv.org/html/2302.02740v1) motivate fixed enrollment
  evidence and unseen-user evaluation. Their systems are not reproduced.
  That note's HAR/WISDM prefix/purge protocol is within-recording and does not
  replace this study's Santos day-1/day-2 trial protocol.

These are research-informed adaptations, not reproductions or guarantees of
authentication accuracy. No tests, training, GPU work or real-data inspection
were performed to author this guide.
