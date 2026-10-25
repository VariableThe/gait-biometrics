# Frozen DEV nuisance covariance study

This is a bounded adaptation to **already inspected DEV/stress evidence**.
Earlier DEV and TEST inspection is acknowledged. It supplies no independent
confirmation, significance claim, population FAR guarantee, or deployment
adapter. The previously nominated canonical representation remains a research
nomination. Original CALIB and TEST payloads stay closed, and the loader has no
public subject-ID override. No real DEV execution is authorized by implementing
or testing this harness: root must freeze the complete reviewed harness and
source/runtime closure before any real DEV execution.

The historical canonical source reference is
`108a559bc1a83355dc85335e9f640786846f894a`; inspected evidence HEAD is
`1c861bfc2104b204c7262d931d8cffa19cc6e676`. These are provenance labels supplied
with the brief, not a claim that this implementation queried Git or read those
result archives. The run registration hashes the actual complete source closure.

## Fixed design and research basis

The only arms, in order, are `baseline` and `nuisance`. Both use the unchanged
historical **canonical362** representation: equal-sign pooled canonical336 plus
the original raw-window lag-spectrum26. There is no descriptor search. The
historical modules, their globals and files are left unchanged.

[Hatch, Kajarekar and Stolcke (2006), sections 1–2](https://www.isca-archive.org/interspeech_2006/hatch06_interspeech.pdf)
motivate weighting by within-class covariance. This study adapts that mathematics
to gait; it does not reproduce their speaker recognition system.
[Um et al. (2017), introduction and section 3.2](https://arxiv.org/html/1706.00527v2)
motivate controlled sensor augmentations while explaining that transformations
need not preserve labels. Fixed synthetic perturbations here are engineering
probes, not new independent walks or measured phone-placement distributions.

Use the inherited twenty DEV-only crossfit folds without changes. Each arm fits
four models on nine TRAIN identities, scores their disjoint clean held triples,
and pools those scores to obtain the policy-C empirical FAR <= 1% cutoff. It then
refits on the twelve nonvalidation TRAIN identities and applies that frozen
cutoff to the outer validation triple. The metric fit budget is exactly
**200 actual fits** (`20 × 2 × 5`), with **160 inner and 280 outer score sets**.
Replay scores all **440** sets and fits no model. Each DEV identity validates
four times; people, folds and claims are dependent.

Both arms complete pooled clean calibration before either twelve-person refit
begins. A synthetic spy checks this actual call order and the pooled source
graph; refit independence does not replace this registered sequencing rule.

## Separate TRAIN transform

Every model has its own nine- or twelve-person TRAIN partition. Use only its
original rows, **both days**, with exactly four paired raw views in this order:

1. `gain09`: multiply every coordinate, including acceleration DC, by 0.9.
2. `gain11`: multiply every coordinate by 1.1.
3. `noise_plus`: `X + 0.01 * s * Z`.
4. `noise_minus`: `X - 0.01 * s * Z`.

For every view, convert source float32 to float64 for arithmetic, then cast back
to source float32 before feature extraction. Both noise signs use the original
float64 X and the same unrounded displacement; never derive minus from a rounded
plus window. `s` is exactly the inherited `training_scale(data, train_ids)`:
center each TRAIN window/channel over time, average residual squares across
TRAIN rows, xyz and time within each triad, take the square root, repeat across
xyz. Inner models use nine-person scales, never outer twelve-person scales.

Seed **20261026** drives independent row-keyed PCG64 generators. Each generator
is initialized with the integer SHA256 digest of compact sorted canonical JSON
`['imu-evaluation-stress-v1', 20261026, ['str', key]]`. Keys are the original
`subject/day/trial/start` strings; batching or reordering never resets them.
The seed convention matches `data/imu_perturbations.py`; the TRAIN transform is
separately implemented and registered. That module's `evaluation_only` metadata
is never used as TRAIN evidence.

Only original clean TRAIN rows determine feature mean, standard deviation and
automatic identity-balanced WCCN shrinkage. Let `C_B` be that clean covariance,
and `delta_ik = (view_ik - clean_i) / clean_std`. The nuisance second moment is

`V = mean_identity mean_original_row,view(delta_ik delta_ik.T)`.

It is **uncentered**, identity balanced, with fixed coefficient one. The nuisance
arm whitens `C_A = C_B + V` using the existing floor
`max(max_eigenvalue * 1e-5, 1e-6)`. The baseline uses `V = 0` and the unchanged
clean metric. No covariance is recovered by inverting a floored baseline
projection. Views supply no extra clean rows or independently labeled samples.

## Geometry and source coverage

Registration binds the fixed plan, runtime versions, and every inherited
`SOURCES` entry plus the new study, metric and selector modules/tests/docs, before
the DEV loader runs. Source coverage derives exact enrollment and claim graphs
from original subject/day/trial inventories and retains loader exclusions,
including zero-window recordings. Every scored graph must match exactly.

Before **any descriptor or fit**, preflight persists all **541 assigned context records**:
one clean-all-DEV context, twenty outer folds × seven validation conditions,
and all hundred TRAIN assignments × four raw views. TRAIN assignments include
baseline and nuisance explicitly, sharing the same four view records once per
partition. The explicit fold and inner/refit role is preserved even when TRAIN
identity sets repeat. No statistics or fit-budget entries are deduplicated.
For positive finite TRAIN references, all 541 contexts materialize exactly the
registered windows and geometry. In failed runs, 541 counts assigned records,
including blocked contexts; it does not claim 541 geometry materializations.

`development.json` records the original data/member fingerprint and complete
loader metadata, source graphs, TRAIN IDs, inner/refit scales, clean row/window
hashes and every view's geometry record. The inherited lossless geometry NPZs
contain ordered source keys, frames, validity masks, reasons and relative
diagnostics, including NaNs for undefined geometry. Records hash both ordered
keys and transformed windows. The fixed relative geometry guards remain
`1e-6` gravity, `1e-12` horizontal energy and `1e-6` eigengap.

Any unidentifiable window aborts the **whole study**, after registering all
geometry contexts and persisting their diagnostics and development manifest,
before the first feature or fit. There is no dropping, imputation or fallback.
Preserve failed roots; never overwrite registration to accept a changed harness.

Invalid own-TRAIN dynamic RMS references follow the same persistence rule,
including both inner nine-person and refit twelve-person assignments. A zero or
nonfinite triad RMS is an error even if canonical frames are identifiable (for
example, zero gyro dynamics with valid acceleration). The registered plan binds
this failure policy before DEV loading. Public `training_scale`,
`training_transform` and `_training_inputs` retain their strict positive finite
RMS requirement. Preflight may recompute the original raw RMS only to record the
failure; it never substitutes a positive scale or supplies an invalid reference
to a noise transform or training operation.

On such a failure, `development.json` retains original source metadata,
exclusions, exact graphs, keys, all 541 records, and geometry status
`failed_invalid_train_scale`. Its `scale_errors` identify every affected fold,
inner/refit role, TRAIN ID list, raw six-coordinate scale, failed triad and reason.
Each affected TRAIN assignment also has `scale_status=invalid_train_scale`.
Finite zero RMS values remain zero in diagnostics; nonfinite RMS values serialize
as JSON `null` with reason `nonfinite`, never as usable references.

Scale-independent clean, rotation and gain contexts still materialize. The
affected TRAIN `noise_plus`/`noise_minus` views and, for invalid refit references,
outer `noise01`/`noise05` conditions have status
`blocked_by_invalid_train_scale` and `materialized=false`. Their diagnostic NPZs
retain all ordered source keys, false validity masks, the explicit blocked reason,
and NaN frame pairs and ratios. These records bind `source_windows_sha256` but
set the transformed-window field `windows_sha256=null`: no transformed payload
or evaluated geometry is claimed. Failure manifests separately count
`blocked_contexts` and `materialized_contexts`; `invalid_context_windows` includes
blocked source rows. After saving the complete manifest, the run raises an
explicit scale error before any feature or fit. The failed root remains intact
and a repeated run refuses it as nonempty.

## Validation stress and exact selection

The inherited conditions stay `clean`, `rotation30`, `rotation90`, `gain09`,
`gain11`, `noise01`, `noise05`. Validation perturbations affect **day2 only**;
day1 enrollment stays bit-identical. Noise retains seed **20261025** and each
outer model's twelve-TRAIN dynamic triad scales. Raw mean/two-medoid blend scores
and clean calibration cutoffs are frozen for every condition.

The sole selection authority is
[`summarize_nuisance_study`](../engine/robustness_selection.py). This harness calls
it and does not duplicate its gates. Rows extend the historical `_row` fields
with exact overall/per-probe-person EER and AUC from `ranking_metrics`, plus the
canonical array hash of float64 scores compared `>=` the cutoff.

The root selector's frozen contract requires clean mean FAR <= baseline and
<= 1%, no more bad folds, no greater mean FRR or pooled rejects, at least 10/15
person means no worse, no person with more than one additional reject or a new
all-reject outcome, and mean exact EER <= baseline. The robust objective requires
mean FRR over gain09/gain11/noise01 to improve by at least 0.05, none of those
conditions worse, at least 10/15 person mean stress rates no worse, and maximum
person regression <= 0.05. Every condition must retain FAR <= 1%, bad folds no
worse than baseline, and each claimant's false accepts no greater than baseline.
Both rotations must have decisions identical to that arm's clean decisions.
Noise05 is reported and passes FAR gates but is excluded from the improvement
objective. All decisions use exact fractions/counts; display floats do not gate.
This objective intentionally adapts to inspected stress, and cannot be described
as stress-independent selection.

## Artifacts and verification

The CLI accepts `--stage {run,verify}`, `--archive-path`, `--out-root`. Python
callers supply `archive_path` and `out_root`; `run` returns the selector summary
on success and `verify` returns its report. Preflight failures persist diagnostics
and raise without creating a result. Every run requires a fresh empty root.

Save 200 lossless numeric model states, 440 label/score caches, original score
metadata, ordered threshold files, complete rows and exact selector summary.
Each model stores mean, scale, projection, training IDs, shrinkage and all numeric
diagnostics: `clean_covariance`, `nuisance_second_moment`, unfloored `eigenvalues`,
`eigenvalue_floor`, `original_rows`, `original_subjects`, `views_per_row`.
Model records bind dimensions, counts, TRAIN row keys and clean/paired-view
feature hashes. Artifact assignments and hashes are unique and complete.

Verification refuses changed source or runtime **before archive loading**. It
rebuilds all source coverage/scales/geometry and features. Without calling any
estimator fit, deterministic residual sums validate clean covariance and the
saved automatic shrinkage scalar; recreated paired features validate V. Baseline
V must be zero. Shape, finiteness, symmetry, PSD, counts, TRAIN IDs, feature
hashes, `C_B+V` eigenvalues, floor and projection are checked. This is deterministic
algebra on recreated TRAIN features, not a new learned covariance fit. Saved C
and V cannot be swapped merely because their sum produces the same projection.
Algebra comparisons use `atol=1e-12, rtol=1e-10`; row/count/assignment and
mean/scale checks are exact.

All 440 score sets replay with inherited `atol=rtol=1e-6`, exact labels/metadata,
and identical fresh/cache cutoff decisions, including within numerical tolerance.
Pool thresholds, cached exact ranking/counts/decision hashes and selection must
recompute exactly. Numerical replay does not independently validate raw truth.

## Synthetic checks and freeze handoff

Tests use only synthetic `(N,6,256)` signals and temporary output artifacts.
Real canonical/view integration requires no fits. The end-to-end fixture uses
distinct five-dimensional clean/view features and exactly 200 tiny actual metric
fits, followed by 440 fit-free score replays, complete geometry coverage, early
refusal and assignment/digest/scale/threshold/decision/diagnostic tampering.
Independent failure checks cover zero acceleration RMS for an inner nine-person
TRAIN partition, zero gyro RMS for a refit twelve-person partition with otherwise
valid geometry, and zero RMS across all DEV. They check complete source/scale
diagnostics, all 541 row-preserving records, blocked noise with no materialization
claim, zero features/fits, strict public transforms, and preserved failed roots.

Use the existing shared environment, one CPU thread, no installation:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
PYTHONDONTWRITEBYTECODE=1 UV_OFFLINE=1 \
uv run --no-project --python ../.venv/bin/python python -B -m pytest \
  -p no:cacheprovider tests/test_nuisance_covariance_study.py
ruff check --no-cache scripts/nuisance_covariance_study.py tests/test_nuisance_covariance_study.py
ruff format --check --no-cache scripts/nuisance_covariance_study.py tests/test_nuisance_covariance_study.py
ty check --python ../.venv/bin/python scripts/nuisance_covariance_study.py tests/test_nuisance_covariance_study.py
```

Ruff/ty versions must match the existing project pins. Run these checks only
after concurrent metric/selector modules, tests and docs exist; registration
intentionally refuses an incomplete source closure. Root owns CI, review,
freezing, real DEV execution, disposition and publication. No archive, historical
result payload, real metric fit, SSH or child-agent execution is needed for this
implementation handoff.
