# Joint-sensor repeat feasibility — 25 October 2026

**The joint score resolves the prescribed synthetic half-period ambiguity; it
does not establish gait recognition or authentication improvement. Retain
Canonical362 + automatic WCCN / raw mean_multi / policy C.**

One frozen attempt evaluates 2,880 synthetic inputs and retains all 8,640
extraction records. Before candidate work, all 5,120 historical control records
and available input/correlation/feature bits reproduce exactly. No sensor
archive, original CALIB/TEST payload, biometric model, quality fit or new DEV
experiment is used. The 768-sample factor remains diagnostic only: the
[completed metadata inventory](../2026-10-25-support/README.md) found one such
block and no identity with both-day support.

## Mechanism and extraction are separate results

The primary panel has 96 balanced-peak fixtures at each length. Acceleration
repeats at P/2, while gyro contains P. Both acceleration-only controls return
96 finite half-period outputs per length. The sole joint candidate estimates
exactly P in all 96 cases at each length, but does not always find boundaries.

| Primary panel, denominator 96 at each length | 256 samples | 768 samples, diagnostic only |
| --- | ---: | ---: |
| Control finite one-P outputs / half-P outputs | 0 / 96 | 0 / 96 |
| Joint lag exactly P | 96 | 96 |
| Joint finite one-P outputs | 64 | 95 |
| Joint missing-pair failures | 32 | 1 |

Missing pairs are failures, not corrected extractions. The full legacy panel
also retains the 320 deliberately degenerate-gyro fixtures at each length.

| All 1,280 legacy inputs per length | N256 | N768 |
| --- | ---: | ---: |
| Control finite one-P / finite half-P | 752 / 96 | 863 / 96 |
| Control missing pairs / channel-guard failures | 192 / 240 | 1 / 320 |
| Joint finite one-P / finite half-P | 816 / 0 | 958 / 0 |
| Joint missing pairs / no valid joint lag | 144 / 320 | 2 / 320 |

All historical finite one-P outputs are preserved. Total finite coverage still
falls from 848 to 816 at N256 and 959 to 958 at N768: a finite half-period output
and a missing output are different outcomes. The known alternating P180/phase20
strict-minimum hole persists at N768; balanced P200/phase22 also lacks a pair.
Whole-window degeneracy remains recorded even when another failure is reported
first. No failed inputs are removed from denominators.

## Negative controls and invariance

- Both-noise-only inputs produce finite candidate features in **10/12 at N256
  and 12/12 at N768**. Numerical validity is not a walking detector.
- Replacing the gyro periodic component with noise resolves stipulated P within
  one sample in only **3/12 at each length**. Equal channel weighting supplies
  no signal-quality guarantee.
- All **48 constant/near-constant gyro stress inputs** reject every candidate
  lag. No acceleration-only fallback is used.
- All **24 both-half-period inputs** select the observable P/2 repeat. The
  stipulated anatomical P is not inferred.
- All 128 prescribed transformations per arm preserve selected lag, pair and
  validity. Candidate feature bits differ in 72 comparisons; all 88 comparisons
  with both features valid agree within the declared 1e-10 tolerance, maximum
  difference **8.26e-16**. This finite fixture panel establishes no general
  invariance or phone-placement guarantee.

## Fixed method and research basis

The [registered brief](source/BRIEF.md) and exact source snapshots specify the
candidate. It averages acceleration/gyro Pearson correlations at the same
lag, searches 80–200 samples, and takes the smallest lag within absolute
1e-10 of the finite maximum. Every correlation vector must pass its finite,
positive-RMS and relative-standard-deviation guards. Both magnitudes use the
frozen second-order 3 Hz Butterworth filter with odd padding and padlen 9.
Undefined modality evidence makes the joint lag undefined.

All main arms retain strict acceleration minima, ten-sample margins, pair
constraints and deterministic ranking. Magnitudes share the selected inclusive
interval and the old natural-cubic, endpoint-inclusive 200-point mapping.
The outcome-informed plateau convention is restricted to helper regressions;
it is not applied to this measured candidate. Quality cutoff stays null.

[IDNet §3](https://arxiv.org/html/1606.03238v3#S3) motivates repeat normalization
and uses acceleration-only adaptive template segmentation, followed by eight
orientation-processed channels and spline normalization. Its gyro feature
channels do not determine segmentation. [Barth et al. §2.4](https://pmc.ncbi.nlm.nih.gov/articles/PMC4435165/)
combine sensor-axis distances before shared DTW segmentation, using labeled
shoe-mounted templates. [Matrix Profile VI §§III–IV](https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf)
studies normalized multichannel motif distances and the effect of irrelevant
dimensions; its distance differs from this equal-mean correlation score.
These papers guide this specified adaptation. None of their full pipelines,
datasets or reported authentication accuracies is reproduced here.

## Evidence, audits and execution

The [provenance](provenance.json) binds the freeze, original report, summary and
original artifact index. Every numeric array is published: the complete
335,376,264-byte NPZ compresses losslessly to 52,799,276 bytes, split into three
parts. All 30 NPZ entries are byte-identical to the original standalone NPY
files. Compressed JSON snapshots retain every input key, record, failure,
diagnostic, source/runtime/parent binding and original attempt marker/receipt.
The original 2,493 regression files and runtime binaries are not bundled;
their original hash records remain available. They are not extra observations.

[Full original saved-only replay](full_verification.json) passes for all saved
correlations, statistics, masks, pairs and features, plus NPZ/NPY identity and
complete original index bindings. It does not regenerate signals or re-filter.
An independent review checks all counts, selections, guards, hashes and 50
saved numerical inputs, including 100 filter traces and 1,718 centered-dot
correlations. See [audit history and limits](audit.md).

| Recorded job boundary | Process CPU seconds | Wall seconds |
| --- | ---: | ---: |
| Measurement through complete event | 320.643605 | 323.768351 |
| Separate full saved-correlation verification | 327.848626 | 344.160049 |

The measurement uses one CPU and no GPU, within its 600 CPU/900 wall-second
ceiling. The complete event precedes final index hashing; uv CPU is excluded.
Verification wall time starts at common-module initialization. Preparation,
packaging and verification are separate work. These timings are not a speedup
claim. A failed actual attempt would remain consumed; no measurement retry
occurred. Sources, earlier failures and frozen parent evidence remain unchanged.

## Replay the published numeric evidence

Use the recorded Python 3.14.8, NumPy 2.5.4 and SciPy 1.18.1 environment. From
the repository, choose a fresh output directory with about 0.75 GiB free:

```sh
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-project --offline --python ../.venv/bin/python \
  python -B reports/2026-10-25-joint-repeat/replay.py \
  --out-root /absolute/path/to/a-new-joint-numeric-replay --full
```

The helper validates the exact publication inventory, restores source/metadata
bytes and every original numeric array, then replays correlations, guards,
features, pairs and the complete summary. Omitting `--full` checks the saved
relationships and feature/pair replay without recomputing every correlation.
Existing directories and changed/missing/extra published files are refused.
Failures retain their outputs. No signals are generated, filtered or fitted.

Publication integration checks pass: main full correlation replay, Student B default replay, five root refusal probes and 31 independent failure probes. Ruff 0.17.1 and ty 0.0.90 pass for the two new helpers. [Validation records](publication_validation.json) and the [publication review](publication_review/audit-result.json) retain the evidence. Student B uses the same synthetic inputs and is a mirror, not independent data replication.

This is a **relocated numeric replay**, not verification of the original
absolute-path runtime/cache closure, original regression tree or OS/kernel.
The original strict verification is preserved separately. A local inventory
cannot authenticate an owner rewriting all trust anchors together. Frozen
`source/README.md` describes the preparation snapshot, not current execution
status. The [independent count script](count_reconstruction.py) retains its
original workspace paths; use the replay command above for the publication.

## Next scope

The [secondary metadata planning arithmetic](support_planning.json) identifies
299 segments of length at least 256, spanning 446–931 samples and preserving all
historical trials. They contain 183,128 valid rows, 32,600 more than historical
coverage. A matched canonical cover has 885 windows and exact reciprocal
row-multiplicity weights, averaged before learned processing. This is support
planning, not a newly registered performance experiment.

One separately reviewed addendum is being prepared at **N446 only**, with an
explicit plateau convention and a **linear, endpoint-exclusive 400D mapping**.
That mapping is not validated by this study. Passing its future synthetic gates
would justify preparing a DEV registration, not executing DEV automatically.
Any later comparison must freeze shared source-row perturbations, matched
support/availability, trial/enrollment failure rules, TRAIN owners, policy-C
calibration and all 33 safety gates against matched and historical controls.
Original CALIB/TEST stay closed. Reused DEV and synthetic fixtures cannot
establish population FAR, anatomical stride or deployment readiness.
