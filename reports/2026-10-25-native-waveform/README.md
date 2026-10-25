# N446 waveform feasibility — stopped candidate

**STOP: four of 960 required nondegenerate fixtures lack admissible boundaries.**
The fixed all-960 gate fails. Canonical362 + automatic WCCN / raw mean_multi /
policy C remains the exploratory authentication nominee. No DEV registration
eligibility, model fit, original CALIB/TEST read or authentication improvement
follows from this result. The consumed attempt is not retried or retuned.

One reviewed, frozen synthetic attempt retains all **1,440 inputs**: 1,280 legacy
fixtures, 96 stress inputs and 64 transformations, all at N446. This is the
minimum eligible segment length in the completed
[metadata inventory](../2026-10-25-support/README.md). The representation and
boundary convention were specified separately from the earlier
[joint-repeat study](../2026-10-25-joint-repeat/README.md); outcomes are adaptive
exploratory evidence, not independent confirmation.

## Registered gates and outcome

| Fixed gate | Observed result | Disposition |
| --- | --- | --- |
| Fixed numerical/mapping checks | 68 preparation checks pass | Pass |
| All 960 nondegenerate legacy inputs valid, lag and duration errors ≤1 sample | All 960 lags equal P; **956 valid**, four missing pairs | **Fail — STOP** |
| Weak gyro and constant/near-constant stress inputs invalid | 344/344 invalid | Pass |
| Fixed transformations preserve decisions and features within tolerance | 64/64 pass; 48 both-valid feature comparisons | Pass |
| Valid mapping means, variances and norm meet fixed verification tolerances | All valid outputs pass | Pass |

The balanced half-period-admissible subset retains all 96 fixtures: **94 valid,
two missing pairs**, with all 96 selected lags equal P. These are synthetic
repeat intervals, not anatomical event labels.

| Legacy family, 320 inputs each | Valid | Missing pair | No joint lag |
| --- | ---: | ---: | ---: |
| Gravity periodic six-axis | 320 | 0 | 0 |
| Alternating-step harmonic | 318 | 2 | 0 |
| Balanced periodic peaks | 318 | 2 | 0 |
| Weak/degenerate gyro | 0 | 0 | 320 |

The four failed keys are P200 at phase indices **2 and 18** in the alternating
and balanced families. Their saved strict and plateau minima are identical:
`[87, 187, 288, 388]`. Possible pair durations are `{100, 101, 201, 301}`.
None meets both registered conditions: duration 80–200 and within 10% of lag200.
Correct lag selection therefore does not produce an admissible waveform.
This diagnosis reads saved evidence only. No boundary-limit adjustment,
tolerance change, fallback, length expansion or repaired rerun is performed.

At N446, ideal P≤200 minima have enough continuous phase room inside the
ten-sample margins. That geometric statement does not guarantee sampled,
filtered minima satisfying the complete numerical pair rule. The failed gate
is retained even though the other 956 nondegenerate fixtures succeed.

## Stress diagnostics remain unconditional

Each row below retains twelve inputs; stress results do not rescue a failed gate.

| Condition | Valid features | Lag within one sample of stipulated P |
| --- | ---: | ---: |
| Clean analytic joint repeat | 12 | 12 |
| Both modalities, 0.1× AC-RMS noise | 12 | 12 |
| Gyro, 4× AC-RMS noise | 12 | 9 |
| Gyro periodic component replaced by noise | 12 | 2 |
| Both periodic components replaced by noise | **12** | 0 |
| Constant gyro | 0 | 0 |
| Nearly constant gyro | 0 | 0 |
| Both modalities repeating at P/2 | 12 | 0 |

All twelve noise-only signals produce finite features. Numerical validity is
not walking detection. The both-half condition exposes a genuine available
repeat at P/2; stipulated anatomical P is not identifiable from those signals.
No quality gate is learned or applied. All 64 prescribed rotation/gain
transformations preserve lag, pair and validity; maximum feature difference
among the 48 both-valid comparisons is **1.322e-14**, below the fixed 1e-10 bound.
This finite synthetic panel supplies no general phone-placement guarantee.

## Exact adaptation

The [brief](source/BRIEF.md) freezes the sole candidate before extraction:
equal-mean acceleration/gyro Pearson correlation at the same lag, 80–200;
second-order 3 Hz Butterworth filtering at 100 Hz, odd padding and padlen9;
finite/positive-RMS/relative-standard-deviation guards; smallest lag within
absolute 1e-10 of the finite maximum. Either undefined modality makes the
joint lag undefined, with no modality dropping.

An exact-equality plateau rule takes the lower-middle representative only when
both exterior neighbors are strictly higher. All minimum pairs are enumerated,
including pairs with intervening minima. The original strict detector is a
diagnostic and cannot rescue extraction. The plateau convention was informed
by the earlier targeted regression and is explicitly declared adaptive.

The two **unfiltered** magnitudes share integer knots from a through b. Linear
interpolation evaluates `a + (b-a)*k/200`, k=0…199: the right endpoint is excluded
from output phase samples but remains an interpolation knot. Each channel is
centered and divided by its population standard deviation; acceleration comes
first, gyro second. There is no global normalization or appended duration,
amplitude or quality feature. The 400D vector's theoretical norm is20. This
mapping differs from the old natural-cubic, endpoint-inclusive representation.

[IDNet §3](https://arxiv.org/html/1606.03238v3#S3) motivates cycle normalization
but uses acceleration-only adaptive template segmentation and eight processed
channels. [Barth et al. §2.4](https://pmc.ncbi.nlm.nih.gov/articles/PMC4435165/)
uses labeled shoe-mounted templates with joint-axis DTW. The repeat score and
phase mapping here reproduce neither complete system nor their accuracy.

## Evidence, execution and verification

Before candidate extraction, all 1,280 analytically generated N446 legacy
fixtures reproduce their N256 prefixes bitwise using the unchanged generator.
This is a generator-compatibility check, not a cached historical-outcome replay.
The [provenance](provenance.json) binds all 18 frozen sources, metadata, complete
records, failure masks, gates, attempt receipts and original inventory.
All **21 numeric arrays** are published losslessly: the 86,655,858-byte NPZ
compresses to **15,194,716 bytes**. Every NPZ member is byte-identical to its
original standalone NPY file.

The run reaches its report boundary at **89.331602 process CPU / 90.314713 wall
seconds**, on one CPU with no GPU, within the fixed 600 CPU/900 wall-second
ceiling. Its in-run saved replay takes 39.547100 CPU seconds. The report boundary
precedes final index hashing and excludes uv CPU; packaging is separate work.
No speedup claim is made. An additional original saved-only verification passes
and reproduces the gate failure; its external timing was not instrumented.

[Original verification](full_verification.json) checks every saved correlation,
statistic, mask, pair, coordinate, mapped sample, feature and gate without
generating or filtering signals. It deliberately starts from the saved filtered
magnitudes. [Independent count reconstruction](count_reconstruction.json) confirms
the four failed keys and their pair durations. [Independent concrete-evidence review](audit.md) clears the retained negative result: all 1,440 records, complete gates and 21 array pairs agree, with independent scalar/filter references. It provides no authorization to progress this stopped candidate.

## Replay the negative result

Use Python3.14.8 / NumPy2.5.4 / SciPy1.18.1 and a fresh output directory with
about 0.25GiB free. From the repository:

```sh
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-project --offline --python ../.venv/bin/python \
  python -B reports/2026-10-25-native-waveform/replay.py \
  --out-root /absolute/path/to/a-new-native-waveform-replay
```

The helper validates the exact publication inventory, restores source/metadata
and all original array bytes, then replays all correlations, mapping and gates.
Both accepted forms, with or without `--full`, perform full numeric replay.
A verified gate failure is a successful reproduction of **STOP**, not a pass
for the candidate. Existing output directories and changed/missing/extra files
are refused; failed restorations remain. Nothing is generated, filtered or fitted.

Main and Student B relocated full numeric replay both pass and reproduce STOP. Five root missing/corrupt/extra-file/existing-directory probes pass. Ruff 0.17.1 and ty 0.0.90 pass for the two new publication helpers. Student B is a byte-identical mirror, not independent data replication.

Independent publication review clears the retained negative result: all 51 mirrored files, restored payloads and numeric arrays match, all 123 checked local links resolve, and 41 isolated helper checks pass. Its numerical calls were stubbed; the two actual full integration replays above supply that separate evidence. See the [review record](audit.md).

This is **relocated numeric replay**, not verification of the original complete
absolute-path runtime/cache closure or preparation-history tree. Their original
hash records are retained; their payloads are not bundled. Mutable local trust
anchors cannot authenticate an owner rewriting them all. Frozen
`source/README.md` describes preparation and is preserved unchanged.

This candidate stops here. Any separate nonsegmented support comparison needs
its own hypothesis, controls, registration and review. It is not a repaired
waveform attempt. No CALIB/TEST reopening or real DEV execution is authorized
by this result, and Canonical362 remains retained.
