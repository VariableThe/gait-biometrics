# Longer cycle support: coverage improves, strict minima still fail

**Retain Canonical362 and defer a cycle DEV experiment.** The fixed 768-sample
comparison removes most missing-pair failures, but leaves one failure despite
adequate geometric support. It retains every degenerate-channel failure and all
half-period aliases. No biometric model is fitted, no real dataset payload is
read, and no TEST result or threshold changes.

## Registered recreation

The same four analytic families, ten stipulated periods and 32 phases from the
[short-window study](../2026-10-25-cycle/README.md) are generated at 768 samples.
Their nested 256-sample prefixes match independently generated short signals
bitwise. All **1,280 short exact-rule records** reproduce the original cached
fields and floating-point bits before any long extraction. This guards against
silently changing the control when extending the support.

Only lengths 256 and 768 were registered, at 100 Hz. Two fixed lag rules are
compared: smallest exact correlation maximum, and smallest lag within absolute
`1e-10` of that maximum with zero relative tolerance. The remaining filter,
strict-minimum detector, pair ordering, spline-200/channel normalization and
degeneracy guard remain unchanged. The longer length was selected geometrically
before execution, not by a length search. There are **2,560 fixture/length
combinations and 5,120 rule-specific records**.

Both lag rules produce identical outcomes, lags, pairs and finite features on
the measurement grid. Counts below are **per rule, out of 1,280**:

| Outcome before any quality gate | 256 samples | 768 samples |
| --- | ---: | ---: |
| Finite features | 848 | 959 |
| No eligible minima pair | 192 | 1 |
| Degenerate-channel failure | 240 | 320 |
| Finite one-P intervals | 752 | 863 |
| Finite half-P aliases | 96 | 96 |

The 80 additional guard failures are the weak-gyro fixtures that previously
failed earlier for lack of a pair. All 320 weak-gyro cases still fail; the longer
support does not rescue a degenerate signal. Aliases remain finite outputs and
are not relabelled as extraction failures. The quality cutoff remains null and
unapplied. These are constructed-signal outcomes, not gait or authentication
error rates. Reusing the same fixtures is recreation and sensitivity analysis,
not independent confirmation.

The independent continuous raw-minimum reference has 194 phase holes at N256
and zero at N768. Two short filtered extractions occur just inside the margin
where a continuous raw minimum lies outside it. Raw continuous geometry and
sampled, filtered minima are different objects; neither is a physiological
footfall annotation or an unconditional detector guarantee.

## The remaining failure is a discrete minimum tie

At N768, `alternating_step_harmonic/P180/phase20` selects lag 180 under both rules,
with correlation `0.9999701790460247`. Its strict detected minima are
`[67,157,608,698]`, giving no eligible 180-sample pair. The continuous prescribed
minima lie at `67.5 + 90k`, admitting six pairs of length 180 inside the margin.

Single-fixture reconstruction reproduces all 121 saved extraction correlations
bitwise. A separately coded forward/backward direct-form-II filter reproduces
SciPy's filtered vector bitwise. At four interior troughs, neighboring samples
are exactly equal:

| Equal sample indexes | Shared binary64 value, hexadecimal |
| --- | --- |
| 247, 248 | `0x1.0f78ca92c5066p+3` |
| 337, 338 | `0x1.1f724fd4e6dffp+3` |
| 427, 428 | `0x1.0f78ca92c5063p+3` |
| 517, 518 | `0x1.1f724fd4e6dffp+3` |

Neither member is strictly below both neighbors. All four troughs disappear
from the strict detector. This is separate from correlation-maximum ties.
An 80-digit mpmath reference on the reconstructed binary64 magnitude samples
resolves eight strict minima. Its maximum absolute difference from the float64
filter is about `6.25888e-14`; the interior right-minus-left gaps range from
approximately +0.017 to −0.108 float64 ULP. Near-identical signal shapes can
therefore produce different discrete events. These high-precision references
are not interval-certified error bounds or anatomical event labels.

[The original probe transcript](raw/phase-precision-original-transcript.txt),
[portable input-path adaptation](raw/phase_precision_probe.py),
[one saved six-axis fixture](raw/single-fixture.npy) and
[root reproduction output](phase-precision-root.log) retain this diagnosis.
The adaptation changes the input to the copied single fixture and adds its path
import; numerical calculations are otherwise the reviewer's executed probe.
It is an outcome-informed diagnostic, separate from the frozen measurement.

The separate outcome-informed regression compares strict minima with one
exactly-equal-run plateau convention on four families, P180 and phases 19/20/21
at N768. It retains both lag rules and all guards. A maximal equal-valued run
is one minimum only when both exterior neighbors exist and are strictly higher;
its lower-middle representative must meet the original margin. There is no
epsilon, filter change or tolerance sweep.

| Outcome per lag rule, out of 12 saved fixtures | Strict | Plateau |
| --- | ---: | ---: |
| Finite features | 8 | 9 |
| Missing pair | 1 | 0 |
| Gyro guard failure | 3 | 3 |
| Finite half-period aliases | 3 | 3 |

The known alternating-family phase20 case gains pair `[337,517]`. It is the
only fixture whose selected pair changes; the three degenerate gyro failures
and three finite half-period aliases remain. All 24 cached strict baseline rows
match bitwise before candidate checks. All 16 fixed helper cases pass. The panel
contains 48 rows across two lag rules and two minimum policies; it was selected
after the failure was observed and supports a numerical regression only. It is
not another parameter search, a full-grid plateau result or a basis for
cycle-model promotion. None of it changes the completed length-study counts.

[Complete regression evidence](plateau/RESULTS.md), including its lossless NPZ,
partial arrays, sources and fingerprint self-checks, is published separately
inside this bundle. Its run uses 1.879324401 CPU seconds; combined
preparation/freeze/run through verification uses 3.959992010 CPU seconds. Its
JSON fingerprint contract checks string keys and exact type-preserving round
trips before hashing, avoiding the older serialization defect without modifying
that older report. The frozen preparation script refers to the original local
parent artifact directory; it is retained for provenance, not presented as a
portable fresh-run command. Completed regression directories refuse reruns.

## Ambiguity and numerical checks

All 18 registered tile-sine/equal-maxima, one-ULP and reversed-enumeration checks
pass for the near-maximum convention. Exact and candidate selections differ on
nine fixed vectors, as expected. This is a narrow numerical property; it does
not establish stability at arbitrary `1e-10` membership boundaries. Neither
rule changes an authentication-score threshold or tie rule.

Long-window diagnostics use exactly 348 common anchors at each lag 40–400 for
raw/filtered acceleration and gyro magnitudes. They never rescue extraction.
Balanced acceleration has observable period P/2, while that fixture's gyro
magnitude contains P. Acceleration alone is ambiguous; the joint magnitudes
in this fixture are **not** unidentifiable. More support and a unique numerical
winner do not establish anatomical stride. All 96 finite half-period extractions
remain in the results.

[IDNet §3](https://arxiv.org/html/1606.03238#S3) uses adaptive template matching,
orientation processing and eight normalized signals absent here. Adding gyro
features after acceleration-only segmentation would not by itself reproduce
IDNet or fix its segmentation ambiguities. [Derawi et al. (2010), §IV](https://www.researchgate.net/publication/221566748_Improved_Cycle_Detection_for_Accelerometer_Based_Gait_Authentication)
search for a lowest value around predicted endpoints, differing from this strict
immediate-neighbor rule. Neither inspected paper specifies this implementation's
floating-point plateau convention. The present choices are engineering
adaptations; their authentication results are not reproduced.

## Preserved fingerprint defect and independent review

The original combined fingerprint was computed with integer histogram keys.
JSON saves them as strings, and canonical sorting changes their order. The
frozen `audit.py` therefore **fails its scientific fingerprint assertion** even
though individual file hashes pass. This failed audit remains part of the record;
the source and measured report were not edited and the attempt was not retried.

The separate erratum verifies both forms:

```text
Original integer-key form:
51a085a519c36fb465a0b69812c61bed11cac5bca9995c31d1da15d8a7af801b
Saved JSON form:
9061bf1d4df7f88a56ef9dd156e55625a4b44933867e87fd55bc5083cf890112
```

Restoring only the disclosed histogram keys reproduces the original hash exactly.
This explains a serialization defect, not altered numeric results. The original
audit's failure is not relabelled a strict pass. A separately saved post-run audit
checks the remaining arrays with that explicit qualification.

An independent reviewer checked all individual source, manifest, runtime,
reference, measured-file and partial-array digests, all 5,120 record mappings,
correlation selections, pair rankings, masks, guard decisions and counts. Every
partial NPY agrees bitwise with its NPZ counterpart. Eight saved fixtures at both
lengths independently reproduce 1,936 extraction and 11,552 diagnostic
correlations, ten finite feature vectors, four guard failures and two missing-pair
cases bitwise. All continuous reference geometries and 18 fixed numerical cases
check out. The reviewer did not rerun the complete grid or adopt the implementer's
supplemental audit as independent verification. See [audit](audit.md).

The single frozen run records **64.069864588 process CPU seconds** and
**66.662142669 wall seconds after imports**, one CPU and disabled CUDA. The
implementer's saved-array audit adds 2.834259962 CPU seconds separately. These
are workload costs, not inference latency or speedup claims. Runtime is Python
3.14.8, NumPy 2.5.4 and SciPy 1.18.1; the precision probe uses existing mpmath 1.3.0.

## Compact publication and reproduction

The bundle preserves raw sources, cache, complete measured JSONL rows, report,
erratum, original/supplemental audit outputs and the complete local artifact hash
index. Raw files are copied exactly or losslessly gzip-compressed. The full
81,296,238-byte `audit_arrays.npz` and partial NPYs remain local and are **not**
embedded here. The compact checker verifies published bytes, original hashes,
rows, counts and both fingerprint forms; it does **not** replay omitted arrays:

```bash
uv run --no-project --python ../.venv/bin/python python \
  reports/2026-10-25-cycle-length/verify_compact.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  uv run --no-project --python ../.venv/bin/python python \
  reports/2026-10-25-cycle-length/raw/phase_precision_probe.py
```

The full synthetic inputs and arrays can be regenerated from preserved source
in a **new nonexistent directory** with the pinned runtime. Do not run the
measurement in the published raw directory or reset the original attempt:

```bash
uv run --no-project --python ../.venv/bin/python python - <<'PY'
import gzip, shutil
from pathlib import Path
raw = Path('reports/2026-10-25-cycle-length/raw')
target = Path('../artifacts/cycle-length-recreation-new')
target.mkdir()  # Refuse an existing directory, even when empty.
for name in ('prototype.py', 'audit.py', 'README.md'):
    shutil.copyfile(raw / name, target / name)
(target / 'cached_reference.json').write_bytes(
    gzip.decompress((raw / 'cached_reference.json.gz').read_bytes()))
PY
uv run --no-project --python ../.venv/bin/python python \
  ../artifacts/cycle-length-recreation-new/prototype.py --freeze-only
uv run --no-project --python ../.venv/bin/python python \
  ../artifacts/cycle-length-recreation-new/prototype.py --run
```

Each new attempt captures its own runtime, marker, timestamps and digests. The
source enforces its baseline-recreation gate, exclusive attempt and five-CPU-minute
cap. It uses Linux resource/affinity APIs and installs nothing. The historical
combined-fingerprint bug is intentionally retained; the raw audit's known stop
must be interpreted using the separate erratum, not hidden or bypassed. Matching
version strings alone do not guarantee cross-platform bitwise equality. The
published single-fixture probe was reproduced by root; the full new-directory
measurement command is provided for recreation, not claimed as an additional run.

The new compact/precision helpers are report tools. Production modules, frozen
historical sources, data splits and published thresholds remain unchanged. No
new full-suite run is needed for this evidence-only addendum; the preceding
utility PR's CPU CI passed in both code repositories before the addendum.
