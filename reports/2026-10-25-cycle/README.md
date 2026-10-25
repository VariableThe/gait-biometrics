# Short-window cycle feasibility and failure-inclusive evaluation

**Keep Canonical362. Redesign the proposed cycle baseline before a DEV fit.**
A fixed synthetic test of a two-magnitude cycle descriptor exposes missing
boundaries, degenerate channels and half-stride aliases in 2.56-second windows.
This round adds an exact-count evaluation utility so extraction failures remain
in the error denominators. It produces **no new gait accuracy, DEV model or TEST
result**. The [preceding nuisance study](../2026-10-25-nuisance/README.md) and its
baseline-retention decision are unchanged.

## Fixed synthetic experiment

The preregistered grid contains ten stipulated stride periods (80, 90, 100, 110,
120, 128, 140, 160, 180, 200 samples), 32 phases and four signal families at
100 Hz: **1,280 windows**, each with six channels and 256 samples. The full
equations, assumptions and per-window diagnostics are in the preserved
[report](raw/report.json.gz) and [prototype](raw/prototype.py).

The detector smooths acceleration magnitude with a second-order 3 Hz Butterworth
filter, searches Pearson correlations at lags 80–200, and chooses a pair of strict
minima close to the selected lag. It spline-resamples raw acceleration and gyro
magnitudes to 200 samples each, centers and normalizes the two channels, and
unit-normalizes their concatenation. Flat or numerically degenerate channels
fail. The proposed TRAIN-derived quality cutoff is **null and unapplied** here.

| Outcome before quality gating | Windows |
| --- | ---: |
| Finite features | 848 / 1,280 |
| No eligible minima pair | 192 / 1,280 |
| Channel-degeneracy failure | 240 / 1,280 |
| Half-stride extraction among finite outputs | 96 / 848 |

All 96 finite half-stride aliases belong to the balanced-peak family at periods
160, 180 and 200. There are no double-stride classifications in this filtered
grid. A separate **unfiltered** period-100 diagnostic selects 200 with
`numpy.corrcoef` because its computed correlations are
`0.9999999999999999` and `1.0`; a centered-dot reference ties and selects 100.
That diagnostic is a numerical tie warning, separate from the filtered results.

For the gravity family, finite extraction by period is:

| Period, samples | 80 | 90 | 100 | 110 | 120 | 128 | 140 | 160 | 180 | 200 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Finite / 32 phases | 32 | 32 | 32 | 32 | 31 | 27 | 22 | 16 | 10 | 6 |

These are deliberately constructed signals, not observed gait or authentication
success rates. Extracted segments are **putative cycles**: magnitude periodicity
does not uniquely identify a physical stride.

An interval must span at least twice an ideal stride period to contain a complete
boundary-aligned cycle at every phase. The actual sample endpoints span 2.55 s;
the allowed minima span only 2.35 s. Their ideal all-phase stride limit is
therefore 1.175 s, before filtering, alias choice and channel guards. Longer raw
support may address coverage, while harmonic ambiguity needs its own policy.

Limited controls preserve selected lag/pair under gains 0.9/1.1 and common proper
rotations of 30°/90°, with maximum saved feature difference `5.01005e-15`.
Batch/reorder and outside-window future checks agree. These controls establish
properties of the stipulated signals, not phone robustness or a causal samplewise
streaming filter: extraction uses only the completed window.

## Counting failed claims

The new [gated ROC utility](../../docs/gated_roc.md) preserves every original
genuine/impostor claim. Failed claims carry an explicit false validity mask and
`None` score, and always reject. An explicit reject-all policy avoids invented
score sentinels. Tied scores enter together; integer counts and exact fractions
determine EER crossings and empirical FAR-constrained thresholds.

| Synthetic example | Full, failure-inclusive EER | Conditional EER, successful claims only |
| --- | --- | --- |
| Every claim fails | Undefined; terminal FAR/FRR = 0 / 1 | Undefined |
| 20% valid in each class, perfectly ranked valid scores | Undefined; terminal FAR/FRR = 1/5 / 4/5 | 0 |
| Crossing toy, 75% valid in each class | 1/2 | Exact 1/3 at score .6 |
| Audit's terminal-equality regression | Exact 1/3 at score .6 | 1/2, interpolated |

The last row uses genuine scores `[.9, .6, None]` and impostor scores
`[.8, None, None]`. At its terminal state, exactly two of three genuine claims
accept and one of three impostor claims accepts. Full FAR and FRR are both 1/3.
See [corrected examples](corrected-roc-examples.json) for every attainable policy,
mask, count, rational crossing and conditional diagnostic.

When genuine and impostor failure fractions are `fg` and `fi`, the terminal state
accepts every valid claim and has FAR `1-fi` and FRR `fg`; failed claims still
reject. The separate reject-all state has FAR 0 and FRR 1. A full EER crossing exists precisely
when their valid coverages sum to at least one. Otherwise EER is undefined;
forcing an accept-all endpoint would change the system under evaluation.
Conditional EER cannot replace the operational coverage penalty. These are
empirical metrics under a specified policy, not population guarantees or an
assertion of ISO/NIST conformance.

**Erratum to preserved r1:** its floating subtraction incorrectly labels the
crossing toy's exact conditional 1/3 as interpolation. The same bug can label an
exact terminal 1/3 crossing undefined, as the audit regression shows. The stored
all-reject and poor-coverage non-crossing conclusions remain correct. Use the
new utility and corrected examples for ROC arithmetic; the raw prototype/report
remain byte-for-byte historical evidence. The raw README's instruction to reset
the old report **is superseded: never reset or overwrite frozen r1**.

Any future cycle comparison needs a full-coverage canonical control and a
canonical control with the exact same validity/quality masks as the cycle arm.
It must retain original claim denominators, report enrollment/probe failures and
per-person coverage, and freeze calibration independently. Available-window
enrollment, canonical fallback and reacquisition are different protocols; none
was selected or tested here. For illustration only, independent 5% window failure
would invalidate an all-window enrollment with 20 windows about 64.15% of the
time; without independence the sharp bounds are 5%–100%. No real failure rate is
estimated by that example.

## Research basis and reproduction

[IDNet §3](https://arxiv.org/html/1606.03238#S3) motivates cycle extraction and
normalization. Its adaptive templates, eight normalized signals, CNN, one-class
SVM and sequential system are absent here. This is a limited adaptation, not an
IDNet reproduction. [Derawi, Bours and Holien (2010)](https://doi.org/10.1109/IIHMSP.2010.84)
studied cycle detection with longer walks; their observations do not establish
short-window coverage. [Wayman, Possolo and Mansfield](https://www.nist.gov/system/files/documents/2021/05/26/fundamentalissues_final.pdf)
explain why technical test measurements need uncertainty and an explicit link to
the operational question. Synthetic feasibility cannot supply that link alone.

The original run recorded **10.265 CPU seconds**, Python 3.14.8, NumPy 2.5.4,
SciPy 1.18.1, one CPU and disabled CUDA. This is the measured synthetic workload
cost, not an inference benchmark or a speedup. The [independent audit](audit.md)
checked records, masks, bounds and hashes without rerunning the measurement grid.
A separate root replay reproduced all registered scientific fields exactly in
the same numerical runtime; timings, output path and source fingerprint differ.
No cross-platform bitwise promise is made for numerically near-tied signals.

Root validation passes **69 gated-ROC tests**, including the new terminal-equality
regression; the focused suite with existing exact-ranking/selection tests passes
**268 tests in each code repository**. Ruff and ty pass. Raw logs are retained in
this bundle. No new full-suite result is claimed for this isolated addition;
the preceding full-suite results and CI remain in the preceding report.

The raw prototype is tied to its original absolute path. Use the supplied helper
to create a **new, nonexistent** output directory and change only that assignment
in a copied prototype. The new source hash is recorded explicitly:

```bash
# From the repository root, using an existing compatible environment.
uv run --no-project --python ../.venv/bin/python python \
  reports/2026-10-25-cycle/portable_replay.py --out ../artifacts/cycle-replay-new

uv run --no-project --python ../.venv/bin/python python \
  reports/2026-10-25-cycle/recompute_roc_examples.py --out /tmp/corrected-roc-new.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  uv run --no-project --python ../.venv/bin/python python -m pytest -q \
  tests/test_gated_roc.py tests/test_robustness_selection.py tests/test_calibration_selection.py
```

The helper requires the prototype's SciPy 1.18.1 and Linux resource/affinity APIs;
it installs nothing, rejects existing output directories and preserves incomplete
attempts. It intentionally reproduces the raw ROC bug as part of historical
equality. The separate corrected helper uses the new module.

[Provenance](provenance.json) and [inventory](inventory.json) bind raw bytes,
decompressed report, source closure, replay and corrected examples. The compressed
report includes the full grid but not full per-lag arrays or feature vectors.
Raw source permits reconstruction; saved records alone cannot independently
recompute those omitted arrays. No TRAIN/DEV/CALIB/TEST payload was read and no
biometric model was fitted in this round. Previous TEST inspection and all earlier
dependent DEV limitations remain disclosed by their original reports.
