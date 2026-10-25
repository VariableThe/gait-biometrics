# Synthetic cycle feasibility, round 1

Synthetic protocol evidence only. No TRAIN/DEV data, fits, installs, Git, SSH or
child agents. Only these three files are created. The prototype uses the shared
Python environment, one pinned CPU and single-thread numerical libraries; CUDA
is disabled. `report.json` contains all 1,280 planned fixtures, per-window
diagnostics, controls, reference checks, failure-inclusive ROC toys and assumptions.
Its source and manifest SHA-256 fingerprints bind the results to the prior freeze.

Exact invocation from any working directory (first freezes, second measures):

```bash
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 UV_NO_SYNC=1 /home/sherlock/.local/bin/uv run --no-project --no-sync --python /home/sherlock/Code/github/VariableThe/.venv/bin/python python -B /home/sherlock/Code/github/VariableThe/artifacts/cycle-feasibility-r1/prototype.py --freeze-only
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 UV_NO_SYNC=1 /home/sherlock/.local/bin/uv run --no-project --no-sync --python /home/sherlock/Code/github/VariableThe/.venv/bin/python python -B /home/sherlock/Code/github/VariableThe/artifacts/cycle-feasibility-r1/prototype.py --run
```

These are the commands used for the completed attempt. The script refuses to
overwrite/re-execute an existing report; a fresh reproduction requires resetting
`report.json` at this same path. Scientific contents are deterministic; timestamps,
CPU affinity and timings vary. Soft/hard CPU limits are 290/299 seconds, with a
cooperative stop at 280; checkpoints preserve incomplete attempts. No environment
sync occurs (`uv` warns that `--no-sync` is redundant with `--no-project`).

The 10-period x 32-phase x 4-family grid and every limited control are fixed in
`MANIFEST` before measurement. Ground truth is stipulated physical stride phase,
not real gait validation. Alias labels use a reporting-only 10% tolerance; the
detector uses exact Pearson maxima/ties. Quality thresholds remain null for both
own9 and own12 TRAIN. Pre-quality extraction success is not authentication success.

Natural spline endpoint inclusion, raw-channel RMS interpretation of the relative
1e-12 guard, undefined Pearson handling and EER interpolation conventions are
explicit in the report. Failed claims retain null scores and validity masks;
there is no numeric sentinel or fictitious accept-all endpoint.

Primary references and verified API semantics are linked in `report.json`.
IDNet is substantially richer than this two-magnitude representation. The cited
Derawi *Improved Cycle Detection* paper is published in 2010, not 2009, and its
longer-walk observations do not establish short-window cycle coverage.

Completed on one CPU in **10.265 CPU seconds**: 848/1280 finite features before
quality gating, 192 missing pairs and 240 channel-degeneracy failures. Of the 848
finite outputs, 96 are half-stride extractions, all balanced-peak periods
160/180/200. These are synthetic counts, not gait error rates. Gravity-family
success drops to 6/32 phases at period 200. The fixed unfiltered period-100
diagnostic chooses lag 200 with `corrcoef` (r100=0.9999999999999999, r200=1.0),
while the centered-dot reference ties and chooses 100. This diagnostic is separate
from the filtered grid, which produced no double-stride lag classifications.

Hand/reference filter, spline, Pearson and tie checks pass. Gains 0.9/1.1 and
proper common rotations 30/90 degrees preserve lag/pair and features to L2 error
at most 5.1e-15; batch/reorder/outside-window-future checks are bitwise identical.
A separate saved-record audit verifies all fixture keys, hashes, bounds, guards,
ROC masks/denominators and threshold acceptance counts without new measurements.

All-reject ROC: EER undefined, terminal FAR/FRR=0/1. At 20% valid coverage in
both classes, conditional EER=0 but full EER is undefined, terminal FAR/FRR=.2/.8.
The crossing gated toy has full EER=.5. Minor preserved prototype issue: its
conditional EER is numerically 1/3, but floating subtraction labels it interpolated;
exact integer-count fractions identify equality at threshold .6. This does not
affect either non-crossing result. Near-zero accepted channels can also amplify
floating centering error (the fixed relative-1e-9 control remains finite).

Recommendation: preregistration redesign before production. Keep the full
Canonical362 control and add a canonical control with identical cycle validity
masks, always retaining original claims. Available-window enrollment aggregation,
canonical fallback or declared reacquisition are possible new protocols, never
reasons to delete failed claims. None is implemented or selected here.
