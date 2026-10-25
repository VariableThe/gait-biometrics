# Frozen synthetic length recreation r1

This new directory owns one exclusive standalone synthetic attempt. Nothing is
imported from the original prototype: its top-level output and resource side
effects must never execute here. `cached_reference.json` contains the original
report's 1,280 records and provenance hashes for the three required original
files. Original sources are only read while preparing that portable cache.

The script's output root is its own parent from the start. Freeze and run from
any working directory using the existing shared environment, without installs:

```bash
env PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 /home/sherlock/.local/bin/uv run --no-project --no-sync --python /home/sherlock/Code/github/VariableThe/.venv/bin/python python -B /home/sherlock/Code/github/VariableThe/artifacts/cycle-length-feasibility-r1/prototype.py --freeze-only
env PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 /home/sherlock/.local/bin/uv run --no-project --no-sync --python /home/sherlock/Code/github/VariableThe/.venv/bin/python python -B /home/sherlock/Code/github/VariableThe/artifacts/cycle-length-feasibility-r1/prototype.py --run
env PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 /home/sherlock/.local/bin/uv run --no-project --no-sync --python /home/sherlock/Code/github/VariableThe/.venv/bin/python python -B /home/sherlock/Code/github/VariableThe/artifacts/cycle-length-feasibility-r1/audit.py
```

`--freeze-only` hashes the source, audit, this README, portable cached reference,
complete manifest, interpreter and numerical runtime files before any fixture
measurement. `--run` verifies all hashes, then creates an exclusive attempt
marker. A running, failed, incomplete or completed attempt cannot retry in place.
Preserve incomplete roots. Do not delete or reset their report or marker.
The UV redundant `--no-sync` warning is expected with `--no-project`.

Pinned runtime: Python 3.14.8, NumPy 2.5.4, SciPy 1.18.1, float64, one allowed
CPU, numerical thread environment one, CUDA disabled. There is a five CPU minute
budget: cooperative stop at 270 process CPU seconds, soft limit 290, hard 299.
Array memory maps and JSONL checkpoints retain partial observations.

The four original equations, ten periods, 32 phases, seed 20261025 and time origins
are fixed. Each six-axis 768-sample fixture is generated analytically once; its
256-prefix must match an independently generated 256-sample fixture bitwise.
No old windows are concatenated. All exact-rule N256 records are independently
computed and all cached keys/values compared with exact floating bits. The
required baseline is 848 finite, 192 missing pairs, 240 channel-degenerate,
including 96 finite half aliases. Stop before long extraction on any mismatch.

There are exactly 2,560 fixture/length combinations and two rules per combination:
5,120 records. The original rule selects the smallest exact floating maximum.
The candidate includes every finite lag within absolute 1e-10 of the maximum,
zero relative tolerance, and chooses the lowest. It reports a putative repeat
interval. Both use the original 3Hz, second-order 100Hz Butterworth odd-pad9
filter; np.corrcoef at lags80..200; strict minima at 10..N-11; and the original
pair ordering, inclusive natural splines, per-magnitude normalization, and
1e-12 raw RMS guard. No quality cutoff is trained, guessed or applied.

Only N768 has supplementary correlations: raw acceleration and gyro magnitudes
and their separately filtered versions, lags40..400, always the same 348 anchor
samples t=10..357. These diagnostics log all nearmax sets, gaps and harmonic
alternatives and have no extraction effect. The two-maxima tile-sine and equal
synthetic vectors have fixed +/-one ULP checks and reversed enumeration; other
lag values remain unchanged. Candidate stability is required only for these
fixed checks, not arbitrary values at the nearset boundary.

Independent geometric availability enumerates continuous minima of each actual
prescribed raw acceleration magnitude, then all eligible reference pairs at the
selected lag. Gravity/weak minima use a fixed, preregistered derivative-root
enumeration; alternating/balanced minima have exact phases zero and half.
This differs from stipulated theta-zero physical stride events and from observed
filtered minima. Reports retain all reference pairs, phase holes and disagreements.

`records.jsonl` and `diagnostics.jsonl` preserve per-fixture outcomes, including
every failure. `audit_arrays.npz` is lossless float64 with all six input axes,
both-length extraction correlations, four-channel long diagnostics, computed
400-vectors, invalid NaNs and separate masks. Its record mapping gives fixture,
length, rule and correlation row; diagnostic rows use fixture index. No pickle,
authentication scores or score sentinels occur. `arrays/*.npy` preserve partial
state even if interrupted. File hashes and a named scientific fingerprint bind
the outputs in `report.json`. `audit.py` checks saved evidence without extraction;
it is a local consistency audit, not the root's independent scientific review.

Balanced acceleration has observable period P/2, so physical P cannot be inferred
from acceleration alone. Its gyro magnitude contains P. The joint magnitudes in
this fixture are therefore not unidentifiable; general ambiguity of identical
joint magnitudes is a separate theoretical issue with no new generator here.
Acceleration ambiguity, full-signal periodicity and extracted aliases are
separately flagged. A finite aliased extraction remains a success, and a unique
nearmax set is never evidence of anatomical stride. Near-flat gyro hard failures
remain failures without imputation.

Researcher-selected N768 is the first multiple of256 greater than
3*Pmax+2*margin=620. This is geometric motivation only. These four noiseless
analytic fixtures provide no authentication accuracy or general gait claim.
[IDNet section3](https://arxiv.org/html/1606.03238) includes eight normalized
channels, orientation transforms and adaptive templates absent here. Derawi's
cycle-detection paper is 2010, DOI10.1109/IIHMSP.2010.84, whose metadata is supplied
in the original context (the DOI endpoint was unavailable during preparation).

No real TRAIN/DEV/data/archive/CALIB/TEST, fitting, installs, Git, SSH, child
agents, gain/noise/new conditions, parameter or seed search, publication, or old
artifact changes are authorized or performed. Root reviews this standalone
evidence before any publication or real DEV plan. Results, if completed, are in
`RESULTS.md`; exact denominators and all family/period/rule/length counts remain
in `report.json`.
