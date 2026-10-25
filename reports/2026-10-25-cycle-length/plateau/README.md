# Outcome-informed numerical regression r1

One exclusive attempt, using only twelve already saved six-axis float64 windows
from cycle-length-feasibility-r1: its four original families, P180, phases19/20/21,
N768. The known alternating-family phase20 strict-minimum failure motivated this
panel and the exact plateau convention. This is a repeated-interval normalization
regression, not anatomical stride annotation or a new feasibility study.

Preparation reads saved arrays, copies AST-delimited pure function source text,
and records provenance without importing the old prototype or running its
top-level code. No fixture generator, continuous-root search, full grid, seed or
parameter search is executed. All twenty-four strict baseline rows must reproduce
every cached field with exact JSON type and float64 bits, plus correlation,
feature and mask bytes, before any candidate helper or fixture check executes.

The candidate scans maximal exactly equal runs across the entire finite 1D
filtered acceleration magnitude, requires existing strictly higher exteriors,
chooses floor((a+b)/2), then applies margin10..N-11 to that representative only.
There is no epsilon or ULP tolerance. The copied extraction code object is reused
with only its minima-and-pair dependency replaced. Pair enumeration and ranking,
3Hz order2 100Hz Butterworth SOS odd-pad9, Pearson lags80..200, exact/near1e-10
lag rules, raw magnitude inclusive natural200 splines, population std, unit norm
and 1e-12 raw RMS guard remain frozen. Finite aliases stay finite. Degenerate gyro
is never rescued. There is no quality cutoff or biometric score.

Freeze source, manifest, selected inputs/cache and numerical runtime before
measurement. The exclusive attempt marker blocks retries after any failure.
Partial arrays and journals preserve failures. All JSON hashing requires string
keys and exact type-preserving round trips before hashing, with explicit
post-save fingerprint verification. The combined preparation/freeze/run CPU
budget is strictly below60 seconds, CPU affinity is one allowed core, numerical
threads are1, CUDA is disabled, and only the existing .venv is used through uv.
The completed run includes a saved-evidence consistency audit, not an independent
review and not another extraction attempt.

Run once, without installs, from this directory (absolute paths also work):

```
env PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 /home/sherlock/.local/bin/uv run --no-project --no-sync --python /home/sherlock/Code/github/VariableThe/.venv/bin/python python -B prepare.py
env PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 /home/sherlock/.local/bin/uv run --no-project --no-sync --python /home/sherlock/Code/github/VariableThe/.venv/bin/python python -B runner.py --freeze-only
env PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 /home/sherlock/.local/bin/uv run --no-project --no-sync --python /home/sherlock/Code/github/VariableThe/.venv/bin/python python -B runner.py --run
```

The fixed synthetic helper cases only validate the convention, including both
margin edges, exact shoulders and nonfinite rejection. Independent groupby
references use hand-declared expectations. The original eighteen correlation
nearmax checks concern another numerical issue and are not run or counted here.

IDNet arxiv1606.03238 section3's correlation templates, updates and eight channels
are absent; neither IDNet nor Derawi2010 prescribes this plateau convention.
This is our engineering rule motivated by a known floating equality failure.
Balanced acceleration remains half-period ambiguous, while its gyro contains P;
there is no joint-unidentifiability claim. Canonical362 is retained unchanged.
Passing this small selected panel supports a unit regression and scoping further
work only. It provides no real DEV authorization, gait/authentication claim or
permission to publish. Root must independently review before publication.

All writes are exclusively inside this new directory. No old file, repository
state, dependency installation, SSH, child task, real data, fit or score is used.
