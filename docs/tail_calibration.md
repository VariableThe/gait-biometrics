# Exploratory tail calibration

`engine.tail_calibration` supplies pure NumPy/SciPy threshold functions. It reads
no datasets or artifacts, fits no models, and changes no frozen study sources.
All examples and tests use synthetic scores. These functions support a DEV-only
threshold comparison; they do not alter the existing calibration protocol.

**No iid guarantee is claimed for the current clustered gait claims.** Claims
reuse probes, enrolled identities, trials and fitted models; their score count
is not a count of independent calibration observations. Correlation cannot be
fixed by relabeling the number of claims as an effective independent sample
size. The nominal calculations below are reference diagnostics, not certified
population FAR bounds for this experiment.

## Frozen API

```text
nominal_order_statistic_threshold(impostor_scores, *, alpha=0.01, delta=0.1) -> dict
nominal_violation_bound(n, *, allowed_count=0, alpha=0.01) -> float
empirical_threshold(labels, scores, target_far=0.01) -> dict
```

Scores must be nonempty, one-dimensional, finite real numeric vectors
representable in float64. Ranking and ROC selection convert them to float64
without changing the input. Boolean, complex, string and object score arrays
are rejected. Both binary label classes are required for the empirical function. `alpha` and
`delta` must be real scalars strictly inside `(0, 1)`; booleans, strings and
arrays, including zero-dimensional arrays, are rejected. `target_far` uses the
same scalar validation but permits the endpoints `[0, 1]`.

### Nominal order-statistic selector

For `n` impostor scores, choose the maximum integer `j` in `0 .. n-1` satisfying

```text
B(n, j, alpha) = scipy.stats.binom.cdf(j, n, alpha) <= delta.
```

The threshold is `nextafter(the (j+1)-th highest score, +infinity)`. Decisions
accept `score >= threshold`. Every calibration score tied at the boundary is
rejected, so at most `j` calibration impostors are accepted. Rounding uses
`nextafter`, not addition of an arbitrary epsilon.

Apply a nominal threshold using
`np.asarray(scores, dtype=np.float64) >= result["threshold"]`, after checking
`status`. Its reported count uses this comparison. A lower-precision NumPy
array can round a Python-float threshold downward during comparison.

The returned dictionary is safe for `json.dumps(result, allow_nan=False)`:

- `status`: `ok`, `insufficient_independent_calibration`, or
  `threshold_not_representable`. Only `ok` supplies a usable threshold.
- `threshold`: a finite Python float on success, otherwise `None`.
- `count`: actual accepted calibration count on success, otherwise `None`.
- `rank`: one-based **ascending** rank `n-j`, or `None` when infeasible.
- `allowed_count`: the selected `j`, or `None` when infeasible.
- `n`, `alpha`, `delta`: the sample count and requested nominal parameters.
- `nominal_violation_bound`: `B(n,j,alpha)` for the selected rank; `None` if
  infeasible. It still uses `j` when ties reduce `count`.
- `best_nominal_violation_bound`: `B(n,0,alpha)`, including on insufficiency.
- `order_statistic`: the finite selected boundary score, or `None` if infeasible.
- `decision_rule`: `score >= threshold`.
- `interpretation`: `nominal_iid_reference_only`.
- `assumptions`: explicit unverified independence, fixed-rule, continuity and
  clustered-data limitations.

No feasible rank returns `insufficient_independent_calibration`, not an infinite
or fabricated zero threshold. A selected boundary equal to float64's largest
finite value has no finite successor; `threshold_not_representable` retains its
rank and nominal bound but returns `threshold=None` and `count=None`. Callers
must check `status` before applying a threshold. `n` counts supplied scores; the
function does not infer or validate independent sampling units.

`nominal_violation_bound` returns the scalar `B` at a fixed allowed count. `n`
must be a positive integer and `allowed_count` an integer in `[0,n-1]`, excluding
booleans. This helper summarizes the implied nominal confidence `1-B`; it does not
inspect the data or adjust for dependence or selection.

The sample size, allowed count, and nominal targets must be chosen independently
of observed calibration outcomes for the reference calculation to apply.

### Primary reference and interpretation

[Tong, Feng and Li (2018), Algorithm 1 and Proposition 1](https://arxiv.org/pdf/1608.03109)
use held-out class-0 scores independent of fitting. Their ascending rank `k`
has violation bound `sum(i=k..n) choose(n,i)(1-alpha)^i alpha^(n-i)`. Substitution
`j=n-k` gives the binomial CDF above. Their strict `score > boundary` becomes
our float64 `>= nextafter(boundary,+infinity)` rule. Under independent,
identically distributed continuous scores and an independently fixed scoring
rule, the ideal calculation concerns
`P_calibration(population FAR(random threshold) > alpha) <= B <= delta`.
Rounding upward can be conservative. This is a repeated-sampling statement,
**not a posterior probability** that a fitted threshold is safe. The journal
reference is *Science Advances* 4(2), eaao1659,
[DOI 10.1126/sciadv.aao1659](https://doi.org/10.1126/sciadv.aao1659);
the preprint is [arXiv:1608.03109](https://arxiv.org/abs/1608.03109).

At `alpha=0.01`, the calculations give:

- `n=60`: best violation bound about `0.547157`; `n=196`: about `0.139476`.
  Both fail the nominal 90% request (`delta=0.1`).
- `n=230` and `240`: only `j=0` is feasible at 90%; ascending `rank=n` and no
  accepted calibration scores. Their bounds are about `0.099105` and `0.089629`.
- `n=298` fails 95%; `n=299` supports `j=0` with bound about `0.049536`.
- `n=496` permits `j=1` at 95%, with bound about `0.041109`, even if ties lead
  to fewer than one accepted calibration score.

These numerical examples assume independent observations for their nominal
interpretation; the current claim graph does not establish that assumption.

### Exact empirical comparison

`empirical_threshold` reproduces the frozen
`scripts.mentor_backend_study.calibrate` result at its default `target_far=0.01`.
It returns the same metrics: `threshold`, `far`, `frr`, `false_accepts`,
`false_rejects`, `n_genuine`, `n_impostor`, `decision_rule`, `target_far`, and
`selection="empirical_far_at_most_target"`.

The candidate thresholds are the finite reject-all successor of the maximum
score followed by all distinct scores in descending order. Ties enter together.
The function selects the last attainable grouped ROC point with empirical
FAR at most the target. This can be a genuine-only score group: with labels
`[1,0,1,0]` and scores `[0.9,0.8,0.8,0.1]`, the threshold is exactly `0.9`, not
the successor of the maximum impostor score `0.8`. Equality at `0.9` is accepted.
For other valid targets the same grouped rule applies. Like the original,
it raises `ValueError` if no finite reject-all successor exists, even when a
different candidate might otherwise be usable.

Exact result parity also retains the original `fixed_metrics` comparison
against the input score dtype. NumPy can round the Python-float reject-all
threshold downward for float16/float32 arrays; the original then reports
accepted ties despite selecting the reject-all ROC point. This inherited
behavior is covered by parity tests. Use float64 inputs for the intended
grouped decisions; the nominal selector always computes counts in float64.

No population FAR guarantee follows from its empirical FAR. For a fixed rule
and iid continuous impostor scores, the ideal strict maximum-order cutoff has
expected population tail probability `1/(n+1)`. That expectation does **not**
apply to the current grouped-ROC-selected threshold, and independence is not
established for the clustered claims. Nor does an expectation supply the
requested high-probability bound or a posterior claim.

Pooling scores from different fitted models does not establish independent,
identically distributed calibration scores for one fixed rule. It does not
produce CV+ or jackknife+ guarantees. A DEV-only pooled comparison remains
exploratory; reusing its results for model or threshold selection also requires
accounting for that selection before any independent inferential claim.

## Synthetic verification

From the project directory, use the existing shared environment with CPU
threads limited to one and bytecode writes disabled:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 UV_OFFLINE=1 \
uv run --no-project /home/sherlock/Code/github/VariableThe/.venv/bin/python -B \
  -m unittest discover -s tests -p test_tail_calibration.py -v
```

The suite checks the sample-size boundaries, independent binomial-sum and
uniform-order-statistic identities, maximal feasible ranks, confidence
monotonicity, rejected whole ties, float64 roundoff/extremes, strict validation,
JSON serialization, unchanged inputs, and exact original empirical parity on
synthetic continuous and tied scores. It reads no real dataset, CALIB or TEST
payloads and runs no fitted or GPU jobs.
