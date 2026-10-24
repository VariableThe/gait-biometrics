# Exploratory frozen-cohort score normalization

`engine.cohort_normalization.cohort_normalized_scores` is a NumPy-only plain
S-norm backend. It fits no representation and learns no threshold. Supply
authentic, frozen **DEV-train IDs only** as the normalization cohort, disjoint
from every target ID. The API cannot verify split or recording provenance.
Previously inspected CALIB/TEST data remain inspected; this follow-up is
exploratory. Do not fit or select cohorts, methods, floors or thresholds using
TEST, or use CALIB to fit the representation/cohort.

```python
from engine.cohort_normalization import cohort_normalized_scores

labels, scores, metadata = cohort_normalized_scores(
    embeddings, subjects, days, trials,
    cohort_embeddings=dev_train_embeddings,
    cohort_subjects=dev_train_subjects,
    cohort_days=dev_train_days,
    cohort_trials=dev_train_trials,
    template_method="multi", std_floor=1e-3,
)
```

The target and cohort each pass through `engine.template_scoring.score_trials`
with the same selected `mean`, `robust` or `multi` method. This supplies actual
day1 prototype vectors and baseline eligibility. Each set needs at least two
eligible identities. Every supplied cohort ID must remain eligible on both
days; excluded cohort IDs cause an error. Target exclusions and coverage follow
the baseline exactly. Trials need not cover all ten trial IDs. Input shape,
integer labels, trial/day restrictions and cancellation behavior follow the
baseline. Target and cohort embedding dimensions must match.

Windows are normalized individually. Their arithmetic mean per
subject/day/trial is normalized again, using the baseline helpers. Every
eligible target day2 trial is a probe. No target day2 values enter a template.
Raw scores are clipped cosine minus one, taking maximum prototype cosine for
`multi`. For a target claim template `e`, score all eligible cohort day2 trial
vectors against `e` to obtain population mean `mu_e` and standard deviation
`std_e`. Every trial has equal weight, including multiple trials per cohort ID.
For target probe `p`, score it against each cohort identity's day1 template to
obtain `mu_p`, `std_p`, with one score per identity. Then:

```text
normalized = 0.5 * ((raw - mu_e) / max(std_e, std_floor)
                  + (raw - mu_p) / max(std_p, std_floor))
```

`std_floor` is a positive finite float64-compatible scalar, default `1e-3`;
standard deviations use `ddof=0`. Nonfinite inputs and arithmetic overflow
raise `ValueError`. Finite extreme windows retain the baseline's safe cosine
direction. A degenerate cohort distribution uses the declared floor rather
than estimating a threshold. There is no top-k cohort selection.

Output labels are int64 and scores float64. Labels, score order,
`claim_subject_ids`, `probe_subject_ids`, `probe_trial_ids`, `coverage`, and
target `templates` match the base scorer for the selected method. Metadata
includes aligned `raw_scores`, a distinct protocol and calibration candidate,
and `normalization` containing frozen sorted cohort IDs, actual cohort
templates/coverage, cohort probe order, keyed Z/T statistics, population sizes,
raw standard deviations, effective denominators and floor indicators. All
metadata serialize with `json.dumps(metadata, allow_nan=False)`.
`cohort_frozen` describes using the full supplied cohort unchanged within this
call; the caller must freeze it across evaluation calls and record provenance.

T-norm uses the current probe deterministically for its own statistics. It
does not pool target probes or adapt enrollment. Provided eligibility stays
unchanged, changing one probe cannot change another probe's score, target
prototypes, cohort prototypes or Z statistics. The graph remains directed:
different enrollment/probe aggregation and cohort day populations do not imply
invariance to swapping enrollment and probe. Scores have no cosine-minus-one
range. Obtain a separate threshold
on the authorized calibration split for each frozen backend, floor, template
method and cohort configuration. No improvement in FRR or deployment FAR is
established by synthetic checks, especially with a small cohort such as 12 IDs.

## Primary research

Matějka, Novotný, Plchot, Burget, Diez Sánchez and Černocký (2017),
*Analysis of Score Normalization in Multilingual Speaker Recognition*,
Interspeech, pp. 1567–1571, DOI
[10.21437/Interspeech.2017-803](https://doi.org/10.21437/Interspeech.2017-803).
The [primary paper](https://www.isca-archive.org/interspeech_2017/matejka17_interspeech.pdf),
§2.1–2.2, equations (1)–(4), defines the enrollment/test cohort statistics;
§2.4, equation (5), p. 1568, averages Z-norm and T-norm. Section 2.5,
equations (7)–(8), distinguishes adaptive S-norm variants that select top
scoring cohort members. This implementation uses the entire supplied cohort,
so it is **plain S-norm**, not either adaptive variant. The use of gait
day1 templates and day2 trial vectors is this repository's adaptation;
the paper's speaker-recognition results do not establish gait performance.

Shum, Dehak, Dehak and Glass (2010), *Unsupervised Speaker Adaptation based
on the Cosine Similarity for Text-Independent Speaker Verification*, Odyssey,
pp. 76–82, [primary paper](https://www.isca-archive.org/odyssey_2010/shum10_odyssey.pdf),
§4.3, equation (22), p. 79, describes S-norm using a sum of standardized
enrollment/test scores and a union of impostor utterances/models. The factor
`0.5` here follows Matějka et al. equation (5) rather than that sum convention.
We implement the requested two day-specific cohort populations and no
unsupervised model adaptation from Shum et al.

## Tiny synthetic verification

The 14 tests in `tests/test_cohort_normalization.py` import no Torch and cover
the manual formula for every template method, population/trial weights, graph
and coverage parity, probe isolation, row permutation, no input mutation,
disjoint/all-days cohort checks, finite extreme windows, floors and overflow.
Run only this test file with the prescribed interpreter and thread limits:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/home/sherlock/Code/github/VariableThe/artifacts/tmp \
uv run --no-project --python ../.venv/bin/python \
python -m unittest discover -s tests -p test_cohort_normalization.py -v
```
