# Santos DEV continuity/support inventory

`santos_dev_continuity_support_inventory_v1` is one newly registered,
metadata-only inventory. It does not select a representation, run a model,
evaluate scores, or authorize materialization. Implementation and synthetic
tests are in [the script](../scripts/santos_support_inventory.py) and
[the tests](../tests/test_santos_support_inventory.py). All three files form its
source closure. Historical artifacts and source are left unchanged.

Implementation/schema inspection is separate from running this inventory.
**An independent root review must precede the actual public-metadata job.**
No actual support distribution was computed during implementation. Inspection
found the required source-coverage, fold-role and graph fields in the public
schema; their full content has not been validated by an actual inventory run.
Any missing necessary fact makes the code stop, with no raw-data fallback.

## Inputs and access boundary

The intended inputs are the frozen public JSON files:

* `reports/2026-10-25-canonical/evidence/development.json.gz`
* `reports/2026-10-25-canonical/evidence/registration.json.gz`

The former was saved before feature extraction/fits and includes complete
`source_coverage`, including zero-window source recordings. The latter freezes
the original cohort order and all twenty outer folds, their four nine-person
inner TRAIN owners and twelve-person refit TRAIN owners. Neither old runners
nor `data.santos` are imported. Imports are Python standard library only.

The fixed sorted cohort is `[1,3,5,7,9,11,13,14,16,17,19,24,25,27,29]`.
Historical registration retains its original unsorted cohort order. Every
source member must match exactly:

```text
A multi-sensor human gait dataset/processed_data/userXX/capture_userXX_dayD_TTTT_imu_walk.csv
```

Directory and filename IDs must agree; `D` is 1 or 2, and `TTTT` is `0001`
through `0010`. This is validation of a **metadata string**. No member path or
historical `archive_path` value is ever opened. There is no archive argument,
archive parser, dataset access, signal statistic, filtering, cycle extraction,
spline, learned quality rule, feature extractor, model, score or fit path. Raw
member read budget and fit budget are both zero. The only filesystem reads are
the three source-closure files, explicit metadata JSON inputs and saved attempt
JSON files. JSON inputs must be nonsymlink regular `.json` or `.json.gz` files.

## Three definitions, frozen before viewing support results

Each valid continuity segment has original half-open row bounds `[a,b)`, length
`L=b-a`, original inclusive frame endpoints, and preserved boundary reasons.
The original segment metadata, recording key, member string/hash, continuity
breaks and invalid offsets remain in the output.

| Definition | Units | Unique original row coverage | Unused valid rows |
| --- | --- | --- | --- |
| `historical256` | `floor(L/256)` tiles, anchor `a`, stride 256 | `256*floor(L/256)` | `L mod 256` |
| `fixed768` | `floor(L/768)` blocks, anchor `a`, stride 768 | `768*floor(L/768)` | `L mod 768` |
| `complete_segment` | One conceptual unit for the entire nonempty valid segment | `L` | 0 |

Every 768 block records the indices of its three exact historical 256 tiles.
Segments, trials and days are never joined. An 800-row recording split into
400+400 supports zero 768 blocks; an unbroken 800-row segment supports one.
Whole segments retain valid tails unused by historical tiling. A one-row
segment is one conceptual unit, with its length eligibility recorded separately;
it does not establish a trainable feature or covariance. One long segment can
also be only one conceptual unit despite ample samples. Counts by day expose
this even when two-day totals meet the proposed count gate.

Only `<256`, `256..767`, and `>=768` bands are reported. Exact segment lengths
and fixed minimum/median/maximum summaries are retained; there is no length
sweep. Medians use exact numerator/denominator pairs. Empty groups have null
extrema and median.

`plan()` freezes flags, definitions, budgets and interpretations before parsing
the input metadata:

* Per definition, record each identity's supported day1 enrollment trials and
  day2 probe trials. `some_identity_support` means at least one of the fixed
  fifteen has support on both days; `all_identity_support` requires all fifteen.
* `complete_historical_trial_support` requires support for **every original
  historical eligible trial**, with per-identity flags. It does not discard a
  trial that fails under another definition.
* For every frozen nine-person inner and twelve-person refit TRAIN owner,
  retain ordered per-identity day counts, total units, historical-graph units,
  row coverage and unused rows. `proposed_min_two_units` requires at least two
  units for that identity over both days. `all_min_two_units` combines those
  flags for that owner. This is a conservative proposed WCCN count gate, with
  no proof of covariance rank, independence, extraction or fitting feasibility.
* Complete-segment flags describe raw support only. No extraction eligibility
  claim is made. Row-count eligibility at 256/768 is separately labeled and
  does not introduce an extraction policy.

The all-fifteen cohort and twenty outer folds remain fixed. None of these flags
permits cohort reduction, adaptive protocol choice or promotion.

## Original graph and integrity projection

The authoritative graph is `development.coverage`; its original arrays, counts,
trial dictionaries and ordering are copied into `original_graphs`. Fold roles
are copied from `registration.plan.folds`. No random generator is used.

To validate that those graphs match the historical source metadata, the script
implements the following explicit projection of the inspected source:

1. In `data/santos.py`, `load_santos_windows` orders recordings numerically by
   subject/day/trial. `_parse_recording` orders segments and historical windows
   within each segment, restarting the 256 stride at each segment start. A trial
   enters the historical graph iff its saved historical window count is positive.
2. `scripts/crossfit_threshold_study.py:expected_graph` sorts a role's identity
   IDs numerically. For each probe identity, it sorts eligible day2 trials, then
   emits every claim identity in sorted order. It records integer labels,
   `claim_subject_ids`, `probe_subject_ids`, `probe_trial_ids`, both-day trial
   dictionaries and genuine/impostor counts. These are checked in full/order.
3. `pooled_graph` concatenates the four inner arrays in frozen inner order and
   sorts the unique directed impostor-pair strings lexically. The registration's
   four five-fold partitions, next-triple inner order, nine/twelve TRAIN sets
   and original TRAIN identity ordering are validated without regenerating
   any random splits.
4. `scripts/claimant_normalization_study.py:coverage_plan` supplies the stored
   `inner`, `pool`, `validation`, `cohort_inner` and `cohort_refit` graphs used
   by the canonical study. All are necessary; missing graphs stop the job.

These sources are read for implementation understanding, never imported or
executed by the inventory. Reconstruction validates the saved graph; it does
not replace it with a new graph. Source-present trials absent from the original
graph have a separate ledger and do not enter its claims. Missing trials are
the complement of the exact official DEV subject/day/trial domain **only after
that complement matches the historical explicit `missing_recordings` ledger**.
There is no inference that an absent metadata entry was a missing raw member.

Each original graph gets support vectors in the same claim order. They record
whether the original probe has units and whether any/all original enrollment
trials have units. Failed original trial rows are explicit. The any/all vectors
are raw-support diagnostics, not an enrollment weighting, cancellation or
partial-failure decision. Original genuine/impostor denominators stay intact.

Available integrity links checked without reading their referenced payloads:

* `development.registration_sha256` must equal SHA-256 of the **decompressed
  historical registration bytes**, including its original newline. Exact input
  hashes separately cover compressed bytes when inputs are gzip files.
* The ordered seven-field member manifest is hashed using historical canonical
  JSON and must match both `source_coverage.members_sha256` and
  `data.members_sha256`. The five-field `data.members` projection must match.
* If present, `geometry.clean.row_keys_sha256`, its columns and row count are
  checked against the reconstructed historical ordered `(subject,day,trial,start)`
  metadata keys. The historical algorithm prefixes canonical `['int64',[N,4]]`
  and hashes little-endian signed int64 bytes. The little-endian interpretation
  is the explicit historical Linux array projection, not a signal read.

Historical member payload hashes, data/window signal fingerprints, feature,
model, score and geometry artifact hashes cannot be reverified with a zero raw
or artifact read budget. They remain historical assertions; the output labels
this limit. Source hash entries in the historical registration are preserved
by the saved input binding, not checked against potentially changed local source.
The **new** three-file source closure is checked against the current files.

Validation rejects duplicate keys/identities, ambiguous paths, missing fields,
nonfinite numbers, float-valued integers, booleans in integer fields, signed
integer `-0`, invalid signed int64 frames, inconsistent frame lengths, invalid
or overlapping row intervals, incomplete row partitions, unexplained splits,
incorrect break reasons/order, and inconsistent counts/coverage/exclusions.
Every JSON fingerprint first requires string keys and a type-preserving JSON
roundtrip. `1`, `1.0`, `true`, `0.0` and `-0.0` retain their types/sign; no key
coercion or numerical truncation is accepted. Validation and exact comparison
walk the trees directly, with auxiliary storage bounded by nesting depth;
they do not construct full parallel type-signature trees. Canonical hashing
keeps the original sorted, compact JSON byte encoding and explicitly compares
one decoded roundtrip against the original, including float hexadecimal values
for signed zero. Direct equivalence ignores object key order only and avoids
serializing two entire trees.

The public `length_summary()` helper also rejects any nonpositive or nonexact
integer length, even when called outside validated derivation. An empty
enrollment list makes both claim raw-support flags false. Registered graphs
already require nonempty eligible trials on both days; this helper guard does
not change accepted graphs, denominators or enrollment policy.

The predeclared 588 historical windows, 28 continuity breaks and 33,810 dropped
tail rows are reconciliation references from the brief. The metadata job must
match them exactly. That comparison is explicitly **metadata reconciliation**,
not raw verification. No actual reconciliation outcome is reported here.

## CLI, attempts and saved result schema

After root review, run from the project with its shared uv Python environment:

```sh
uv run --no-project --offline python scripts/santos_support_inventory.py \
  --stage run \
  --historical-registration-json reports/2026-10-25-canonical/evidence/registration.json.gz \
  --development-metadata-json reports/2026-10-25-canonical/evidence/development.json.gz \
  --out-root /absolute/path/to/a-new-inventory-attempt

uv run --no-project --offline python scripts/santos_support_inventory.py \
  --stage verify --out-root /absolute/path/to/a-new-inventory-attempt
```

`run(args)` and `verify(args)` expose the same Python API. `plan()`,
`canonical()`, `fingerprint()`, `segment_support()`, `length_summary()`,
`validate_metadata()` and `derive_inventory()` are metadata helpers. The pure
derivation helper does not reserve an attempt itself; real jobs must use `run`.
The command accepts no archive input and no cohort, fold, length, minimum-unit
or budget overrides. Verify accepts only the output root.

The Linux process is restricted to one available CPU, CUDA is hidden, common
thread counts are set to one, CPU time is at most 60 seconds, and address space
is at most 512 MiB. A soft CPU warning at 59 seconds permits a failure diagnostic
before the 60-second hard stop. Existing stricter limits are respected. Frozen file/decoded
JSON limits are 32/64 MiB and each definition is bounded to 100,000 units.
Budget failure cannot trigger expansion or an automatic retry.

The root must not already exist, even if empty. `attempt.json` reserves it;
failed attempts remain consumed. All writes use exclusive creation. After
hashing source files and exact input bytes, `registration.json` is written
**before any input JSON parsing, metadata validation or support derivation**.
The input bytes are then saved as `historical_registration.input.json[.gz]`
and `development_metadata.input.json[.gz]`. Input failures before a complete
registration retain an explicitly incomplete registration with only available
hashes and failure diagnostics; nothing is parsed or derived. No file is overwritten.

Successful attempts retain:

| File/key | Contract |
| --- | --- |
| `registration.json` | Full plan/flags/budgets, Python runtime, three source SHA-256 values, exact input sizes/hashes/paths/snapshot assignments |
| `result.json` | Exactly `registration_sha256`, `inputs_sha256`, and an `inventory` object |
| `diagnostics.json` | Complete/failed-consumed status; completion binds registration/result byte hashes; failure retains phase and exception |
| `inventory.original_windows` | Historical ordered keys, row/frame bounds, segment and recording indices |
| `inventory.original_recording_order` | Original ordered source-trial keys/member paths, including zero-window recordings |
| `inventory.segments` | Exact original segment metadata, boundary reasons, bands, length eligibility and all three supports |
| `inventory.trials` | All 300 frozen-domain trial slots with explicit missing/zero units, fragmentation, invalid offsets, breaks and support totals |
| `inventory.subject_days` | Both days for all fifteen identities, trial presence/absence/zero/fragmentation and support totals |
| `inventory.folds` | Same twenty folds, nine/twelve TRAIN-owner counts/coverage/gates and original graph support/failure vectors |
| `inventory.source_present_absent_original_graph` | Source trials excluded by historical zero-window tiling; never graph expansion |
| `inventory.subject19_day2_trial1` | Explicit inventory row for the requested recording slot, including missing status if historically recorded missing |
| `inventory.flags`, `totals`, `length_summary`, `length_bands` | Only the predeclared support flags and summaries |
| `inventory.metadata_reconciliation`, `integrity_limits` | Metadata-only checks and explicit unverified signal/member claims |

Unique row coverage is summed only over disjoint original trial/segment row
domains inside an owner. Overlapping fold/owner totals must not be pooled as
unique dataset rows. Verify reads saved snapshots, checks current source/runtime
and plan **before deriving**, checks every binding/result byte hash, then
rejects extra/missing result keys or wrong binding/inventory types before
loading the saved historical inputs. It deterministically reconstructs and
compares the entire three-key result object, including the inventory. It never fits
or reads the original input paths. A changed result is refused, including when
its diagnostic hash is also updated. This is reproducibility binding, not a
cryptographic signature or external proof against replacement of every file.

## Synthetic validation and scientific limits

The stdlib `unittest` suite uses synthetic JSON and temporary attempt roots.
It covers 400+400 versus 800, short segments/tail recovery, original graph
failures/newly source-present trials, exact cohort/fold/order, malformed
identities/paths/hashes/frames/partitions/counts, exact JSON numeric/key behavior,
registration-before-parse/derive, import/read prohibition spies, source/runtime
changes, consumed failures, historical hash links, and tampered inputs/results.
Result tampering tests recompute the diagnostic result hash, including for
unknown top-level keys and type-only changes, so replay must independently
reject them. Canonical bytes/fingerprints have literal expected-byte cases;
direct equivalence is checked independently for booleans, integers, large
integers, floats, signed zero, structure and object key order. Hashing must
reject a type/value change in its explicit decoded roundtrip.

A persistent full-domain synthetic fixture contains all 300 recording slots
(15 identities, two days, ten trials), 328 segments, 588 historical windows,
28 continuity breaks and 33,810 historical tail rows. Recording indices 0..27
start with segment lengths `[256,256]`, 28..287 with `[512]`, and 288..299 with
`[256]`; the first 26 segments globally receive 104 extra rows and every other
segment receives 103. Its manifests, coverage, row-key hashes and complete
twenty-fold graphs use trials 1..10. API run/replay and separate fresh-process
CLI run/replay must succeed with actual 60-second CPU, 512-MiB address-space,
one-CPU and CUDA-off restrictions, without mocked limits or reconciliation
constants. Replay deletes the original synthetic input files first. These
temporary synthetic attempts say nothing about the public support distribution.
Use the shared Python with no installs, plus pinned Ruff 0.17.1 and ty 0.0.90.

Real administrative ceilings are applied only in fresh isolated interpreters,
including the full-domain API run/replay worker. The CLI worker starts separate
fresh CLI run and saved-only replay processes, which inherit the worker's exact
59/60-second soft/hard CPU limits, 512-MiB soft/hard address-space limits,
one-CPU affinity and CUDA-off environment and enforce production limits again.
These workers check the actual OS settings without mocking enforcement or the
reconciliation constants. Behavior/schema tests mock only `enforce_limits()`;
a dedicated test checks that both production `run()` and `verify()` call it.
The limiter unit test mocks OS calls and restores its environment, and the
in-process CLI parser test isolates its signal-handler registration.

A persistent integration regression reserves 768 MiB of virtual address space
in a separate parent interpreter to model an already large scientific test
runner, without reading data or importing scientific dependencies. It runs the
entire contract suite except the integration wrapper itself, checks actual
CPU/address-space limits, CPU affinity, the complete environment and the CPU
signal handler before and after every test and the suite, then allocates more
memory to exercise later work. The reservation uses sparse anonymous memory;
it does not commit 768 MiB of physical memory. Real-limit workers start by exec
and do not inherit that mapping. No test hard-caps the shared suite interpreter.

The research brief's new 256/768 synthetic comparison at 5,120 rows and
outcome-informed twelve-fixture plateau regression clear a numerical issue
only. They are not new DEV cycle-fit or full-grid plateau evidence. The prior
canonical study selected Canonical362; matched canonical comparison requires
256-sample input. A 768-sample definition can exceed available continuous
recordings, motivating this inventory before any whole-segment choice.
[IDNet, arXiv:1606.03238, section 3](https://arxiv.org/abs/1606.03238) and
Derawi (2010) are motivation identified in the brief, not reproduction claims.

A future explicit protocol may authorize old raw tails. This job does not.
Before any future materialization, freeze cycle rules, ties, aggregation,
partial-failure weights, cancellation handling, quality policy learned only
from each owner's TRAIN identities, policy C and its separate cutoffs.
Invalid claims must retain `scores=None`, exact gated ROC accounting and the
original denominators. No inventory flag establishes extraction feasibility,
permission to fit, or a protocol nomination.
