# Full-support Canonical362: retain the frozen baseline

**The sole full-support candidate F does not improve the registered support outcome and fails 14 of 33 unchanged replacement gates. Retain R (Canonical362 / auto WCCN / rawmean_multi / policy C).** H is a diagnostic aggregation control and cannot be nominated. This completed single attempt is exploratory evidence on reused DEV; it supplies no independent confirmation.

Full DEV raw-source/feature/model/score verification and independent concrete review have **passed**. Both retain R; the separate publication-layer review is pending. The compact replay below verifies cached decisions, thresholds, metrics and gates only. It cannot establish raw-feature extraction, model inference, the historical R numeric copies, or the original executable runtime from this portable bundle.

## Registered comparison and primary result

R preserves 100 frozen states and their cutoffs: standardize/WCCN the original 588 windows, unit window embeddings, then normalized trial means. H averages the same 588 window feature rows per historical trial before its own clean standardization/WCCN. F uses the unchanged Canonical362 extractor on 885 covers: 588 old tiles plus 297 deduplicated final endcaps. Exact multiplicity weights are `w_j = (1/L) sum_{r in W_j} 1/m(r)`, with deterministic float64 aggregation before its own clean transform.

F covers 183,128 valid rows in the same 299 historical trials, adding 32,600 rows over the old 150,528. No person, trial or claim is added. There is no joining across gaps/trials, resizing, padding or repeat-period/segmentation change. Twenty-nine short fragments (1,210 rows: 740 from historical trials and 470 from subject 19/day 2/trial 1) remain excluded; the latter recording stays outside the old claim graph. Window lengths stay 256; eligible segment lengths range 446–931.

The primary exact equal-fold mean `F clean FRR − H clean FRR` is **+133/8700**, or **+1.528736 percentage points**. Favoring support required a strictly negative difference and F clean FAR no greater than H and no greater than 1%. F FAR is 3/400; H FAR is 7/1200. The primary does **not** favor F. Replacement is a separate F-versus-R decision; every one of the 33 existing gates remains unchanged.

R→H changes aggregation and training units. H→F changes the full-support estimator, including endcaps and weights. F−R is not a pure tail-row effect. Means of nonlinear window features do not equal a uniform-row descriptor; overlap weighting establishes no independence.

## All seven conditions

Each cell is **FAR / FRR / EER (%)**, calculated from the exact fractions in [summary.json](summary.json). FAR and FRR are equal-weight means of 20 fold error fractions. EER is the equal-fold mean of exact rational interpolations of tied-score ROC endpoints, not an error count, pooled ROC, or approximate EER count. Display decimals are rounded only here. Exact AUC, every person's four-appearance means, claimant counts, and H−R/F−R differences are also in the summary.

| Condition | R | H | F |
| --- | ---: | ---: | ---: |
| clean | 0.500000 / 14.787356 / 5.876437 | 0.583333 / 15.126437 / 5.551724 | 0.750000 / 16.655172 / 6.551724 |
| rotation30 | 0.500000 / 14.787356 / 5.876437 | 0.583333 / 15.126437 / 5.551724 | 0.750000 / 16.655172 / 6.551724 |
| rotation90 | 0.500000 / 14.787356 / 5.876437 | 0.583333 / 15.126437 / 5.551724 | 0.750000 / 16.655172 / 6.551724 |
| gain09 | 0.000000 / 45.206897 / 7.129310 | 0.000000 / 83.057471 / 11.433908 | 0.000000 / 75.333333 / 9.956897 |
| gain11 | 0.166667 / 39.827586 / 6.451149 | 0.166667 / 70.201149 / 10.488506 | 0.000000 / 70.844828 / 8.652299 |
| noise01 | 0.250000 / 52.281609 / 7.304598 | 0.333333 / 46.591954 / 7.557471 | 0.083333 / 50.534483 / 8.316092 |
| noise05 | 0.000000 / 100.000000 / 33.902299 | 0.000000 / 100.000000 / 35.563218 | 0.000000 / 100.000000 / 35.002874 |

All arms retain 596 genuine and 1,192 impostor repeated outer claims **per condition**. Clean pooled counts are R 6/1,192 false accepts and 88/596 rejects, H 7/1,192 and 90/596, and F 9/1,192 and 99/596. These dependent pooled counts differ from equal-fold rates; they are not independent trials or a population FAR estimate.

Only day 2 probes are perturbed; day 1 enrollment and clean-calibrated cutoffs remain fixed. Both rotations preserve each arm's ordered clean acceptance decisions. The proper common 30°/90° frame transforms do not model physical phone displacement, dynamic rotation, altered motion, or independent sensor misalignment. Gain 0.9/1.1 and Gaussian noise 1%/5% are engineering sensitivity probes, not measured phone perturbations. Severe noise rejects every genuine claim; low FAR with 100% FRR is unusable.

The new noise field is generated once per complete physical source row/channel and reused across covers, arms, folds and both noise magnitudes (seed 20261025). Its per-fold dynamic triad RMS uses only the twelve TRAIN identities' clean historical-window subset, on both days, in native SI acceleration / deg/s angular velocity units. Tails and held probes cannot fit RMS; inner nine-person fits/calibration stay clean. The old WINDOW-keyed noise cache is a different realization and never supplies paired gate truth. Its payload is omitted; its frozen provenance and diagnostic-only scope remain in the R restore record.

## People and claimants

Clean entries below are repeated reject count / genuine appearances, followed by false accepts targeting that claimant / impostor appearances. Subject 19 has 36 genuine appearances; all others have 40. The summary retains all seven conditions, person EER/AUC, directed claim pairs and exact rates.

| DEV ID | R rejects; claimant FA | H rejects; claimant FA | F rejects; claimant FA |
| --- | ---: | ---: | ---: |
| 1 | 1/40; 0/80 | 3/40; 0/80 | 4/40; 0/80 |
| 3 | 3/40; 0/80 | 3/40; 0/80 | 1/40; 0/80 |
| 5 | 17/40; 0/79 | 21/40; 0/79 | 20/40; 0/79 |
| 7 | 0/40; 0/79 | 0/40; 0/79 | 0/40; 0/79 |
| 9 | 40/40; 1/79 | 40/40; 1/79 | 40/40; 0/79 |
| 11 | 1/40; 0/79 | 5/40; 0/79 | 8/40; 0/79 |
| 13 | 13/40; 0/80 | 6/40; 1/80 | 5/40; 0/80 |
| 14 | 10/40; 0/78 | 8/40; 0/78 | 7/40; 0/78 |
| 16 | 0/40; 0/80 | 0/40; 1/80 | 1/40; 1/80 |
| 17 | 2/40; 0/79 | 1/40; 0/79 | 4/40; 1/79 |
| 19 | 0/36; 0/80 | 0/36; 0/80 | 2/36; 0/80 |
| 24 | 0/40; 5/80 | 0/40; 2/80 | 0/40; 5/80 |
| 25 | 0/40; 0/79 | 3/40; 0/79 | 0/40; 2/79 |
| 27 | 0/40; 0/80 | 0/40; 1/80 | 0/40; 0/80 |
| 29 | 1/40; 0/80 | 0/40; 1/80 | 7/40; 0/80 |

Person 9 remains rejected in all 40 clean appearances in every arm. Claimant 24 receives five clean false accepts in R and F, and two in H. Other claimant increases still matter: aggregate FAR cannot establish uniform claimant protection.

## Replacement failures (14/33)

- `clean_at_least_10_people_no_worse`
- `clean_each_claimant_false_accepts_no_worse`
- `clean_eer_no_worse`
- `clean_far_no_worse_and_at_most_1pct`
- `clean_frr_no_worse`
- `clean_no_person_more_than_one_extra_reject`
- `clean_repeated_rejects_no_worse`
- `noise01_each_claimant_false_accepts_no_worse`
- `robust_at_least_10_people_no_worse`
- `robust_each_condition_frr_no_worse`
- `robust_mean_frr_improves_at_least_5pp`
- `robust_no_person_worsens_more_than_5pp`
- `rotation30_each_claimant_false_accepts_no_worse`
- `rotation90_each_claimant_false_accepts_no_worse`

The complete gate booleans and exact objectives are in `summary.replacement`; the unchanged [gate specification](docs/robustness_selection.md) and pure [authority source](authority/engine/robustness_selection.py) are included. The objective averages gain0.9, gain1.1 and noise1% FRR and requires at least a five-point improvement, with the existing clean/person/EER/claimant/FAR-tail/rotation safety gates. Severe noise remains a reported safety endpoint. There is no H fallback and no relaxation after seeing outcomes.

## Execution and review boundary

One completed full-support attempt used 200 actual new H/F fits (160 inner nine-person fits, 40 twelve-person refits), 100 fit-free R state restores, 160 new inner score sets, 280 new outer score sets, 80 R clean inner replays and 140 matched R outer sets. Both own H/F pooled policy-C cutoffs were frozen before either refit in each fold. Inner training has 179/180 trial units; refits have 239/240. The original R cutoff is preserved. There is no threshold transfer or stress recalibration. Preflight records 11,373 required windows across 121 contexts, zero invalid windows and zero invalid scales before candidate fits; these are diagnostics, not learned quality or walking gates.

The fixed one-attempt budget was 1,800 process CPU seconds / 3,600 bounded wall seconds, one CPU and numerical thread, CUDA hidden, offline and no bytecode writes. The root's whole-launch measurement is 459.43 user + 5.81 system = **465.24 CPU seconds**, **7:59.90 wall**. The narrower worker interval in results is a different scope. No speed improvement is claimed. The separate full-verification launch measured 371.88 user + 3.47 system = **375.35 CPU seconds**, **6:31.19 wall**, with zero fits. This is whole-command timing, not inference throughput. No automatic retry, parameter change or budget expansion occurred. The unrelated STOPPED N446 failure (4/960 at P200 phase 2/18) retains its final `STOP_NO_RETRY_NO_PARAMETER_CHANGE` disposition.

R1 systems review fixed P1 (transitive executable/cache closure) and P2 (explicit immutable-original-R numeric/metadata bindings); a 158-file R1 snapshot was preserved. R2 systems review was CLEAR with 2,865 loaded module paths, 8,656 package executable paths, 190 mapped natives and 30 real-worker tamper checks, as reported by root. The exact [fix record](snapshot/review_fixes.json.gz) and preparation history remain included. The completed full DEV verification and independent concrete-run review also pass; their exact evidence hashes and scopes are recorded in [review-status.json](review-status.json).

## Portable cached replay

The bundle contains **all 660 current lossless int64-label/float64-score NPZs**, 40 H/F and 20 R cutoffs, complete cached scoring metadata/results, registrations, source-row/cover/claim/fit/RMS ledgers, all named preflight diagnostics, geometry/feature inventories, complete preparation Python/shell/vendor sources, public metadata inputs, original source/runtime/protocol hash manifests and exact original artifact indexes. Large JSON is gzip-compressed with `mtime=0`; its decompressed bytes retain the original hashes. [scope.json](scope.json) enumerates every copied and omitted original path, byte size and digest. The exact [publication inventory](inventory.json) excludes only itself.

Raw source segments, physical-field/sample arrays, raw-feature/geometry arrays, 300 R/H/F learned model-state NPZs, TRAIN-RMS NPZs, runtime binaries/caches and the 140 historical outer score payloads are omitted. Original R metadata, thresholds, historical cache assignments/hashes and nonnoise metadata parity records remain; historical array equality is a recorded full-run obligation, not independently re-established by this helper. No payload is quantized. Original numerics remain at workspace-local `artifacts/full-support-canonical-v1-run-r1` and original R `artifacts/canonical-orientation-v1`, with preparation at `artifacts/full-support-canonical-v1`; these location hints are not portable file links or bundled copies.

This helper performs **complete cached-decision/threshold/metric/gate replay**, not standalone raw-feature/model inference or full-runtime verification. It rebuilds all ordered claim graphs/labels and denominators, recomputes policy C from 240 inner caches, uses all 420 outer caches at frozen cutoffs, checks ordered decisions, exact FAR/FRR/EER/AUC/person/claimant/pair metrics, primary/secondary and all 33 gates. Frozen metadata and original indexes pin cache maps and types even if a local publication inventory is rehashed. It rejects missing, extra, tampered, symlinked or unsafe paths, dtype/shape/key/type/signed-zero changes and existing output directories. The verifier and its external inventory digest are the trust boundary; replacing both code and trusted external anchors is outside that guarantee.

Calculations reuse unchanged pure frozen vendor functions plus exact extracted kernel functions; this is production-authority reuse, **not independent replication**. It imports no prior study entrypoint, opens no sensor archive/model/raw-feature NPZ, fits nothing and does not rerun the study. Source snapshots are preserved evidence and are not executed. Failed helper outputs remain consumed. Each repository mirror has its own cached-replay command and external receipt; both mirrors represent the same experiment. The helper reports full inference and external review as outside its own scope, independently of the recorded successful root checks.

Use the existing shared Python 3.14.8 / NumPy 2.5.4 environment (full pinned requirements and lock are under `environment/`; no install is needed here). Replace the interpreter, CPU and output paths as appropriate. Choose an available single CPU and a fresh absolute output outside this report directory:

```sh
env UV_OFFLINE=1 UV_PYTHON_DOWNLOADS=never PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 OMP_THREAD_LIMIT=1 \
  taskset -c 0 uv run --no-project --offline --python ../.venv/bin/python \
  python -B reports/2026-10-25-full-support/verify_cached_decisions.py \
  --out /absolute/path/to/new-cached-replay
```

`--bundle /absolute/report/path` is optional and defaults to the helper's directory. Outputs are strict `verification.json`, all replayed `rows.json`, ordered `decisions.json`, and an exact self-excluded inventory outside the bundle. `test_cached_decisions.py --out /absolute/new/test/path` runs only tiny synthetic and refusal checks. Full raw/model inference needs the retained complete numeric bundle and frozen runtime, or a separately registered reproduction with the authentic dataset, original R states and original protocol prerequisites; do not treat this portable cache replay as that reproduction.

## Provenance and scientific limits

Dataset: [Santos multi-sensor gait v7](https://doi.org/10.6084/m9.figshare.14727231.v7), by **Geise Santos, Marcelo M. Wanderley, Tiago Tavares and Anderson Rocha**, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The frozen [public provenance](docs/source_provenance.json), metadata member assertions and prior official-authenticity evidence are carried. Cache integrity alone does not re-establish raw authenticity. Only exact allowlisted DEV MCU/processed imu_walk payloads were allowed during the run; no original CALIB/TEST payload, archive testzip, whole-archive hash or unselected-member hashing belongs to this attempt.

[IDNet §6](https://arxiv.org/html/1606.03238v3#S6) motivates evidence accumulation; it does not prove this feature-mean estimator. [Hatch et al., WCCN §2](https://www.isca-archive.org/interspeech_2006/hatch06_interspeech.pdf) motivates TRAIN covariance. [Bolle et al.](https://research.ibm.com/publications/an-evaluation-of-error-confidence-interval-estimation-methods) discusses dependent repeated biometric claims. This keeps Canonical362 and the existing metric/scoring rules; it is not a full IDNet system replication. Fifteen reused DEV identities, overlapping windows and twenty dependent folds give no iid FAR guarantee, independent confirmation, SOTA or deployment claim. Previously inspected TEST cannot become fresh confirmation, and its historical **3/200 = 1.5% FAR** is unchanged.

Permanent prior context: [canonical orientation](../2026-10-25-canonical/README.md), [nuisance robustness](../2026-10-25-nuisance/README.md), [continuity support inventory](../2026-10-25-support/README.md), and [official authenticity](../2026-10-23-cross-day/README.md). Historical canonical source is [108a559bc1a83355dc85335e9f640786846f894a](https://github.com/VariableThe/gait-biometrics/commit/108a559bc1a83355dc85335e9f640786846f894a). Later validation edits mean no original runtime-closure equivalence is asserted; R clean replay must satisfy inherited 1e-6 score tolerance and identical decisions/counts. All original source, run, freeze and result anchors remain unchanged.
