# Prepared N446 addendum

**Preparation only; post previous outcomes, adaptive exploratory.** No full-panel measurement or freeze is authorized for the implementation agent. See BRIEF.md and manifest.proposed.json for the exact sole candidate, fixed1440 panel, gates and interpretation.

Run commands from `/home/sherlock/Code/github/VariableThe` with the existing shared `.venv`. The verbatim reviewed launcher uses uv offline, Python -B, no installs, one CPU/numerical thread and CUDAoff. All console logs belong outside this artifact tree.

```sh
artifacts/native-waveform-feasibility-r1/run.sh prepare
artifacts/native-waveform-feasibility-r1/run.sh check-preparation
```

ROOT future stages after independent review only:

```sh
artifacts/native-waveform-feasibility-r1/run.sh freeze
artifacts/native-waveform-feasibility-r1/run.sh execute
artifacts/native-waveform-feasibility-r1/run.sh verify
```

`verify --replay-correlations` is also accepted; both verify forms perform full correlation replay. Stage must come first; unknown arguments fail before Python imports. An execute attempt is permanently consumed even on import/validation failure. The shell reservation and exclusive worker receipt are created before dependency imports/validation. A dead/reused launcher PID is insufficient: starttime and live ancestry must match; receipt exclusive creation prevents reuse by the same live ancestor. Direct Python execute consumes the attempt and is rejected because it cannot cover uv startup. Do not remove receipts or retry.

API: `panel.keys()` returns metadata only; `panel.generate(item, stress_cache)` is future measurement only. `kernel.extract(item,raw)` returns `(record,array_bundle,mapping)` for 6x446. `kernel.decide()` is the saved-evidence decision API. `evidence.replay()` recomputes correlations, stats/masks, plateau and diagnostic strict boundaries, linear mapping, features, decisions and full gates without generator/filter calls. It uses saved filtered arrays; it does not independently authenticate or regenerate the filter result. `driver.prefix_gate()` is a future pre-candidate full generator prefix check; it does not reproduce old outcome controls or access any parent measurements.

`copy_provenance.json` binds every copied function, complete inert generator SOURCE, R3 launcher, entry guard prefix, parent SOURCE hashes and reviewed stress BRIEF. `legacy.py` deliberately exposes only the copied generator/control helpers; the old cubic feature extractor exists only in inert `legacy_parent.source`, never imported. `bootstrap.py` is a one-time construction script and refuses existing output files. No measurement entrypoint from any parent is imported/executed.

Preparation writes proposed manifest, metadata keys/schema, source/parent/runtime indexes, selftests with source-bound history, preparation receipt and exact artifact index. Histories snapshot source before each check and retain failures. `check-preparation` is read-only. A changed runtime, source or parent requires explicit review; do not repair the index to conceal a mismatch.

Future execution saves one row per1440 inputs, lossless arrays in both NPZ and NPY, records, prefix gate, all fixed gates, saved replay and a fully bound report. All noise diagnostics are unconditional, and gate failure yields STOP. Passing authorizes only consideration of DEV registration preparation. It is not evidence of native feasibility, gait accuracy or anatomical event recovery.

Known limits: no preparation evidence establishes full-panel gates, runtime budget sufficiency or actual support-waveform minima. No OS/network sandbox or cryptographic process authenticity is implemented; the worker is cooperative, external mutation can forge content, and code review/root operational control remain required. Runtime closure is conservative for named source/cache/native trees and loaded libraries, not a full kernel/firmware/unloaded-library closure. Post-run replay deliberately starts from saved filtered magnitudes. Independent root review of source, protocol, runtime and execution controls is required before any freeze; no DEV execution is available in this CLI.
