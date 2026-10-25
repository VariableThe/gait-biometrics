# Review and validation history

All reviews below preceded the actual metadata run unless explicitly labeled
as evidence review. Reviewers used synthetic inputs and current source only.

1. **R1 requested changes.** Full-300 valid synthetic metadata exhausted the
   512 MiB ceiling because canonical serialization constructed parallel type
   trees. Replay also accepted an unknown outer result field after its byte
   hash was updated. Both were reproduced before actual input execution.
2. **R2 cleared both fixes.** Depth-bounded validation/comparison preserves
   canonical bytes and explicit type/signed-zero roundtrips. Verification checks
   the exact result schema before loading inputs, then compares the complete
   reconstructed result. Independent full-300 API/CLI checks passed under real
   OS ceilings; nineteen rehashed outer-result mutations were rejected early.
3. **Root found a suite integration blocker.** Tests imposed real hard limits
   on the shared scientific test runner. Behavior tests now mock enforcement,
   while real full-300 API/CLI tests execute in fresh interpreters. A dedicated
   check still requires both production entrypoints to call enforcement.
4. **R3 cleared isolation.** Twenty targeted tests and 183 subtests passed,
   including actual child-process limits. A parent reserving 768 MiB retained
   its limits, affinity, complete environment and SIGXCPU handler after every
   selected contract, then allocated another 16 MiB. R1 fixes remained effective.

The source closure was frozen as main
`7e6aabe0c28950c8732380419bec58f9c7d45063` and Student B
`97a47d0b1b8d2b3ac47a43039654639ceb3014a1`.

Root full suites passed: main 1,272 tests in 336.83 seconds, Student B 1,268
in 335.12 seconds, each with 1,805 subtests and eight earlier warnings. They ran
concurrently. Full logs are retained in `validation/`. These are total suite
timings, separate from the inventory's process CPU/job wall measurements.

The actual metadata run and strict main/Student B saved-only replays completed.
Their published access reports, source/input/result hashes and original
reservation are the primary execution evidence. File-access guards and review
do not independently authenticate raw sensor payloads.

## Independent concrete evidence audit

The evidence reviewer found no material inconsistency. Without calling the
production inventory or running its tests, independent metadata arithmetic
matched every result field, including all owner ledgers, graph ordering,
failure ledgers and support masks. Snapshot/source/runtime links, the historical
member projections and little-endian row-key hash, the exact three-key result
schema, and completion hashes passed.

All 220 original graphs retain their arrays and denominators. Across repeated
occurrences, genuine/impostor counts are: inner 2,384/4,768; pool 2,384/4,768;
validation 596/1,192; cohort inner 7,152/57,216; cohort refit 2,384/26,224.
Each any/all support mask has 109,068 positions: all true for historical256
and complete segments, all false for fixed768. These repeated graph occurrences
are not unique observations. All 960 training-owner identity occurrences pass
the proposed count flag for historical256 and complete segments; none passes
for fixed768.

The sole long block is recording index 87, subject 9/day 1/trial 8, segment 0,
rows `[0,768)`, matching historical window indices 170–172. The six fragmented
trials, in subject/day/trial order, are `(1,2,5)`, `(5,1,8)`, `(5,1,9)`,
`(16,2,5)`, `(19,2,1)`, `(24,2,2)`; their segment counts 3/2/2/2/22/3 account
for all 28 gap breaks.

The review establishes metadata consistency only. It did not validate signal
payloads, extraction eligibility, learned models, authentication outcomes or
the then-unfinished publication helper.
