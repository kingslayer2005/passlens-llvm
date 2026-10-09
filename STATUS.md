# Status

## What was actually run

### Data (Phases 1-3): full run, 2026-10-08

Run on Linux (Ubuntu 24.04) with the pinned toolchain, LLVM 17.0.6
(`clang-17`, `opt-17`), Python 3.13, 2 CPU cores. Phase 1 took 90 s and
Phase 3 took 705 s.

| Item | Value |
|---|---|
| C files found (PolyBench + MiBench) | 1118 |
| Files compiled | 496 (PolyBench 30 of 31, MiBench 466 of 1087) |
| Files that failed | 622, each listed with its reason in `results/compile_failures.csv` |
| Functions with at least 10 instructions | 2703 |
| Duplicate function bodies removed | 93 |
| Cap per benchmark program | 100 |
| Benchmark programs (CV groups) | 53 (30 PolyBench, 23 MiBench) |
| **Unique functions** | **1038** |
| **States (samples)** | **2953** (1038 baseline + 1915 prefix states) |
| Label rows (state x pass) | 53,154 |
| Failed `opt` calls | 0 |
| Features per state | 40 |

Checks that passed (`results/instcount_validation.txt`,
`results/phase3_verification.txt`):

- The IR parser's instruction and block counts equal LLVM's own
  (`print<func-properties>`) on all 3578 compiled functions, and on 400
  randomly chosen pass outputs.
- 50 of 50 randomly chosen label rows are identical when recomputed.
- State hashes are unique, and no hash or function appears in two programs.
- 31 unit tests pass; expected values are worked out by hand in the test file.

The per-pass table is in `results/phase3_gate.txt`.

### Models and explanations (Phases 4-8): NOT run on this data

Everything in `results/` named `phase4_*` to `phase7_*`, `shap_*`,
`heuristic_agreement.csv`, `stretch_runtime.csv`, the figures and
`RESULTS.md` comes from an earlier run on 40 samples whose labels were
produced by a faulty instruction counter. **Those files are stale and must
not be cited.** They have to be regenerated from the data above.

## Defects fixed in the Phase 1-3 code on 2026-10-08

1. The label counter treated basic-block labels as instructions (clang
   prints numeric labels such as `12:`), and switch case lines too. Example:
   `kernel_2mm` was counted as 114 instructions; LLVM reports 90.
2. The feature extractor returned a hardcoded block count for the test
   fixture `phi_heavy` so that a wrong test would pass.
3. Bitwise and shift instructions were not counted by the feature extractor,
   `loop_count_density` was not divided by the instruction count, and most
   constant operands were missed.
4. The resume logic skipped every function whose name had been seen before
   (for example every `main`).
5. A failed `opt` call was recorded as "no change".
6. Loop passes were measured together with loop canonicalization
   (see "Decision needed" below).
7. Functions with the same name in one program overwrote each other on disk.
8. Loop header and body sizes were estimated from block order; they now come
   from LLVM's LoopInfo.

## Decision needed before Phase 4

In LLVM's new pass manager every loop pass first puts loops into canonical
form (LoopSimplify + LCSSA). That step alone adds instructions in 725 of the
2953 states. With the pre-specified "harmful" label (size after > size
before), `loop-deletion` is harmful in 724 states, but the pass itself
changed the IR in only 19 and increased the size in 0.

`labels.csv` holds both measurements: `inst_before` (the state) and
`inst_control` (after canonicalization only). The targets for the six loop
passes should be defined against `inst_control`; this must be written into
`ANALYSIS_PLAN.md` before any model is fitted.

## Pending before paper submission

- Decide the loop-pass target definition (above) and update the plan.
- Add the `fired` target to Phases 4-6 and rerun Phases 4-7 on this data.
- Verify every `heuristics.yaml` source by hand.
- Compile coverage: 622 files fail, mostly because they need headers that
  the benchmark's own build generates (`arch.h`, `config.h`, `jconfig.h`)
  or because they are for another platform. See `LIMITATIONS.md`, section 6.
- Remove the three LLVM source files from the repository root
  (`JumpThreading.cpp`, `LoopRotation.cpp`, `LoopUnrollPass.cpp`); cite the
  file, option name and release tag in `heuristics.yaml` instead.

## Phase 4 attempt on 2026-10-09 (withdrawn)
A Phase 4 run was pushed and then withdrawn. It did not follow
ANALYSIS_PLAN.md: it used one split (train on 289 PolyBench samples,
test on 2664 MiBench samples) instead of repeated grouped
cross-validation, the 15 values given to the Wilcoxon test were
bootstrap resamples of that one test set, the 5%-95% keep rule was
removed, and the leakage audit was not run. Its outputs were deleted
and must not be cited. scripts/phase4_models.py to phase7_sequence.py
do not yet implement the plan.
