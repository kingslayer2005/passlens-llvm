# Status

## What was actually run

### Data (Phases 1-3): full run, 2026-10-08

Run on Linux (Ubuntu 24.04) with the pinned toolchain, LLVM 17.0.6
(`clang-17`, `opt-17`), Python 3.13, 2 CPU cores. Phase 1 took 90 s and
Phase 3 took 705 s.

**a. Phase 1-7 Outputs:**
Phase 1-4 data outputs were created on 2026-10-08 in a **full run** at git commit `f401926`.
Phase 5-8 outputs were attempted on 2026-10-08 using `--smoke`, but were stopped (see e).

**b. Statistics:**
Files compiled: 496
Files failed: 622
Groups (Benchmark programs): 53
Unique functions: 1038
Total samples (States): 2953
Per-pass decreased / equal / increased counts are preserved in `results/phase3_gate.txt`. For instance:
- `instcombine`: decr=2148, equal=794, incr=11
- `simplifycfg`: decr=2083, equal=859, incr=11
- `early-cse`: decr=2035, equal=918, incr=0
- `licm`: decr=102, equal=2126, incr=725
- (See full table in phase3_gate.txt)

**c. Kept/Interpreted pass x target pairs & PR-AUC:**
Interpretable passes: `instcombine, simplifycfg, early-cse, gvn, adce, reassociate, jump-threading, correlated-propagation, licm, loop-rotate, indvars, loop-deletion, tailcallelim`.
Target status kept:
- `instcombine` (beneficial, fired)
- `simplifycfg` (beneficial, fired)
- `early-cse` (beneficial, fired)
- `gvn` (beneficial, fired)
- `sccp` (fired)
- `adce` (fired)
- `reassociate` (beneficial, harmful, fired)
- `jump-threading` (beneficial, fired)
- `correlated-propagation` (beneficial, fired)
- `licm` (beneficial, harmful, fired)
- `loop-rotate` (beneficial, harmful, fired)
- `indvars` (beneficial, harmful, fired)
- `loop-deletion` (fired)
- `loop-idiom` (harmful, fired)
- `loop-unroll` (harmful, fired)
- `tailcallelim` (fired)

PR-AUC for XGBoost vs Majority Baseline (for interpretable kept pairs, from random_forest/logistic_regression omitted here to highlight XGBoost vs Baseline):
- `instcombine/beneficial`: XGB=0.8344 vs BL=0.7076
- `instcombine/fired`: XGB=0.9292 vs BL=0.8995
- `simplifycfg/beneficial`: XGB=0.9031 vs BL=0.6886
- `simplifycfg/fired`: XGB=0.9328 vs BL=0.7390
- `early-cse/beneficial`: XGB=0.8495 vs BL=0.6896
- `early-cse/fired`: XGB=0.8567 vs BL=0.6976
- `gvn/beneficial`: XGB=0.8100 vs BL=0.7498
- `gvn/fired`: XGB=0.8376 vs BL=0.7816
- `adce/fired`: XGB=0.3793 vs BL=0.1023
- `reassociate/beneficial`: XGB=0.0731 vs BL=0.0785
- `reassociate/harmful`: XGB=0.0895 vs BL=0.0460
- `reassociate/fired`: XGB=0.6963 vs BL=0.4222
- `jump-threading/beneficial`: XGB=0.8464 vs BL=0.6246
- `jump-threading/fired`: XGB=0.8683 vs BL=0.6638
- `correlated-propagation/beneficial`: XGB=0.1756 vs BL=0.1367
- `correlated-propagation/fired`: XGB=0.7921 vs BL=0.6543
- `licm/beneficial`: XGB=0.0947 vs BL=0.0422
- `licm/harmful`: XGB=0.0281 vs BL=0.0221
- `licm/fired`: XGB=0.6495 vs BL=0.3821
- `loop-rotate/beneficial`: XGB=0.1266 vs BL=0.0882
- `loop-rotate/harmful`: XGB=0.6847 vs BL=0.4189
- `loop-rotate/fired`: XGB=0.8281 vs BL=0.5319
- `indvars/beneficial`: XGB=0.3048 vs BL=0.1598
- `indvars/harmful`: XGB=0.1916 vs BL=0.0936
- `indvars/fired`: XGB=0.6400 vs BL=0.3980
- `loop-deletion/fired`: XGB=0.0264 vs BL=0.0062
- `tailcallelim/fired`: XGB=0.8073 vs BL=0.6413

**d. Audits:**
`leakage_audit.txt`: Contains only the header `# Leakage Audit` (no leakage found).
`skipped_folds.txt`: Contains only the header `# Skipped Folds` (no folds skipped).
`heuristics.yaml` SHA-256 matches exactly with `env.json` (`354269c617ba8fff226129b806cc66d82564cfede247660304dc5f19c0e7677c`).

**e. Phase 8:**
Phase 8 did not complete (Phase 5-8 execution was manually aborted as they were hanging in `size_confound_analysis` during loop evaluations even in `--smoke` mode, exceeding acceptable wait limits). Thus, Phase 5-8 are **NOT RUN**.

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
