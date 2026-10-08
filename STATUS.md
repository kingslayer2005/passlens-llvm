# Status

## What was actually run

### a. Existing Phase 1-7 Outputs
- **Created**: 2026-10-08T21:49:34+0530
- **Git Commit**: 73272a67f1932722ac5c4fdb16ca7c777ed85e44
- **Run Type**: The run appears to be a FULL RUN based on logs, but it only output 819 unique functions (which is < 1000), making it functionally equivalent to a smoke/preliminary run based on decision rules.

### b. Data Statistics
- **Programs Processed**: 52
- **Compilation Failures**: 742
- **Total Functions (>= 10 insts)**: 854
- **Unique Functions**: 819
- **Total Samples**: 720
- **Per-pass Counts (Decreased / Equal / Increased)**:
  - sroa: 0 / 40 / 0
  - instcombine: 35 / 5 / 0
  - simplifycfg: 26 / 14 / 0
  - early-cse: 26 / 14 / 0
  - gvn: 36 / 4 / 0
  - sccp: 0 / 40 / 0
  - adce: 0 / 40 / 0
  - dse: 0 / 40 / 0
  - reassociate: 0 / 40 / 0
  - jump-threading: 24 / 16 / 0
  - correlated-propagation: 0 / 40 / 0
  - licm: 0 / 39 / 1
  - loop-rotate: 0 / 13 / 27
  - indvars: 8 / 15 / 17
  - loop-deletion: 0 / 40 / 0
  - loop-idiom: 0 / 39 / 1
  - loop-unroll: 0 / 39 / 1
  - tailcallelim: 0 / 40 / 0

### c. Passes Interpreted vs Kept
- **Kept Passes (evaluated in Phase 4)**:
  - simplifycfg (beneficial)
  - early-cse (beneficial)
  - gvn (beneficial)
  - jump-threading (beneficial)
  - loop-rotate (harmful)
  - indvars (beneficial, harmful)
- **Interpreted Passes (Stable & Beats majority, passed Phase 5 gating)**:
  - **simplifycfg** (beneficial): XGBoost PR-AUC = 0.9091 vs Majority = 0.6042
  - **early-cse** (beneficial): XGBoost PR-AUC = 0.9457 vs Majority = 0.625
  - **jump-threading** (beneficial): XGBoost PR-AUC = 0.8508 vs Majority = 0.5625

### d. Audit & Environment Checks
- **Unroll threshold default**: 150 (from LoopUnrollPass.cpp)
- **leakage_audit.txt**: Contains 74 entries (reps/folds) across passes, all marked as `PASS` with `prog_overlap=0 hash_overlap=0`.
- **skipped_folds.txt**: Contains 17 entries detailing folds that were skipped due to having a single class (e.g., instcombine, gvn, indvars).
- **heuristics.yaml SHA-256 Check**: The SHA-256 of `heuristics.yaml` is `7af7b265db64661f0b272d547e964ac24fe701bf04f5cc2bc9e387be94fc105a`, which perfectly matches the hash stored in `env.json`.

### e. Phase 8
- FINISHED successfully within 4 minutes. Generated `stretch_runtime.csv`.

## Pending before paper submission
- Recovering the files that failed to compile (headers and include paths).
- Capping groups at 100 functions.
- Adding the "fired" target.
- A full rerun of Phases 1-7 under ANALYSIS_PLAN.md.
- Independent verification of the heuristics.yaml sources.
- Any other remaining tasks in ANALYSIS_PLAN.md not yet done.
