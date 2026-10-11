# PassLens: Interpretable Compiler Pass Selection

Explaining ML-guided LLVM optimization decisions with SHAP on LLVM IR features.

**Authors:** Aarush Gupta

> **Status.** Data (Phases 1-3): full run complete and verified, 1038
> functions and 2953 states. Models and explanations (Phases 4-7): code
> complete and tested, **full run not yet done**. There are no model, SHAP or
> heuristic-agreement results yet. See [STATUS.md](STATUS.md).

## What the project asks

Machine-learning models can predict which compiler optimization pass will
help a piece of code, but they do not say why. This project trains such
models on LLVM IR and asks three questions:

1. **RQ1.** Which static IR features drive a model's prediction that a pass
   will change, shrink or grow a function?
2. **RQ2.** Do those features agree with the conditions and thresholds in
   LLVM's own implementation of the pass?
3. **RQ3.** Can the explanation be used to build a smaller model that is
   nearly as accurate?

The objective throughout is code size (instruction count), not running time.

## Pipeline

| Phase | Script | What it does |
|---|---|---|
| 1 | `scripts/phase1_setup_data.py` | Fetches PolyBench/C and MiBench at pinned commits, compiles each C file to LLVM IR, extracts functions, removes duplicate bodies, caps each program at 100 functions. Checks the instruction counter against LLVM's own count on every function. |
| 2 | `scripts/phase2_features.py` | The feature extractor: 40 size-independent features per function (opcode and CFG densities, block sizes, loop features from LLVM's LoopInfo). |
| 3 | `scripts/phase3_labels.py` | For each function: the baseline and up to three states reached by seeded random pass prefixes. Runs each of 18 passes on each state and records the instruction count before, after, and after the pass's control run. |
| 4 | `scripts/phase4_models.py` | Per pass and target (beneficial / harmful / fired): majority, logistic regression, random forest and XGBoost under 3 x 5 repeated cross-validation grouped by benchmark program, with nested tuning and a leakage audit. |
| 5 | `scripts/phase5_shap.py` | TreeSHAP on held-out folds, trust checks (stability, dual method, permutation importance, label-shuffle control, size confound, faithfulness), and agreement with `heuristics.yaml`. |
| 6 | `scripts/phase6_pruning.py` | Refits with the top-k features chosen by SHAP, gain, mutual information or at random. |
| 7 | `scripts/phase7_sequence.py` | Model-guided greedy pass sequences on held-out programs against oracle, random, `-Oz` and `-O2`. |
| | `scripts/generate_results_md.py` | Builds `results/RESULTS.md` from the result files only. |

Shared code: `scripts/targets.py` (the one definition of every label),
`scripts/analysis_common.py` (settings, folds, models, statistics),
`scripts/utils.py` (LLVM tools, the IR parser).

The analysis is pre-specified in [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md).
`heuristics.yaml`, the list of expectations the models are compared with, is
frozen by its SHA-256 in `env.json`; `run_all.py` stops if it changes.

## Requirements

- Linux, or Windows with WSL2 Ubuntu. Run everything inside WSL, not with
  Windows Python.
- LLVM 17 (`clang-17`, `opt-17`, `llvm-extract-17`) from the distribution's
  packages. Do not build LLVM from source.
- Python 3.10 or newer with the packages in `requirements.txt`.
- About 1 GB of disk. Each parallel worker needs roughly 1-2 GB of memory;
  `--jobs` is lowered automatically when memory is short.

```bash
bash setup_wsl.sh            # once: installs LLVM 17 and creates .venv
source .venv/bin/activate
```

## Running

```bash
python -m unittest discover tests            # unit tests (about 2 seconds)
python run_all.py --smoke --from-phase 4     # checks Phases 4-7 run; NOT results
python run_all.py --from-phase 4 --jobs 2    # full analysis on the existing data
python -m scripts.check_phase4               # structural check of the Phase 4 output
```

Other options: `--phase N` runs one phase, `--only gvn,licm` restricts
Phases 4-6 to some passes, `--force` ignores the resume cache. A run can be
interrupted and restarted; finished (pass, target) pairs are cached in
`data/cache/` as small JSON files.

Phases 1-3 need not be rerun to reproduce the analysis if `data/` holds
`features/features.csv`, `labels/labels.csv`, `function_index.csv` and
`ir/`. Rerunning them fetches the benchmark sources (see `data/SOURCES.md`).

## Outputs

- `results/RESULTS.md`: every table, generated from the files below.
- `results/master_table.csv`: one row per pass x target with its status
  (dropped / not interpreted / interpreted) and the reason.
- `results/phase4_*.csv`, `leakage_audit.txt`, `skipped_folds.txt`: models.
- `results/shap_*.csv`, `heuristic_*.csv`, `threshold_*.csv`, `faithfulness.csv`,
  `size_confound.csv`, `novel_drivers.csv`: explanations.
- `results/phase6_*.csv`, `results/phase7_*.csv`: pruning and sequences.
- `results/figures/`: PNG figures; each has a CSV with the same numbers.
- `data/` is not tracked, except `data/SOURCES.md` and `data/manifest.csv`.

## Limitations

See [LIMITATIONS.md](LIMITATIONS.md). In short: code size only, two C
benchmark suites with uneven program sizes, static features, single-pass
effects, and a heuristics list that is only partly verified against LLVM's
source.

## License

MIT for the code in this repository. The benchmark suites have their own
licenses and are not redistributed here.
