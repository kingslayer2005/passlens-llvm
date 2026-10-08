# Data: full run (1038 functions, 2953 states). Models and explanations: not yet run on this data.
# Interpretable Compiler Pass Selection: Explaining ML-Guided Optimization Decisions using SHAP on LLVM IR Features
**Authors:** Aarush Gupta (23BDS0219) + [teammates TBD]

This repository contains the complete, reproducible research codebase for studying how machine learning models make compiler optimization decisions. We use XGBoost and TreeSHAP to interpret the feature importance and feature interactions that drive optimization predictions on LLVM IR.

## Research Questions

1. **RQ1:** Which static IR features drive a model's decision that pass $P$ is beneficial?
2. **RQ2:** Do those learned drivers agree with the documented hand-written LLVM heuristics?
3. **RQ3:** Can SHAP-guided feature pruning give a smaller model with comparable accuracy?

## System Requirements

- **OS:** Linux (or Windows via WSL2 Ubuntu)
- **CPU only** (No GPU required)
- **Disk space:** ~2.5 GB (Code + Data + Artifacts stay under the 3GB limit)
- **Dependencies:**
  - Python 3.10+
  - Prebuilt LLVM toolchain (clang, opt, llvm-extract, llvm-dis) >= version 17

*Note: Do NOT build LLVM from source. We rely on the prebuilt binaries provided by your package manager or apt.llvm.org.*

## Installation and Setup

### 1. WSL2 / Ubuntu Setup
If you are on Windows, ensure you are running inside a WSL2 Ubuntu terminal. Run the one-time setup script which installs LLVM 17 and creates a Python virtual environment:

```bash
bash setup_wsl.sh
```

### 2. Activate Virtual Environment
Activate the environment containing the pinned Python dependencies:

```bash
source .venv/bin/activate
```

## Running the Pipeline

The entire pipeline is orchestrated by `run_all.py`. It uses a caching mechanism to avoid re-running successful phases. 

### Quick Test (Smoke Mode)
To ensure everything works on your machine, run the smoke test. This processes only 3 programs and completes in under 5 minutes:

```bash
python run_all.py --smoke
```

### Full Execution
To run the full pipeline (downloads all benchmarks, extracts features, trains models, computes SHAP values, and generates final results):

```bash
python run_all.py
```

### Running Specific Phases
You can run an individual phase (e.g., Phase 5 SHAP analysis):
```bash
python run_all.py --phase 5
```
Or force a re-run of the entire pipeline, ignoring cached outputs:
```bash
python run_all.py --force
```

## Pipeline Architecture

The pipeline is split into explicit, readable, intern-level Python scripts located in `scripts/`:

- **Phase 1 (`phase1_setup_data.py`):** Downloads PolyBench/C and MiBench, compiles C code to LLVM IR, extracts individual functions, and deduplicates them by IR hash.
- **Phase 2 (`phase2_features.py`):** Text-based extraction of ~60 static Autophase-style features from the LLVM IR, such as opcode distributions and CFG properties.
- **Phase 3 (`phase3_labels.py`):** Generates (function, pass) state transitions by applying a random prefix of passes, followed by the target pass. Labels passes as "beneficial" if they reduce the instruction count by $\ge 1\%$.
- **Phase 4 (`phase4_models.py`):** Trains an XGBoost classifier for each pass using GroupKFold cross-validation (grouped by program). Flags passes that fail to outperform majority-class baselines.
- **Phase 5 (`phase5_shap.py`):** *Core Contribution.* Applies TreeSHAP to the models on held-out folds. Conducts faithfulness checks (rank stability, permutation importance), clusters correlated features, evaluates agreement with `heuristics.yaml`, and identifies novel drivers.
- **Phase 6 (`phase6_pruning.py`):** Retrains models iteratively using the top-$k$ features identified by SHAP vs. XGBoost gain vs. Mutual Information.
- **Phase 7 (`phase7_sequence.py`):** Evaluates a greedy pass selector guided by the trained models on held-out programs, comparing final sequence reduction against `-Oz`, `-O2`, random, and oracle baselines.
- **Results Generator (`generate_results_md.py`):** Automatically compiles all CSV outputs and plots into a unified `results/RESULTS.md` report.

## Deliverables

- `data/` — Contains raw tarballs, extracted LLVM IR functions, and cached opt outputs.
- `results/` — Contains all generated CSVs, figures, and the auto-generated `RESULTS.md`.
- `heuristics.yaml` — Hand-annotated mapping of passes to their documented expected feature dependencies.
- `env.json` — Generated metadata recording dependency versions and seeds for reproducibility.
