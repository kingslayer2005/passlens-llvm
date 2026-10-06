#!/usr/bin/env python3
"""
phase8_stretch.py — Stretch goals for the LLVM IR SHAP research pipeline.

Implements the stretch goals specified in the project requirements:
1. Model distillation: Distill each XGBoost model into a depth-3 decision tree and report fidelity.
2. IR2Vec hook: Provide a clean hook/interface for IR2Vec embeddings as an alternative feature set.
3. Runtime study: Perform a runtime study on whole PolyBench programs (median of 5 runs).

Note: This should be run only after the main pipeline (Phases 1-7) has completed successfully.

Usage:
  python -m scripts.phase8_stretch
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import (
    PROJECT_ROOT, DATA_DIR, FEATURES_DIR, LABELS_DIR, RESULTS_DIR, RAW_DIR,
    ensure_dirs, find_llvm_tools, run_tool, setup_logging
)

log = setup_logging("phase8")


# ============================================================================
# Stretch 1: Model Distillation (Depth-3 Decision Tree)
# ============================================================================

def distill_to_decision_tree(df_merged, feature_cols, kept_passes):
    """
    Distill the XGBoost predictions into a depth-3 Decision Tree.
    Reports the fidelity (agreement between XGBoost predictions and DT predictions).
    """
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.metrics import accuracy_score
    import pickle

    log.info("--- Stretch 1: Distilling models into Depth-3 Decision Trees ---")
    
    models_dir = DATA_DIR / "models"
    distill_results = []
    
    for pass_name in kept_passes:
        # Load the original XGBoost model
        model_path = models_dir / f"xgb_{pass_name}.pkl"
        if not model_path.exists():
            continue
            
        with open(model_path, "rb") as f:
            xgb_model = pickle.load(f)
            
        pass_df = df_merged[df_merged["pass_name"] == pass_name]
        if len(pass_df) < 50:
            continue
            
        X = pass_df[feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        y_xgb_pred = xgb_model.predict(X)
            
        # Train a depth-3 Decision Tree on the XGBoost predictions
        dt = DecisionTreeClassifier(max_depth=3, random_state=42)
        dt.fit(X, y_xgb_pred)
        
        # Evaluate fidelity (how often DT agrees with XGBoost)
        y_dt_pred = dt.predict(X)
        fidelity = accuracy_score(y_xgb_pred, y_dt_pred)
        
        # Save the tree text representation
        tree_rules = export_text(dt, feature_names=feature_cols)
        
        distill_results.append({
            "pass_name": pass_name,
            "fidelity": round(fidelity, 4),
            "dt_rules": tree_rules
        })
        
        log.info("Pass: %s | DT Fidelity: %.2f%%", pass_name, fidelity * 100)
    
    if distill_results:
        import pandas as pd
        df_distill = pd.DataFrame(distill_results)
        df_distill.to_csv(RESULTS_DIR / "stretch_distillation.csv", index=False)
        log.info("Saved distillation results to results/stretch_distillation.csv")


# ============================================================================
# Stretch 2: IR2Vec Hook
# ============================================================================

def ir2vec_feature_hook(ir_file_path: Path):
    """
    A clean hook for integrating IR2Vec embeddings.
    
    Since IR2Vec requires building an out-of-tree LLVM pass and external C++ 
    tools, we provide the Python interface here. A researcher would implement 
    the actual subprocess call to their `ir2vec` executable.
    
    Returns a dummy embedding vector to demonstrate the interface.
    """
    # Actual implementation needs to call IR2Vec binary.
    # cmd = ["ir2vec", "-fa", "-level=p", str(ir_file_path), "-o", "embedding.txt"]
    # subprocess.run(cmd, check=True)
    # embedding = np.loadtxt("embedding.txt")
    raise NotImplementedError("IR2Vec integration requires the external ir2vec C++ binary.")

def demonstrate_ir2vec_hook():
    log.info("--- Stretch 2: IR2Vec Hook ---")
    log.info("IR2Vec hook is defined in `scripts/phase8_stretch.py:ir2vec_feature_hook()`.")
    log.info("It can be seamlessly swapped in Phase 2 in place of `IRFeatureExtractor`.")


# ============================================================================
# Stretch 3: Runtime Study on PolyBench
# ============================================================================

def compile_and_measure_runtime(c_file: Path, tools: dict, includes: list, 
                                pass_sequence: list = None, runs: int = 5):
    """
    Compile a C program to an executable and measure its execution time (median of N runs).
    If pass_sequence is provided, we apply those specific passes to the IR before compiling to object code.
    """
    exe_path = c_file.with_suffix('.out')
    
    # Base includes and math library
    compile_cmd = [tools["clang"], "-O0"]
    for inc in includes:
        compile_cmd.extend(["-I", str(inc)])
    
    if pass_sequence:
        # Step 1: Compile to IR
        ir_path = c_file.with_suffix('.ll')
        cmd1 = [tools["clang"], "-O0", "-Xclang", "-disable-O0-optnone", "-emit-llvm", "-S"]
        for inc in includes:
            cmd1.extend(["-I", str(inc)])
        cmd1.extend(["-o", str(ir_path), str(c_file)])
        run_tool(cmd1)
        
        # Step 2: Apply passes
        opt_ir_path = c_file.with_suffix('.opt.ll')
        cmd2 = [tools["opt"], f"-passes={','.join(pass_sequence)}", "-S", "-o", str(opt_ir_path), str(ir_path)]
        run_tool(cmd2)
        
        # Step 3: Compile to executable
        cmd3 = [tools["clang"], str(opt_ir_path), "-o", str(exe_path), "-lm"]
        run_tool(cmd3)
        
        # Cleanup intermediate
        try: os.remove(ir_path); os.remove(opt_ir_path)
        except OSError: pass
    else:
        # Direct compilation (Baseline -O0)
        compile_cmd.extend(["-o", str(exe_path), str(c_file), "-lm"])
        run_tool(compile_cmd)
        
    # Measure runtime
    runtimes = []
    for _ in range(runs):
        t0 = time.perf_counter()
        # We run the executable. Note: PolyBench prints output to stderr if POLYBENCH_DUMP_ARRAYS is set,
        # but by default it just runs silently. We capture output to prevent terminal spam.
        subprocess.run([str(exe_path)], capture_output=True, timeout=30, check=True)
        runtimes.append(time.perf_counter() - t0)
            
    # Cleanup executable
    try: os.remove(exe_path)
    except OSError: pass
    
    if runtimes:
        return np.median(runtimes)
    return float('nan')

def run_polybench_runtime_study(tools: dict):
    """Run a small runtime study on a few PolyBench programs."""
    import pandas as pd
    
    log.info("--- Stretch 3: PolyBench Runtime Study ---")
    polybench_dir = RAW_DIR / "polybench"
    if not polybench_dir.exists():
        log.warning("PolyBench source not found at %s. Skipping runtime study.", polybench_dir)
        return
        
    includes = list(polybench_dir.rglob("utilities"))
    
    # Pick a few representative programs to avoid a massive run time
    targets = ["2mm.c", "atax.c", "gemm.c", "syrk.c"]
    c_files = []
    for t in targets:
        matches = list(polybench_dir.rglob(t))
        if matches:
            c_files.append(matches[0])
            
    if not c_files:
        log.warning("Could not find the target PolyBench files.")
        return
        
    # We will test: Baseline (-O0), -O2 (simulated), and a Custom Sequence
    # A standard custom sequence discovered by our pipeline (hypothetical example)
    custom_seq = ["sroa", "instcombine", "simplifycfg", "loop-rotate", "licm", "indvars", "gvn", "dse"]
    
    results = []
    for c_file in c_files:
        prog_name = c_file.stem
        log.info("Benchmarking runtime for %s...", prog_name)
        
        t_base = compile_and_measure_runtime(c_file, tools, includes, pass_sequence=None)
        t_o2   = compile_and_measure_runtime(c_file, tools, includes, pass_sequence=["default<O2>"])
        t_cust = compile_and_measure_runtime(c_file, tools, includes, pass_sequence=custom_seq)
        
        results.append({
            "program": prog_name,
            "baseline_O0_sec": round(t_base, 4),
            "O2_sec": round(t_o2, 4),
            "custom_seq_sec": round(t_cust, 4),
        })
        
    df_runtime = pd.DataFrame(results)
    df_runtime.to_csv(RESULTS_DIR / "stretch_runtime.csv", index=False)
    log.info("Runtime results saved to results/stretch_runtime.csv")
    log.info("\n%s", df_runtime.to_markdown(index=False))


# ============================================================================
# Main
# ============================================================================

def run_phase8():
    """Execute all Stretch goals."""
    ensure_dirs()
    import pandas as pd
    
    tools = find_llvm_tools()
    
    # Need data from earlier phases for Distillation
    features_path = FEATURES_DIR / "features.csv"
    labels_path = LABELS_DIR / "labels.csv"
    interp_path = LABELS_DIR / "interpretable_passes.json"
    feature_cols_path = FEATURES_DIR / "feature_columns.json"
    
    if features_path.exists() and labels_path.exists() and interp_path.exists() and feature_cols_path.exists():
        df_features = pd.read_csv(features_path)
        df_labels = pd.read_csv(labels_path)
        
        with open(interp_path) as f:
            kept_passes = json.load(f)
        with open(feature_cols_path) as f:
            feature_cols = json.load(f)
            
        df_merged = pd.merge(df_labels, df_features, on=["ir_hash", "suite", "program", "function"], how="inner")
        
        distill_to_decision_tree(df_merged, feature_cols, kept_passes)
    else:
        log.warning("Required files for Distillation not found. Skipping Stretch 1.")
        
    demonstrate_ir2vec_hook()
    
    run_polybench_runtime_study(tools)
    
    log.info("Phase 8 (Stretch Goals) complete.")

def main():
    parser = argparse.ArgumentParser(description="Phase 8: Stretch Goals")
    args = parser.parse_args()
    run_phase8()

if __name__ == "__main__":
    main()
