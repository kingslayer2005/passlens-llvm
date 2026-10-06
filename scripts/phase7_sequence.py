#!/usr/bin/env python3
"""
phase7_sequence.py — Sequence-level pass ordering evaluation.

Greedy selector:
  1. Extract features from the current IR state.
  2. Apply the pass with the highest predicted probability of being beneficial.
  3. Repeat up to 8 steps or until no pass exceeds probability 0.5.
  4. Compare final instruction count with:
     - -Oz (LLVM size-optimized)
     - -O2 (LLVM default optimization)
     - A random sequence of equal length
     - Oracle greedy (try all passes, pick the one that actually reduces most)
  5. Evaluate on held-out programs only.

Output:
  results/phase7_sequence.csv     — per-program sequence results
  results/phase7_summary.csv      — aggregate comparison
  results/figures/sequence_*.png  — comparison plots

Usage:
  python -m scripts.phase7_sequence [--smoke]
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import (
    PROJECT_ROOT, DATA_DIR, IR_DIR, FEATURES_DIR, LABELS_DIR,
    RESULTS_DIR, FIGURES_DIR, CACHE_DIR,
    ensure_dirs, find_llvm_tools, run_tool,
    count_instructions, read_ir_file, write_ir_file,
    setup_logging, PASS_LIST,
)
from scripts.phase2_features import extract_features_from_ir_text
from scripts.phase3_labels import apply_pass

log = setup_logging("phase7")
warnings.filterwarnings("ignore", category=FutureWarning)

MAX_STEPS = 8
PROB_THRESHOLD = 0.5
RANDOM_SEED = 42


# ============================================================================
# Load trained models
# ============================================================================

def load_models(models_dir: Path, passes: list) -> dict:
    """Load saved XGBoost models for each pass."""
    import pickle
    models = {}
    for pass_name in passes:
        model_path = models_dir / f"xgb_{pass_name}.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                models[pass_name] = pickle.load(f)
        else:
            log.warning("No model found for pass %s", pass_name)
    return models


# ============================================================================
# Greedy selector
# ============================================================================

def greedy_sequence(ir_text: str, models: dict, feature_cols: list,
                     tools: dict, func_name: str = "unknown") -> dict:
    """
    Greedily apply passes using model predictions.
    Returns dict with sequence info and final instruction count.
    """
    current_ir = ir_text
    sequence = []
    inst_counts = [count_instructions(current_ir)]

    for step in range(MAX_STEPS):
        # Extract features from current state
        # Write to temp file for loop analysis
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ll", delete=False,
                                          dir=str(CACHE_DIR)) as tmp:
            tmp.write(current_ir)
            tmp_path = Path(tmp.name)

        try:
            features = extract_features_from_ir_text(
                current_ir, tools, ll_file=tmp_path, func_name=func_name
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if not features:
            break

        # Build feature vector in the same order as training
        feat_vector = np.array(
            [features.get(col, 0.0) for col in feature_cols],
            dtype=np.float32,
        ).reshape(1, -1)
        feat_vector = np.nan_to_num(feat_vector, nan=0.0, posinf=0.0, neginf=0.0)

        # Predict probability for each pass
        best_pass = None
        best_prob = 0.0

        for pass_name, model in models.items():
            prob = model.predict_proba(feat_vector)[0, 1]
            if prob > best_prob:
                best_prob = prob
                best_pass = pass_name

        # Stop if no pass exceeds threshold
        if best_pass is None or best_prob < PROB_THRESHOLD:
            break

        # Apply the best pass
        result_ir = apply_pass(current_ir, best_pass, tools)
        if result_ir is None:
            break

        new_count = count_instructions(result_ir)
        current_ir = result_ir
        sequence.append(best_pass)
        inst_counts.append(new_count)

        # Stop if no reduction
        if new_count >= inst_counts[-2]:
            break

    return {
        "sequence": sequence,
        "inst_counts": inst_counts,
        "initial_count": inst_counts[0],
        "final_count": inst_counts[-1],
        "n_steps": len(sequence),
    }


# ============================================================================
# Baseline sequences
# ============================================================================

def apply_opt_level(ir_text: str, level: str, tools: dict) -> int:
    """Apply -Oz or -O2 and return final instruction count."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ll", delete=False,
                                      dir=str(CACHE_DIR)) as tmp:
        tmp.write(ir_text)
        tmp_path = tmp.name

    try:
        out_path = tmp_path + f".{level}.ll"
        cmd = [tools["opt"], f"-{level}", "-S", "-o", out_path, tmp_path]
        result = run_tool(cmd, timeout=30, check=False)

        if result.returncode != 0 or not os.path.exists(out_path):
            return count_instructions(ir_text)

        output_ir = Path(out_path).read_text(encoding="utf-8", errors="replace")
        return count_instructions(output_ir)
    finally:
        for p in [tmp_path, tmp_path + f".{level}.ll"]:
            try:
                os.unlink(p)
            except OSError:
                pass


def random_sequence(ir_text: str, length: int, tools: dict, seed: int) -> dict:
    """Apply a random sequence of passes."""
    rng = random.Random(seed)
    current_ir = ir_text
    sequence = []
    inst_counts = [count_instructions(current_ir)]

    for _ in range(length):
        pass_name = rng.choice(PASS_LIST)
        result_ir = apply_pass(current_ir, pass_name, tools)
        if result_ir is None:
            break
        current_ir = result_ir
        sequence.append(pass_name)
        inst_counts.append(count_instructions(current_ir))

    return {
        "sequence": sequence,
        "inst_counts": inst_counts,
        "initial_count": inst_counts[0],
        "final_count": inst_counts[-1],
    }


def oracle_greedy_sequence(ir_text: str, tools: dict, max_steps: int = 8) -> dict:
    """
    Oracle greedy: at each step, try ALL passes and pick the one
    that actually reduces instruction count the most.
    """
    current_ir = ir_text
    sequence = []
    inst_counts = [count_instructions(current_ir)]

    for _ in range(max_steps):
        best_pass = None
        best_count = inst_counts[-1]
        best_ir = None

        for pass_name in PASS_LIST:
            result_ir = apply_pass(current_ir, pass_name, tools)
            if result_ir is None:
                continue
            new_count = count_instructions(result_ir)
            if new_count < best_count:
                best_count = new_count
                best_pass = pass_name
                best_ir = result_ir

        if best_pass is None or best_count >= inst_counts[-1]:
            break

        current_ir = best_ir
        sequence.append(best_pass)
        inst_counts.append(best_count)

    return {
        "sequence": sequence,
        "inst_counts": inst_counts,
        "initial_count": inst_counts[0],
        "final_count": inst_counts[-1],
    }


# ============================================================================
# Main pipeline
# ============================================================================

def run_phase7(smoke: bool = False):
    """Execute Phase 7: sequence-level evaluation."""
    ensure_dirs()
    import pandas as pd

    tools = find_llvm_tools()

    # Load function index (held-out programs only — use test fold from phase 4)
    index_path = DATA_DIR / "function_index.csv"
    if not index_path.exists():
        log.error("Function index not found — run Phase 1 first.")
        sys.exit(1)

    df_index = pd.read_csv(index_path)
    df_index = df_index[df_index["is_duplicate"] == False].copy()

    # For held-out evaluation, use a simple split: last 20% of programs
    programs = sorted(df_index["program"].unique())
    n_held_out = max(1, len(programs) // 5)
    held_out_programs = programs[-n_held_out:]
    df_held_out = df_index[df_index["program"].isin(held_out_programs)]

    if smoke:
        df_held_out = df_held_out.head(5)

    log.info("Phase 7: evaluating on %d held-out functions from %d programs",
             len(df_held_out), len(held_out_programs))

    # Load models
    models_dir = DATA_DIR / "models"
    interp_path = LABELS_DIR / "interpretable_passes.json"
    if interp_path.exists():
        with open(interp_path) as f:
            interp_passes = json.load(f)
    else:
        interp_passes = PASS_LIST

    models = load_models(models_dir, interp_passes)
    if not models:
        log.error("No models found — run Phase 4 first.")
        sys.exit(1)

    # Load feature columns
    feature_cols_path = FEATURES_DIR / "feature_columns.json"
    if feature_cols_path.exists():
        with open(feature_cols_path) as f:
            feature_cols = json.load(f)
    else:
        log.error("Feature columns not found — run Phase 4 first.")
        sys.exit(1)

    # Evaluate each held-out function
    results = []

    for idx, row in df_held_out.iterrows():
        ll_path = PROJECT_ROOT / row["path"]
        if not ll_path.exists():
            continue

        ir_text = read_ir_file(ll_path)
        func_name = row["function"]
        program = row["program"]
        suite = row["suite"]
        initial_count = count_instructions(ir_text)

        if initial_count < 10:
            continue

        log.info("  Evaluating %s/%s (initial: %d inst)", program, func_name, initial_count)

        # 1. Greedy selector
        greedy = greedy_sequence(ir_text, models, feature_cols, tools, func_name)

        # 2. -Oz
        oz_count = apply_opt_level(ir_text, "Oz", tools)

        # 3. -O2
        o2_count = apply_opt_level(ir_text, "O2", tools)

        # 4. Random sequence (same length as greedy)
        rand_result = random_sequence(ir_text, greedy["n_steps"], tools, RANDOM_SEED)

        # 5. Oracle greedy
        oracle = oracle_greedy_sequence(ir_text, tools, MAX_STEPS)

        results.append({
            "suite": suite,
            "program": program,
            "function": func_name,
            "initial_count": initial_count,
            "greedy_count": greedy["final_count"],
            "greedy_steps": greedy["n_steps"],
            "greedy_sequence": ";".join(greedy["sequence"]),
            "oz_count": oz_count,
            "o2_count": o2_count,
            "random_count": rand_result["final_count"],
            "random_steps": len(rand_result["sequence"]),
            "oracle_count": oracle["final_count"],
            "oracle_steps": len(oracle["sequence"]),
            "oracle_sequence": ";".join(oracle["sequence"]),
            # Relative reductions from initial
            "greedy_reduction": round((initial_count - greedy["final_count"]) / initial_count, 4),
            "oz_reduction": round((initial_count - oz_count) / initial_count, 4),
            "o2_reduction": round((initial_count - o2_count) / initial_count, 4),
            "random_reduction": round((initial_count - rand_result["final_count"]) / initial_count, 4),
            "oracle_reduction": round((initial_count - oracle["final_count"]) / initial_count, 4),
        })

    if not results:
        log.error("No sequence results generated!")
        sys.exit(1)

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_DIR / "phase7_sequence.csv", index=False)
    log.info("Sequence results saved to %s", RESULTS_DIR / "phase7_sequence.csv")

    # Summary statistics
    summary = {
        "method": ["greedy", "Oz", "O2", "random", "oracle"],
        "mean_reduction": [
            df_results["greedy_reduction"].mean(),
            df_results["oz_reduction"].mean(),
            df_results["o2_reduction"].mean(),
            df_results["random_reduction"].mean(),
            df_results["oracle_reduction"].mean(),
        ],
        "median_reduction": [
            df_results["greedy_reduction"].median(),
            df_results["oz_reduction"].median(),
            df_results["o2_reduction"].median(),
            df_results["random_reduction"].median(),
            df_results["oracle_reduction"].median(),
        ],
        "mean_final_count": [
            df_results["greedy_count"].mean(),
            df_results["oz_count"].mean(),
            df_results["o2_count"].mean(),
            df_results["random_count"].mean(),
            df_results["oracle_count"].mean(),
        ],
    }
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(RESULTS_DIR / "phase7_summary.csv", index=False)
    log.info("Summary:\n%s", df_summary.to_string())

    # ---- Plot comparison ----
    seq_fig_dir = FIGURES_DIR / "sequence"
    seq_fig_dir.mkdir(parents=True, exist_ok=True)

    # Bar chart of mean reductions
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = df_summary["method"]
    reductions = df_summary["mean_reduction"] * 100
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9E9E9E", "#E91E63"]
    bars = ax.bar(methods, reductions, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Mean Instruction Reduction (%)")
    ax.set_title("Sequence-Level Optimization Comparison")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, reductions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(seq_fig_dir / "sequence_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Box plot of per-function reductions
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [
        df_results["greedy_reduction"] * 100,
        df_results["oz_reduction"] * 100,
        df_results["o2_reduction"] * 100,
        df_results["random_reduction"] * 100,
        df_results["oracle_reduction"] * 100,
    ]
    bp = ax.boxplot(data, tick_labels=["Greedy", "-Oz", "-O2", "Random", "Oracle"],
                     patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Instruction Reduction (%)")
    ax.set_title("Per-Function Instruction Reduction Distribution")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(seq_fig_dir / "sequence_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Gate report
    report = f"""
╔══════════════════════════════════════════════════════╗
║              PHASE 7 GATE REPORT                     ║
╠══════════════════════════════════════════════════════╣
║ Held-out functions:     {len(results):>8}                     ║
║ Greedy mean reduction:  {df_results['greedy_reduction'].mean()*100:>7.1f}%%                    ║
║ -Oz mean reduction:     {df_results['oz_reduction'].mean()*100:>7.1f}%%                    ║
║ -O2 mean reduction:     {df_results['o2_reduction'].mean()*100:>7.1f}%%                    ║
║ Random mean reduction:  {df_results['random_reduction'].mean()*100:>7.1f}%%                    ║
║ Oracle mean reduction:  {df_results['oracle_reduction'].mean()*100:>7.1f}%%                    ║
╚══════════════════════════════════════════════════════╝
"""
    log.info(report)
    (RESULTS_DIR / "phase7_gate.txt").write_text(report, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description="Phase 7: Sequence-level evaluation")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_phase7(smoke=args.smoke)


if __name__ == "__main__":
    main()
