#!/usr/bin/env python3
"""
phase3_labels.py — Generate optimization labels for each function × pass.

For each unique function:
  1. Start from the baseline IR (from Phase 1).
  2. Create 3 additional states by applying random prefixes of 1-3 passes (seeded).
  3. For each (function, state) and each of the 18 passes:
     a. Run: opt -passes=<pass> -S -o output.ll input.ll
     b. Count instructions before and after.
     c. rel_reduction = (before - after) / before
     d. Label "beneficial" if rel_reduction >= 0.01
  4. Parallelize with joblib, 10s timeout per opt call, cache by IR hash.
  5. Drop passes with positive rate < 5% or > 95%.

Output:
  data/labels/labels.csv — columns: function, program, suite, ir_hash, state,
      pass_name, inst_before, inst_after, rel_reduction, beneficial
  data/labels/label_distribution.csv — per-pass label statistics
  results/phase3_gate.txt — gate report

Usage:
  python -m scripts.phase3_labels [--smoke] [--jobs N]
"""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import (
    PROJECT_ROOT, DATA_DIR, IR_DIR, LABELS_DIR, CACHE_DIR, RESULTS_DIR,
    ensure_dirs, find_llvm_tools, run_tool,
    count_instructions, hash_ir, read_ir_file, write_ir_file,
    setup_logging, PASS_LIST, verify_passes,
)

log = setup_logging("phase3")

PASS_WRAPPERS = {
    "licm": "loop-mssa(licm)",
    "loop-rotate": "loop(loop-rotate)",
    "loop-deletion": "loop(loop-deletion)",
    "loop-idiom": "loop(loop-idiom)",
    "loop-unroll": "loop-unroll",
}

# ============================================================================
# Configuration
# ============================================================================

BENEFICIAL_THRESHOLD = 0.01   # 1% relative reduction
OPT_TIMEOUT = 10              # seconds per opt call
NUM_RANDOM_STATES = 3         # additional states per function
MAX_PREFIX_LEN = 3            # max passes in random prefix
RANDOM_SEED = 42
DROP_POS_RATE_LOW = 0.05      # drop passes with pos rate < 5%
DROP_POS_RATE_HIGH = 0.95     # drop passes with pos rate > 95%


# ============================================================================
# Opt wrapper with caching
# ============================================================================

def apply_pass(ir_text: str, pass_name: str, tools: dict) -> Optional[str]:
    """
    Apply a single optimization pass to IR text using stdin/stdout in memory.
    Returns the optimized IR text.
    """
    cmd = [
        tools["opt"],
        f"-passes={pass_name}",
        "-S"
    ]
    try:
        result = run_tool(cmd, timeout=OPT_TIMEOUT, check=True, input_data=ir_text)
        return result.stdout
    except subprocess.CalledProcessError:
        return None
    except subprocess.TimeoutExpired:
        return None


def apply_pass_sequence(ir_text: str, passes: List[str], tools: dict) -> Optional[str]:
    """Apply a sequence of passes to IR text, one at a time."""
    current_ir = ir_text
    for p in passes:
        result = apply_pass(current_ir, p, tools)
        if result is None:
            return None
        current_ir = result
    return current_ir


# ============================================================================
# State generation
# ============================================================================

def generate_random_states(ir_text: str, func_name: str, seed: int,
                            tools: dict) -> List[Tuple[str, str, List[str]]]:
    """
    Generate random prefix states for a function.
    Returns list of (state_name, ir_text, passes_applied).
    """
    rng = random.Random(seed)
    states = []

    for i in range(NUM_RANDOM_STATES):
        # Pick a random prefix length (1-3)
        prefix_len = rng.randint(1, MAX_PREFIX_LEN)
        # Pick random passes for the prefix
        prefix_passes = [rng.choice(PASS_LIST) for _ in range(prefix_len)]

        state_name = f"prefix_{i}"
        result_ir = apply_pass_sequence(ir_text, prefix_passes, tools)

        if result_ir is not None:
            states.append((state_name, result_ir, prefix_passes))
        else:
            log.debug("Failed to generate state %s for %s with passes %s",
                      state_name, func_name, prefix_passes)

    return states


# ============================================================================
# Label generation for one function
# ============================================================================

def label_one_function(func_info: dict, tools: dict, active_passes: List[str]) -> Tuple[List[dict], List[dict]]:
    """
    Generate labels for one function across all states and passes.
    Returns (labels_list, features_list).
    """
    ll_path = PROJECT_ROOT / func_info["path"]
    if not ll_path.exists():
        log.warning("IR file not found: %s", ll_path)
        return []

    ir_text = read_ir_file(ll_path)
    func_name = func_info["function"]
    program = func_info["program"]
    suite = func_info["suite"]
    ir_h = func_info["ir_hash"]

    # Seed for random states: hash-based so it's deterministic per function
    seed = int(ir_h[:8], 16) % (2**31)

    # States: baseline + random prefixes
    states = [("baseline", ir_text, [])]
    states.extend(generate_random_states(ir_text, func_name, seed, tools))

    labels = []
    features_list = []

    # Import locally to avoid circular dependency
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.phase2_features import extract_features_from_ir_text
    from scripts.utils import normalize_ir, hash_ir

    for state_name, state_ir, prefix_passes in states:
        inst_before = count_instructions(state_ir)
        if inst_before < 1:
            continue

        state_ir_h = hash_ir(state_ir)
        
        # Compute features for this state
        try:
            feats = extract_features_from_ir_text(state_ir, tools, func_name=func_name)
            feats["suite"] = suite
            feats["program"] = program
            feats["function"] = func_name
            feats["ir_hash"] = state_ir_h
            features_list.append(feats)
        except Exception as e:
            log.warning("Failed to extract features for %s/%s state=%s: %s", program, func_name, state_name, e)
            continue

        # Get the normalized no-op ir for 'fired' comparison
        noop_res = run_tool([tools["opt"], "-S"], input_data=state_ir, timeout=10, check=False)
        if noop_res.returncode == 0:
            noop_ir_norm = normalize_ir(noop_res.stdout)
        else:
            noop_ir_norm = ""

        for pass_name in active_passes:
            real_pass = PASS_WRAPPERS.get(pass_name, pass_name)
            result_ir = apply_pass(state_ir, real_pass, tools)

            if result_ir is None:
                # opt failed — record as no change
                inst_after = inst_before
                outcome = "equal"
                fired = False
            else:
                inst_after = count_instructions(result_ir)
                if inst_after < inst_before:
                    outcome = "decreased"
                elif inst_after > inst_before:
                    outcome = "increased"
                else:
                    outcome = "equal"
                    
                result_ir_norm = normalize_ir(result_ir)
                fired = (result_ir_norm != noop_ir_norm) if noop_ir_norm else False

            rel_reduction = (inst_before - inst_after) / inst_before
            beneficial = 1 if rel_reduction >= BENEFICIAL_THRESHOLD else 0

            labels.append({
                "suite": suite,
                "program": program,
                "function": func_name,
                "ir_hash": state_ir_h,
                "state": state_name,
                "prefix_passes": ";".join(prefix_passes),
                "pass_name": pass_name,
                "inst_before": inst_before,
                "inst_after": inst_after,
                "rel_reduction": round(rel_reduction, 6),
                "outcome": outcome,
                "fired": fired,
                "beneficial": beneficial,
            })

    return labels, features_list


# ============================================================================
# Determinism check
# ============================================================================

def check_determinism(all_labels: list, tools: dict, n_check: int = 50) -> bool:
    """
    Rerun 50 random samples and verify labels are identical.
    Returns True if all match.
    """
    log.info("Running determinism check on %d samples...", n_check)
    rng = random.Random(12345)

    if len(all_labels) < n_check:
        n_check = len(all_labels)

    samples = rng.sample(all_labels, n_check)
    mismatches = 0

    for sample in samples:
        # Reconstruct the state
        ll_path = PROJECT_ROOT / "data" / "ir" / sample["suite"] / sample["program"] / f"{sample['function']}.ll"
        if not ll_path.exists():
            continue

        ir_text = read_ir_file(ll_path)

        # Apply prefix passes if any
        if sample["prefix_passes"]:
            prefix = sample["prefix_passes"].split(";")
            ir_text = apply_pass_sequence(ir_text, prefix, tools)
            if ir_text is None:
                continue

        inst_before = count_instructions(ir_text)
        result_ir = apply_pass(ir_text, sample["pass_name"], tools)
        inst_after = count_instructions(result_ir) if result_ir else inst_before

        rel_reduction = (inst_before - inst_after) / inst_before if inst_before > 0 else 0
        beneficial = 1 if rel_reduction >= BENEFICIAL_THRESHOLD else 0

        if beneficial != sample["beneficial"]:
            mismatches += 1
            log.warning("Determinism mismatch: %s/%s pass=%s state=%s "
                        "original=%d recomputed=%d",
                        sample["program"], sample["function"],
                        sample["pass_name"], sample["state"],
                        sample["beneficial"], beneficial)

    if mismatches == 0:
        log.info("Determinism check PASSED: all %d samples match.", n_check)
    else:
        log.warning("Determinism check: %d/%d mismatches!", mismatches, n_check)

    return mismatches == 0


# ============================================================================
# Main pipeline
# ============================================================================

def run_phase3(smoke: bool = False, n_jobs: int = 1):
    """Execute Phase 3: generate optimization labels."""
    ensure_dirs()
    tools = find_llvm_tools()

    # Verify pass names
    invalid = verify_passes(tools)
    if invalid:
        log.warning("Invalid passes (will skip): %s", invalid)

    active_passes = [p for p in PASS_LIST if p not in invalid]
    log.info("Active passes (%d): %s", len(active_passes), active_passes)

    # Read function index
    import pandas as pd
    index_path = DATA_DIR / "function_index.csv"
    if not index_path.exists():
        log.error("Function index not found — run Phase 1 first.")
        sys.exit(1)

    df_index = pd.read_csv(index_path)
    df_index = df_index[df_index["is_duplicate"] == False].copy()

    if smoke:
        # In smoke mode, take only first 10 functions
        df_index = df_index.head(10)
        log.info("Smoke mode: using %d functions", len(df_index))

    func_list = df_index.to_dict("records")
    log.info("Generating labels for %d functions × %d passes × ~4 states = ~%d samples",
             len(func_list), len(active_passes),
             len(func_list) * len(active_passes) * 4)

    # Generate labels (sequential for now — joblib parallelization below)
    import time
    # Resumable from cache
    processed_hashes = set()
    from scripts.utils import DATA_DIR
    features_path = DATA_DIR / "features" / "features.csv"
    labels_path = LABELS_DIR / "labels.csv"
    if labels_path.exists() and features_path.exists():
        try:
            df_old = pd.read_csv(labels_path)
            processed_hashes = set(df_old["ir_hash"].unique())
            log.info("Found %d already processed states in cache.", len(processed_hashes))
        except Exception:
            pass
            
    # We filter functions if ALL their possible states would be in the cache. 
    # For a simple resumability, we just check if the baseline hash is in the cache.
    # Actually, we can just process all and let label_one_function skip, or just skip if the function's baseline is in labels.csv.
    # But since baseline hash is modified to state hash, we can check if the function's base IR hash is in labels.csv by matching function name, or just skip if function is in labels.csv.
    if labels_path.exists():
        try:
            df_old = pd.read_csv(labels_path)
            processed_funcs = set(df_old["function"].unique())
            func_list = [f for f in func_list if f["function"] not in processed_funcs]
            log.info("Resuming: %d functions already processed, %d remaining.", len(processed_funcs), len(func_list))
        except Exception:
            pass

    start_time = time.time()
    total_opt_calls = len(func_list) * (1 + NUM_RANDOM_STATES) * len(active_passes)
    all_labels = []
    all_features = []
    
    # Load old data if resuming
    if labels_path.exists() and features_path.exists():
        try:
            all_labels = pd.read_csv(labels_path).to_dict('records')
            all_features = pd.read_csv(features_path).to_dict('records')
        except Exception:
            all_labels = []
            all_features = []

    if len(func_list) > 0:
        if n_jobs == 1:
            for i, func_info in enumerate(func_list):
                labels, feats = label_one_function(func_info, tools, active_passes)
                all_labels.extend(labels)
                all_features.extend(feats)
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    rem = (len(func_list) - (i + 1)) / rate
                    progress_msg = f"Processed {i+1}/{len(func_list)} | Elapsed: {elapsed:.1f}s | ETA: {rem:.1f}s\n"
                    with open(RESULTS_DIR / "phase3_progress.txt", "a") as f:
                        f.write(progress_msg)
        else:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(label_one_function)(fi, tools, active_passes)
                for fi in func_list
            )
            for i, (r_labels, r_feats) in enumerate(results):
                all_labels.extend(r_labels)
                all_features.extend(r_feats)
                if (i + 1) % 10 == 0 or (i + 1) == len(func_list):
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    rem = (len(func_list) - (i + 1)) / rate
                    progress_msg = f"Processed {i+1}/{len(func_list)} | Elapsed: {elapsed:.1f}s | ETA: {rem:.1f}s\n"
                    with open(RESULTS_DIR / "phase3_progress.txt", "a") as f:
                        f.write(progress_msg)
                    
                    # Periodically save to allow resuming if killed
                    pd.DataFrame(all_labels).to_csv(labels_path, index=False)
                    pd.DataFrame(all_features).drop_duplicates(subset=["ir_hash"]).to_csv(features_path, index=False)

    elapsed = time.time() - start_time
    if elapsed > 0:
        calls_sec = total_opt_calls / elapsed
        log.info("Total labels generated: %d in %.1fs (%.1f calls/sec)", len(all_labels), elapsed, calls_sec)

    # Save features CSV
    from scripts.utils import DATA_DIR
    features_path = DATA_DIR / "features" / "features.csv"
    if all_features:
        df_features = pd.DataFrame(all_features).drop_duplicates(subset=["ir_hash"])
        df_features.to_csv(features_path, index=False)
        log.info("Saved features to %s (%d rows)", features_path, len(df_features))

    # Save labels CSV
    labels_path = LABELS_DIR / "labels.csv"
    if all_labels:
        df_labels = pd.DataFrame(all_labels)
        df_labels.to_csv(labels_path, index=False)
        log.info("Saved labels to %s", labels_path)

        # ---- Label distribution table ----
        dist_rows = []
        target_status = {}
        for pass_name in active_passes:
            pass_df = df_labels[df_labels["pass_name"] == pass_name]
            n_total = len(pass_df)
            if n_total == 0:
                continue

            # Outcomes
            n_dec = (pass_df["outcome"] == "decreased").sum()
            n_eq = (pass_df["outcome"] == "equal").sum()
            n_inc = (pass_df["outcome"] == "increased").sum()
            n_fired = pass_df["fired"].sum()

            # Targets
            n_ben = pass_df["beneficial"].sum()
            n_harm = n_inc

            pos_rate_ben = n_ben / n_total
            pos_rate_harm = n_harm / n_total

            drop_ben = pos_rate_ben < DROP_POS_RATE_LOW or pos_rate_ben > DROP_POS_RATE_HIGH
            drop_harm = pos_rate_harm < DROP_POS_RATE_LOW or pos_rate_harm > DROP_POS_RATE_HIGH

            dist_rows.append({
                "pass_name": pass_name,
                "total_samples": n_total,
                "fired": int(n_fired),
                "decreased": int(n_dec),
                "equal": int(n_eq),
                "increased": int(n_inc),
                "beneficial_rate": round(pos_rate_ben, 4),
                "harmful_rate": round(pos_rate_harm, 4),
                "drop_beneficial": drop_ben,
                "drop_harmful": drop_harm,
            })

            if not drop_ben:
                target_status[f"{pass_name}_beneficial"] = "kept"
            else:
                target_status[f"{pass_name}_beneficial"] = "dropped"

            if not drop_harm:
                target_status[f"{pass_name}_harmful"] = "kept"
            else:
                target_status[f"{pass_name}_harmful"] = "dropped"

        df_dist = pd.DataFrame(dist_rows)
        dist_path = LABELS_DIR / "label_distribution.csv"
        df_dist.to_csv(dist_path, index=False)
        log.info("Label distribution saved to %s", dist_path)

        # Print distribution
        log.info("\n--- Label Distribution ---")
        for _, row in df_dist.iterrows():
            stat_ben = "DROPPED" if row["drop_beneficial"] else "KEPT"
            stat_harm = "DROPPED" if row["drop_harmful"] else "KEPT"
            log.info("  %-25s fired=%-4d dec=%-4d eq=%-4d inc=%-4d | ben_rate=%5.1f%% [%-7s] harm_rate=%5.1f%% [%-7s]",
                     row["pass_name"], row["fired"], row["decreased"], row["equal"], row["increased"],
                     row["beneficial_rate"] * 100, stat_ben,
                     row["harmful_rate"] * 100, stat_harm)

        # List dropped passes (if a pass has BOTH targets dropped, it's totally dropped)
        dropped_passes = df_dist[(df_dist["drop_beneficial"] == True) & (df_dist["drop_harmful"] == True)]["pass_name"].tolist()
        kept_passes = df_dist[(df_dist["drop_beneficial"] == False) | (df_dist["drop_harmful"] == False)]["pass_name"].tolist()

        # Save target status for Phase 4
        with open(LABELS_DIR / "target_status.json", "w") as f:
            json.dump(target_status, f, indent=2)

        # ---- Determinism check ----
        check_determinism(all_labels, tools, n_check=min(50, len(all_labels)))

        # ---- Gate report ----
        report = f"""
╔══════════════════════════════════════════════════════╗
║              PHASE 3 GATE REPORT                     ║
╠══════════════════════════════════════════════════════╣
║ Total samples:          {len(all_labels):>8}                     ║
║ Functions:              {len(func_list):>8}                     ║
║ Passes (active):        {len(active_passes):>8}                     ║
║ Passes (kept):          {len(kept_passes):>8}                     ║
║ Passes (dropped):       {len(dropped_passes):>8}                     ║
╠══════════════════════════════════════════════════════╣
║ Dropped passes: {', '.join(dropped_passes) if dropped_passes else 'none':<37}║
╚══════════════════════════════════════════════════════╝
"""
        log.info(report)
        gate_path = RESULTS_DIR / "phase3_gate.txt"
        gate_path.write_text(report, encoding='utf-8')

        # Save kept passes list for downstream phases
        kept_path = LABELS_DIR / "kept_passes.json"
        with open(kept_path, "w") as f:
            json.dump(kept_passes, f)
        log.info("Kept passes saved to %s", kept_path)

    else:
        log.error("No labels generated!")
        sys.exit(1)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Generate optimization labels")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Number of parallel jobs")
    args = parser.parse_args()
    run_phase3(smoke=args.smoke, n_jobs=args.jobs)


if __name__ == "__main__":
    main()
