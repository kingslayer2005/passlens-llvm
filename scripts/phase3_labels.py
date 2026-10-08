#!/usr/bin/env python3
"""
phase3_labels.py — Generate features and optimization labels.

For each unique function from Phase 1:
  1. Start from the baseline IR.
  2. Create up to 3 more states by applying seeded random prefixes of 1-3
     passes.  States whose IR is identical to an earlier state are dropped.
  3. For each state compute the full feature vector (before the IR is
     thrown away), then for each of the 18 passes run opt once and record:
       inst_before   instructions in the state
       inst_control  instructions after the pass's CONTROL pipeline (below)
       inst_after    instructions after the pass
       rel_reduction (inst_before - inst_after) / inst_before
       outcome       decreased / equal / increased   (after vs before)
       beneficial    1 if rel_reduction >= 1%
       harmful       1 if outcome == increased
       changed_ir    1 if the output differs from a no-op opt run
       fired         1 if the output differs from the CONTROL output

CONTROL pipeline.  In LLVM's new pass manager a loop pass is always wrapped
in an adaptor that first puts every loop into canonical form (LoopSimplify
+ LCSSA).  That canonicalization alone can add blocks and phi nodes, even
when the pass itself does nothing.  To tell the two apart we also run the
same adaptor with a pass that does nothing ("no-op-loop").  For passes that
are not loop passes, the control is the plain no-op run.

All IR stays in memory: opt reads stdin and writes stdout.  Nothing
transformed is written to disk.

Output:
  data/features/features.csv        one row per state
  data/labels/labels.csv            one row per state x pass
  data/labels/label_distribution.csv
  data/labels/target_status.json, kept_passes.json
  results/phase3_failures.csv       every opt call that failed (never hidden)
  results/phase3_gate.txt

Usage:
  python -m scripts.phase3_labels [--smoke] [--jobs N] [--force]
"""

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import (
    PROJECT_ROOT, DATA_DIR, FEATURES_DIR, LABELS_DIR, CACHE_DIR, RESULTS_DIR,
    ensure_dirs, find_llvm_tools, require_pinned_llvm, run_tool,
    count_instructions, hash_ir, normalize_ir, read_ir_file,
    setup_logging, PASS_LIST, verify_passes,
)
from scripts.phase2_features import (
    extract_features_from_ir_text, run_noop_with_loops,
)

log = setup_logging("phase3")

# ============================================================================
# Configuration
# ============================================================================

BENEFICIAL_THRESHOLD = 0.01   # 1% relative reduction
OPT_TIMEOUT = 10              # seconds per opt call
NUM_RANDOM_STATES = 3         # additional states per function
MAX_PREFIX_LEN = 3            # max passes in random prefix
DROP_POS_RATE_LOW = 0.05      # a target is dropped below this positive rate
DROP_POS_RATE_HIGH = 0.95     # ... or above this one
DETERMINISM_SAMPLES = 50

# The exact pipeline string given to `opt -passes=` for each pass.
# Loop passes must be wrapped in a loop adaptor; LICM needs MemorySSA.
PASS_PIPELINES = {
    "sroa": "sroa",
    "instcombine": "instcombine",
    "simplifycfg": "simplifycfg",
    "early-cse": "early-cse",
    "gvn": "gvn",
    "sccp": "sccp",
    "adce": "adce",
    "dse": "dse",
    "reassociate": "reassociate",
    "jump-threading": "jump-threading",
    "correlated-propagation": "correlated-propagation",
    "licm": "loop-mssa(licm)",
    "loop-rotate": "loop(loop-rotate)",
    "indvars": "loop(indvars)",
    "loop-deletion": "loop(loop-deletion)",
    "loop-idiom": "loop(loop-idiom)",
    "loop-unroll": "loop-unroll",
    "tailcallelim": "tailcallelim",
}

# Which control each pass is compared against to decide "fired".
#   "noop"      plain opt run, no passes
#   "loop"      loop(no-op-loop)       -> LoopSimplify + LCSSA only
#   "loop-mssa" loop-mssa(no-op-loop)  -> the same, with MemorySSA available
# loop-unroll is a function pass, but it puts every loop into the same
# canonical form before it decides anything, so it uses the loop control.
PASS_CONTROL = {
    "licm": "loop-mssa",
    "loop-rotate": "loop",
    "indvars": "loop",
    "loop-deletion": "loop",
    "loop-idiom": "loop",
    "loop-unroll": "loop",
}
CONTROL_PIPELINES = {
    "loop": "loop(no-op-loop)",
    "loop-mssa": "loop-mssa(no-op-loop)",
}

# Kept for scripts that still import the old name
PASS_WRAPPERS = {name: pipe for name, pipe in PASS_PIPELINES.items() if pipe != name}


def config_signature(llvm_version: str) -> str:
    """
    Short id of everything that decides a label.  The resume cache is only
    reused when this id is unchanged.
    """
    payload = json.dumps({
        "llvm": llvm_version,
        "pipelines": PASS_PIPELINES,
        "controls": [PASS_CONTROL, CONTROL_PIPELINES],
        "states": [NUM_RANDOM_STATES, MAX_PREFIX_LEN],
        "threshold": BENEFICIAL_THRESHOLD,
        "format": 3,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# ============================================================================
# Running opt
# ============================================================================

def run_pipeline(ir_text: str, pipeline: str, tools: dict) -> Tuple[Optional[str], str]:
    """
    Run `opt -passes=<pipeline> -S` on IR text (stdin -> stdout).
    Returns (output_ir, "") on success, or (None, reason) on failure.
    """
    cmd = [tools["opt"], f"-passes={pipeline}", "-S"]
    try:
        result = run_tool(cmd, timeout=OPT_TIMEOUT, check=False, input_data=ir_text)
    except subprocess.TimeoutExpired:
        return None, f"timeout after {OPT_TIMEOUT}s"
    if result.returncode != 0:
        lines = result.stderr.strip().splitlines()
        return None, (lines[0][:200] if lines else f"exit code {result.returncode}")
    return result.stdout, ""


def apply_pass(ir_text: str, pass_name: str, tools: dict) -> Optional[str]:
    """Apply one pass (by its name in PASS_LIST).  Returns None on failure."""
    output, _ = run_pipeline(ir_text, PASS_PIPELINES.get(pass_name, pass_name), tools)
    return output


def apply_pass_sequence(ir_text: str, passes: List[str], tools: dict) -> Tuple[Optional[str], str]:
    """Apply passes one after another.  Returns (ir, "") or (None, reason)."""
    current_ir = ir_text
    for p in passes:
        output, reason = run_pipeline(current_ir, PASS_PIPELINES[p], tools)
        if output is None:
            return None, f"{p}: {reason}"
        current_ir = output
    return current_ir, ""


# ============================================================================
# States
# ============================================================================

def draw_prefixes(seed: int) -> List[List[str]]:
    """The seeded random prefixes for one function (1-3 passes each)."""
    rng = random.Random(seed)
    prefixes = []
    for _ in range(NUM_RANDOM_STATES):
        prefix_len = rng.randint(1, MAX_PREFIX_LEN)
        prefixes.append([rng.choice(PASS_LIST) for _ in range(prefix_len)])
    return prefixes


def seed_for_function(ir_hash: str) -> int:
    """Seed from the function's baseline hash: same function, same prefixes."""
    return int(ir_hash[:8], 16) % (2 ** 31)


# ============================================================================
# Measuring one state
# ============================================================================

def measure_state(state_ir: str, func_name: str, tools: dict,
                  active_passes: List[str]) -> Tuple[dict, List[dict], List[dict]]:
    """
    Compute the features of one state and the effect of every pass on it.
    Returns (features, pass_rows, failures).
    Raises if the no-op run itself fails (then the state cannot be used).
    """
    failures = []

    # One opt call gives the no-op reprint and LoopInfo
    noop_ir, loops = run_noop_with_loops(state_ir, tools, timeout=OPT_TIMEOUT)
    features = extract_features_from_ir_text(state_ir, tools, func_name=func_name, loops=loops)

    inst_before = count_instructions(state_ir)
    noop_norm = normalize_ir(noop_ir)

    # Control runs (loop canonicalization without any transformation)
    control_norm = {"noop": noop_norm}
    control_count = {"noop": inst_before}
    for control_name, pipeline in CONTROL_PIPELINES.items():
        control_ir, reason = run_pipeline(state_ir, pipeline, tools)
        if control_ir is None:
            failures.append({"stage": f"control:{control_name}", "reason": reason})
            control_norm[control_name] = None
            control_count[control_name] = None
        else:
            control_norm[control_name] = normalize_ir(control_ir)
            control_count[control_name] = count_instructions(control_ir)

    pass_rows = []
    for pass_name in active_passes:
        control_name = PASS_CONTROL.get(pass_name, "noop")
        row = {
            "pass_name": pass_name,
            "inst_before": inst_before,
            "inst_control": control_count[control_name],
            "inst_after": None,
            "rel_reduction": None,
            "outcome": "error",
            "beneficial": None,
            "harmful": None,
            "changed_ir": None,
            "fired": None,
        }

        result_ir, reason = run_pipeline(state_ir, PASS_PIPELINES[pass_name], tools)
        if result_ir is None:
            # A failed opt call is recorded as an error, never as "no change"
            failures.append({"stage": f"pass:{pass_name}", "reason": reason})
            pass_rows.append(row)
            continue

        inst_after = count_instructions(result_ir)
        if inst_after < inst_before:
            outcome = "decreased"
        elif inst_after > inst_before:
            outcome = "increased"
        else:
            outcome = "equal"
        rel_reduction = (inst_before - inst_after) / inst_before
        result_norm = normalize_ir(result_ir)

        row["inst_after"] = inst_after
        row["rel_reduction"] = round(rel_reduction, 6)
        row["outcome"] = outcome
        row["beneficial"] = 1 if rel_reduction >= BENEFICIAL_THRESHOLD else 0
        row["harmful"] = 1 if outcome == "increased" else 0
        row["changed_ir"] = 1 if result_norm != noop_norm else 0
        if control_norm[control_name] is not None:
            row["fired"] = 1 if result_norm != control_norm[control_name] else 0
        pass_rows.append(row)

    return features, pass_rows, failures


# ============================================================================
# One function (runs in a worker process)
# ============================================================================

def label_one_function(func_info: dict, tools: dict, active_passes: List[str]) -> dict:
    """
    Generate all states of one function and measure them.
    Returns a dict with keys: func_id, labels, features, failures, n_states_dropped.
    """
    func_id = func_info["func_id"]
    func_name = func_info["function"]
    meta = {
        "func_id": func_id,
        "suite": func_info["suite"],
        "program": func_info["program"],
        "function": func_name,
    }
    out = {"func_id": func_id, "labels": [], "features": [], "failures": [],
           "n_states_dropped": 0}

    ll_path = PROJECT_ROOT / func_info["path"]
    baseline_ir = read_ir_file(ll_path)

    # ---- Build the states: baseline + seeded random prefixes ----
    states = [("baseline", baseline_ir, [])]
    for i, prefix in enumerate(draw_prefixes(seed_for_function(func_info["ir_hash"]))):
        state_ir, reason = apply_pass_sequence(baseline_ir, prefix, tools)
        if state_ir is None:
            out["failures"].append({**meta, "state": f"prefix_{i}",
                                    "stage": "prefix:" + ";".join(prefix), "reason": reason})
            continue
        states.append((f"prefix_{i}", state_ir, prefix))

    # ---- Measure each distinct state ----
    seen_hashes = set()
    for state_name, state_ir, prefix in states:
        if count_instructions(state_ir) < 1:
            out["n_states_dropped"] += 1
            continue
        state_hash = hash_ir(state_ir, func_name)
        if state_hash in seen_hashes:
            # The prefix changed nothing (or gave the same IR as another
            # prefix): an exact copy adds no information.
            out["n_states_dropped"] += 1
            continue
        seen_hashes.add(state_hash)

        try:
            features, pass_rows, failures = measure_state(state_ir, func_name, tools, active_passes)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as e:
            out["failures"].append({**meta, "state": state_name,
                                    "stage": "state", "reason": str(e)[:200]})
            continue

        for failure in failures:
            out["failures"].append({**meta, "state": state_name, **failure})

        out["features"].append({**features, **meta, "ir_hash": state_hash, "state": state_name})
        for row in pass_rows:
            out["labels"].append({**meta, "ir_hash": state_hash, "state": state_name,
                                  "prefix_passes": ";".join(prefix), **row})
    return out


# ============================================================================
# Determinism check
# ============================================================================

def check_determinism(df_labels, index_by_id: dict, tools: dict, n_check: int) -> Tuple[int, int]:
    """
    Recompute n_check random (state, pass) rows from the baseline file and
    compare every measured value.  Returns (n_checked, n_mismatches).
    """
    import pandas as pd
    ok_rows = df_labels[df_labels["outcome"] != "error"]
    n_check = min(n_check, len(ok_rows))
    sample = ok_rows.sample(n=n_check, random_state=12345)
    mismatches = 0

    for _, row in sample.iterrows():
        func_info = index_by_id[row["func_id"]]
        ir_text = read_ir_file(PROJECT_ROOT / func_info["path"])
        prefix = row["prefix_passes"].split(";") if isinstance(row["prefix_passes"], str) and row["prefix_passes"] else []
        if prefix:
            ir_text, _ = apply_pass_sequence(ir_text, prefix, tools)
            if ir_text is None:
                mismatches += 1
                continue
        _, pass_rows, _ = measure_state(ir_text, row["function"], tools, [row["pass_name"]])
        new = pass_rows[0]
        def as_plain(value):
            """Turn a pandas missing value into None so == is a plain compare."""
            return None if pd.isna(value) else int(value)

        same = (hash_ir(ir_text, row["function"]) == row["ir_hash"]
                and new["inst_before"] == as_plain(row["inst_before"])
                and new["inst_after"] == as_plain(row["inst_after"])
                and new["inst_control"] == as_plain(row["inst_control"])
                and new["fired"] == as_plain(row["fired"])
                and new["changed_ir"] == as_plain(row["changed_ir"]))
        if not same:
            mismatches += 1
            log.warning("Determinism mismatch: %s state=%s pass=%s", row["func_id"],
                        row["state"], row["pass_name"])
    return n_check, mismatches


# ============================================================================
# Main pipeline
# ============================================================================

def run_phase3(smoke: bool = False, n_jobs: int = 1, force: bool = False):
    """Execute Phase 3: features and labels for every state."""
    import pandas as pd

    ensure_dirs()
    tools = find_llvm_tools()
    llvm_version = require_pinned_llvm(tools)

    # ---- Check the pass names, then really run every pipeline once ----
    invalid = verify_passes(tools)
    if invalid:
        raise RuntimeError(f"Pass names not known to this opt: {invalid}")
    active_passes = list(PASS_LIST)

    index_path = DATA_DIR / "function_index.csv"
    if not index_path.exists():
        log.error("Function index not found — run Phase 1 first.")
        sys.exit(1)
    df_index = pd.read_csv(index_path)
    df_index = df_index[df_index["is_duplicate"] == False].copy()
    if smoke:
        df_index = df_index.head(10)
        log.info("Smoke mode: using %d functions", len(df_index))
    func_list = df_index.to_dict("records")
    index_by_id = {f["func_id"]: f for f in func_list}

    preflight_ir = read_ir_file(PROJECT_ROOT / func_list[0]["path"])
    for name, pipeline in list(PASS_PIPELINES.items()) + list(CONTROL_PIPELINES.items()):
        output, reason = run_pipeline(preflight_ir, pipeline, tools)
        if output is None:
            raise RuntimeError(f"Pipeline '{pipeline}' for '{name}' does not run: {reason}")
    log.info("All %d pass pipelines and %d control pipelines run.",
             len(PASS_PIPELINES), len(CONTROL_PIPELINES))

    # ---- Resume cache: one JSON line per finished function ----
    signature = config_signature(llvm_version)
    cache_path = CACHE_DIR / f"phase3_{'smoke_' if smoke else ''}{signature}.jsonl"
    if force and cache_path.exists():
        cache_path.unlink()
    done = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue   # a half-written last line from an interrupted run
                done[record["func_id"]] = record
        log.info("Resuming: %d functions already in the cache.", len(done))

    todo = [f for f in func_list if f["func_id"] not in done]
    log.info("Functions: %d total, %d to process, %d passes, %d jobs",
             len(func_list), len(todo), len(active_passes), n_jobs)

    # ---- Process ----
    start_time = time.time()
    progress_path = RESULTS_DIR / "phase3_progress.txt"
    calls_per_function = (1 + NUM_RANDOM_STATES) * (1 + len(CONTROL_PIPELINES) + len(active_passes))

    if todo:
        if n_jobs == 1:
            results = (label_one_function(fi, tools, active_passes) for fi in todo)
        else:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(label_one_function)(fi, tools, active_passes) for fi in todo)

        with open(cache_path, "a", encoding="utf-8") as cache_file:
            for i, record in enumerate(results):
                done[record["func_id"]] = record
                cache_file.write(json.dumps(record) + "\n")
                if (i + 1) % 25 == 0 or (i + 1) == len(todo):
                    cache_file.flush()
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    remaining = (len(todo) - (i + 1)) / rate
                    message = (f"{time.strftime('%H:%M:%S')} processed {i + 1}/{len(todo)} "
                               f"functions | elapsed {elapsed:.0f}s | ETA {remaining:.0f}s | "
                               f"~{rate * calls_per_function:.0f} opt calls/s (upper bound)")
                    log.info(message)
                    with open(progress_path, "a", encoding="utf-8") as pf:
                        pf.write(message + "\n")
    elapsed = time.time() - start_time

    # ---- Assemble in the fixed index order ----
    all_labels, all_features, all_failures = [], [], []
    n_states_dropped_same = 0
    n_states_dropped_global = 0
    seen_state_hashes = set()
    for func_info in func_list:
        record = done[func_info["func_id"]]
        n_states_dropped_same += record["n_states_dropped"]
        all_failures.extend(record["failures"])
        for feat in record["features"]:
            if feat["ir_hash"] in seen_state_hashes:
                # The same IR was already produced by another function:
                # keep it once only, so a hash can never be in two programs.
                n_states_dropped_global += 1
                continue
            seen_state_hashes.add(feat["ir_hash"])
            all_features.append(feat)
            all_labels.extend(l for l in record["labels"]
                              if l["ir_hash"] == feat["ir_hash"] and l["state"] == feat["state"])

    if not all_labels:
        log.error("No labels generated!")
        sys.exit(1)

    df_features = pd.DataFrame(all_features)
    df_labels = pd.DataFrame(all_labels)
    for column in ("inst_control", "inst_after", "beneficial", "harmful", "changed_ir", "fired"):
        df_labels[column] = df_labels[column].astype("Int64")   # integers that allow missing
    assert df_features["ir_hash"].is_unique, "state hashes must be unique"

    features_path = FEATURES_DIR / "features.csv"
    labels_path = LABELS_DIR / "labels.csv"
    df_features.to_csv(features_path, index=False)
    df_labels.to_csv(labels_path, index=False)
    log.info("Saved %d feature rows to %s", len(df_features), features_path)
    log.info("Saved %d label rows to %s", len(df_labels), labels_path)

    failure_fields = ["func_id", "suite", "program", "function", "state", "stage", "reason"]
    with open(RESULTS_DIR / "phase3_failures.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=failure_fields)
        writer.writeheader()
        writer.writerows(all_failures)

    # ---- Label distribution ----
    dist_rows = []
    target_status = {}
    for pass_name in active_passes:
        pass_df = df_labels[df_labels["pass_name"] == pass_name]
        ok = pass_df[pass_df["outcome"] != "error"]
        n_ok = len(ok)
        n_err = len(pass_df) - n_ok
        counts = {
            "fired": int(ok["fired"].sum()),
            "changed_ir": int(ok["changed_ir"].sum()),
            "decreased": int((ok["outcome"] == "decreased").sum()),
            "equal": int((ok["outcome"] == "equal").sum()),
            "increased": int((ok["outcome"] == "increased").sum()),
        }
        # Size went up although the pass itself changed nothing: the growth
        # comes from loop canonicalization, not from the pass.
        increased_not_fired = int(((ok["outcome"] == "increased") & (ok["fired"] == 0)).sum())
        rates = {
            "beneficial": ok["beneficial"].mean() if n_ok else float("nan"),
            "harmful": ok["harmful"].mean() if n_ok else float("nan"),
            "fired": ok["fired"].mean() if n_ok else float("nan"),
        }
        row = {"pass_name": pass_name, "samples": n_ok, "errors": n_err, **counts,
               "increased_not_fired": increased_not_fired}
        for target, rate in rates.items():
            keep = bool(n_ok) and DROP_POS_RATE_LOW <= rate <= DROP_POS_RATE_HIGH
            row[f"{target}_rate"] = round(float(rate), 4) if n_ok else None
            row[f"keep_{target}"] = keep
            target_status[f"{pass_name}_{target}"] = "kept" if keep else "dropped"
        dist_rows.append(row)

    df_dist = pd.DataFrame(dist_rows)
    df_dist.to_csv(LABELS_DIR / "label_distribution.csv", index=False)
    df_dist.to_csv(RESULTS_DIR / "phase3_label_distribution.csv", index=False)
    with open(LABELS_DIR / "target_status.json", "w") as f:
        json.dump(target_status, f, indent=2)
    kept_passes = [r["pass_name"] for r in dist_rows
                   if r["keep_beneficial"] or r["keep_harmful"] or r["keep_fired"]]
    with open(LABELS_DIR / "kept_passes.json", "w") as f:
        json.dump(kept_passes, f)

    # ---- Determinism check ----
    n_checked, n_mismatch = check_determinism(df_labels, index_by_id, tools, DETERMINISM_SAMPLES)

    # ---- Gate report ----
    def mark(flag):
        return "keep" if flag else "drop"

    lines = []
    lines.append("PHASE 3 GATE REPORT")
    lines.append("=" * 100)
    lines.append(f"LLVM version:                 {llvm_version}")
    lines.append(f"Mode:                         {'SMOKE' if smoke else 'FULL'}")
    lines.append(f"Functions:                    {len(func_list)}")
    lines.append(f"Programs (groups):            {df_features['program'].nunique()}")
    lines.append(f"States (samples):             {len(df_features)}")
    lines.append(f"  baseline states:            {(df_features['state'] == 'baseline').sum()}")
    lines.append(f"  prefix states:              {(df_features['state'] != 'baseline').sum()}")
    lines.append(f"  dropped, same IR as an earlier state of the function: {n_states_dropped_same}")
    lines.append(f"  dropped, same IR as a state of another function:      {n_states_dropped_global}")
    lines.append(f"Label rows (state x pass):    {len(df_labels)}")
    lines.append(f"Failed opt calls:             {len(all_failures)}  (see results/phase3_failures.csv)")
    lines.append(f"Feature columns:              {len(df_features.columns) - 6}")
    lines.append(f"Determinism check:            {n_checked - n_mismatch}/{n_checked} rows identical on rerun")
    lines.append(f"Run time this session:        {elapsed:.0f} s for {len(todo)} functions")
    lines.append("")
    lines.append(f"{'pass':<24}{'n':>6}{'err':>5}{'fired':>7}{'decr':>7}{'equal':>7}{'incr':>7}"
                 f"{'incr,not fired':>16}  {'beneficial':>16}{'harmful':>16}{'fired':>16}")
    for r in dist_rows:
        def cell(target):
            rate = r[f"{target}_rate"]
            return f"{rate * 100:5.1f}% {mark(r[f'keep_{target}'])}" if rate is not None else "n/a"
        lines.append(f"{r['pass_name']:<24}{r['samples']:>6}{r['errors']:>5}{r['fired']:>7}"
                     f"{r['decreased']:>7}{r['equal']:>7}{r['increased']:>7}"
                     f"{r['increased_not_fired']:>16}  {cell('beneficial'):>16}"
                     f"{cell('harmful'):>16}{cell('fired'):>16}")
    lines.append("")
    lines.append("decr / equal / incr compare the instruction count after the pass with the count before.")
    lines.append("fired = the pass output differs from its control run (see the file header).")
    lines.append("'incr,not fired' = size grew only because of loop canonicalization, not the pass.")
    lines.append(f"A target is kept when its positive rate is between "
                 f"{DROP_POS_RATE_LOW:.0%} and {DROP_POS_RATE_HIGH:.0%}.")
    report = "\n".join(lines) + "\n"
    log.info("\n%s", report)
    (RESULTS_DIR / "phase3_gate.txt").write_text(report, encoding="utf-8")

    if n_mismatch:
        raise RuntimeError(f"Determinism check failed on {n_mismatch} of {n_checked} rows.")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 3: features and optimization labels")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--force", action="store_true", help="Ignore the resume cache")
    args = parser.parse_args()
    run_phase3(smoke=args.smoke, n_jobs=args.jobs, force=args.force)


if __name__ == "__main__":
    main()
