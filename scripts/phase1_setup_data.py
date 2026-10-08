#!/usr/bin/env python3
"""
phase1_setup_data.py — Fetch benchmarks, compile to IR, extract functions.

Pipeline:
  1. Fetch PolyBench/C 4.2.1 and MiBench (pinned commits, see SOURCES below).
  2. Compile every .c file to LLVM IR:
       clang -O0 -Xclang -disable-O0-optnone -emit-llvm -S -w -I<dirs> in.c
     with -I for every directory that holds headers inside the SAME benchmark.
     If that fails, retry ONCE in a legacy C dialect (-std=gnu89 and the
     pre-clang-16 behaviour for implicit declarations), because MiBench is
     pre-C99 code.  Which mode was used is recorded per function.
  3. Run mem2reg to get the baseline IR.
  4. Extract every defined function into its own module (llvm-extract).
  5. Cross-check our instruction counter against LLVM's own count.
  6. Drop functions with fewer than 10 instructions.
  7. Deduplicate by hash of the normalized function body.
  8. Cap each benchmark program at N functions (seeded sample).
  9. Write the selected baseline IR files and the function index.

Nothing from the benchmarks is ever executed; files are only compiled.

Output:
  data/ir/<suite>/<program>/<id>.ll   one baseline function per file
  data/function_index.csv             one row per selected function
  data/manifest.csv                   same table (small, safe to commit)
  data/SOURCES.md                     where the benchmark sources came from
  results/compile_failures.csv        every file that failed, with the reason
  results/instcount_validation.txt    our counter vs LLVM's counter
  results/phase1_gate.txt             gate report

Usage:
  python -m scripts.phase1_setup_data [--smoke] [--jobs N] [--cap N]
"""

import argparse
import csv
import hashlib
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import (
    PROJECT_ROOT, DATA_DIR, RAW_DIR, IR_DIR, RESULTS_DIR,
    ensure_dirs, find_llvm_tools, require_pinned_llvm, run_tool,
    hash_ir, count_instructions,
    setup_logging, SMOKE_PROGRAMS,
)
from scripts.phase2_features import reachable_counts

log = setup_logging("phase1")

# ============================================================================
# Benchmark sources (pinned to exact commits so every run sees the same code)
# ============================================================================

SOURCES = {
    "polybench": {
        "repo": "https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1",
        "commit": "3e872547cef7e5c9909422ef1e6af03cf4e56072",
    },
    "mibench": {
        "repo": "https://github.com/embecosm/mibench",
        "commit": "0f3cbcf6b3d589a2b0753cfb9289ddf40b6b9ed8",
    },
}

# ============================================================================
# Configuration
# ============================================================================

MIN_INSTRUCTIONS = 10          # drop functions smaller than this
DEFAULT_CAP = 100              # max functions per benchmark program
ESCALATED_CAP = 200            # used only if DEFAULT_CAP gives < MIN_UNIQUE
MIN_UNIQUE_FUNCTIONS = 1000    # Phase 1 gate
SAMPLE_SEED = 42               # seed for the per-program sample
COMPILE_TIMEOUT = 60           # seconds per clang call

# Flags for the first compile attempt
BASE_FLAGS = ["-O0", "-Xclang", "-disable-O0-optnone", "-emit-llvm", "-S", "-w"]

# Extra flags for the single retry.  clang 16+ turned several old-C habits
# into hard errors; these flags restore the older behaviour (warnings).
LEGACY_C_FLAGS = [
    "-std=gnu89",
    "-Wno-error=implicit-function-declaration",
    "-Wno-error=implicit-int",
    "-Wno-error=int-conversion",
    "-Wno-error=incompatible-function-pointer-types",
    "-Wno-error=incompatible-pointer-types",
    "-Wno-error=return-type",
]


# ============================================================================
# Download helpers
# ============================================================================

def download_file(url: str, dest: Path, label: str = "") -> bool:
    """Download a file from URL to dest.  Returns True on success."""
    if dest.exists():
        log.info("Already downloaded: %s", dest.name)
        return True
    log.info("Downloading %s from %s ...", label or dest.name, url)
    # Use urllib to avoid extra dependencies
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
    log.info("Downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return True


def fetch_suite(suite: str) -> Path:
    """
    Make sure data/raw/<suite> exists and return its path.
    If the folder is already there (from an earlier run or a manual copy),
    it is used as it is.  Otherwise the pinned commit is downloaded.
    """
    suite_dir = RAW_DIR / suite
    if suite_dir.exists():
        log.info("Using existing sources: %s", suite_dir)
        return suite_dir

    info = SOURCES[suite]
    url = f"{info['repo']}/archive/{info['commit']}.tar.gz"
    tarball = RAW_DIR / f"{suite}-{info['commit'][:12]}.tar.gz"
    download_file(url, tarball, suite)

    # Extract into a temporary folder, then rename the single top-level
    # folder inside the archive to data/raw/<suite>
    tmp_dir = RAW_DIR / f"_{suite}_extract"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    log.info("Extracting %s ...", tarball.name)
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(tmp_dir)
    top_level = [d for d in tmp_dir.iterdir() if d.is_dir()]
    if len(top_level) != 1:
        raise RuntimeError(f"Unexpected archive layout in {tarball}: {top_level}")
    top_level[0].rename(suite_dir)
    tmp_dir.rmdir()
    tarball.unlink()   # the archive is no longer needed (saves disk space)
    return suite_dir


def write_sources_md():
    """Record where the benchmark sources came from."""
    lines = ["# Benchmark sources", "",
             "Fetched by `scripts/phase1_setup_data.py`. The sources themselves",
             "are not part of this repository (each suite has its own license).",
             ""]
    for suite, info in SOURCES.items():
        lines.append(f"- **{suite}**: {info['repo']} at commit `{info['commit']}`")
    (DATA_DIR / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================================
# Benchmark discovery
# ============================================================================

def find_c_files(suite: str, suite_dir: Path, smoke: bool = False) -> List[dict]:
    """
    Find every .c file of a suite.  Returns a list of dicts:
      suite, program, c_file (Path), rel_path (str), include_dirs (list)

    "program" is the benchmark program the file belongs to.  It is the
    grouping unit for cross-validation, so files of one benchmark must
    never be split across two program names.
      PolyBench: the kernel folder                  (e.g. "2mm")
      MiBench:   <category>/<benchmark> -> benchmark (e.g. "ghostscript")
    """
    entries = []

    for c_file in sorted(suite_dir.rglob("*.c")):
        rel = c_file.relative_to(suite_dir)
        parts = rel.parts
        if ".git" in parts:
            continue

        if suite == "polybench":
            # utilities/ holds the shared harness (polybench.c) and a
            # template file; they are not benchmark kernels.
            if parts[0] == "utilities":
                continue
            program = c_file.parent.name
            benchmark_root = c_file.parent
            extra_include_roots = [suite_dir / "utilities"]
        else:
            # MiBench layout: <category>/<benchmark>/.../file.c
            if len(parts) < 3:
                continue
            program = parts[1]
            benchmark_root = suite_dir / parts[0] / parts[1]
            extra_include_roots = []

        if smoke and program not in SMOKE_PROGRAMS:
            continue

        # Include path: the file's own folder first, then every folder
        # inside the same benchmark that contains at least one header.
        header_dirs = sorted(set(h.parent for h in benchmark_root.rglob("*.h")))
        include_dirs = [c_file.parent]
        for d in header_dirs + extra_include_roots:
            if d not in include_dirs:
                include_dirs.append(d)

        entries.append({
            "suite": suite,
            "program": program,
            "c_file": c_file,
            "rel_path": rel.as_posix(),
            "include_dirs": include_dirs,
        })

    log.info("Found %d .c files in %s%s", len(entries), suite,
             " (smoke mode)" if smoke else "")
    return entries


# ============================================================================
# Compilation pipeline
# ============================================================================

def first_error_line(stderr_text: str) -> str:
    """Return the first compiler error message, shortened, for the failure log."""
    for line in stderr_text.splitlines():
        if "error:" in line:
            message = line.split("error:", 1)[1].strip()
            return message[:160]
    stripped = stderr_text.strip().splitlines()
    return stripped[0][:160] if stripped else "unknown error"


def compile_to_ir(c_file: Path, include_dirs: List[Path], tools: dict,
                  legacy: bool) -> Tuple[Optional[str], str]:
    """
    Compile one C file to LLVM IR text (written to stdout, not to disk).
    Returns (ir_text, "") on success or (None, reason) on failure.
    """
    cmd = [tools["clang"]] + BASE_FLAGS
    if legacy:
        cmd += LEGACY_C_FLAGS
    for inc in include_dirs:
        cmd += ["-I", str(inc)]
    cmd += ["-o", "-", str(c_file)]

    try:
        result = run_tool(cmd, timeout=COMPILE_TIMEOUT, check=False)
    except subprocess.TimeoutExpired:
        return None, f"timeout after {COMPILE_TIMEOUT}s"
    if result.returncode != 0:
        return None, first_error_line(result.stderr)
    return result.stdout, ""


def run_mem2reg(ir_text: str, tools: dict) -> Tuple[Optional[str], str]:
    """Run mem2reg (promote allocas to SSA registers) on IR text."""
    cmd = [tools["opt"], "-passes=mem2reg", "-S"]
    try:
        result = run_tool(cmd, timeout=COMPILE_TIMEOUT, check=False, input_data=ir_text)
    except subprocess.TimeoutExpired:
        return None, f"mem2reg timeout after {COMPILE_TIMEOUT}s"
    if result.returncode != 0:
        return None, "mem2reg: " + first_error_line(result.stderr)
    return result.stdout, ""


def extract_function_names(ir_text: str) -> List[str]:
    """
    Return the names of all defined (non-declaration) functions.
    Looks for lines like: define ... @function_name(...)
    """
    names = []
    for match in re.finditer(r"^define\s+.*?@([a-zA-Z_.$][a-zA-Z0-9_.$]*)\s*\(",
                             ir_text, re.MULTILINE):
        names.append(match.group(1))
    return names


def extract_single_function(module_ir: str, func_name: str, tools: dict) -> Optional[str]:
    """
    Extract one function from a module with llvm-extract.
    Returns the IR text of a module that contains only that definition
    (everything it calls becomes a declaration), or None on failure.
    """
    cmd = [tools["llvm-extract"], "--func", func_name, "-S", "-o", "-", "-"]
    try:
        result = run_tool(cmd, timeout=30, check=False, input_data=module_ir)
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def llvm_instruction_counts(module_ir: str, tools: dict) -> Dict[str, Tuple[int, int]]:
    """
    Ask LLVM itself how many instructions and basic blocks each function has.
    Returns {function_name: (TotalInstructionCount, BasicBlockCount)}.
    Used only to validate our text-based counter.
    """
    cmd = [tools["opt"], "-passes=print<func-properties>", "-disable-output"]
    result = run_tool(cmd, timeout=COMPILE_TIMEOUT, check=True, input_data=module_ir)
    counts = {}
    current = None
    bb = None
    for line in (result.stderr + result.stdout).splitlines():
        match = re.match(r"Printing analysis results of CFA for function '(.*)':", line)
        if match:
            current = match.group(1)
            bb = None
            continue
        if current is None:
            continue
        if line.startswith("BasicBlockCount:"):
            bb = int(line.split(":")[1])
        elif line.startswith("TotalInstructionCount:"):
            counts[current] = (int(line.split(":")[1]), bb)
    return counts


def process_one_file(entry: dict, tools: dict) -> dict:
    """
    Compile one C file and extract all of its functions.
    Runs in a worker process.  Returns a dict with:
      status         "ok" or "failed"
      stage, reason  where and why it failed (if it failed)
      compile_mode   "default" or "legacy_c89"
      functions      list of dicts (function, ir_text, inst_count, ir_hash)
      mismatches     list of counter disagreements with LLVM
    """
    out = {"entry": entry, "status": "failed", "stage": "", "reason": "",
           "compile_mode": "", "functions": [], "mismatches": [],
           "extract_failures": 0, "n_defined": 0, "with_dead_code": 0}

    # Step A: C -> IR (default dialect, then one legacy retry)
    ir_text, reason = compile_to_ir(entry["c_file"], entry["include_dirs"], tools, legacy=False)
    mode = "default"
    if ir_text is None:
        first_reason = reason
        ir_text, reason = compile_to_ir(entry["c_file"], entry["include_dirs"], tools, legacy=True)
        mode = "legacy_c89"
        if ir_text is None:
            out["stage"] = "compile"
            # Report the error from the legacy attempt; it is the one that
            # remains after old-C habits are allowed.
            out["reason"] = reason
            out["first_reason"] = first_reason
            return out
    out["compile_mode"] = mode

    # Step B: mem2reg
    baseline_ir, reason = run_mem2reg(ir_text, tools)
    if baseline_ir is None:
        out["stage"] = "mem2reg"
        out["reason"] = reason
        return out

    # Step C: LLVM's own counts, to validate our counter
    llvm_counts = llvm_instruction_counts(baseline_ir, tools)

    # Step D: extract each defined function
    func_names = extract_function_names(baseline_ir)
    out["n_defined"] = len(func_names)
    for func_name in func_names:
        func_ir = extract_single_function(baseline_ir, func_name, tools)
        if func_ir is None:
            out["extract_failures"] += 1
            continue

        inst_count = count_instructions(func_ir)

        # Validate our parser against LLVM (every function, not a sample).
        # LLVM's analysis counts only blocks reachable from the entry, so
        # we compare it with OUR reachable counts.  This checks the
        # instruction counter, the block parser and the branch-target
        # parser at the same time.
        reach_insts, reach_blocks = reachable_counts(func_ir, func_name)
        if reach_insts != inst_count:
            out["with_dead_code"] += 1
        if func_name in llvm_counts:
            if llvm_counts[func_name] != (reach_insts, reach_blocks):
                out["mismatches"].append((entry["rel_path"], func_name,
                                          (reach_insts, reach_blocks), llvm_counts[func_name]))
        else:
            out["mismatches"].append((entry["rel_path"], func_name,
                                      (reach_insts, reach_blocks), None))

        if inst_count < MIN_INSTRUCTIONS:
            continue

        out["functions"].append({
            "function": func_name,
            "ir_text": func_ir,
            "inst_count": inst_count,
            "ir_hash": hash_ir(func_ir, func_name),
        })

    out["status"] = "ok"
    return out


# ============================================================================
# Selection: dedup + per-program cap
# ============================================================================

def select_functions(candidates: List[dict], cap: int) -> Tuple[List[dict], int]:
    """
    candidates: every function with >= MIN_INSTRUCTIONS, in a fixed order.
    Returns (selected, n_duplicates_removed).

    1. Deduplicate by ir_hash over the WHOLE dataset (first one wins), so
       the same function body can never sit in two programs.
    2. Cap each program at `cap` functions with a seeded random sample.
       The seed depends only on the program name, so the sample of one
       program does not change when another program is added.
    """
    seen = set()
    unique = []
    for cand in candidates:
        if cand["ir_hash"] in seen:
            continue
        seen.add(cand["ir_hash"])
        unique.append(cand)
    n_duplicates = len(candidates) - len(unique)

    by_program = defaultdict(list)
    for cand in unique:
        by_program[(cand["suite"], cand["program"])].append(cand)

    selected = []
    for key in sorted(by_program):
        funcs = by_program[key]
        if len(funcs) > cap:
            rng = random.Random(f"{SAMPLE_SEED}:{key[0]}:{key[1]}")
            chosen_ids = set(rng.sample(range(len(funcs)), cap))
            funcs = [f for i, f in enumerate(funcs) if i in chosen_ids]
        selected.extend(funcs)
    return selected, n_duplicates


def safe_file_name(func_id: str) -> str:
    """A short, unique, filesystem-safe file name for a function id."""
    digest = hashlib.sha1(func_id.encode("utf-8")).hexdigest()[:10]
    readable = re.sub(r"[^A-Za-z0-9_.-]", "_", func_id.split("::")[-1])[:60]
    return f"{readable}__{digest}.ll"


# ============================================================================
# Main pipeline
# ============================================================================

def run_phase1(smoke: bool = False, n_jobs: int = 1, cap: Optional[int] = None):
    """Execute the full Phase 1 pipeline."""
    start_time = time.time()
    ensure_dirs()
    tools = find_llvm_tools()
    llvm_version = require_pinned_llvm(tools)
    log.info("Using LLVM %s tools: %s", llvm_version, tools)

    # ---- 1. Fetch benchmarks ----
    entries = []
    for suite in SOURCES:
        suite_dir = fetch_suite(suite)
        n_headers = len(list(suite_dir.rglob("*.h")))
        log.info("%s: %d header (.h) files present", suite, n_headers)
        entries.extend(find_c_files(suite, suite_dir, smoke))
    write_sources_md()
    log.info("Total C files to compile: %d", len(entries))

    # ---- 2. Compile + extract (one worker task per C file) ----
    if n_jobs == 1:
        results = [process_one_file(e, tools) for e in entries]
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_one_file)(e, tools) for e in entries)

    # ---- 3. Collect results in a fixed order (input order is sorted) ----
    candidates = []
    failures = []
    mismatches = []
    extract_failures = 0
    stats = defaultdict(Counter)       # suite -> counters
    n_validated = 0
    n_with_dead_code = 0   # functions that contain unreachable blocks

    for res in results:
        entry = res["entry"]
        suite = entry["suite"]
        stats[suite]["files"] += 1
        if res["status"] != "ok":
            stats[suite]["failed"] += 1
            failures.append({
                "suite": suite,
                "program": entry["program"],
                "file": entry["rel_path"],
                "stage": res["stage"],
                "reason": res["reason"],
            })
            continue

        stats[suite]["compiled"] += 1
        stats[suite]["compiled_" + res["compile_mode"]] += 1
        stats[suite]["functions_defined"] += res["n_defined"]
        extract_failures += res["extract_failures"]
        mismatches.extend(res["mismatches"])
        n_validated += res["n_defined"] - res["extract_failures"]
        n_with_dead_code += res["with_dead_code"]

        for func in res["functions"]:
            func_id = f"{suite}/{entry['rel_path']}::{func['function']}"
            candidates.append({
                "func_id": func_id,
                "suite": suite,
                "program": entry["program"],
                "source_file": entry["rel_path"],
                "function": func["function"],
                "ir_hash": func["ir_hash"],
                "inst_count": func["inst_count"],
                "compile_mode": res["compile_mode"],
                "ir_text": func["ir_text"],
            })

    # ---- 4. Counter validation (hard gate) ----
    validation_path = RESULTS_DIR / "instcount_validation.txt"
    with open(validation_path, "w", encoding="utf-8") as f:
        f.write("Check: (reachable instructions, reachable basic blocks) from our\n")
        f.write("text parser versus LLVM's print<func-properties>\n")
        f.write("(TotalInstructionCount, BasicBlockCount), for every function.\n\n")
        f.write(f"Functions checked: {n_validated}\n")
        f.write(f"Mismatches:        {len(mismatches)}\n")
        f.write(f"Functions that also contain unreachable blocks: {n_with_dead_code}\n")
        f.write("(labels count ALL instructions in the IR, including those blocks)\n")
        for rel_path, func_name, ours, llvm in mismatches[:200]:
            f.write(f"{rel_path}::{func_name} ours={ours} llvm={llvm}\n")
    if mismatches:
        raise RuntimeError(
            f"IR parser disagrees with LLVM on {len(mismatches)} of "
            f"{n_validated} functions. See {validation_path}. Stopping.")
    log.info("IR parser matches LLVM (instructions and blocks) on all %d functions.", n_validated)

    # ---- 5. Dedup + cap (escalate the cap once if the gate is missed) ----
    cap_used = cap if cap is not None else DEFAULT_CAP
    selected, n_duplicates = select_functions(candidates, cap_used)
    cap_note = f"{cap_used}"
    if cap is None and not smoke and len(selected) < MIN_UNIQUE_FUNCTIONS:
        log.warning("Only %d functions with cap %d; raising the cap to %d.",
                    len(selected), cap_used, ESCALATED_CAP)
        cap_at_default = len(selected)
        cap_used = ESCALATED_CAP
        selected, n_duplicates = select_functions(candidates, cap_used)
        cap_note = f"{cap_used} (cap {DEFAULT_CAP} gave only {cap_at_default})"

    # ---- 6. Write the selected baseline IR files and the index ----
    # Remove IR from earlier runs so the folder holds exactly this selection
    if IR_DIR.exists():
        shutil.rmtree(IR_DIR)
    index_rows = []
    for cand in selected:
        ll_path = IR_DIR / cand["suite"] / cand["program"] / safe_file_name(cand["func_id"])
        ll_path.parent.mkdir(parents=True, exist_ok=True)
        ll_path.write_text(cand["ir_text"], encoding="utf-8")
        index_rows.append({
            "func_id": cand["func_id"],
            "suite": cand["suite"],
            "program": cand["program"],
            "source_file": cand["source_file"],
            "function": cand["function"],
            "path": ll_path.relative_to(PROJECT_ROOT).as_posix(),
            "ir_hash": cand["ir_hash"],
            "inst_count": cand["inst_count"],
            "compile_mode": cand["compile_mode"],
            "is_duplicate": False,
        })

    fieldnames = ["func_id", "suite", "program", "source_file", "function", "path",
                  "ir_hash", "inst_count", "compile_mode", "is_duplicate"]
    for out_name in ("function_index.csv", "manifest.csv"):
        with open(DATA_DIR / out_name, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(index_rows)

    # ---- 7. Failure log with reasons ----
    with open(RESULTS_DIR / "compile_failures.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["suite", "program", "file", "stage", "reason"])
        writer.writeheader()
        writer.writerows(failures)

    # ---- 8. Gate report ----
    n_groups = len(set((r["suite"], r["program"]) for r in index_rows))
    per_suite_selected = Counter(r["suite"] for r in index_rows)
    reason_counts = Counter(f["reason"] for f in failures)
    failed_programs = Counter((f["suite"], f["program"]) for f in failures)

    lines = []
    lines.append("PHASE 1 GATE REPORT")
    lines.append("=" * 70)
    lines.append(f"LLVM version:                    {llvm_version}")
    lines.append(f"Mode:                            {'SMOKE' if smoke else 'FULL'}")
    lines.append("")
    lines.append(f"{'suite':<12}{'files':>7}{'compiled':>10}{'(default)':>11}"
                 f"{'(legacy)':>10}{'failed':>8}{'selected funcs':>16}")
    for suite in SOURCES:
        st = stats[suite]
        lines.append(f"{suite:<12}{st['files']:>7}{st['compiled']:>10}"
                     f"{st['compiled_default']:>11}{st['compiled_legacy_c89']:>10}"
                     f"{st['failed']:>8}{per_suite_selected[suite]:>16}")
    lines.append("")
    lines.append(f"Functions with >= {MIN_INSTRUCTIONS} instructions:   {len(candidates)}")
    lines.append(f"Duplicates removed (same body):  {n_duplicates}")
    lines.append(f"Cap per program:                 {cap_note}")
    lines.append(f"Groups (benchmark programs):     {n_groups}")
    lines.append(f"UNIQUE FUNCTIONS SELECTED:       {len(index_rows)}")
    lines.append(f"Extraction failures:             {extract_failures}")
    lines.append(f"Counter vs LLVM mismatches:      {len(mismatches)} of {n_validated}")
    lines.append("")
    lines.append("Most common reasons among files that still fail:")
    for reason, count in reason_counts.most_common(5):
        lines.append(f"  {count:>4}  {reason}")
    lines.append("")
    lines.append("Failed files per program:")
    for (suite, program), count in failed_programs.most_common():
        lines.append(f"  {count:>4}  {suite}/{program}")
    lines.append("")
    lines.append("Selected functions per program (10 largest):")
    per_program = Counter((r["suite"], r["program"]) for r in index_rows)
    for (suite, program), count in per_program.most_common(10):
        lines.append(f"  {count:>4}  {suite}/{program}")
    lines.append("")
    lines.append(f"Phase 1 run time: {time.time() - start_time:.0f} s")
    report = "\n".join(lines) + "\n"
    log.info("\n%s", report)
    (RESULTS_DIR / "phase1_gate.txt").write_text(report, encoding="utf-8")

    # ---- 9. Gate ----
    if not smoke and len(index_rows) < MIN_UNIQUE_FUNCTIONS:
        raise RuntimeError(
            f"Only {len(index_rows)} unique functions (< {MIN_UNIQUE_FUNCTIONS}) "
            f"even with cap {cap_used}. Add another source before continuing.")

    return index_rows


# ============================================================================
# CLI entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 1: fetch benchmarks and compile to IR")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: process only 3 PolyBench programs")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--cap", type=int, default=None,
                        help=f"Max functions per program (default {DEFAULT_CAP}, "
                             f"raised to {ESCALATED_CAP} if needed)")
    args = parser.parse_args()
    run_phase1(smoke=args.smoke, n_jobs=args.jobs, cap=args.cap)


if __name__ == "__main__":
    main()
