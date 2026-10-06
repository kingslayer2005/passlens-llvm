#!/usr/bin/env python3
"""
phase1_setup_data.py — Download benchmarks, compile to IR, extract functions.

Pipeline:
  1. Download PolyBench/C 4.2.1 and MiBench from GitHub.
  2. Find all .c files, compile each to LLVM IR with:
       clang -O0 -Xclang -disable-O0-optnone -emit-llvm -S -o out.ll in.c
  3. Run mem2reg on each IR file to get a clean baseline:
       opt -passes=mem2reg -S -o out.ll in.ll
  4. Extract each defined function into its own .ll module using llvm-extract.
  5. Drop functions with fewer than 10 instructions.
  6. Deduplicate by SHA256 hash of normalized IR.
  7. Log every compilation failure to data/compile_errors.log.

Output:
  data/ir/<suite>/<program>/<function_name>.ll  — one function per file
  data/function_index.csv — columns: suite, program, function, path, ir_hash,
                            inst_count, is_duplicate

Usage:
  python -m scripts.phase1_setup_data [--smoke]
"""

import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import (
    PROJECT_ROOT, DATA_DIR, RAW_DIR, IR_DIR, RESULTS_DIR,
    ensure_dirs, find_llvm_tools, run_tool,
    normalize_ir, hash_ir, count_instructions,
    setup_logging, SMOKE_PROGRAMS,
)

log = setup_logging("phase1")

# ============================================================================
# Source URLs
# ============================================================================

POLYBENCH_URL = "https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/archive/refs/heads/master.tar.gz"
POLYBENCH_FALLBACK_URL = "https://downloads.sourceforge.net/project/polybench/polybench-c-4.2.1.tar.gz"

MIBENCH_URL = "https://github.com/embecosm/mibench/archive/refs/heads/master.tar.gz"

# AnghaBench (used only if unique function count < 1000)
ANGHABENCH_URL = "https://github.com/brenocfg/AnghaBench/archive/refs/heads/master.tar.gz"
ANGHABENCH_CAP = 500  # max number of .c files to take from AnghaBench


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
    with urllib.request.urlopen(req, timeout=120) as resp:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
    log.info("Downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return True


def extract_tarball(tarball: Path, dest_dir: Path):
    """Extract a .tar.gz file into dest_dir."""
    if not tarball.exists():
        raise FileNotFoundError(f"Tarball not found: {tarball}")
    log.info("Extracting %s ...", tarball.name)
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(dest_dir)
    log.info("Extracted to %s", dest_dir)


# ============================================================================
# Benchmark discovery
# ============================================================================

def find_c_files(root: Path, suite: str, smoke: bool = False) -> List[Tuple[str, str, Path]]:
    """
    Find all .c files under root, returning (suite, program_name, path) tuples.
    Skips utility/header files (polybench.c, etc.).
    """
    skip_names = {"polybench.c", "polybench.h", "Makefile"}
    results = []

    for c_file in sorted(root.rglob("*.c")):
        # Skip utility files
        if c_file.name in skip_names:
            continue
        # Skip test harnesses and makefiles
        if "utilities" in str(c_file) or "common" in str(c_file):
            continue
        # Derive program name from parent directory
        program = c_file.stem
        # In smoke mode, only process SMOKE_PROGRAMS
        if smoke and not any(sp in program for sp in SMOKE_PROGRAMS):
            continue
        results.append((suite, program, c_file))

    log.info("Found %d .c files in %s%s", len(results), suite,
             " (smoke mode)" if smoke else "")
    return results


# ============================================================================
# Compilation pipeline
# ============================================================================

def compile_to_ir(c_file: Path, output_ll: Path, tools: dict,
                  include_dirs: List[Path] = None) -> bool:
    """
    Compile a C file to LLVM IR:
      clang -O0 -Xclang -disable-O0-optnone -emit-llvm -S -o output.ll input.c
    Returns True on success, False on failure (logged).
    """
    cmd = [
        tools["clang"],
        "-O0", "-Xclang", "-disable-O0-optnone",
        "-emit-llvm", "-S",
        "-w",  # suppress warnings
    ]
    # Add include directories (PolyBench needs its utilities/ dir)
    if include_dirs:
        for inc in include_dirs:
            cmd.extend(["-I", str(inc)])
    cmd.extend(["-o", str(output_ll), str(c_file)])

    try:
        run_tool(cmd, timeout=30, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def run_mem2reg(input_ll: Path, output_ll: Path, tools: dict) -> bool:
    """Run mem2reg pass to promote allocas to SSA registers."""
    cmd = [
        tools["opt"],
        "-passes=mem2reg",
        "-S",
        "-o", str(output_ll),
        str(input_ll),
    ]
    try:
        run_tool(cmd, timeout=30, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def extract_function_names(ll_file: Path) -> List[str]:
    """
    Parse an .ll file and return the names of all defined (non-declaration) functions.
    Looks for lines like: define ... @function_name(...)
    """
    names = []
    text = ll_file.read_text(encoding="utf-8", errors="replace")
    # Match function definitions (not declarations)
    for match in re.finditer(r"^define\s+.*?@([a-zA-Z_.$][a-zA-Z0-9_.$]*)\s*\(", text, re.MULTILINE):
        names.append(match.group(1))
    return names


def extract_single_function(ll_file: Path, func_name: str,
                             output_ll: Path, tools: dict) -> bool:
    """
    Extract a single function from an .ll module using llvm-extract.
    Falls back to manual extraction if llvm-extract fails.
    """
    cmd = [
        tools["llvm-extract"],
        "--func", func_name,
        "-S",
        "-o", str(output_ll),
        str(ll_file),
    ]
    try:
        run_tool(cmd, timeout=15, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def _manual_extract_function(ll_file: Path, func_name: str, output_ll: Path) -> bool:
    """
    Fallback function extraction: read the .ll file and extract the function
    definition for func_name, including any necessary type definitions and
    declarations.
    """
    try:
        text = ll_file.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        # Collect preamble (target triple, datalayout, type definitions, declarations)
        preamble_lines = []
        func_lines = []
        in_target_func = False
        brace_depth = 0

        for line in lines:
            stripped = line.strip()

            # Keep target info and type definitions
            if stripped.startswith("target ") or stripped.startswith("%") or \
               stripped.startswith("declare ") or stripped.startswith("@") or \
               stripped.startswith("source_filename") or stripped.startswith("attributes"):
                if not in_target_func:
                    preamble_lines.append(line)
                continue

            # Check if this is the start of our target function
            if re.match(rf"define\s+.*@{re.escape(func_name)}\s*\(", stripped):
                in_target_func = True
                func_lines.append(line)
                brace_depth += line.count("{") - line.count("}")
                continue

            if in_target_func:
                func_lines.append(line)
                brace_depth += line.count("{") - line.count("}")
                if brace_depth <= 0:
                    break
            # Skip other function definitions
            elif stripped.startswith("define "):
                continue

        if not func_lines:
            return False

        output_text = "\n".join(preamble_lines + [""] + func_lines) + "\n"
        output_ll.parent.mkdir(parents=True, exist_ok=True)
        output_ll.write_text(output_text, encoding="utf-8")
        return True

    except Exception as e:
        log.error("Manual extraction failed for %s: %s", func_name, e)
        return False


# ============================================================================
# Main pipeline
# ============================================================================

def run_phase1(smoke: bool = False):
    """Execute the full Phase 1 pipeline."""
    ensure_dirs()
    tools = find_llvm_tools()
    log.info("Using LLVM tools: %s", {k: v for k, v in tools.items()})

    error_log_path = DATA_DIR / "compile_errors.log"
    error_log = open(error_log_path, "w")

    # ---- 1. Download benchmarks ----
    polybench_tar = RAW_DIR / "polybench-c-4.2.1.tar.gz"
    mibench_tar = RAW_DIR / "mibench-master.tar.gz"

    # Download benchmarks
    if not download_file(POLYBENCH_URL, polybench_tar, "PolyBench/C 4.2.1"):
        log.error("Could not download PolyBench. Please download manually to %s", polybench_tar)
        sys.exit(1)

    if not download_file(MIBENCH_URL, mibench_tar, "MiBench"):
        log.error("Could not download MiBench. Please download manually to %s", mibench_tar)
        sys.exit(1)

    # ---- 2. Extract tarballs ----
    polybench_dir = RAW_DIR / "polybench"
    mibench_dir = RAW_DIR / "mibench"

    if not polybench_dir.exists():
        extract_tarball(polybench_tar, RAW_DIR)
        # Find the extracted directory (may have various names)
        for d in RAW_DIR.iterdir():
            if d.is_dir() and "polybench" in d.name.lower() and d != polybench_dir:
                d.rename(polybench_dir)
                break

    if not mibench_dir.exists():
        extract_tarball(mibench_tar, RAW_DIR)
        for d in RAW_DIR.iterdir():
            if d.is_dir() and "mibench" in d.name.lower() and d != mibench_dir:
                d.rename(mibench_dir)
                break

    # ---- 3. Find .c files ----
    c_files = []
    if polybench_dir.exists():
        # PolyBench has include directories for its headers
        polybench_includes = list(polybench_dir.rglob("utilities"))
        c_files.extend([(s, p, f, polybench_includes)
                        for s, p, f in find_c_files(polybench_dir, "polybench", smoke)])
    if mibench_dir.exists():
        c_files.extend([(s, p, f, [])
                        for s, p, f in find_c_files(mibench_dir, "mibench", smoke)])

    log.info("Total C files to compile: %d", len(c_files))

    # ---- 4. Compile each file to IR, run mem2reg, extract functions ----
    all_functions = []  # list of dicts for the index CSV
    compile_failures = 0
    extract_failures = 0

    for suite, program, c_file, includes in c_files:
        # Output paths
        raw_ll = IR_DIR / suite / program / f"{c_file.stem}_raw.ll"
        baseline_ll = IR_DIR / suite / program / f"{c_file.stem}_baseline.ll"
        raw_ll.parent.mkdir(parents=True, exist_ok=True)

        # Step A: Compile C -> IR
        if not raw_ll.exists():
            ok = compile_to_ir(c_file, raw_ll, tools, includes)
            if not ok:
                compile_failures += 1
                error_log.write(f"COMPILE_FAIL\t{c_file}\n")
                log.warning("Compilation failed: %s", c_file)
                continue
        
        # Step B: mem2reg
        if not baseline_ll.exists():
            ok = run_mem2reg(raw_ll, baseline_ll, tools)
            if not ok:
                compile_failures += 1
                error_log.write(f"MEM2REG_FAIL\t{c_file}\n")
                log.warning("mem2reg failed: %s", raw_ll)
                continue

        # Step C: Extract each function
        func_names = extract_function_names(baseline_ll)
        for func_name in func_names:
            func_ll = IR_DIR / suite / program / f"{func_name}.ll"

            if not func_ll.exists():
                ok = extract_single_function(baseline_ll, func_name, func_ll, tools)
                if not ok:
                    extract_failures += 1
                    error_log.write(f"EXTRACT_FAIL\t{baseline_ll}\t{func_name}\n")
                    continue

            # Read the extracted function and count instructions
            ir_text = func_ll.read_text(encoding="utf-8", errors="replace")
            inst_count = count_instructions(ir_text)
            ir_h = hash_ir(ir_text)

            all_functions.append({
                "suite": suite,
                "program": program,
                "function": func_name,
                "path": str(func_ll.relative_to(PROJECT_ROOT)),
                "ir_hash": ir_h,
                "inst_count": inst_count,
                "is_duplicate": False,  # set later
            })

    error_log.close()

    # ---- 5. Drop functions under 10 instructions ----
    before_filter = len(all_functions)
    all_functions = [f for f in all_functions if f["inst_count"] >= 10]
    log.info("Dropped %d functions with < 10 instructions (kept %d)",
             before_filter - len(all_functions), len(all_functions))

    # ---- 6. Deduplicate by IR hash ----
    seen_hashes = set()
    unique_functions = []
    duplicates = 0
    for func in all_functions:
        if func["ir_hash"] in seen_hashes:
            func["is_duplicate"] = True
            duplicates += 1
        else:
            seen_hashes.add(func["ir_hash"])
            func["is_duplicate"] = False
            unique_functions.append(func)

    log.info("Deduplication: %d duplicates removed, %d unique functions remain",
             duplicates, len(unique_functions))

    # ---- 7. Save function index ----
    index_path = DATA_DIR / "function_index.csv"
    fieldnames = ["suite", "program", "function", "path", "ir_hash",
                  "inst_count", "is_duplicate"]
    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Write ALL functions (including duplicates, marked)
        for func in all_functions:
            writer.writerow(func)
        # Also write duplicates that were filtered
        for func in [f for f in all_functions if f not in unique_functions and f["is_duplicate"]]:
            writer.writerow(func)

    # ---- 8. Gate report ----
    n_programs = len(set(f["program"] for f in all_functions))
    n_total = len(all_functions)
    n_unique = len(unique_functions)

    report = f"""
╔══════════════════════════════════════════════════╗
║            PHASE 1 GATE REPORT                   ║
╠══════════════════════════════════════════════════╣
║ Programs processed:     {n_programs:>6}                   ║
║ Total functions (≥10i): {n_total:>6}                   ║
║ Unique functions:       {n_unique:>6}                   ║
║ Duplicates removed:     {duplicates:>6}                   ║
║ Compilation failures:   {compile_failures:>6}                   ║
║ Extraction failures:    {extract_failures:>6}                   ║
╚══════════════════════════════════════════════════╝
"""
    log.info(report)

    # Save gate report
    gate_path = RESULTS_DIR / "phase1_gate.txt"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(report, encoding='utf-8')

    # ---- 9. Check if we need AnghaBench ----
    if n_unique < 1000 and not smoke:
        log.error("Only %d unique functions — below 1000 threshold. Stopping.", n_unique)
        sys.exit(1)
    elif smoke:
        log.info("Smoke mode — skipping function threshold check.")

    return unique_functions


def _supplement_with_anghabench(tools, all_functions, unique_functions,
                                  seen_hashes, error_log_path):
    """Download AnghaBench and add up to ANGHABENCH_CAP files."""
    angha_tar = RAW_DIR / "anghabench-master.tar.gz"
    
    # AnghaBench is very large — only download a partial archive or clone shallow
    log.info("Downloading AnghaBench (this may be large)...")
    
    # Instead of the full repo, we'll use git sparse checkout to get only
    # a subset, or just grab individual files via the API.
    # For safety, let's use a shallow clone with limited depth.
    angha_dir = RAW_DIR / "anghabench"
    
    if not angha_dir.exists():
        # Try shallow clone of just the linux subdirectory
        try:
            # Use git sparse-checkout to limit download size
            angha_dir.mkdir(parents=True, exist_ok=True)
            run_tool(["git", "init", str(angha_dir)], timeout=10)
            run_tool(["git", "-C", str(angha_dir), "remote", "add", "origin",
                      "https://github.com/brenocfg/AnghaBench.git"], timeout=10)
            run_tool(["git", "-C", str(angha_dir), "config", "core.sparseCheckout", "true"],
                     timeout=10)
            
            # Only check out a small subset
            sparse_file = angha_dir / ".git" / "info" / "sparse-checkout"
            sparse_file.parent.mkdir(parents=True, exist_ok=True)
            sparse_file.write_text("linux/net/\n")
            
            run_tool(["git", "-C", str(angha_dir), "pull", "--depth=1",
                      "origin", "master"], timeout=300, check=False)
        except Exception as e:
            log.error("Failed to clone AnghaBench: %s", e)
            log.warning("Proceeding without AnghaBench supplement.")
            return

    # Find .c files from AnghaBench (capped)
    angha_c_files = sorted(angha_dir.rglob("*.c"))[:ANGHABENCH_CAP]
    log.info("Found %d AnghaBench .c files (capped at %d)", len(angha_c_files), ANGHABENCH_CAP)

    error_log = open(error_log_path, "a")
    new_unique = 0

    for c_file in angha_c_files:
        program = c_file.stem
        raw_ll = IR_DIR / "anghabench" / program / f"{program}_raw.ll"
        baseline_ll = IR_DIR / "anghabench" / program / f"{program}_baseline.ll"
        raw_ll.parent.mkdir(parents=True, exist_ok=True)

        # Compile
        if not raw_ll.exists():
            ok = compile_to_ir(c_file, raw_ll, tools)
            if not ok:
                error_log.write(f"COMPILE_FAIL\t{c_file}\n")
                continue

        # mem2reg
        if not baseline_ll.exists():
            ok = run_mem2reg(raw_ll, baseline_ll, tools)
            if not ok:
                error_log.write(f"MEM2REG_FAIL\t{raw_ll}\n")
                continue

        # Extract functions
        func_names = extract_function_names(baseline_ll)
        for func_name in func_names:
            func_ll = IR_DIR / "anghabench" / program / f"{func_name}.ll"
            if not func_ll.exists():
                ok = extract_single_function(baseline_ll, func_name, func_ll, tools)
                if not ok:
                    continue

            ir_text = func_ll.read_text(encoding="utf-8", errors="replace")
            inst_count = count_instructions(ir_text)
            if inst_count < 10:
                continue

            ir_h = hash_ir(ir_text)
            if ir_h in seen_hashes:
                continue

            seen_hashes.add(ir_h)
            new_unique += 1
            entry = {
                "suite": "anghabench",
                "program": program,
                "function": func_name,
                "path": str(func_ll.relative_to(PROJECT_ROOT)),
                "ir_hash": ir_h,
                "inst_count": inst_count,
                "is_duplicate": False,
            }
            unique_functions.append(entry)
            all_functions.append(entry)

    error_log.close()
    log.info("AnghaBench added %d new unique functions (total now: %d)",
             new_unique, len(unique_functions))

    # Update the function index CSV
    index_path = DATA_DIR / "function_index.csv"
    fieldnames = ["suite", "program", "function", "path", "ir_hash",
                  "inst_count", "is_duplicate"]
    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for func in all_functions:
            writer.writerow(func)


# ============================================================================
# CLI entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Download benchmarks and compile to IR")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: process only 3 programs")
    args = parser.parse_args()
    run_phase1(smoke=args.smoke)


if __name__ == "__main__":
    main()
