#!/usr/bin/env python3
"""
run_all.py — Master orchestrator for the LLVM IR SHAP research pipeline.

Runs every phase in order with caching (skips phases whose outputs already exist).

Flags:
  --smoke    Run the full pipeline on 3 programs in <5 minutes.
  --phase N  Run only phase N (1-7).
  --force    Force re-run even if outputs exist.
  --jobs N   Number of parallel jobs for Phase 3.

Usage:
  python run_all.py --smoke          # fast end-to-end test
  python run_all.py                  # full run
  python run_all.py --phase 4        # run only Phase 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path so scripts can find each other
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import (
    DATA_DIR, FEATURES_DIR, LABELS_DIR, RESULTS_DIR,
    ensure_dirs, find_llvm_tools, save_env_json, setup_logging,
)

log = setup_logging("run_all")

# ============================================================================
# Phase gate checks: does the output from this phase already exist?
# ============================================================================

def phase1_done() -> bool:
    return (DATA_DIR / "function_index.csv").exists()

def phase2_done() -> bool:
    return (FEATURES_DIR / "features.csv").exists()

def phase3_done() -> bool:
    return (LABELS_DIR / "labels.csv").exists()

def phase4_done() -> bool:
    return (RESULTS_DIR / "phase4_summary.csv").exists()

def phase5_done() -> bool:
    return (RESULTS_DIR / "shap_rankings.csv").exists()

def phase6_done() -> bool:
    return (RESULTS_DIR / "phase6_pruning.csv").exists()

def phase7_done() -> bool:
    return (RESULTS_DIR / "phase7_sequence.csv").exists()

PHASE_CHECKS = {
    1: phase1_done,
    2: phase2_done,
    3: phase3_done,
    4: phase4_done,
    5: phase5_done,
    6: phase6_done,
    7: phase7_done,
}


# ============================================================================
# Phase runners
# ============================================================================

def run_phase(phase_num: int, smoke: bool = False, force: bool = False,
              n_jobs: int = 1):
    """Run a single phase, skipping if output exists and force=False."""
    check_fn = PHASE_CHECKS.get(phase_num)
    if check_fn and check_fn() and not force:
        log.info("Phase %d: outputs already exist, skipping. Use --force to re-run.",
                 phase_num)
        return True

    log.info("=" * 60)
    log.info("PHASE %d — Starting", phase_num)
    log.info("=" * 60)

    t0 = time.time()
    try:
        if phase_num == 1:
            from scripts.phase1_setup_data import run_phase1
            run_phase1(smoke=smoke)

        elif phase_num == 2:
            from scripts.phase2_features import run_phase2
            run_phase2(smoke=smoke)

        elif phase_num == 3:
            from scripts.phase3_labels import run_phase3
            run_phase3(smoke=smoke, n_jobs=n_jobs)

        elif phase_num == 4:
            from scripts.phase4_models import run_phase4
            run_phase4(smoke=smoke)

        elif phase_num == 5:
            from scripts.phase5_shap import run_phase5
            run_phase5(smoke=smoke)

        elif phase_num == 6:
            from scripts.phase6_pruning import run_phase6
            run_phase6(smoke=smoke)

        elif phase_num == 7:
            from scripts.phase7_sequence import run_phase7
            run_phase7(smoke=smoke)

        else:
            log.error("Unknown phase: %d", phase_num)
            return False

        elapsed = time.time() - t0
        log.info("Phase %d completed in %.1f seconds", phase_num, elapsed)
        return True

    except Exception as e:
        elapsed = time.time() - t0
        log.error("Phase %d FAILED after %.1f seconds: %s", phase_num, elapsed, e)
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LLVM IR SHAP Research Pipeline — Master Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py --smoke          # Fast end-to-end test (~5 min)
  python run_all.py                  # Full pipeline run
  python run_all.py --phase 5        # Run only Phase 5 (SHAP)
  python run_all.py --force          # Force re-run all phases
        """,
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: process only 3 programs, <5 min target")
    parser.add_argument("--phase", type=int, choices=range(1, 8),
                        help="Run only the specified phase (1-7)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if outputs exist")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Number of parallel jobs for Phase 3")
    args = parser.parse_args()

    # ---- Setup ----
    ensure_dirs()
    total_start = time.time()

    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║   LLVM IR SHAP Research Pipeline                       ║")
    log.info("║   Mode: %s                                    ║",
             "SMOKE TEST" if args.smoke else "FULL RUN  ")
    log.info("╚══════════════════════════════════════════════════════════╝")

    # ---- Find and verify LLVM tools ----
    try:
        tools = find_llvm_tools()
        log.info("LLVM tools found: %s", list(tools.keys()))
    except RuntimeError as e:
        log.error(str(e))
        log.error("Please run setup_wsl.sh first to install LLVM tools.")
        sys.exit(1)

    # ---- Save environment info ----
    seeds = [42, 43, 44, 45, 46]  # seeds used across the pipeline
    save_env_json(tools, seeds)

    # ---- Run phases ----
    if args.phase:
        # Run only the specified phase
        success = run_phase(args.phase, smoke=args.smoke, force=args.force,
                            n_jobs=args.jobs)
        if not success:
            sys.exit(1)
    else:
        # Run all phases in order
        for phase_num in range(1, 8):
            success = run_phase(phase_num, smoke=args.smoke, force=args.force,
                                n_jobs=args.jobs)
            if not success:
                log.error("Pipeline stopped at Phase %d. Fix the error and re-run.",
                          phase_num)
                sys.exit(1)

    # ---- Generate RESULTS.md ----
    log.info("Generating RESULTS.md ...")
    try:
        from scripts.generate_results_md import generate
        generate()
    except Exception as e:
        log.warning("RESULTS.md generation failed: %s", e)

    # ---- Summary ----
    total_elapsed = time.time() - total_start
    log.info("")
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║   Pipeline complete!                                   ║")
    log.info("║   Total time: %6.0f seconds (%4.1f minutes)             ║",
             total_elapsed, total_elapsed / 60)
    log.info("║   Results: results/RESULTS.md                          ║")
    log.info("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
