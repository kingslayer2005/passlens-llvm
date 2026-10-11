#!/usr/bin/env python3
"""
run_all.py — Runs the pipeline phases in order.

  Phase 1  fetch benchmarks, compile to IR, select functions
  Phase 2  baseline feature table (a check of the extractor)
  Phase 3  states, features and labels          -> data/features, data/labels
  Phase 4  models and cross-validation          -> results/master_table.csv
  Phase 5  SHAP, trust checks, heuristic agreement
  Phase 6  explanation-guided feature pruning
  Phase 7  sequence-level evaluation
  then     results/RESULTS.md is rebuilt from the result files

Flags:
  --smoke          tiny settings to check that the code runs (NOT results)
  --phase N        run only phase N
  --from-phase N   run phases N..7
  --jobs N         parallel workers (default 2; lowered if memory is short)
  --only a,b       Phases 4-6: only these passes (for testing)
  --force          redo a phase even if its outputs / cache exist
  --refreeze-heuristics   accept a changed heuristics.yaml (see below)

heuristics.yaml is the list of expectations the model is compared with.  Its
SHA-256 is stored in env.json; if the file changes afterwards the run stops,
because changing expectations after seeing results would make the comparison
meaningless.  Use --refreeze-heuristics only before any result exists.

Examples:
  python run_all.py --smoke --from-phase 4     # quick check of Phases 4-7
  python run_all.py --from-phase 4 --jobs 2    # full analysis on existing data
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import (
    DATA_DIR, FEATURES_DIR, LABELS_DIR, RESULTS_DIR,
    check_disk_space, ensure_dirs, find_llvm_tools, save_env_json, setup_logging,
)

log = setup_logging("run_all")

ENV_PATH = PROJECT_ROOT / "env.json"
HEURISTICS_PATH = PROJECT_ROOT / "heuristics.yaml"
LAST_PHASE = 7


def data_phase_done(phase: int) -> bool:
    """Phases 1-3 are skipped when their output file already exists."""
    outputs = {
        1: DATA_DIR / "function_index.csv",
        2: FEATURES_DIR / "features_baseline.csv",
        3: LABELS_DIR / "labels.csv",
    }
    return phase in outputs and outputs[phase].exists()


def freeze_heuristics(refreeze: bool, previous_hash: str):
    """Store the hash of heuristics.yaml in env.json, or stop if it changed."""
    current = hashlib.sha256(HEURISTICS_PATH.read_bytes()).hexdigest()
    if previous_hash and previous_hash != current and not refreeze:
        raise RuntimeError(
            "heuristics.yaml has changed since it was frozen "
            f"(stored {previous_hash[:12]}, now {current[:12]}). Restore the file, or "
            "rerun with --refreeze-heuristics if no result has been produced yet.")
    env = json.loads(ENV_PATH.read_text(encoding="utf-8")) if ENV_PATH.exists() else {}
    env["heuristics_sha256"] = current
    ENV_PATH.write_text(json.dumps(env, indent=2), encoding="utf-8")
    log.info("heuristics.yaml SHA-256: %s", current)


def run_phase(phase: int, args) -> None:
    """Run one phase.  Errors are not caught: a failed phase stops the run."""
    if phase <= 3 and data_phase_done(phase) and not args.force:
        log.info("Phase %d: output exists, skipping (use --force to redo).", phase)
        return

    check_disk_space(min_free_gb=1.0)
    log.info("=" * 60)
    log.info("PHASE %d — starting", phase)
    log.info("=" * 60)
    started = time.time()

    if phase == 1:
        from scripts.phase1_setup_data import run_phase1
        run_phase1(smoke=args.smoke, n_jobs=args.jobs)
    elif phase == 2:
        from scripts.phase2_features import run_phase2
        run_phase2(smoke=args.smoke)
    elif phase == 3:
        from scripts.phase3_labels import run_phase3
        run_phase3(smoke=args.smoke, n_jobs=args.jobs, force=args.force)
    elif phase == 4:
        from scripts.phase4_models import run_phase4
        run_phase4(smoke=args.smoke, n_jobs=args.jobs, only=args.only, force=args.force)
    elif phase == 5:
        from scripts.phase5_shap import run_phase5
        run_phase5(smoke=args.smoke, n_jobs=args.jobs, only=args.only, force=args.force)
    elif phase == 6:
        from scripts.phase6_pruning import run_phase6
        run_phase6(smoke=args.smoke, n_jobs=args.jobs, only=args.only, force=args.force)
    elif phase == 7:
        from scripts.phase7_sequence import run_phase7
        run_phase7(smoke=args.smoke, n_jobs=args.jobs)

    log.info("Phase %d finished in %.0f s", phase, time.time() - started)


def folder_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main():
    parser = argparse.ArgumentParser(description="Run the pipeline phases in order",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--phase", type=int, choices=range(1, LAST_PHASE + 1))
    parser.add_argument("--from-phase", type=int, choices=range(1, LAST_PHASE + 1), default=1)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--refreeze-heuristics", action="store_true")
    parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args()

    if args.log_file:
        from logging.handlers import RotatingFileHandler
        handler = RotatingFileHandler(args.log_file, maxBytes=20 * 1024 * 1024, backupCount=1)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logging.getLogger().addHandler(handler)   # every phase logger passes through the root

    ensure_dirs()
    total_start = time.time()
    log.info("Mode: %s", "SMOKE TEST (not results)" if args.smoke else "FULL RUN")

    # Environment record and the heuristics freeze
    previous_hash = ""
    if ENV_PATH.exists():
        previous_hash = json.loads(ENV_PATH.read_text(encoding="utf-8")).get("heuristics_sha256", "")
    tools = find_llvm_tools()
    save_env_json(tools, seeds=[42])
    freeze_heuristics(args.refreeze_heuristics, previous_hash)

    phases = [args.phase] if args.phase else list(range(args.from_phase, LAST_PHASE + 1))
    for phase in phases:
        run_phase(phase, args)

    # RESULTS.md is always rebuilt from whatever result files exist
    from scripts.generate_results_md import generate
    generate()

    log.info("Done in %.0f s. results/: %.1f MB, data/: %.1f MB",
             time.time() - total_start, folder_size_mb(RESULTS_DIR), folder_size_mb(DATA_DIR))


if __name__ == "__main__":
    main()
