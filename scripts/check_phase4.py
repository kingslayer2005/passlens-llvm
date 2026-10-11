#!/usr/bin/env python3
"""
check_phase4.py — Structural check of the Phase 4 outputs.

It cannot tell whether the numbers are good, but it catches the shortcuts
that make them meaningless: a single train/test split instead of
cross-validation, test folds larger than training folds, missing audit
lines, missing models, or targets that slipped past the keep rule.

Run it after a FULL Phase 4 run (it is expected to fail on a --smoke run):
  python -m scripts.check_phase4
Exit code 0 = passed, 1 = problems found.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.analysis_common import FULL, MODEL_NAMES
from scripts.utils import RESULTS_DIR


def main():
    problems = []
    results = pd.read_csv(RESULTS_DIR / "phase4_results.csv")
    master = pd.read_csv(RESULTS_DIR / "master_table.csv")
    params = json.loads((RESULTS_DIR / "phase4_tuned_params.json").read_text(encoding="utf-8"))
    audit = (RESULTS_DIR / "leakage_audit.txt").read_text(encoding="utf-8").splitlines()
    audit = [line for line in audit if line and not line.startswith("#")]

    expected_folds = FULL.n_repeats * FULL.n_outer_folds

    if params.get("settings") != "full":
        problems.append(f"Phase 4 was run with '{params.get('settings')}' settings, not 'full'")

    # 1. Real repeated cross-validation
    n_fold_ids = results[["repeat", "fold"]].drop_duplicates().shape[0]
    if n_fold_ids != expected_folds:
        problems.append(f"expected {expected_folds} (repeat, fold) pairs, found {n_fold_ids}")

    # 2. Training folds are larger than test folds (5-fold CV gives about 4:1)
    ratio = (results["n_train"] / results["n_test"]).min()
    if ratio < 2:
        problems.append(f"smallest n_train / n_test is {ratio:.2f}; a 5-fold split gives about 4")

    # 3. All four models in every scored fold
    per_fold = results.groupby(["pass_name", "target", "repeat", "fold"])["model"].nunique()
    if (per_fold != len(MODEL_NAMES)).any():
        problems.append("some folds do not have all four models")

    # 4. Enough scored folds per target, and no duplicated rows
    per_model = results.groupby(["pass_name", "target", "model"]).size()
    if per_model.max() > expected_folds:
        problems.append("a model has more rows than there are folds (duplicates)")
    few = per_model[per_model < FULL.min_valid_folds]
    n_scored = results[["pass_name", "target"]].drop_duplicates().shape[0]

    # 5. Keep rule: scored targets are exactly a subset of the kept ones, and
    #    every kept target has a positive rate inside the band
    kept = master[master["keep_status"] == "kept"]
    outside = kept[(kept["positive_rate"] < FULL.keep_low) | (kept["positive_rate"] > FULL.keep_high)]
    if len(outside):
        problems.append(f"{len(outside)} kept target(s) have a positive rate outside 5%-95%")
    scored = set(zip(results["pass_name"], results["target"]))
    if not scored <= set(zip(kept["pass_name"], kept["target"])):
        problems.append("a target was fitted although the keep rule dropped it")
    if len(master) != 54:
        problems.append(f"master table has {len(master)} rows, expected 54 (18 passes x 3 targets)")

    # 6. Leakage audit: one line per fold of every kept target, all PASS
    n_fail = sum(line.startswith("FAIL") for line in audit)
    n_pass = sum(line.startswith("PASS") for line in audit)
    if n_fail:
        problems.append(f"leakage audit has {n_fail} FAIL line(s)")
    if n_pass != len(kept) * expected_folds:
        problems.append(f"leakage audit has {n_pass} PASS lines, expected "
                        f"{len(kept) * expected_folds} ({len(kept)} kept targets x {expected_folds})")

    print(f"targets_kept={len(kept)} targets_scored={n_scored} fold_ids={n_fold_ids} "
          f"min_train_test_ratio={ratio:.2f} min_rows_per_model={per_model.min()} "
          f"audit_pass={n_pass} audit_fail={n_fail} "
          f"targets_with_few_folds={len(few) // len(MODEL_NAMES)}")
    if problems:
        print("PROBLEMS:")
        for problem in problems:
            print(" -", problem)
        sys.exit(1)
    print("CHECK PASSED")


if __name__ == "__main__":
    main()
