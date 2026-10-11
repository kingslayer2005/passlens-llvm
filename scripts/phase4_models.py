#!/usr/bin/env python3
"""
phase4_models.py — Can a model predict what a pass will do?

For every pass and each of the three targets (beneficial / harmful / fired):

  1. Apply the keep rule: a target whose positive rate is outside 5 %-95 %
     is dropped and nothing is fitted for it.
  2. Evaluate four models with repeated, grouped, stratified cross-validation
     (3 repeats x 5 folds, grouped by benchmark program, so a program is
     never in both the training and the test fold):
        majority baseline, logistic regression, random forest, XGBoost.
     XGBoost hyperparameters are tuned inside each training fold (nested CV).
  3. Write one audit line per fold proving that no program, function or IR
     hash is on both sides.
  4. Compare XGBoost with each baseline:
        - Wilcoxon signed-rank test over the fold PR-AUCs, Holm-corrected
          across targets
        - bootstrap confidence interval of the PR-AUC difference, resampling
          whole programs
     A target "beats the majority baseline" only if BOTH agree.
  5. Cross-suite check: train on one benchmark suite, test on the other.

Output (all small text files):
  results/master_table.csv            one row per pass x target, with status
  results/phase4_results.csv          one row per pass, target, model, repeat, fold
  results/phase4_summary.csv          mean / std / bootstrap CI per model
  results/phase4_wilcoxon.csv         XGBoost vs each baseline
  results/phase4_cross_suite.csv      train on one suite, test on the other
  results/phase4_tuned_params.json    hyperparameters chosen in each fold
  results/leakage_audit.txt           one line per fold
  results/skipped_folds.txt           folds that could not be scored, with reason
  results/phase4_gate.txt             gate report

Usage:
  python -m scripts.phase4_models [--smoke] [--jobs N] [--only p1,p2] [--force]
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.analysis_common import (
    MASTER_TABLE, MODEL_NAMES, TUNED_PARAMS, DiskGuard, Settings,
    build_pair_data, cache_dir, candidate_configs, feature_columns, fold_is_usable,
    fold_key, get_settings, holm_correction, load_dataset, make_xgb, outer_splits,
    pair_key, percentile_ci, pr_auc, program_bootstrap_indices, safe_n_jobs,
    select_passes, target_table, tune_xgb, write_json_small,
)
from scripts.utils import RESULTS_DIR, ensure_dirs, setup_logging

log = setup_logging("phase4")


# ============================================================================
# One (pass, target) pair — runs in a worker process
# ============================================================================

def evaluate_pair(pass_name: str, target: str, cfg: Settings) -> dict:
    """
    Cross-validate the four models on one (pass, target) pair.

    The worker loads the two CSV files itself.  Nothing large is passed in
    or returned: the result holds fold metrics, the chosen hyperparameters,
    audit lines and the out-of-fold predictions (a few thousand numbers).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, matthews_corrcoef
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    start = time.time()
    features, labels = load_dataset()
    data = build_pair_data(features, labels, pass_name, target)
    X, y, groups = data.X, data.y, data.groups
    configs = candidate_configs(pass_name, target, cfg)

    fold_rows = []       # one dict per model per usable fold
    audit_lines = []     # one line per fold
    skipped = []         # folds without both classes
    tuned = {}           # fold key -> chosen hyperparameters
    leak_found = False

    # Out-of-fold predicted probabilities: [repeat][model] -> one value per
    # sample (NaN where the sample's fold was skipped)
    oof = np.full((cfg.n_repeats, len(MODEL_NAMES), len(y)), np.nan, dtype=np.float32)

    for repeat in range(cfg.n_repeats):
        for fold, (train_idx, test_idx) in enumerate(outer_splits(y, groups, repeat, cfg)):
            key = fold_key(repeat, fold)
            seed = cfg.base_seed + 1000 * repeat + fold

            # ---- Leakage audit: nothing may be on both sides ----
            program_overlap = len(set(groups[train_idx]) & set(groups[test_idx]))
            function_overlap = len(set(data.func_ids[train_idx]) & set(data.func_ids[test_idx]))
            hash_overlap = len(set(data.hashes[train_idx]) & set(data.hashes[test_idx]))
            clean = program_overlap == 0 and function_overlap == 0 and hash_overlap == 0
            leak_found = leak_found or not clean
            audit_lines.append(
                f"{'PASS' if clean else 'FAIL'} {pass_name} {target} {key} "
                f"n_train={len(train_idx)} n_test={len(test_idx)} "
                f"train_programs={len(set(groups[train_idx]))} "
                f"test_programs={len(set(groups[test_idx]))} "
                f"program_overlap={program_overlap} function_overlap={function_overlap} "
                f"hash_overlap={hash_overlap}")

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            usable, why = fold_is_usable(y_train, y_test)
            if not usable:
                skipped.append(f"{pass_name} {target} {key}: {why}")
                continue

            # ---- Fit the four models on the training fold ----
            probabilities = {}

            # Majority baseline: predicts the training positive rate for
            # every sample, so its PR-AUC equals the test positive rate.
            probabilities["majority"] = np.full(len(y_test), y_train.mean())

            logistic = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))
            logistic.fit(X_train, y_train)
            probabilities["logistic_regression"] = logistic.predict_proba(X_test)[:, 1]

            forest = RandomForestClassifier(
                n_estimators=100, max_depth=10, class_weight="balanced",
                n_jobs=1, random_state=seed)
            forest.fit(X_train, y_train)
            probabilities["random_forest"] = forest.predict_proba(X_test)[:, 1]

            # XGBoost: tune on the training fold only, then refit on all of it
            best_params, tuning_note = tune_xgb(
                X_train, y_train, groups[train_idx], configs, cfg, seed)
            booster = make_xgb(best_params, y_train, seed)
            booster.fit(X_train, y_train)
            probabilities["xgboost"] = booster.predict_proba(X_test)[:, 1]
            tuned[key] = {"params": best_params, "note": tuning_note}

            # ---- Score on the test fold ----
            for model_index, model_name in enumerate(MODEL_NAMES):
                prob = probabilities[model_name]
                oof[repeat, model_index, test_idx] = prob
                predicted = (prob >= 0.5).astype(int)
                fold_rows.append({
                    "pass_name": pass_name, "target": target, "model": model_name,
                    "repeat": repeat, "fold": fold,
                    "pr_auc": round(pr_auc(y_test, prob), 6),
                    "f1": round(float(f1_score(y_test, predicted, zero_division=0)), 6),
                    "mcc": round(float(matthews_corrcoef(y_test, predicted)), 6),
                    "n_train": len(train_idx), "n_test": len(test_idx),
                    "pos_rate_train": round(float(y_train.mean()), 4),
                    "pos_rate_test": round(float(y_test.mean()), 4),
                })

    # ---- Bootstrap over programs on the out-of-fold predictions ----
    # For each resample of programs: PR-AUC of every model, averaged over the
    # repeats.  The same resample is used for all models, so differences
    # between models are paired.
    boot = []
    if fold_rows:
        for idx in program_bootstrap_indices(groups, cfg.n_bootstrap, cfg.base_seed):
            per_model = []
            for model_index in range(len(MODEL_NAMES)):
                values = []
                for repeat in range(cfg.n_repeats):
                    prob = oof[repeat, model_index, idx]
                    ok = ~np.isnan(prob)
                    if ok.sum() and y[idx][ok].min() != y[idx][ok].max():
                        values.append(pr_auc(y[idx][ok], prob[ok]))
                per_model.append(np.mean(values) if values else np.nan)
            boot.append(per_model)
    boot = np.array(boot, dtype=float).reshape(-1, len(MODEL_NAMES))

    bootstrap = {}
    xgb_column = MODEL_NAMES.index("xgboost")
    for model_index, model_name in enumerate(MODEL_NAMES):
        low, high = percentile_ci(boot[:, model_index], cfg.alpha) if len(boot) else (np.nan, np.nan)
        entry = {"ci_low": low, "ci_high": high}
        if model_name != "xgboost" and len(boot):
            diff = boot[:, xgb_column] - boot[:, model_index]
            entry["diff_ci_low"], entry["diff_ci_high"] = percentile_ci(diff, cfg.alpha)
        bootstrap[model_name] = entry

    # ---- Cross-suite check with the most frequently chosen configuration ----
    cross_suite = []
    modal_params = None
    if tuned:
        counts = Counter(json.dumps(v["params"], sort_keys=True) for v in tuned.values())
        modal_params = json.loads(counts.most_common(1)[0][0])
        suites = sorted(set(data.suites))
        for train_suite in suites:
            for test_suite in suites:
                if train_suite == test_suite:
                    continue
                tr = data.suites == train_suite
                te = data.suites == test_suite
                row = {"pass_name": pass_name, "target": target,
                       "train_suite": train_suite, "test_suite": test_suite,
                       "n_train": int(tr.sum()), "n_test": int(te.sum()),
                       "pos_rate_test": round(float(y[te].mean()), 4),
                       "xgboost_pr_auc": None, "note": ""}
                usable, why = fold_is_usable(y[tr], y[te])
                if usable:
                    model = make_xgb(modal_params, y[tr], cfg.base_seed)
                    model.fit(X[tr], y[tr])
                    row["xgboost_pr_auc"] = round(pr_auc(y[te], model.predict_proba(X[te])[:, 1]), 6)
                else:
                    row["note"] = why
                cross_suite.append(row)

    return {
        "pass_name": pass_name, "target": target,
        "n_samples": int(len(y)), "n_features": int(X.shape[1]),
        "n_errors": data.n_errors,
        "fold_rows": fold_rows, "audit_lines": audit_lines, "skipped": skipped,
        "tuned": tuned, "modal_params": modal_params,
        "bootstrap": bootstrap, "cross_suite": cross_suite,
        "leak_found": leak_found,
        "seconds": round(time.time() - start, 1),
    }


# ============================================================================
# Statistics across pairs
# ============================================================================

def compare_with_baselines(results_df: pd.DataFrame, pair_results: dict,
                           cfg: Settings) -> pd.DataFrame:
    """
    XGBoost against each baseline, per (pass, target):
      - paired Wilcoxon signed-rank test on the fold PR-AUCs (two-sided)
      - Holm correction across all (pass, target) pairs, per baseline
      - bootstrap CI of the PR-AUC difference (from evaluate_pair)
    """
    from scipy.stats import wilcoxon

    rows = []
    for (pass_name, target), group in results_df.groupby(["pass_name", "target"], sort=False):
        wide = group.pivot_table(index=["repeat", "fold"], columns="model", values="pr_auc")
        boot = pair_results[(pass_name, target)]["bootstrap"]
        for baseline in MODEL_NAMES:
            if baseline == "xgboost":
                continue
            diff = (wide["xgboost"] - wide[baseline]).to_numpy()
            p_value = float("nan")
            if len(diff) >= 6 and np.any(diff != 0):
                p_value = float(wilcoxon(wide["xgboost"], wide[baseline],
                                         zero_method="wilcox", alternative="two-sided").pvalue)
            rows.append({
                "pass_name": pass_name, "target": target, "baseline": baseline,
                "n_folds": len(diff),
                "xgboost_mean": round(float(wide["xgboost"].mean()), 6),
                "baseline_mean": round(float(wide[baseline].mean()), 6),
                "mean_difference": round(float(diff.mean()), 6),
                "p_value": p_value,
                "diff_ci_low": boot[baseline].get("diff_ci_low"),
                "diff_ci_high": boot[baseline].get("diff_ci_high"),
            })
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return comparison

    # Holm correction across pairs, separately for each baseline
    comparison["p_holm"] = np.nan
    for baseline, group in comparison.groupby("baseline"):
        comparison.loc[group.index, "p_holm"] = holm_correction(group["p_value"].tolist())

    # "Better" needs the test AND the program bootstrap to agree.
    # (Smoke settings have too few folds for the test and use the bootstrap
    #  alone; smoke numbers are not results.)
    test_passed = comparison["p_holm"] < cfg.alpha
    if not cfg.require_wilcoxon:
        test_passed = pd.Series(True, index=comparison.index)
    comparison["xgboost_better"] = (
        (comparison["mean_difference"] > 0)
        & test_passed
        & (comparison["diff_ci_low"] > 0))
    return comparison


def summarise(results_df: pd.DataFrame, pair_results: dict) -> pd.DataFrame:
    """Mean and spread of the fold metrics, plus the program-bootstrap CI."""
    rows = []
    for (pass_name, target, model), group in results_df.groupby(
            ["pass_name", "target", "model"], sort=False):
        boot = pair_results[(pass_name, target)]["bootstrap"][model]
        rows.append({
            "pass_name": pass_name, "target": target, "model": model,
            "n_folds": len(group),
            "pr_auc_mean": round(group["pr_auc"].mean(), 4),
            "pr_auc_std": round(group["pr_auc"].std(ddof=1), 4) if len(group) > 1 else None,
            "pr_auc_ci_low": None if boot["ci_low"] is None else round(boot["ci_low"], 4),
            "pr_auc_ci_high": None if boot["ci_high"] is None else round(boot["ci_high"], 4),
            "f1_mean": round(group["f1"].mean(), 4),
            "mcc_mean": round(group["mcc"].mean(), 4),
        })
    return pd.DataFrame(rows)


# ============================================================================
# Main
# ============================================================================

def run_phase4(smoke: bool = False, n_jobs: int = 2, only: Optional[str] = None,
               force: bool = False):
    """Execute Phase 4."""
    ensure_dirs()
    cfg = get_settings(smoke)
    guard = DiskGuard()
    start = time.time()

    features, labels = load_dataset()
    feature_names, constant_columns = feature_columns(features)
    passes = select_passes(smoke, only)

    # ---- Keep rule first: nothing is fitted for a dropped target ----
    master = target_table(labels, cfg, passes)
    kept = master[master["keep_status"] == "kept"]
    pairs = list(zip(kept["pass_name"], kept["target"]))
    log.info("Settings: %s | passes: %d | targets kept: %d of %d | features: %d "
             "(dropped constant: %s)", cfg.name, len(passes), len(pairs), len(master),
             len(feature_names), constant_columns or "none")
    log.info("\n%s", master[["pass_name", "target", "positive_rate", "keep_status",
                             "keep_reason"]].to_string(index=False))
    if not pairs:
        raise RuntimeError("No target passes the keep rule.")

    # ---- Resume cache: one small JSON file per finished pair ----
    cache = cache_dir("phase4", cfg)
    pair_results = {}
    todo = []
    for pass_name, target in pairs:
        path = cache / f"{pass_name}__{target}.json"
        if path.exists() and not force:
            pair_results[(pass_name, target)] = json.loads(path.read_text(encoding="utf-8"))
        else:
            todo.append((pass_name, target))
    log.info("Pairs: %d cached, %d to run.", len(pair_results), len(todo))

    if todo:
        jobs = safe_n_jobs(n_jobs)
        if jobs == 1:
            finished = (evaluate_pair(p, t, cfg) for p, t in todo)
        else:
            from joblib import Parallel, delayed
            # max_nbytes=None: never write arguments to memory-mapped temp files
            finished = Parallel(n_jobs=jobs, return_as="generator_unordered",
                                max_nbytes=None)(
                delayed(evaluate_pair)(p, t, cfg) for p, t in todo)
        run_start = time.time()
        for done_count, result in enumerate(finished, start=1):
            key = (result["pass_name"], result["target"])
            pair_results[key] = result
            write_json_small(cache / f"{key[0]}__{key[1]}.json", result, max_mb=2.0)
            guard.check()
            elapsed = time.time() - run_start
            remaining = elapsed / done_count * (len(todo) - done_count)
            log.info("[%d/%d] %s %s done in %.0fs | elapsed %.0fs | remaining ~%.0fs",
                     done_count, len(todo), key[0], key[1], result["seconds"],
                     elapsed, remaining)

    # ---- Collect, in the fixed order of the master table ----
    fold_rows, audit_lines, skipped, cross_suite = [], [], [], []
    tuned_all = {}
    for pass_name, target in pairs:
        result = pair_results[(pass_name, target)]
        fold_rows.extend(result["fold_rows"])
        audit_lines.extend(result["audit_lines"])
        skipped.extend(result["skipped"])
        cross_suite.extend(result["cross_suite"])
        tuned_all[pair_key(pass_name, target)] = {
            "folds": result["tuned"], "modal_params": result["modal_params"]}

    results_df = pd.DataFrame(fold_rows)
    comparison = compare_with_baselines(results_df, pair_results, cfg)
    summary = summarise(results_df, pair_results)

    # ---- Write the result files ----
    results_df.to_csv(RESULTS_DIR / "phase4_results.csv", index=False)
    summary.to_csv(RESULTS_DIR / "phase4_summary.csv", index=False)
    comparison.to_csv(RESULTS_DIR / "phase4_wilcoxon.csv", index=False)
    pd.DataFrame(cross_suite).to_csv(RESULTS_DIR / "phase4_cross_suite.csv", index=False)
    TUNED_PARAMS.write_text(json.dumps({
        "settings": cfg.name, "feature_names": feature_names,
        "dropped_constant_columns": constant_columns, "pairs": tuned_all,
    }, indent=1), encoding="utf-8")
    (RESULTS_DIR / "leakage_audit.txt").write_text(
        "# One line per outer fold: PASS means no program, function or IR hash "
        "is in both the training and the test fold.\n" + "\n".join(audit_lines) + "\n",
        encoding="utf-8")
    (RESULTS_DIR / "skipped_folds.txt").write_text(
        "# Outer folds that could not be scored, with the reason.\n"
        + "\n".join(skipped) + ("\n" if skipped else ""), encoding="utf-8")

    # ---- Master table: add the Phase 4 verdict to every pass x target ----
    xgb_summary = summary[summary["model"] == "xgboost"].set_index(["pass_name", "target"])
    vs_majority = (comparison[comparison["baseline"] == "majority"]
                   .set_index(["pass_name", "target"]) if not comparison.empty else None)
    extra = []
    for row in master.itertuples():
        key = (row.pass_name, row.target)
        info = {"valid_folds": None, "xgboost_pr_auc": None, "majority_pr_auc": None,
                "p_holm_vs_majority": None, "diff_ci_low_vs_majority": None,
                "beats_majority": None, "model_status": "", "model_reason": ""}
        if row.keep_status != "kept":
            info["model_status"] = "not fitted"
            info["model_reason"] = "dropped by the keep rule"
        elif key not in xgb_summary.index:
            info["model_status"] = "not scored"
            info["model_reason"] = "no outer fold had both classes"
        else:
            cmp_row = vs_majority.loc[key]
            n_valid = int(xgb_summary.loc[key, "n_folds"])
            info.update({
                "valid_folds": n_valid,
                "xgboost_pr_auc": xgb_summary.loc[key, "pr_auc_mean"],
                "majority_pr_auc": round(float(cmp_row["baseline_mean"]), 4),
                "p_holm_vs_majority": cmp_row["p_holm"],
                "diff_ci_low_vs_majority": cmp_row["diff_ci_low"],
                "beats_majority": bool(cmp_row["xgboost_better"]) and n_valid >= cfg.min_valid_folds,
            })
            if n_valid < cfg.min_valid_folds:
                info["model_status"] = "not interpretable"
                info["model_reason"] = (f"only {n_valid} usable outer folds "
                                        f"(minimum {cfg.min_valid_folds})")
            elif info["beats_majority"]:
                info["model_status"] = "beats majority"
            else:
                info["model_status"] = "not interpretable"
                info["model_reason"] = ("XGBoost does not beat the majority baseline "
                                        "(Holm-corrected Wilcoxon and program bootstrap)")
        extra.append(info)
    master = pd.concat([master, pd.DataFrame(extra)], axis=1)
    master.to_csv(MASTER_TABLE, index=False)

    # ---- Gate report ----
    n_fail = sum(line.startswith("FAIL") for line in audit_lines)
    n_beats = int((master["model_status"] == "beats majority").sum())
    lines = [
        "PHASE 4 GATE REPORT",
        "=" * 70,
        f"Settings:                         {cfg.name}"
        + ("   (SMOKE: these numbers are not results)" if smoke else ""),
        f"Passes considered:                {len(passes)}",
        f"Pass x target pairs:              {len(master)}",
        f"  kept by the 5%-95% rule:        {len(pairs)}",
        f"  dropped by the rule:            {len(master) - len(pairs)}",
        f"Outer CV:                         {cfg.n_repeats} repeats x {cfg.n_outer_folds} folds, "
        f"grouped by program",
        f"Tuning:                           {cfg.n_tuning_configs} configurations, "
        f"{cfg.n_inner_folds} inner folds",
        f"Features:                         {len(feature_names)}"
        f" (dropped constant: {', '.join(constant_columns) or 'none'})",
        f"Outer folds audited:              {len(audit_lines)}  (FAIL: {n_fail})",
        f"Outer folds skipped (one class):  {len(skipped)}",
        f"Targets where XGBoost beats the majority baseline: {n_beats} of {len(pairs)}",
        f"Run time this session:            {time.time() - start:.0f} s",
        "",
        master[["pass_name", "target", "positive_rate", "keep_status", "valid_folds",
                "xgboost_pr_auc", "majority_pr_auc", "p_holm_vs_majority",
                "model_status"]].to_string(index=False),
    ]
    report = "\n".join(lines) + "\n"
    (RESULTS_DIR / "phase4_gate.txt").write_text(report, encoding="utf-8")
    log.info("\n%s", report)

    if n_fail or any(r["leak_found"] for r in pair_results.values()):
        raise RuntimeError("Leakage audit FAILED. See results/leakage_audit.txt.")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 4: models and cross-validation")
    parser.add_argument("--smoke", action="store_true", help="tiny settings, a few passes")
    parser.add_argument("--jobs", type=int, default=2, help="parallel worker processes")
    parser.add_argument("--only", type=str, default=None, help="comma-separated pass names")
    parser.add_argument("--force", action="store_true", help="ignore the resume cache")
    args = parser.parse_args()
    run_phase4(smoke=args.smoke, n_jobs=args.jobs, only=args.only, force=args.force)


if __name__ == "__main__":
    main()
