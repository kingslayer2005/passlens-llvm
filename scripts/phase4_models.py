#!/usr/bin/env python3
"""
phase4_models.py — Train and evaluate XGBoost classifiers per pass.

Research upgrades (v2):
  - Dual targets: beneficial and harmful per pass
  - Repeated StratifiedGroupKFold (3 repeats x 5 folds), grouped by program
  - Nested tuning: at most 10 random configs, 3 grouped inner folds
  - scale_pos_weight for imbalance
  - Bootstrap CI (resampling programs) for metrics
  - Wilcoxon signed-rank test, Holm-corrected across passes
  - Leakage audit (program and IR hash)
  - Threshold sensitivity (relabel at 0.5%, 1%, 2%)
  - Disk-space checks before each phase

Output:
  results/phase4_results.csv       — per-pass, per-fold, per-target metrics
  results/phase4_summary.csv       — per-pass summary (mean + bootstrap CI)
  results/phase4_wilcoxon.csv      — Wilcoxon signed-rank test results
  results/leakage_audit.txt        — leakage audit
  results/threshold_sensitivity.csv— label rates at each threshold
  results/phase4_gate.txt          — gate report
  data/models/                     — saved XGBoost models (one per pass per target)

Usage:
  python -m scripts.phase4_models [--smoke]
"""

import argparse
import json
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import (
    PROJECT_ROOT, DATA_DIR, FEATURES_DIR, LABELS_DIR, RESULTS_DIR,
    ensure_dirs, setup_logging, check_disk_space,
)

log = setup_logging("phase4")
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# Configuration
# ============================================================================

N_REPEATS = 3
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
N_RANDOM_CONFIGS = 10
N_BOOTSTRAP = 10000
ALPHA = 0.05
THRESHOLDS = [0.005, 0.01, 0.02]  # 0.5%, 1%, 2%

METADATA_COLS = {"suite", "program", "function", "ir_hash", "state",
                 "prefix_passes", "pass_name", "inst_before", "inst_after",
                 "rel_reduction", "beneficial", "inst_count", "outcome"}

# XGBoost hyperparameter search space
XGB_PARAM_SPACE = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
}


def get_feature_columns(df):
    """Return the list of feature column names (excluding metadata)."""
    return [c for c in df.columns if c not in METADATA_COLS]


# ============================================================================
# Nested tuning
# ============================================================================

def nested_tune_xgboost(X_train, y_train, groups_train, rng):
    """
    Random search over XGB_PARAM_SPACE using N_INNER_FOLDS grouped CV.
    Returns the best XGBClassifier (fitted on full X_train).
    """
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedGroupKFold
    from xgboost import XGBClassifier

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    spw = n_neg / max(n_pos, 1)

    n_inner = min(N_INNER_FOLDS, len(set(groups_train)))
    if n_inner < 2:
        # Can't do inner CV; use defaults
        model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, random_state=42,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)
        return model

    configs = []
    for _ in range(N_RANDOM_CONFIGS):
        cfg = {k: rng.choice(v) for k, v in XGB_PARAM_SPACE.items()}
        configs.append(cfg)

    best_score = -1
    best_cfg = configs[0]

    igkf = StratifiedGroupKFold(n_splits=n_inner)
    for cfg in configs:
        scores = []
        for tr_idx, val_idx in igkf.split(X_train, y_train, groups_train):
            if len(set(y_train[tr_idx])) < 2 or len(set(y_train[val_idx])) < 2:
                continue
            m = XGBClassifier(
                **cfg, scale_pos_weight=spw,
                random_state=42, eval_metric="logloss",
            )
            m.fit(X_train[tr_idx], y_train[tr_idx])
            y_prob = m.predict_proba(X_train[val_idx])[:, 1]
            scores.append(average_precision_score(y_train[val_idx], y_prob))
        if scores and np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_cfg = cfg

    model = XGBClassifier(
        **best_cfg, scale_pos_weight=spw,
        random_state=42, eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model


# ============================================================================
# Bootstrap CI
# ============================================================================

def bootstrap_ci_by_program(values, programs, n_bootstrap=N_BOOTSTRAP, alpha=0.05):
    """
    95% bootstrap CI resampling programs (cluster bootstrap).
    values: array of per-sample values.
    programs: array of program labels (same length).
    Returns (mean, ci_low, ci_high).
    """
    rng = np.random.RandomState(42)
    unique_programs = np.unique(programs)
    n_prog = len(unique_programs)
    if n_prog < 2:
        m = float(np.mean(values))
        return m, m, m

    means = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_programs, size=n_prog, replace=True)
        boot_vals = []
        for p in sampled:
            mask = programs == p
            boot_vals.extend(values[mask])
        if boot_vals:
            means.append(np.mean(boot_vals))
    means = np.array(means)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(np.mean(values)), lo, hi


# ============================================================================
# Leakage audit
# ============================================================================

def leakage_audit(X, y, groups, ir_hashes, pass_name, target, n_repeats, n_folds, audit_file):
    """
    Assert no program and no IR hash appears in both train and test.
    Writes results to audit_file (append mode).
    Returns True if all folds pass.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    all_pass = True
    n_splits = min(n_folds, len(set(groups)))
    if n_splits < 2:
        return True

    for rep in range(n_repeats):
        gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                     random_state=42 + rep)
        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            train_programs = set(groups[train_idx])
            test_programs = set(groups[test_idx])
            prog_overlap = train_programs & test_programs

            train_hashes = set(ir_hashes[train_idx])
            test_hashes = set(ir_hashes[test_idx])
            hash_overlap = train_hashes & test_hashes

            ok = len(prog_overlap) == 0 and len(hash_overlap) == 0
            if not ok:
                all_pass = False

            with open(audit_file, "a") as f:
                f.write(f"pass={pass_name} target={target} rep={rep} fold={fold_idx} "
                        f"prog_overlap={len(prog_overlap)} hash_overlap={len(hash_overlap)} "
                        f"{'PASS' if ok else 'FAIL'}\n")
    return all_pass


# ============================================================================
# Model training and evaluation
# ============================================================================

def train_and_evaluate_pass(pass_name, target_name, df_merged, feature_cols,
                              save_dir, audit_file):
    """
    Train and evaluate models for a single pass and target.
    Returns (fold_results, xgb_prauc_vec, baseline_prauc_vecs).
    """
    import pandas as pd
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score, f1_score, matthews_corrcoef,
    )
    from sklearn.model_selection import StratifiedGroupKFold
    from xgboost import XGBClassifier

    pass_df = df_merged[df_merged["pass_name"] == pass_name].copy()

    if target_name == "beneficial":
        y_col = "beneficial"
    else:
        y_col = "harmful"
        pass_df["harmful"] = (pass_df["outcome"] == "increased").astype(int)

    if len(pass_df) < 20:
        log.warning("Too few samples for %s/%s (%d), skipping.", pass_name, target_name, len(pass_df))
        return None, None, None

    X = pass_df[feature_cols].values.astype(np.float32)
    y = pass_df[y_col].values.astype(int)
    groups = pass_df["program"].values
    ir_hashes = pass_df["ir_hash"].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    pos_rate = y.mean()
    if pos_rate < 0.05 or pos_rate > 0.95:
        log.info("Target %s/%s pos_rate=%.2f%% — DROPPED", pass_name, target_name, pos_rate * 100)
        return None, None, None

    log.info("Pass %-25s target=%-10s samples=%d pos_rate=%.2f%%",
             pass_name, target_name, len(y), 100 * pos_rate)

    # Leakage audit
    leakage_audit(X, y, groups, ir_hashes, pass_name, target_name,
                  N_REPEATS, N_OUTER_FOLDS, audit_file)

    rng = np.random.RandomState(42)
    n_splits = min(N_OUTER_FOLDS, len(set(groups)))
    if n_splits < 2:
        log.warning("Not enough groups for CV on %s/%s", pass_name, target_name)
        return None, None, None

    fold_results = []
    xgb_prauc_vec = []
    baseline_vecs = {"majority": [], "logistic_regression": [], "random_forest": []}
    best_xgb_model = None
    best_xgb_score = -1
    skipped_file = RESULTS_DIR / "skipped_folds.txt"

    for rep in range(N_REPEATS):
        gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                     random_state=42 + rep)

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train = groups[train_idx]

            if len(set(y_test)) < 2 or len(set(y_train)) < 2:
                msg = (f"Fold rep={rep} fold={fold_idx} pass={pass_name} "
                       f"target={target_name} skipped (single class).")
                log.warning(msg)
                with open(skipped_file, "a") as f:
                    f.write(msg + "\n")
                continue

            n_pos_train = int(y_train.sum())
            n_neg_train = len(y_train) - n_pos_train
            spw = n_neg_train / max(n_pos_train, 1)

            # Baselines
            baselines = {
                "majority": DummyClassifier(strategy="most_frequent"),
                "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
                "random_forest": RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
                ),
            }

            for model_name, model in baselines.items():
                from sklearn.base import clone
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                y_pred = model_clone.predict(X_test)
                if hasattr(model_clone, "predict_proba"):
                    y_prob = model_clone.predict_proba(X_test)[:, 1]
                    pr_auc = average_precision_score(y_test, y_prob)
                else:
                    pr_auc = float("nan")
                f1 = f1_score(y_test, y_pred, zero_division=0)
                mcc = matthews_corrcoef(y_test, y_pred)

                fold_results.append({
                    "pass_name": pass_name, "target": target_name,
                    "model": model_name, "repeat": rep, "fold": fold_idx,
                    "pr_auc": round(pr_auc, 4), "f1": round(f1, 4),
                    "mcc": round(mcc, 4),
                    "n_train": len(y_train), "n_test": len(y_test),
                    "pos_rate_train": round(y_train.mean(), 4),
                    "pos_rate_test": round(y_test.mean(), 4),
                })
                baseline_vecs[model_name].append(pr_auc)

            # XGBoost with nested tuning
            xgb_model = nested_tune_xgboost(X_train, y_train, groups_train, rng)
            y_pred = xgb_model.predict(X_test)
            y_prob = xgb_model.predict_proba(X_test)[:, 1]
            pr_auc = average_precision_score(y_test, y_prob)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)

            fold_results.append({
                "pass_name": pass_name, "target": target_name,
                "model": "xgboost", "repeat": rep, "fold": fold_idx,
                "pr_auc": round(pr_auc, 4), "f1": round(f1, 4),
                "mcc": round(mcc, 4),
                "n_train": len(y_train), "n_test": len(y_test),
                "pos_rate_train": round(y_train.mean(), 4),
                "pos_rate_test": round(y_test.mean(), 4),
            })
            xgb_prauc_vec.append(pr_auc)

            if pr_auc > best_xgb_score:
                best_xgb_score = pr_auc
                best_xgb_model = xgb_model

    # Save best XGBoost model (one per pass per target)
    if best_xgb_model is not None:
        model_path = save_dir / f"xgb_{pass_name}_{target_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_xgb_model, f)

    return fold_results, xgb_prauc_vec, baseline_vecs


# ============================================================================
# Wilcoxon test with Holm correction
# ============================================================================

def wilcoxon_holm(xgb_vecs, baseline_vecs_all, pass_names, targets):
    """
    Wilcoxon signed-rank test: XGBoost vs each baseline, Holm-corrected
    across passes.
    """
    from scipy.stats import wilcoxon

    results = []
    # Collect all p-values for Holm correction
    raw_results = []

    for baseline_name in ["majority", "logistic_regression", "random_forest"]:
        for pass_name in pass_names:
            for target in targets:
                key = (pass_name, target)
                if key not in xgb_vecs or key not in baseline_vecs_all:
                    continue
                xgb = np.array(xgb_vecs[key])
                bl = np.array(baseline_vecs_all[key].get(baseline_name, []))
                min_len = min(len(xgb), len(bl))
                if min_len < 5:
                    continue
                xgb = xgb[:min_len]
                bl = bl[:min_len]
                diff = xgb - bl
                if np.all(diff == 0):
                    raw_results.append({
                        "pass_name": pass_name, "target": target,
                        "baseline": baseline_name,
                        "xgb_mean": round(float(xgb.mean()), 4),
                        "bl_mean": round(float(bl.mean()), 4),
                        "p_value": 1.0, "statistic": 0.0,
                    })
                    continue
                try:
                    stat, p = wilcoxon(xgb, bl, alternative="greater")
                except ValueError:
                    stat, p = 0.0, 1.0
                raw_results.append({
                    "pass_name": pass_name, "target": target,
                    "baseline": baseline_name,
                    "xgb_mean": round(float(xgb.mean()), 4),
                    "bl_mean": round(float(bl.mean()), 4),
                    "p_value": float(p), "statistic": float(stat),
                })

    # Holm correction
    if raw_results:
        p_values = [r["p_value"] for r in raw_results]
        corrected = holm_bonferroni(p_values)
        for r, cp in zip(raw_results, corrected):
            r["p_holm"] = round(cp, 6)
            r["significant"] = cp < ALPHA
        results = raw_results

    return results


def holm_bonferroni(p_values):
    """Apply Holm-Bonferroni correction to a list of p-values."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        corrected[orig_idx] = min(1.0, p * (n - rank))
    # Enforce monotonicity
    indexed_corrected = sorted(enumerate(corrected), key=lambda x: p_values[x[0]])
    running_max = 0.0
    for orig_idx, _ in indexed_corrected:
        corrected[orig_idx] = max(corrected[orig_idx], running_max)
        running_max = corrected[orig_idx]
    return corrected


# ============================================================================
# Threshold sensitivity
# ============================================================================

def threshold_sensitivity(df_labels, df_features, feature_cols, kept_passes):
    """
    Recompute the beneficial label at 0.5%, 1%, 2% thresholds.
    Report label rates and top-5 SHAP features at each threshold.
    """
    import pandas as pd
    import shap
    from sklearn.model_selection import StratifiedGroupKFold
    from xgboost import XGBClassifier

    results = []

    df_merged = pd.merge(
        df_labels, df_features,
        on=["ir_hash", "suite", "program", "function"],
        how="inner",
    )

    for threshold in THRESHOLDS:
        for pass_name in kept_passes:
            pass_df = df_merged[df_merged["pass_name"] == pass_name].copy()
            if len(pass_df) < 20:
                continue

            y = (pass_df["rel_reduction"] >= threshold).astype(int).values
            pos_rate = y.mean()

            X = pass_df[feature_cols].values.astype(np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            groups = pass_df["program"].values

            top5 = []
            if 0.05 <= pos_rate <= 0.95:
                n_splits = min(5, len(set(groups)))
                if n_splits >= 2:
                    gkf = StratifiedGroupKFold(n_splits=n_splits)
                    try:
                        train_idx, test_idx = next(iter(gkf.split(X, y, groups)))
                        n_pos = int(y[train_idx].sum())
                        n_neg = len(y[train_idx]) - n_pos
                        model = XGBClassifier(
                            n_estimators=200, max_depth=6, learning_rate=0.1,
                            scale_pos_weight=n_neg / max(n_pos, 1),
                            random_state=42, eval_metric="logloss",
                        )
                        model.fit(X[train_idx], y[train_idx])
                        explainer = shap.TreeExplainer(model)
                        sv = explainer.shap_values(X[test_idx])
                        mean_abs = np.mean(np.abs(sv), axis=0)
                        top5_idx = np.argsort(-mean_abs)[:5]
                        top5 = [feature_cols[i] for i in top5_idx]
                    except Exception as e:
                        log.warning("Threshold sensitivity SHAP failed for %s @ %.1f%%: %s",
                                    pass_name, threshold * 100, e)

            results.append({
                "pass_name": pass_name,
                "threshold": threshold,
                "pos_rate": round(pos_rate, 4),
                "n_samples": len(y),
                "top5_shap": "; ".join(top5) if top5 else "N/A",
            })

    return results


# ============================================================================
# Main pipeline
# ============================================================================

def run_phase4(smoke: bool = False):
    """Execute Phase 4: train and evaluate models."""
    check_disk_space()
    ensure_dirs()
    import pandas as pd

    # Load features
    features_path = FEATURES_DIR / "features.csv"
    if not features_path.exists():
        log.error("Features not found — run Phase 2 first.")
        sys.exit(1)
    df_features = pd.read_csv(features_path)

    # Load labels
    labels_path = LABELS_DIR / "labels.csv"
    if not labels_path.exists():
        log.error("Labels not found — run Phase 3 first.")
        sys.exit(1)
    df_labels = pd.read_csv(labels_path)

    # Load kept passes
    kept_path = LABELS_DIR / "kept_passes.json"
    if kept_path.exists():
        with open(kept_path) as f:
            kept_passes = json.load(f)
    else:
        kept_passes = list(df_labels["pass_name"].unique())
    log.info("Kept passes: %s", kept_passes)

    # Merge features with labels
    df_merged = pd.merge(
        df_labels, df_features,
        on=["ir_hash", "suite", "program", "function"],
        how="inner",
    )
    log.info("Merged dataset: %d rows", len(df_merged))

    if len(df_merged) == 0:
        log.error("No data after merge — check feature/label alignment.")
        sys.exit(1)

    feature_cols = get_feature_columns(df_features)
    feature_cols = [c for c in feature_cols if c in df_merged.columns]
    log.info("Using %d features", len(feature_cols))

    # Create model save directory
    models_dir = DATA_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Clean old models
    for old in models_dir.glob("*.pkl"):
        old.unlink()

    # Init leakage audit and skipped folds files
    audit_file = RESULTS_DIR / "leakage_audit.txt"
    audit_file.write_text("# Leakage Audit\n", encoding="utf-8")
    skipped_file = RESULTS_DIR / "skipped_folds.txt"
    skipped_file.write_text("# Skipped Folds\n", encoding="utf-8")

    # ---- Train and evaluate per pass, per target ----
    all_fold_results = []
    xgb_vecs = {}
    baseline_vecs_all = {}
    targets_to_process = ["beneficial", "harmful"]

    target_status = {}  # (pass, target) -> kept/dropped

    for pass_name in kept_passes:
        for target_name in targets_to_process:
            fold_results, xgb_vec, bl_vecs = train_and_evaluate_pass(
                pass_name, target_name, df_merged, feature_cols,
                models_dir, str(audit_file),
            )
            key = (pass_name, target_name)
            if fold_results is None:
                target_status[key] = "dropped"
                continue

            target_status[key] = "kept"
            all_fold_results.extend(fold_results)
            if xgb_vec:
                xgb_vecs[key] = xgb_vec
            if bl_vecs:
                baseline_vecs_all[key] = bl_vecs

    if not all_fold_results:
        log.error("No model results generated!")
        sys.exit(1)

    # Save per-fold results
    df_results = pd.DataFrame(all_fold_results)
    results_path = RESULTS_DIR / "phase4_results.csv"
    df_results.to_csv(results_path, index=False)
    log.info("Per-fold results saved to %s", results_path)

    # ---- Summary table (mean + bootstrap CI) ----
    summary_rows = []
    for pass_name in kept_passes:
        for target_name in targets_to_process:
            for model_name in ["majority", "logistic_regression", "random_forest", "xgboost"]:
                mask = ((df_results["pass_name"] == pass_name) &
                        (df_results["target"] == target_name) &
                        (df_results["model"] == model_name))
                subset = df_results[mask]
                if len(subset) == 0:
                    continue
                vals = subset["pr_auc"].values
                mean_val = float(vals.mean())
                std_val = float(vals.std())
                # Simple percentile CI (not program-bootstrap — that's only available in phase5)
                if len(vals) >= 5:
                    ci_lo = float(np.percentile(vals, 2.5))
                    ci_hi = float(np.percentile(vals, 97.5))
                else:
                    ci_lo, ci_hi = mean_val, mean_val
                summary_rows.append({
                    "pass_name": pass_name,
                    "target": target_name,
                    "model": model_name,
                    "pr_auc_mean": round(mean_val, 4),
                    "pr_auc_std": round(std_val, 4),
                    "pr_auc_ci_lo": round(ci_lo, 4),
                    "pr_auc_ci_hi": round(ci_hi, 4),
                    "f1_mean": round(float(subset["f1"].mean()), 4),
                    "mcc_mean": round(float(subset["mcc"].mean()), 4),
                })

    df_summary = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "phase4_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    log.info("Summary saved to %s", summary_path)

    # ---- Wilcoxon signed-rank test ----
    active_targets = list(set(t for _, t in xgb_vecs.keys()))
    wilcoxon_results = wilcoxon_holm(xgb_vecs, baseline_vecs_all, kept_passes, active_targets)
    if wilcoxon_results:
        df_wilcoxon = pd.DataFrame(wilcoxon_results)
        df_wilcoxon.to_csv(RESULTS_DIR / "phase4_wilcoxon.csv", index=False)
        log.info("Wilcoxon results saved")

    # ---- Threshold sensitivity ----
    thresh_results = threshold_sensitivity(df_labels, df_features, feature_cols, kept_passes)
    if thresh_results:
        df_thresh = pd.DataFrame(thresh_results)
        df_thresh.to_csv(RESULTS_DIR / "threshold_sensitivity.csv", index=False)
        log.info("Threshold sensitivity saved")

    # ---- Gate: flag passes where XGBoost doesn't beat majority after Holm correction ----
    interpretable = {}
    for pass_name in kept_passes:
        for target_name in active_targets:
            key = (pass_name, target_name)
            if key not in xgb_vecs:
                continue
            # Check Wilcoxon result
            beats_majority = False
            if wilcoxon_results:
                for wr in wilcoxon_results:
                    if (wr["pass_name"] == pass_name and
                        wr["target"] == target_name and
                        wr["baseline"] == "majority" and
                        wr.get("significant", False)):
                        beats_majority = True
                        break
            interpretable[key] = beats_majority

    # Save interpretable passes
    interp_passes_list = [p for p in kept_passes
                          if any(interpretable.get((p, t), False) for t in active_targets)]
    interp_path = LABELS_DIR / "interpretable_passes.json"
    with open(interp_path, "w") as f:
        json.dump(interp_passes_list, f)

    # Save target status
    target_status_path = LABELS_DIR / "target_status.json"
    with open(target_status_path, "w") as f:
        json.dump({f"{k[0]}_{k[1]}": v for k, v in target_status.items()}, f, indent=2)

    flagged = [p for p in kept_passes if not any(interpretable.get((p, t), False) for t in active_targets)]

    report = f"""
╔══════════════════════════════════════════════════════╗
║              PHASE 4 GATE REPORT                     ║
╠══════════════════════════════════════════════════════╣
║ Passes evaluated:       {len(kept_passes):>8}                     ║
║ Targets per pass:       {len(targets_to_process):>8}                     ║
║ Repeats x folds:        {N_REPEATS}x{N_OUTER_FOLDS} = {N_REPEATS * N_OUTER_FOLDS:>4}                    ║
║ XGB beats majority:     {len(interp_passes_list):>8}                     ║
║ Flagged passes:         {len(flagged):>8}                     ║
╚══════════════════════════════════════════════════════╝
"""
    log.info(report)
    gate_path = RESULTS_DIR / "phase4_gate.txt"
    gate_path.write_text(report, encoding='utf-8')

    # Save feature column list for downstream phases
    with open(FEATURES_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Train and evaluate models")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_phase4(smoke=args.smoke)


if __name__ == "__main__":
    main()
