#!/usr/bin/env python3
"""
phase6_pruning.py — Feature pruning study.

For each interpretable pass:
  1. Retrain XGBoost with top-k features (k = 5, 10, 15, 20, 30, all).
  2. Feature selection methods: SHAP ranking, XGBoost gain, mutual information, random.
  3. Feature selection is done INSIDE each training fold (no leakage).
  4. Compare: PR-AUC, model size, inference time.

Output:
  results/phase6_pruning.csv     — per-pass, per-k, per-method results
  results/figures/pruning_*.png  — pruning curves per pass

Usage:
  python -m scripts.phase6_pruning [--smoke]
"""

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import (
    PROJECT_ROOT, DATA_DIR, FEATURES_DIR, LABELS_DIR, RESULTS_DIR, FIGURES_DIR,
    ensure_dirs, setup_logging,
)

log = setup_logging("phase6")
warnings.filterwarnings("ignore", category=FutureWarning)

# Feature counts to test
K_VALUES = [5, 10, 15, 20, 30]  # 'all' is added dynamically

# ============================================================================
# Feature selection methods
# ============================================================================

def select_by_shap(X_train, y_train, feature_cols, k):
    """Select top-k features by SHAP importance (computed on training data)."""
    import shap
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric="logloss", use_label_encoder=False,
    )
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(-mean_abs)[:k]
    return sorted(top_indices)


def select_by_gain(X_train, y_train, feature_cols, k):
    """Select top-k features by XGBoost gain importance."""
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric="logloss", use_label_encoder=False,
    )
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    top_indices = np.argsort(-importances)[:k]
    return sorted(top_indices)


def select_by_mi(X_train, y_train, feature_cols, k):
    """Select top-k features by mutual information."""
    from sklearn.feature_selection import mutual_info_classif

    mi = mutual_info_classif(X_train, y_train, random_state=42)
    top_indices = np.argsort(-mi)[:k]
    return sorted(top_indices)


def select_random(X_train, y_train, feature_cols, k):
    """Select k random features."""
    rng = np.random.RandomState(42)
    n_features = X_train.shape[1]
    indices = rng.choice(n_features, size=min(k, n_features), replace=False)
    return sorted(indices)


SELECTION_METHODS = {
    "shap": select_by_shap,
    "xgb_gain": select_by_gain,
    "mutual_info": select_by_mi,
    "random": select_random,
}


# ============================================================================
# Pruning evaluation
# ============================================================================

def evaluate_pruning_for_pass(pass_name, df_merged, feature_cols):
    """Evaluate feature pruning for one pass across all k values and methods."""
    import pandas as pd
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedGroupKFold
    from xgboost import XGBClassifier

    pass_df = df_merged[df_merged["pass_name"] == pass_name].copy()
    X = pass_df[feature_cols].values.astype(np.float32)
    y = pass_df["beneficial"].values.astype(int)
    groups = pass_df["program"].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n_features = X.shape[1]
    n_splits = min(5, len(set(groups)))
    if n_splits < 2:
        return []

    # Add 'all' to k values
    k_values = [k for k in K_VALUES if k < n_features] + [n_features]

    results = []

    gkf = StratifiedGroupKFold(n_splits=n_splits)

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(set(y_test)) < 2 or len(set(y_train)) < 2:
            msg = f"Fold {fold_idx} for pass {pass_name} skipped in Pruning (single class in train/test)."
            log.warning(msg)
            with open(RESULTS_DIR / "skipped_folds.txt", "a") as f:
                f.write(msg + "\n")
            continue

        for method_name, selector in SELECTION_METHODS.items():
            for k in k_values:
                # Select features on training data only
                selected_indices = selector(X_train, y_train, feature_cols, k)
                actual_k = len(selected_indices)

                X_train_k = X_train[:, selected_indices]
                X_test_k = X_test[:, selected_indices]

                # Train model
                model = XGBClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    random_state=42, eval_metric="logloss",
                    use_label_encoder=False,
                )
                model.fit(X_train_k, y_train)

                # Evaluate
                y_prob = model.predict_proba(X_test_k)[:, 1]
                pr_auc = average_precision_score(y_test, y_prob)

                # Model size (number of trees × nodes, approximate)
                booster = model.get_booster()
                model_size = len(booster.get_dump())

                # Inference time
                t0 = time.perf_counter()
                for _ in range(10):
                    model.predict_proba(X_test_k)
                t1 = time.perf_counter()
                inference_ms = (t1 - t0) / 10 * 1000  # ms per batch

                results.append({
                    "pass_name": pass_name,
                    "method": method_name,
                    "k": actual_k,
                    "fold": fold_idx,
                    "pr_auc": round(pr_auc, 4),
                    "model_size": model_size,
                    "inference_ms": round(inference_ms, 2),
                    "selected_features": ";".join(
                        feature_cols[i] for i in selected_indices),
                })

    return results


# ============================================================================
# Plotting
# ============================================================================

def plot_pruning_curves(df_pruning, pass_name, save_dir):
    """Plot PR-AUC vs k for each selection method."""
    pass_data = df_pruning[df_pruning["pass_name"] == pass_name]
    if pass_data.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in SELECTION_METHODS.keys():
        method_data = pass_data[pass_data["method"] == method]
        if method_data.empty:
            continue
        # Average across folds
        summary = method_data.groupby("k")["pr_auc"].agg(["mean", "std"]).reset_index()
        ax.errorbar(summary["k"], summary["mean"], yerr=summary["std"],
                     marker="o", label=method, capsize=3)

    ax.set_xlabel("Number of Features (k)")
    ax.set_ylabel("PR-AUC")
    ax.set_title(f"Feature Pruning — {pass_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"pruning_{pass_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================================
# Main pipeline
# ============================================================================

def run_phase6(smoke: bool = False):
    """Execute Phase 6: feature pruning study."""
    ensure_dirs()
    import pandas as pd

    # Load data
    features_path = FEATURES_DIR / "features.csv"
    labels_path = LABELS_DIR / "labels.csv"
    interp_path = LABELS_DIR / "interpretable_passes.json"
    feature_cols_path = FEATURES_DIR / "feature_columns.json"

    for p in [features_path, labels_path]:
        if not p.exists():
            log.error("Required file not found: %s", p)
            sys.exit(1)

    df_features = pd.read_csv(features_path)
    df_labels = pd.read_csv(labels_path)

    if interp_path.exists():
        with open(interp_path) as f:
            interp_passes = json.load(f)
    else:
        interp_passes = list(df_labels["pass_name"].unique())

    if feature_cols_path.exists():
        with open(feature_cols_path) as f:
            feature_cols = json.load(f)
    else:
        feature_cols = [c for c in df_features.columns
                        if c not in {"suite", "program", "function", "ir_hash", "inst_count"}]

    # Merge
    df_merged = pd.merge(
        df_labels, df_features,
        on=["ir_hash", "suite", "program", "function"],
        how="inner",
    )
    feature_cols = [c for c in feature_cols if c in df_merged.columns]

    log.info("Phase 6: pruning study for %d passes with %d features",
             len(interp_passes), len(feature_cols))

    # Evaluate
    all_results = []
    for pass_name in interp_passes:
        log.info("=== Pruning study for pass: %s ===", pass_name)
        results = evaluate_pruning_for_pass(pass_name, df_merged, feature_cols)
        all_results.extend(results)

    if all_results:
        df_pruning = pd.DataFrame(all_results)
        df_pruning.to_csv(RESULTS_DIR / "phase6_pruning.csv", index=False)
        log.info("Pruning results saved to %s", RESULTS_DIR / "phase6_pruning.csv")

        # Generate pruning curve plots
        pruning_fig_dir = FIGURES_DIR / "pruning"
        pruning_fig_dir.mkdir(parents=True, exist_ok=True)
        for pass_name in interp_passes:
            plot_pruning_curves(df_pruning, pass_name, pruning_fig_dir)

        # Summary table
        summary = df_pruning.groupby(["pass_name", "method", "k"]).agg({
            "pr_auc": ["mean", "std"],
            "model_size": "mean",
            "inference_ms": "mean",
        }).reset_index()
        summary.columns = ["pass_name", "method", "k",
                           "pr_auc_mean", "pr_auc_std",
                           "model_size_mean", "inference_ms_mean"]
        summary.to_csv(RESULTS_DIR / "phase6_pruning_summary.csv", index=False)
    else:
        log.error("No pruning results generated!")

    report = f"""
╔══════════════════════════════════════════════════════╗
║              PHASE 6 GATE REPORT                     ║
╠══════════════════════════════════════════════════════╣
║ Passes evaluated:       {len(interp_passes):>8}                     ║
║ Total experiments:      {len(all_results):>8}                     ║
║ K values tested:        {str(K_VALUES + ['all']):>30}║
║ Selection methods:      {len(SELECTION_METHODS):>8}                     ║
╚══════════════════════════════════════════════════════╝
"""
    log.info(report)
    (RESULTS_DIR / "phase6_gate.txt").write_text(report, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Feature Pruning")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_phase6(smoke=args.smoke)


if __name__ == "__main__":
    main()
