#!/usr/bin/env python3
"""
phase4_models.py — Train and evaluate XGBoost classifiers per pass.

For each kept pass:
  1. Build feature matrix X from Phase 2 features, label vector y from Phase 3.
  2. StratifiedGroupKFold (k=5) grouped by source program (no program in both train & test).
  3. Train XGBoost binary classifier; also train baselines (majority, LR, RF).
  4. Metrics: PR-AUC, MCC, F1, mean ± std over folds.
  5. Cross-suite evaluation: train on one suite, test on the other.
  6. Flag passes where XGBoost does not beat majority baseline.

Output:
  results/phase4_results.csv       — per-pass, per-fold metrics
  results/phase4_summary.csv       — per-pass summary (mean ± std)
  results/phase4_cross_suite.csv   — cross-suite results
  results/phase4_gate.txt          — gate report
  data/models/                     — saved XGBoost models (joblib)

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
    ensure_dirs, setup_logging,
)

log = setup_logging("phase4")
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# Feature columns (exclude metadata)
# ============================================================================

METADATA_COLS = {"suite", "program", "function", "ir_hash", "state",
                 "prefix_passes", "pass_name", "inst_before", "inst_after",
                 "rel_reduction", "beneficial", "inst_count"}


def get_feature_columns(df):
    """Return the list of feature column names (excluding metadata)."""
    return [c for c in df.columns if c not in METADATA_COLS]


# ============================================================================
# Model training and evaluation
# ============================================================================

def train_and_evaluate_pass(pass_name: str, df_merged, feature_cols: list,
                              save_dir: Path) -> dict:
    """
    Train and evaluate models for a single pass.
    Returns a dict with results.
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

    # Filter data for this pass
    pass_df = df_merged[df_merged["pass_name"] == pass_name].copy()

    if len(pass_df) < 20:
        log.warning("Too few samples for pass %s (%d), skipping.", pass_name, len(pass_df))
        return None

    X = pass_df[feature_cols].values.astype(np.float32)
    y = pass_df["beneficial"].values.astype(int)
    groups = pass_df["program"].values

    # Handle NaN/inf in features
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    log.info("Pass %-25s  samples=%d  pos_rate=%.2f%%",
             pass_name, len(y), 100 * y.mean())

    # Models to evaluate
    models = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
        ),
    }

    # StratifiedGroupKFold cross-validation
    n_splits = min(5, len(set(groups)))
    if n_splits < 2:
        log.warning("Not enough groups for cross-validation on pass %s", pass_name)
        return None

    gkf = StratifiedGroupKFold(n_splits=n_splits)

    fold_results = []
    best_xgb_model = None
    best_xgb_score = -1

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Skip fold if train or test set has only one class
        if len(set(y_test)) < 2 or len(set(y_train)) < 2:
            msg = f"Fold {fold_idx} for pass {pass_name} skipped (single class in train/test)."
            log.warning(msg)
            with open(RESULTS_DIR / "skipped_folds.txt", "a") as f:
                f.write(msg + "\n")
            continue

        for model_name, model in models.items():
            model_clone = _clone_model(model)
            model_clone.fit(X_train, y_train)

            y_pred = model_clone.predict(X_test)

            # For PR-AUC, we need probability estimates
            if hasattr(model_clone, "predict_proba"):
                y_prob = model_clone.predict_proba(X_test)[:, 1]
                pr_auc = average_precision_score(y_test, y_prob)
            else:
                pr_auc = float("nan")

            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)

            fold_results.append({
                "pass_name": pass_name,
                "model": model_name,
                "fold": fold_idx,
                "pr_auc": round(pr_auc, 4),
                "f1": round(f1, 4),
                "mcc": round(mcc, 4),
                "n_train": len(y_train),
                "n_test": len(y_test),
                "pos_rate_train": round(y_train.mean(), 4),
                "pos_rate_test": round(y_test.mean(), 4),
            })

            # Track best XGBoost model
            if model_name == "xgboost" and pr_auc > best_xgb_score:
                best_xgb_score = pr_auc
                best_xgb_model = model_clone

    # Save best XGBoost model
    if best_xgb_model is not None:
        model_path = save_dir / f"xgb_{pass_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_xgb_model, f)

    return fold_results


def _clone_model(model):
    """Clone a sklearn/xgboost model."""
    from sklearn.base import clone
    return clone(model)


# ============================================================================
# Cross-suite evaluation
# ============================================================================

def cross_suite_evaluation(df_merged, feature_cols: list, kept_passes: list) -> list:
    """
    Train on one suite, test on the other.
    Returns list of result dicts.
    """
    import pandas as pd
    from sklearn.metrics import (
        average_precision_score, f1_score, matthews_corrcoef,
    )
    from xgboost import XGBClassifier

    suites = df_merged["suite"].unique()
    if len(suites) < 2:
        log.warning("Only one suite found — skipping cross-suite evaluation.")
        return []

    results = []

    for pass_name in kept_passes:
        pass_df = df_merged[df_merged["pass_name"] == pass_name]

        for train_suite in suites:
            test_suites = [s for s in suites if s != train_suite]
            for test_suite in test_suites:
                train_df = pass_df[pass_df["suite"] == train_suite]
                test_df = pass_df[pass_df["suite"] == test_suite]

                if len(train_df) < 10 or len(test_df) < 10:
                    continue
                if len(set(test_df["beneficial"])) < 2:
                    continue

                X_train = train_df[feature_cols].values.astype(np.float32)
                y_train = train_df["beneficial"].values.astype(int)
                X_test = test_df[feature_cols].values.astype(np.float32)
                y_test = test_df["beneficial"].values.astype(int)

                X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
                X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

                model = XGBClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    random_state=42, eval_metric="logloss",
                    use_label_encoder=False,
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

                results.append({
                    "pass_name": pass_name,
                    "train_suite": train_suite,
                    "test_suite": test_suite,
                    "pr_auc": round(average_precision_score(y_test, y_prob), 4),
                    "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
                    "mcc": round(matthews_corrcoef(y_test, y_pred), 4),
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                })

    return results


# ============================================================================
# Main pipeline
# ============================================================================

def run_phase4(smoke: bool = False):
    """Execute Phase 4: train and evaluate models."""
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

    # Merge features with labels on (function, ir_hash)
    # Features are per-function; labels are per-(function, state, pass)
    # We need to extract features for each state too, but since states
    # are derived from the same function, we use the baseline features
    # as a proxy (the features of the state BEFORE applying the target pass).
    # For a more accurate approach, we'd re-extract features for each state's IR.
    # For now, we join on function identity.

    # Merge on ir_hash (unique function identifier)
    df_merged = pd.merge(
        df_labels,
        df_features,
        on=["ir_hash", "suite", "program", "function"],
        how="inner",
    )
    log.info("Merged dataset: %d rows", len(df_merged))

    if len(df_merged) == 0:
        log.error("No data after merge — check feature/label alignment.")
        sys.exit(1)

    feature_cols = get_feature_columns(df_features)
    # Ensure all feature columns exist in merged df
    feature_cols = [c for c in feature_cols if c in df_merged.columns]
    log.info("Using %d features: %s", len(feature_cols), feature_cols[:10])

    # Create model save directory
    models_dir = DATA_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # ---- Train and evaluate per pass ----
    all_fold_results = []
    for pass_name in kept_passes:
        fold_results = train_and_evaluate_pass(
            pass_name, df_merged, feature_cols, models_dir
        )
        if fold_results:
            all_fold_results.extend(fold_results)

    if not all_fold_results:
        log.error("No model results generated!")
        sys.exit(1)

    # Save per-fold results
    df_results = pd.DataFrame(all_fold_results)
    results_path = RESULTS_DIR / "phase4_results.csv"
    df_results.to_csv(results_path, index=False)
    log.info("Per-fold results saved to %s", results_path)

    # ---- Summary table (mean ± std) ----
    summary_rows = []
    for pass_name in kept_passes:
        for model_name in ["majority", "logistic_regression", "random_forest", "xgboost"]:
            mask = (df_results["pass_name"] == pass_name) & (df_results["model"] == model_name)
            subset = df_results[mask]
            if len(subset) == 0:
                continue
            summary_rows.append({
                "pass_name": pass_name,
                "model": model_name,
                "pr_auc_mean": round(subset["pr_auc"].mean(), 4),
                "pr_auc_std": round(subset["pr_auc"].std(), 4),
                "f1_mean": round(subset["f1"].mean(), 4),
                "f1_std": round(subset["f1"].std(), 4),
                "mcc_mean": round(subset["mcc"].mean(), 4),
                "mcc_std": round(subset["mcc"].std(), 4),
            })

    df_summary = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "phase4_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    log.info("Summary saved to %s", summary_path)

    # ---- Cross-suite evaluation ----
    cross_results = cross_suite_evaluation(df_merged, feature_cols, kept_passes)
    if cross_results:
        df_cross = pd.DataFrame(cross_results)
        cross_path = RESULTS_DIR / "phase4_cross_suite.csv"
        df_cross.to_csv(cross_path, index=False)
        log.info("Cross-suite results saved to %s", cross_path)

    # ---- Gate report: flag passes where XGBoost doesn't beat majority ----
    flagged = []
    for pass_name in kept_passes:
        xgb = df_summary[(df_summary["pass_name"] == pass_name) &
                          (df_summary["model"] == "xgboost")]
        maj = df_summary[(df_summary["pass_name"] == pass_name) &
                          (df_summary["model"] == "majority")]
        if len(xgb) > 0 and len(maj) > 0:
            xgb_prauc = xgb.iloc[0]["pr_auc_mean"]
            maj_prauc = maj.iloc[0]["pr_auc_mean"]
            if xgb_prauc <= maj_prauc:
                flagged.append(pass_name)
                log.warning("FLAGGED: %s — XGBoost (%.4f) does not beat majority (%.4f)",
                            pass_name, xgb_prauc, maj_prauc)

    # Save passes that passed the gate
    interpretable_passes = [p for p in kept_passes if p not in flagged]
    interp_path = LABELS_DIR / "interpretable_passes.json"
    with open(interp_path, "w") as f:
        json.dump(interpretable_passes, f)

    report = f"""
╔══════════════════════════════════════════════════════╗
║              PHASE 4 GATE REPORT                     ║
╠══════════════════════════════════════════════════════╣
║ Passes evaluated:       {len(kept_passes):>8}                     ║
║ Passes where XGB > Maj: {len(interpretable_passes):>8}                     ║
║ Flagged passes:         {len(flagged):>8}                     ║
╠══════════════════════════════════════════════════════╣
║ Flagged: {', '.join(flagged) if flagged else 'none':<44}║
║ (excluded from Phase 5 interpretation)               ║
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
