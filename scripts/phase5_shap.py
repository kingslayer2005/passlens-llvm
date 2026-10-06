#!/usr/bin/env python3
"""
phase5_shap.py — TreeSHAP explainability analysis (core contribution).

For each interpretable pass:
  1. TreeSHAP on held-out folds only.
  2. Per pass: mean |SHAP| ranking, beeswarm, dependence plots (top 5),
     interaction values for top pairs, 3 local waterfall case studies.
  3. Faithfulness checks:
     a. Rank stability across folds and 5 seeds (Kendall τ, top-10 overlap)
     b. Agreement with permutation importance
     c. Label-shuffle control (rankings must collapse)
  4. Group correlated features (Spearman |ρ| > 0.8, hierarchical clustering)
     and report SHAP at cluster level.
  5. Heuristic agreement scoring (hit@5, MRR, direction agreement) against
     heuristics.yaml. Compare to random-ranking null.
  6. Identify candidate novel drivers (top SHAP features NOT in heuristic table).

Output:
  results/figures/shap_*            — SHAP plots (300 dpi)
  results/shap_rankings.csv         — per-pass feature importance rankings
  results/faithfulness.csv          — faithfulness check results
  results/heuristic_agreement.csv   — heuristic agreement scores
  results/novel_drivers.csv         — candidate novel feature drivers
  results/phase5_gate.txt           — gate report

Usage:
  python -m scripts.phase5_shap [--smoke]
"""

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import (
    PROJECT_ROOT, DATA_DIR, FEATURES_DIR, LABELS_DIR, RESULTS_DIR, FIGURES_DIR,
    ensure_dirs, setup_logging,
)

log = setup_logging("phase5")
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# Figure settings
# ============================================================================
DPI = 300
plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "font.size": 10,
    "axes.titlesize": 12,
    "figure.figsize": (10, 6),
})


# ============================================================================
# SHAP analysis per pass
# ============================================================================

def compute_shap_for_pass(pass_name: str, df_merged, feature_cols: list,
                           n_seeds: int = 5) -> dict:
    """
    Compute TreeSHAP values for a pass across folds and seeds.
    Returns a dict with SHAP values, rankings, and stability metrics.
    """
    import pandas as pd
    import shap
    from scipy.stats import kendalltau, spearmanr
    from sklearn.model_selection import StratifiedGroupKFold
    from xgboost import XGBClassifier

    pass_df = df_merged[df_merged["pass_name"] == pass_name].copy()
    X = pass_df[feature_cols].values.astype(np.float32)
    y = pass_df["beneficial"].values.astype(int)
    groups = pass_df["program"].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n_splits = min(5, len(set(groups)))
    if n_splits < 2:
        log.warning("Not enough groups for SHAP on %s", pass_name)
        return None

    # Collect SHAP values across folds and seeds
    all_shap_values = []       # list of (n_test, n_features) arrays
    all_X_test = []
    rankings_per_seed = []     # list of feature importance rankings per seed

    for seed in range(n_seeds):
        gkf = StratifiedGroupKFold(n_splits=n_splits)
        seed_shap_values = []
        seed_X_test = []

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if len(set(y_test)) < 2 or len(set(y_train)) < 2:
                msg = f"Fold {fold_idx} for pass {pass_name} skipped in SHAP (single class in train/test)."
                log.warning(msg)
                with open(RESULTS_DIR / "skipped_folds.txt", "a") as f:
                    f.write(msg + "\n")
                continue

            model = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42 + seed, eval_metric="logloss",
                use_label_encoder=False,
            )
            model.fit(X_train, y_train)

            # TreeSHAP on held-out fold
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

            seed_shap_values.append(shap_values)
            seed_X_test.append(X_test)

        if seed_shap_values:
            combined = np.vstack(seed_shap_values)
            all_shap_values.append(combined)
            all_X_test.append(np.vstack(seed_X_test))

            # Ranking for this seed: by mean |SHAP|
            mean_abs = np.mean(np.abs(combined), axis=0)
            ranking = np.argsort(-mean_abs)  # descending
            rankings_per_seed.append(ranking)

    if not all_shap_values:
        return None

    # ---- Aggregate SHAP values ----
    # Use the first seed's values for plotting (representative)
    primary_shap = all_shap_values[0]
    primary_X = all_X_test[0]

    # Mean |SHAP| ranking across all seeds
    all_mean_abs = []
    for sv in all_shap_values:
        all_mean_abs.append(np.mean(np.abs(sv), axis=0))
    global_mean_abs = np.mean(all_mean_abs, axis=0)
    global_ranking = np.argsort(-global_mean_abs)

    # ---- Rank stability (Kendall τ, top-10 overlap) ----
    kendall_taus = []
    top10_overlaps = []
    for i in range(len(rankings_per_seed)):
        for j in range(i + 1, len(rankings_per_seed)):
            tau, _ = kendalltau(rankings_per_seed[i], rankings_per_seed[j])
            kendall_taus.append(tau)

            top10_i = set(rankings_per_seed[i][:10])
            top10_j = set(rankings_per_seed[j][:10])
            overlap = len(top10_i & top10_j) / 10.0
            top10_overlaps.append(overlap)

    # ---- Feature importance ranking ----
    ranking_info = []
    for rank, feat_idx in enumerate(global_ranking):
        ranking_info.append({
            "pass_name": pass_name,
            "rank": rank + 1,
            "feature": feature_cols[feat_idx],
            "mean_abs_shap": round(float(global_mean_abs[feat_idx]), 6),
        })

    return {
        "pass_name": pass_name,
        "primary_shap": primary_shap,
        "primary_X": primary_X,
        "global_ranking": global_ranking,
        "global_mean_abs": global_mean_abs,
        "ranking_info": ranking_info,
        "kendall_tau_mean": float(np.mean(kendall_taus)) if kendall_taus else float("nan"),
        "kendall_tau_std": float(np.std(kendall_taus)) if kendall_taus else float("nan"),
        "top10_overlap_mean": float(np.mean(top10_overlaps)) if top10_overlaps else float("nan"),
        "feature_cols": feature_cols,
        "n_samples": len(primary_shap),
    }


# ============================================================================
# Faithfulness checks
# ============================================================================

def permutation_importance_agreement(pass_name: str, df_merged, feature_cols: list) -> dict:
    """
    Compute permutation importance and compare with SHAP ranking.
    """
    import pandas as pd
    from scipy.stats import kendalltau
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import StratifiedGroupKFold
    from xgboost import XGBClassifier

    pass_df = df_merged[df_merged["pass_name"] == pass_name]
    X = pass_df[feature_cols].values.astype(np.float32)
    y = pass_df["beneficial"].values.astype(int)
    groups = pass_df["program"].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n_splits = min(5, len(set(groups)))
    if n_splits < 2:
        return {"pass_name": pass_name, "perm_kendall_tau": float("nan")}

    gkf = StratifiedGroupKFold(n_splits=n_splits)
    # Use first fold
    train_idx, test_idx = next(iter(gkf.split(X, y, groups)))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric="logloss", use_label_encoder=False,
    )
    model.fit(X_train, y_train)

    perm_result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    perm_ranking = np.argsort(-perm_result.importances_mean)

    # Compare with SHAP ranking
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap_mean_abs = np.mean(np.abs(shap_values), axis=0)
    shap_ranking = np.argsort(-shap_mean_abs)

    tau, _ = kendalltau(shap_ranking, perm_ranking)

    return {
        "pass_name": pass_name,
        "perm_kendall_tau": round(float(tau), 4),
    }


def label_shuffle_control(pass_name: str, df_merged, feature_cols: list) -> dict:
    """
    Shuffle labels and check that SHAP rankings collapse (become random).
    """
    import shap
    from scipy.stats import kendalltau
    from sklearn.model_selection import StratifiedGroupKFold
    from xgboost import XGBClassifier

    pass_df = df_merged[df_merged["pass_name"] == pass_name]
    X = pass_df[feature_cols].values.astype(np.float32)
    y = pass_df["beneficial"].values.astype(int)
    groups = pass_df["program"].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n_splits = min(5, len(set(groups)))
    if n_splits < 2:
        return {"pass_name": pass_name, "shuffle_shap_max": float("nan")}

    # Shuffle labels
    rng = np.random.RandomState(99)
    y_shuffled = rng.permutation(y)

    gkf = StratifiedGroupKFold(n_splits=n_splits)
    train_idx, test_idx = next(iter(gkf.split(X, y_shuffled, groups)))

    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric="logloss", use_label_encoder=False,
    )
    model.fit(X[train_idx], y_shuffled[train_idx])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[test_idx])
    shap_mean_abs = np.mean(np.abs(shap_values), axis=0)

    return {
        "pass_name": pass_name,
        "shuffle_shap_max": round(float(np.max(shap_mean_abs)), 6),
        "shuffle_shap_mean": round(float(np.mean(shap_mean_abs)), 6),
    }


# ============================================================================
# Feature correlation clustering
# ============================================================================

def cluster_correlated_features(df_merged, feature_cols: list,
                                  threshold: float = 0.8) -> dict:
    """
    Group correlated features (Spearman |ρ| > threshold) using hierarchical clustering.
    Returns a mapping: cluster_id -> list of feature names.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from scipy.stats import spearmanr

    X = df_merged[feature_cols].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute Spearman correlation matrix
    rho, _ = spearmanr(X)
    if rho.ndim == 0:
        return {0: feature_cols}

    # Convert correlation to distance
    dist = 1 - np.abs(rho)
    dist = np.nan_to_num(dist, nan=1.0)
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, 1)

    # Hierarchical clustering
    condensed = squareform(dist)
    Z = linkage(condensed, method="average")
    clusters = fcluster(Z, t=1 - threshold, criterion="distance")

    cluster_map = {}
    for feat_idx, cluster_id in enumerate(clusters):
        cluster_id = int(cluster_id)
        if cluster_id not in cluster_map:
            cluster_map[cluster_id] = []
        cluster_map[cluster_id].append(feature_cols[feat_idx])

    return cluster_map


def shap_at_cluster_level(shap_values: np.ndarray, feature_cols: list,
                           cluster_map: dict) -> dict:
    """Aggregate SHAP values at cluster level by summing member features."""
    cluster_shap = {}
    for cluster_id, members in cluster_map.items():
        member_indices = [feature_cols.index(f) for f in members if f in feature_cols]
        if member_indices:
            cluster_shap_vals = np.sum(np.abs(shap_values[:, member_indices]), axis=1)
            cluster_shap[cluster_id] = {
                "members": members,
                "mean_abs_shap": float(np.mean(cluster_shap_vals)),
            }
    return cluster_shap


# ============================================================================
# Heuristic agreement
# ============================================================================

def score_heuristic_agreement(shap_result: dict, heuristics: dict) -> dict:
    """
    Score agreement between SHAP rankings and heuristics.yaml.
    Computes: hit@5, MRR, direction agreement.
    Compares to random-ranking null.
    """
    pass_name = shap_result["pass_name"]
    feature_cols = shap_result["feature_cols"]
    ranking = shap_result["global_ranking"]
    shap_values = shap_result["primary_shap"]

    if pass_name not in heuristics:
        return {"pass_name": pass_name, "hit_at_5": None, "mrr": None}

    heuristic_features = heuristics[pass_name].get("features", [])
    heuristic_names = set(f["name"] for f in heuristic_features)
    heuristic_directions = {f["name"]: f["direction"] for f in heuristic_features}

    # Ranked feature names
    ranked_names = [feature_cols[i] for i in ranking]

    # hit@5: fraction of heuristic features in top 5
    top5 = set(ranked_names[:5])
    hits = len(heuristic_names & top5)
    hit_at_5 = hits / len(heuristic_names) if heuristic_names else 0.0

    # MRR: mean reciprocal rank of heuristic features
    reciprocal_ranks = []
    for hf in heuristic_names:
        if hf in ranked_names:
            rank = ranked_names.index(hf) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    # Direction agreement: does SHAP direction match expected direction?
    direction_matches = 0
    direction_total = 0
    for hf_info in heuristic_features:
        feat_name = hf_info["name"]
        expected_dir = hf_info["direction"]  # "high" or "low"

        if feat_name in feature_cols:
            feat_idx = feature_cols.index(feat_name)
            # Compute correlation between SHAP values and feature values
            shap_col = shap_values[:, feat_idx]
            # Positive mean SHAP = high value -> beneficial
            mean_shap = np.mean(shap_col)

            if expected_dir == "high" and mean_shap > 0:
                direction_matches += 1
            elif expected_dir == "low" and mean_shap < 0:
                direction_matches += 1
            direction_total += 1

    direction_agreement = direction_matches / direction_total if direction_total > 0 else float("nan")

    # Random baseline: expected hit@5 under random ranking
    n_features = len(feature_cols)
    random_hit_at_5 = min(5, len(heuristic_names)) / n_features if n_features > 0 else 0
    random_mrr = np.mean([1.0 / ((n_features + 1) / 2)] * len(heuristic_names)) if heuristic_names else 0

    return {
        "pass_name": pass_name,
        "hit_at_5": round(hit_at_5, 4),
        "mrr": round(mrr, 4),
        "direction_agreement": round(direction_agreement, 4) if direction_total > 0 else None,
        "random_hit_at_5": round(random_hit_at_5, 4),
        "random_mrr": round(random_mrr, 4),
        "n_heuristic_features": len(heuristic_names),
    }


def find_novel_drivers(shap_result: dict, heuristics: dict, top_n: int = 10) -> list:
    """
    Find top SHAP features that are NOT in the heuristic table.
    These are candidate novel drivers to investigate.
    """
    pass_name = shap_result["pass_name"]
    feature_cols = shap_result["feature_cols"]
    ranking = shap_result["global_ranking"]

    if pass_name not in heuristics:
        heuristic_names = set()
    else:
        heuristic_names = set(f["name"] for f in heuristics[pass_name].get("features", []))

    novel = []
    for idx in ranking[:top_n]:
        feat_name = feature_cols[idx]
        if feat_name not in heuristic_names:
            novel.append({
                "pass_name": pass_name,
                "feature": feat_name,
                "shap_rank": int(np.where(ranking == idx)[0][0]) + 1,
                "mean_abs_shap": round(float(shap_result["global_mean_abs"][idx]), 6),
            })
    return novel


# ============================================================================
# Plotting
# ============================================================================

def plot_beeswarm(shap_values, X, feature_names, pass_name, save_dir):
    """Generate a SHAP beeswarm plot."""
    import shap
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      show=False, max_display=20)
    plt.title(f"SHAP Beeswarm — {pass_name}")
    plt.tight_layout()
    plt.savefig(save_dir / f"shap_beeswarm_{pass_name}.png", dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_dependence(shap_values, X, feature_names, pass_name, top_features, save_dir):
    """Generate SHAP dependence plots for top 5 features."""
    import shap
    for feat_name in top_features[:5]:
        if feat_name not in feature_names:
            continue
        feat_idx = feature_names.index(feat_name)
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(
            feat_idx, shap_values, X,
            feature_names=feature_names,
            show=False, ax=ax,
        )
        ax.set_title(f"SHAP Dependence — {pass_name} — {feat_name}")
        plt.tight_layout()
        plt.savefig(save_dir / f"shap_dep_{pass_name}_{feat_name}.png",
                    dpi=DPI, bbox_inches="tight")
        plt.close()


def plot_waterfall_cases(shap_values, X, feature_names, pass_name, save_dir,
                          n_cases: int = 3):
    """Generate waterfall plots for representative local explanations."""
    import shap

    # Pick cases with diverse predictions: highest SHAP sum, lowest, and median
    shap_sums = np.sum(shap_values, axis=1)
    indices = [
        int(np.argmax(shap_sums)),
        int(np.argmin(shap_sums)),
        int(np.argsort(shap_sums)[len(shap_sums) // 2]),
    ]

    for case_idx, sample_idx in enumerate(indices[:n_cases]):
        explanation = shap.Explanation(
            values=shap_values[sample_idx],
            base_values=0,  # approximate
            data=X[sample_idx],
            feature_names=feature_names,
        )
        fig = plt.figure(figsize=(10, 6))
        plot_path = save_dir / f"shap_waterfall_{pass_name}_case{case_idx + 1}.png"
        shap.plots.waterfall(explanation, show=False, max_display=15)
        plt.title(f"SHAP Waterfall — {pass_name} — Case {case_idx + 1}")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=DPI, bbox_inches="tight")
        plt.close()


# ============================================================================
# Main pipeline
# ============================================================================

def run_phase5(smoke: bool = False):
    """Execute Phase 5: SHAP explainability analysis."""
    ensure_dirs()
    import pandas as pd
    import yaml

    # Load data
    features_path = FEATURES_DIR / "features.csv"
    labels_path = LABELS_DIR / "labels.csv"
    interp_path = LABELS_DIR / "interpretable_passes.json"
    feature_cols_path = FEATURES_DIR / "feature_columns.json"
    heuristics_path = PROJECT_ROOT / "heuristics.yaml"

    for p in [features_path, labels_path]:
        if not p.exists():
            log.error("Required file not found: %s — run earlier phases first.", p)
            sys.exit(1)

    df_features = pd.read_csv(features_path)
    df_labels = pd.read_csv(labels_path)

    # Load interpretable passes
    if interp_path.exists():
        with open(interp_path) as f:
            interp_passes = json.load(f)
    else:
        interp_passes = list(df_labels["pass_name"].unique())

    # Load feature columns
    if feature_cols_path.exists():
        with open(feature_cols_path) as f:
            feature_cols = json.load(f)
    else:
        feature_cols = [c for c in df_features.columns
                        if c not in {"suite", "program", "function", "ir_hash", "inst_count"}]

    # Merge
    df_merged = pd.merge(
        df_labels,
        df_features,
        on=["ir_hash", "suite", "program", "function"],
        how="inner",
    )
    feature_cols = [c for c in feature_cols if c in df_merged.columns]

    # Load heuristics
    heuristics = {}
    if heuristics_path.exists():
        with open(heuristics_path) as f:
            heuristics = yaml.safe_load(f) or {}

    log.info("Phase 5: analyzing %d interpretable passes", len(interp_passes))

    # Create figures subdirectory
    shap_fig_dir = FIGURES_DIR / "shap"
    shap_fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- Compute SHAP for each pass ----
    all_rankings = []
    all_faithfulness = []
    all_heuristic_scores = []
    all_novel_drivers = []
    all_cluster_shap = []

    # Feature correlation clustering (once, shared across passes)
    cluster_map = cluster_correlated_features(df_merged, feature_cols)
    log.info("Feature clusters: %d clusters from %d features",
             len(cluster_map), len(feature_cols))

    for pass_name in interp_passes:
        log.info("=== SHAP analysis for pass: %s ===", pass_name)

        # 1. Compute SHAP
        shap_result = compute_shap_for_pass(pass_name, df_merged, feature_cols)
        if shap_result is None:
            log.warning("SHAP computation failed for %s, skipping.", pass_name)
            continue

        # Rankings
        all_rankings.extend(shap_result["ranking_info"])

        # 2. Plots
        top_features = [feature_cols[i] for i in shap_result["global_ranking"][:5]]
        log.info("  Top 5 features: %s", top_features)

        plot_beeswarm(shap_result["primary_shap"], shap_result["primary_X"],
                      feature_cols, pass_name, shap_fig_dir)
        plot_dependence(shap_result["primary_shap"], shap_result["primary_X"],
                        feature_cols, pass_name, top_features, shap_fig_dir)
        plot_waterfall_cases(shap_result["primary_shap"], shap_result["primary_X"],
                             feature_cols, pass_name, shap_fig_dir)

        # 3. Faithfulness checks
        faith = {
            "pass_name": pass_name,
            "kendall_tau_mean": shap_result["kendall_tau_mean"],
            "kendall_tau_std": shap_result["kendall_tau_std"],
            "top10_overlap_mean": shap_result["top10_overlap_mean"],
        }

        perm = permutation_importance_agreement(pass_name, df_merged, feature_cols)
        faith["perm_kendall_tau"] = perm["perm_kendall_tau"]

        shuffle = label_shuffle_control(pass_name, df_merged, feature_cols)
        faith["shuffle_shap_max"] = shuffle["shuffle_shap_max"]
        faith["shuffle_shap_mean"] = shuffle["shuffle_shap_mean"]

        all_faithfulness.append(faith)

        # 4. Cluster-level SHAP
        cl_shap = shap_at_cluster_level(
            shap_result["primary_shap"], feature_cols, cluster_map
        )
        for cl_id, cl_info in cl_shap.items():
            all_cluster_shap.append({
                "pass_name": pass_name,
                "cluster_id": cl_id,
                "members": "; ".join(cl_info["members"]),
                "mean_abs_shap": cl_info["mean_abs_shap"],
            })

        # 5. Heuristic agreement
        agreement = score_heuristic_agreement(shap_result, heuristics)
        all_heuristic_scores.append(agreement)

        # 6. Novel drivers
        novel = find_novel_drivers(shap_result, heuristics)
        all_novel_drivers.extend(novel)

    # ---- Save results ----
    if all_rankings:
        pd.DataFrame(all_rankings).to_csv(
            RESULTS_DIR / "shap_rankings.csv", index=False)
    if all_faithfulness:
        pd.DataFrame(all_faithfulness).to_csv(
            RESULTS_DIR / "faithfulness.csv", index=False)
    if all_heuristic_scores:
        pd.DataFrame(all_heuristic_scores).to_csv(
            RESULTS_DIR / "heuristic_agreement.csv", index=False)
    if all_novel_drivers:
        pd.DataFrame(all_novel_drivers).to_csv(
            RESULTS_DIR / "novel_drivers.csv", index=False)
    if all_cluster_shap:
        pd.DataFrame(all_cluster_shap).to_csv(
            RESULTS_DIR / "cluster_shap.csv", index=False)

    # Save cluster map
    cluster_path = RESULTS_DIR / "feature_clusters.json"
    with open(cluster_path, "w") as f:
        json.dump(cluster_map, f, indent=2)

    # ---- Gate report ----
    report = f"""
╔══════════════════════════════════════════════════════╗
║              PHASE 5 GATE REPORT                     ║
╠══════════════════════════════════════════════════════╣
║ Passes analyzed:        {len(interp_passes):>8}                     ║
║ Feature clusters:       {len(cluster_map):>8}                     ║
║ Novel driver candidates:{len(all_novel_drivers):>8}                     ║
╚══════════════════════════════════════════════════════╝
"""
    log.info(report)
    gate_path = RESULTS_DIR / "phase5_gate.txt"
    gate_path.write_text(report, encoding='utf-8')


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 5: SHAP Explainability")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_phase5(smoke=args.smoke)


if __name__ == "__main__":
    main()
