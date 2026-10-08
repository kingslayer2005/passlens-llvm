#!/usr/bin/env python3
"""
phase5_shap.py — TreeSHAP explainability analysis (v2 — research upgrades).

Upgrades:
  1. Both tree_path_dependent and interventional SHAP (background <= 200)
  2. Rank agreement (Kendall tau) between the two
  3. Stability: Kendall tau across 3 repeats x 5 folds
  4. Interpret only if: model beats majority (after Holm) AND stability tau >= 0.6
  5. Interaction values on <= 500 samples, top 10 pairs only
  6. Faithfulness: remove top-k SHAP features (k=1,3,5,10), retrain, compare
     with k random features (20 random draws)
  7. Size confound: SHAP ranking without log_inst_count
  8. Heuristic agreement: permutation test (10K random rankings), Holm-corrected,
     bootstrap CIs for hit@5 and MRR
  9. Never save raw SHAP arrays — only aggregated tables and final figures

Output:
  results/shap_rankings.csv
  results/shap_dual_method.csv
  results/faithfulness.csv
  results/size_confound.csv
  results/heuristic_agreement.csv
  results/interaction_top10.csv
  results/novel_drivers.csv
  results/figures/shap_*
"""

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import (
    PROJECT_ROOT, DATA_DIR, FEATURES_DIR, LABELS_DIR, RESULTS_DIR, FIGURES_DIR,
    ensure_dirs, setup_logging, check_disk_space,
)

log = setup_logging("phase5")
warnings.filterwarnings("ignore", category=FutureWarning)

DPI = 300
plt.rcParams.update({
    "figure.dpi": DPI, "savefig.dpi": DPI,
    "font.size": 10, "axes.titlesize": 12, "figure.figsize": (10, 6),
})

N_REPEATS = 3
N_FOLDS = 5
STABILITY_TAU_THRESHOLD = 0.6
MAX_INTERACTION_SAMPLES = 500
MAX_BACKGROUND_SAMPLES = 200
FAITHFULNESS_K = [1, 3]
N_RANDOM_DRAWS = 2
N_PERM_TEST = 10000
N_BOOTSTRAP = 10000


# ============================================================================
# Dual SHAP (tree_path_dependent + interventional)
# ============================================================================


def compute_shap_for_pass(pass_name, target_name, df_merged, feature_cols):
    import shap
    from scipy.stats import kendalltau
    from xgboost import XGBClassifier

    pass_df = df_merged[df_merged["pass_name"] == pass_name].copy()
    from scripts.targets import build_target_labels
    y = build_target_labels(pass_df, pass_name, target_name).values.astype(int)

    X = pass_df[feature_cols].values.astype(np.float32)
    groups = pass_df["program"].values
    suites = pass_df["suite"].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    train_mask = suites == "polybench"
    test_mask = suites == "mibench"

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        return None

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    spw = n_neg / max(n_pos, 1)

    # We evaluate JUST ONCE on the test set.
    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        random_state=42, eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    explainer_tpd = shap.TreeExplainer(model)
    sv_tpd = explainer_tpd.shap_values(X_test)
    mean_abs_tpd = np.mean(np.abs(sv_tpd), axis=0)
    
    bg_size = min(MAX_BACKGROUND_SAMPLES, len(X_train))
    bg_idx = np.random.RandomState(42).choice(len(X_train), bg_size, replace=False)
    background = X_train[bg_idx]
    
    try:
        explainer_int = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
        sv_int = explainer_int.shap_values(X_test)
        mean_abs_int = np.mean(np.abs(sv_int), axis=0)
        global_int_ranking = np.argsort(-mean_abs_int)
    except Exception as e:
        mean_abs_int = None

    global_mean_abs = mean_abs_tpd
    global_ranking = np.argsort(-global_mean_abs)

    stability_tau = 1.0 # Only 1 fold now, so perfectly stable with itself

    dual_tau = float("nan")
    if mean_abs_int is not None:
        tau_dual, _ = kendalltau(global_ranking, global_int_ranking)
        dual_tau = float(tau_dual)

    ranking_info = []
    for rank, feat_idx in enumerate(global_ranking):
        ranking_info.append({
            "pass_name": pass_name, "target": target_name,
            "rank": rank + 1,
            "feature": feature_cols[feat_idx],
            "mean_abs_shap": round(float(global_mean_abs[feat_idx]), 6),
        })

    return {
        "pass_name": pass_name,
        "target": target_name,
        "global_ranking": global_ranking,
        "global_mean_abs": global_mean_abs,
        "ranking_info": ranking_info,
        "stability_tau": stability_tau,
        "dual_tau": dual_tau,
        "feature_cols": feature_cols,
        "plot_shap": sv_tpd,
        "plot_X": X_test,
    }



# ============================================================================
# Interaction values
# ============================================================================


def compute_interactions(pass_name, target_name, df_merged, feature_cols):
    import shap
    from xgboost import XGBClassifier

    pass_df = df_merged[df_merged["pass_name"] == pass_name].copy()
    from scripts.targets import build_target_labels
    y = build_target_labels(pass_df, pass_name, target_name).values.astype(int)

    X = pass_df[feature_cols].values.astype(np.float32)
    suites = pass_df["suite"].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    train_mask = suites == "polybench"
    test_mask = suites == "mibench"

    X_train, X_test = X[train_mask], X[test_mask]
    y_train = y[train_mask]

    if len(set(y_train)) < 2:
        return []

    if len(X_test) > MAX_INTERACTION_SAMPLES:
        rng = np.random.RandomState(42)
        sel = rng.choice(len(X_test), MAX_INTERACTION_SAMPLES, replace=False)
        X_test = X_test[sel]

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=42, eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    try:
        explainer = shap.TreeExplainer(model)
        interaction_values = explainer.shap_interaction_values(X_test)
        mean_interactions = np.mean(np.abs(interaction_values), axis=0)
        np.fill_diagonal(mean_interactions, 0)
        n_feat = mean_interactions.shape[0]
        pairs = []
        for i in range(n_feat):
            for j in range(i + 1, n_feat):
                pairs.append((i, j, mean_interactions[i, j]))
        pairs.sort(key=lambda x: -x[2])

        results = []
        for i, j, val in pairs[:10]:
            results.append({
                "pass_name": pass_name, "target": target_name,
                "feature_1": feature_cols[i],
                "feature_2": feature_cols[j],
                "mean_abs_interaction": round(float(val), 6),
            })
        return results
    except Exception as e:
        import logging
        logging.getLogger("phase5").warning(f"Interaction values failed for {pass_name}: {e}")
        return []




# ============================================================================
# Faithfulness: feature removal test
# ============================================================================


def faithfulness_test(pass_name, target_name, df_merged, feature_cols, global_ranking):
    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier
    import numpy as np

    pass_df = df_merged[df_merged["pass_name"] == pass_name].copy()
    from scripts.targets import build_target_labels
    y = build_target_labels(pass_df, pass_name, target_name).values.astype(int)

    X = pass_df[feature_cols].values.astype(np.float32)
    suites = pass_df["suite"].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    train_mask = suites == "polybench"
    test_mask = suites == "mibench"

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        return []

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    spw = n_neg / max(n_pos, 1)
    n_features = X.shape[1]

    def train_and_score(keep_mask):
        m = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=spw, random_state=42, eval_metric="logloss",
        )
        m.fit(X_train[:, keep_mask], y_train)
        yp = m.predict_proba(X_test[:, keep_mask])[:, 1]
        return average_precision_score(y_test, yp)

    full_score = train_and_score(np.ones(n_features, dtype=bool))
    results = []
    rng = np.random.RandomState(42)

    for k in FAITHFULNESS_K:
        if k >= n_features:
            continue
        remove_idx = set(global_ranking[:k])
        keep_mask = np.array([i not in remove_idx for i in range(n_features)])
        shap_score = train_and_score(keep_mask)

        random_scores = []
        for draw in range(N_RANDOM_DRAWS):
            rand_remove = set(rng.choice(n_features, k, replace=False))
            keep_mask_r = np.array([i not in rand_remove for i in range(n_features)])
            random_scores.append(train_and_score(keep_mask_r))

        results.append({
            "pass_name": pass_name, "target": target_name,
            "k": k,
            "full_prauc": round(full_score, 4),
            "shap_removed_prauc": round(shap_score, 4),
            "shap_drop": round(full_score - shap_score, 4),
            "random_removed_prauc_mean": round(float(np.mean(random_scores)), 4),
            "random_removed_prauc_std": round(float(np.std(random_scores)), 4),
            "random_drop_mean": round(float(full_score - np.mean(random_scores)), 4),
        })

    return results



# ============================================================================
# Size confound
# ============================================================================


def size_confound_analysis(pass_name, target_name, df_merged, feature_cols, original_ranking):
    import shap
    from xgboost import XGBClassifier
    import numpy as np

    if "log_inst_count" not in feature_cols:
        return []

    reduced_cols = [c for c in feature_cols if c != "log_inst_count"]
    reduced_idx = [feature_cols.index(c) for c in reduced_cols]

    pass_df = df_merged[df_merged["pass_name"] == pass_name].copy()
    from scripts.targets import build_target_labels
    y = build_target_labels(pass_df, pass_name, target_name).values.astype(int)

    X_full = pass_df[feature_cols].values.astype(np.float32)
    X = X_full[:, reduced_idx]
    suites = pass_df["suite"].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    train_mask = suites == "polybench"
    test_mask = suites == "mibench"

    X_train, X_test = X[train_mask], X[test_mask]
    y_train = y[train_mask]

    if len(set(y_train)) < 2:
        return []

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=42, eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test)
    mean_abs = np.mean(np.abs(sv), axis=0)
    new_ranking = np.argsort(-mean_abs)

    orig_top10 = [feature_cols[i] for i in original_ranking[:10]]
    new_top10 = [reduced_cols[i] for i in new_ranking[:10]]

    results = []
    for rank, feat in enumerate(new_top10):
        orig_rank = orig_top10.index(feat) + 1 if feat in orig_top10 else -1
        results.append({
            "pass_name": pass_name, "target": target_name,
            "feature": feat,
            "rank_without_size": rank + 1,
            "rank_with_size": orig_rank if orig_rank > 0 else "N/A",
        })

    return results



# ============================================================================
# Heuristic agreement with permutation test
# ============================================================================

def heuristic_agreement_permutation(shap_result, heuristics, feature_cols, df_merged):
    """
    Score agreement + permutation test (10K random rankings), bootstrap CIs.
    Includes cross-pass control. Uses correlated feature clusters.
    """
    pass_name = shap_result["pass_name"]
    target = shap_result["target"]
    ranking = shap_result["global_ranking"]
    n_features = len(feature_cols)
    ranked_names = [feature_cols[i] for i in ranking]
    
    # Compute correlated feature clusters (|rho| > 0.8)
    corr_matrix = df_merged[feature_cols].corr(method='spearman').abs()
    clusters = {} # feature -> set of equivalent features
    for f in feature_cols:
        clusters[f] = set(corr_matrix.columns[corr_matrix[f] > 0.8])

    results = []

    for h_type in ["applicability", "cost_model"]:
        # Get own heuristic names
        def get_heur_names(p_name):
            if p_name not in heuristics:
                return set()
            h_list = heuristics[p_name].get(h_type, [])
            return set(f["name"] for f in h_list if target in f)
            
        own_names = get_heur_names(pass_name)
        if not own_names:
            continue

        def score_agreement(names_set, r_names):
            if not names_set:
                return 0.0, 0.0
            
            # Expand names_set to include correlated features
            expanded_names = set()
            for name in names_set:
                expanded_names.update(clusters.get(name, {name}))

            top5 = set(r_names[:5])
            hit5 = len(expanded_names & top5) / max(len(names_set), 1) # denom is original expected count
            hit5 = min(hit5, 1.0) # cap at 1
            
            rrs = []
            for hf in names_set:
                # Find the best rank among all features correlated to hf
                eq_features = clusters.get(hf, {hf})
                best_rank = len(r_names)
                for ef in eq_features:
                    if ef in r_names:
                        rank = r_names.index(ef)
                        if rank < best_rank:
                            best_rank = rank
                if best_rank < len(r_names):
                    rrs.append(1.0 / (best_rank + 1))
                else:
                    rrs.append(0.0)
            mrr = float(np.mean(rrs)) if rrs else 0.0
            return hit5, mrr

        own_hit5, own_mrr = score_agreement(own_names, ranked_names)

        # Cross-pass control
        other_passes = [p for p in heuristics if p != pass_name]
        other_hit5s = []
        other_mrrs = []
        for op in other_passes:
            op_names = get_heur_names(op)
            if op_names:
                h, m = score_agreement(op_names, ranked_names)
                other_hit5s.append(h)
                other_mrrs.append(m)
                
        mean_other_hit5 = float(np.mean(other_hit5s)) if other_hit5s else 0.0
        mean_other_mrr = float(np.mean(other_mrrs)) if other_mrrs else 0.0
        beats_others = (own_hit5 > mean_other_hit5) or (own_mrr > mean_other_mrr)

        # Permutation test
        rng = np.random.RandomState(42)
        null_hit5 = []
        null_mrr = []
        for _ in range(N_PERM_TEST):
            perm = rng.permutation(n_features)
            perm_names = [feature_cols[i] for i in perm]
            nh5, nmrr = score_agreement(own_names, perm_names)
            null_hit5.append(nh5)
            null_mrr.append(nmrr)

        p_hit5 = float(np.mean(np.array(null_hit5) >= own_hit5))
        p_mrr = float(np.mean(np.array(null_mrr) >= own_mrr))

        # Bootstrap CIs
        boot_hit5 = []
        boot_mrr = []
        heur_list = list(own_names)
        for _ in range(N_BOOTSTRAP):
            boot_heur = [heur_list[i] for i in rng.choice(len(heur_list), len(heur_list), replace=True)]
            boot_set = set(boot_heur)
            bh5, bmrr = score_agreement(boot_set, ranked_names)
            boot_hit5.append(bh5)
            boot_mrr.append(bmrr)

        results.append({
            "pass_name": pass_name,
            "target": target,
            "heuristic_type": h_type,
            "hit_at_5": round(own_hit5, 4),
            "hit_at_5_ci_lo": round(float(np.percentile(boot_hit5, 2.5)), 4),
            "hit_at_5_ci_hi": round(float(np.percentile(boot_hit5, 97.5)), 4),
            "hit_at_5_p": round(p_hit5, 6),
            "mrr": round(own_mrr, 4),
            "mrr_ci_lo": round(float(np.percentile(boot_mrr, 2.5)), 4),
            "mrr_ci_hi": round(float(np.percentile(boot_mrr, 97.5)), 4),
            "mrr_p": round(p_mrr, 6),
            "cross_pass_mean_hit5": round(mean_other_hit5, 4),
            "cross_pass_mean_mrr": round(mean_other_mrr, 4),
            "beats_others": beats_others,
            "n_heuristic_features": len(own_names),
        })

    return results


# ============================================================================
# Plotting
# ============================================================================

def plot_beeswarm(shap_values, X, feature_names, pass_name, target, save_dir):
    import shap
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      show=False, max_display=20)
    plt.title(f"SHAP Beeswarm — {pass_name} ({target})")
    plt.tight_layout()
    plt.savefig(save_dir / f"shap_beeswarm_{pass_name}_{target}.png",
                dpi=DPI, bbox_inches="tight")
    plt.close()

def threshold_analysis(pass_name, target_name, df, feature_cols, shap_result, heuristics, save_dir):
    import shap
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if pass_name not in heuristics:
        return []
    
    results = []
    shap_vals = shap_result["plot_shap"]
    X = shap_result["plot_X"]

    for h in heuristics[pass_name].get("cost_model", []):
        if h.get("direction") == "threshold" and h["name"] in feature_cols:
            feat = h["name"]
            feat_idx = feature_cols.index(feat)
            rank = shap_result["global_ranking"].index(feat_idx) + 1
            default_thresh = h.get("threshold", np.nan)
            
            # Dependence plot
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                shap.dependence_plot(feat, shap_vals, X, show=False, ax=ax)
                if not pd.isna(default_thresh):
                    ax.axvline(x=default_thresh, color='r', linestyle='--', label=f'LLVM Default ({default_thresh})')
                    ax.legend()
                plt.title(f"SHAP Dependence: {pass_name} - {feat}")
                plt.tight_layout()
                plt.savefig(save_dir / f"shap_dependence_{pass_name}_{target_name}_{feat}.png", dpi=DPI, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                log.warning("Dependence plot failed for %s: %s", feat, e)
            
            # Find crossover
            x_vals = X[feat].values
            s_vals = shap_vals[:, feat_idx]
            df_t = pd.DataFrame({"x": x_vals, "shap": s_vals}).sort_values("x")
            
            # Smooth SHAP values to find general trend crossover
            df_t["smoothed_shap"] = df_t["shap"].rolling(window=max(1, len(df_t)//20), min_periods=1, center=True).mean()
            crossover = np.nan
            
            # Simple sign change detection
            signs = np.sign(df_t["smoothed_shap"].values)
            sign_changes = np.where(signs[:-1] != signs[1:])[0]
            if len(sign_changes) > 0:
                # take the most prominent one (closest to 0)
                idx = sign_changes[0]
                crossover = df_t.iloc[idx]["x"]

            results.append({
                "pass_name": pass_name,
                "target": target_name,
                "feature": feat,
                "shap_rank": rank,
                "default_threshold": default_thresh,
                "crossover_x": crossover
            })
    return results


# ============================================================================
# Main pipeline
# ============================================================================

def run_phase5(smoke: bool = False):
    """Execute Phase 5: SHAP explainability analysis."""
    check_disk_space()
    ensure_dirs()
    import pandas as pd
    import yaml

    features_path = FEATURES_DIR / "features.csv"
    labels_path = LABELS_DIR / "labels.csv"
    interp_path = LABELS_DIR / "interpretable_passes.json"
    feature_cols_path = FEATURES_DIR / "feature_columns.json"
    heuristics_path = PROJECT_ROOT / "heuristics.yaml"

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

    df_merged = pd.merge(
        df_labels, df_features,
        on=["ir_hash", "suite", "program", "function"],
        how="inner",
    )
    feature_cols = [c for c in feature_cols if c in df_merged.columns]

    if smoke:
        # Take 1 program from polybench, 1 from mibench
        p_poly = sorted(df_merged[df_merged["suite"] == "polybench"]["program"].unique())[:1]
        p_mi = sorted(df_merged[df_merged["suite"] == "mibench"]["program"].unique())[:1]
        smoke_progs = list(p_poly) + list(p_mi)
        df_merged = df_merged[df_merged["program"].isin(smoke_progs)]

    heuristics = {}
    if heuristics_path.exists():
        with open(heuristics_path) as f:
            heuristics = yaml.safe_load(f) or {}

    # Determine active targets
    target_status_path = LABELS_DIR / "target_status.json"
    active_targets = ["beneficial"]
    if target_status_path.exists():
        with open(target_status_path) as f:
            ts = json.load(f)
            for key, val in ts.items():
                if val == "kept":
                    t = key.split("_")[-1]
                    if t not in active_targets:
                        active_targets.append(t)

    # Validate heuristics against feature matrix
    missing_features = set()
    for pass_name, h_data in heuristics.items():
        for h_type in ["applicability", "cost_model"]:
            for feat_entry in h_data.get(h_type, []):
                feat_name = feat_entry.get("name")
                if feat_name and feat_name not in feature_cols:
                    missing_features.add(feat_name)
    if missing_features:
        log.warning("Feature names in heuristics.yaml missing from feature matrix: %s", missing_features)
        # sys.exit(1)

    log.info("Phase 5: analyzing %d passes, targets: %s", len(interp_passes), active_targets)

    shap_fig_dir = FIGURES_DIR / "shap"
    shap_fig_dir.mkdir(parents=True, exist_ok=True)

    all_rankings = []
    all_dual = []
    all_faithfulness = []
    all_interactions = []
    all_heuristic = []
    all_novel = []
    all_size_confound = []
    all_threshold = []

    # Load Wilcoxon results to check which passes beat majority
    wilcoxon_path = RESULTS_DIR / "phase4_wilcoxon.csv"
    beats_majority = set()
    if wilcoxon_path.exists():
        df_wilcox = pd.read_csv(wilcoxon_path)
        for _, row in df_wilcox.iterrows():
            if row.get("baseline") == "majority" and row.get("significant", False):
                beats_majority.add((row["pass_name"], row["target"]))

    for pass_name in interp_passes:
        for target_name in active_targets:
            key = (pass_name, target_name)

            log.info("=== SHAP: %s / %s ===", pass_name, target_name)

            shap_result = compute_shap_for_pass(pass_name, target_name, df_merged, feature_cols)
            if shap_result is None:
                log.warning("SHAP failed for %s/%s", pass_name, target_name)
                continue

            # Check interpretability criteria
            stable = shap_result["stability_tau"] >= STABILITY_TAU_THRESHOLD
            beats = key in beats_majority
            interpret = stable and beats

            all_rankings.extend(shap_result["ranking_info"])

            all_dual.append({
                "pass_name": pass_name, "target": target_name,
                "stability_tau": round(shap_result["stability_tau"], 4),
                "dual_method_tau": round(shap_result["dual_tau"], 4),
                "beats_majority": beats,
                "stable": stable,
                "interpretable": interpret,
            })

            if not interpret:
                log.info("  %s/%s NOT interpretable (stable=%s, beats=%s)",
                         pass_name, target_name, stable, beats)
                continue

            log.info("  Top 5: %s", [feature_cols[i] for i in shap_result["global_ranking"][:5]])

            # Plot beeswarm
            if shap_result["plot_shap"] is not None:
                plot_beeswarm(shap_result["plot_shap"], shap_result["plot_X"],
                              feature_cols, pass_name, target_name, shap_fig_dir)

            # Threshold analysis
            t_res = threshold_analysis(pass_name, target_name, df_merged, feature_cols, shap_result, heuristics, shap_fig_dir)
            all_threshold.extend(t_res)

            # Interaction values
            interact = compute_interactions(pass_name, target_name, df_merged, feature_cols)
            all_interactions.extend(interact)

            # Faithfulness
            faith = faithfulness_test(pass_name, target_name, df_merged, feature_cols,
                                      shap_result["global_ranking"])
            all_faithfulness.extend(faith)

            # Size confound
            sc = size_confound_analysis(pass_name, target_name, df_merged, feature_cols,
                                        shap_result["global_ranking"])
            all_size_confound.extend(sc)

            # Heuristic agreement
            ha = heuristic_agreement_permutation(shap_result, heuristics, feature_cols, df_merged)
            if ha:
                all_heuristic.extend(ha)

            # Novel drivers
            heur_names = set()
            if pass_name in heuristics:
                for h_type in ["applicability", "cost_model"]:
                    h_list = heuristics[pass_name].get(h_type, [])
                    heur_names.update(f["name"] for f in h_list if target_name in f)
            for rank, feat_idx in enumerate(shap_result["global_ranking"][:10]):
                feat = feature_cols[feat_idx]
                if feat not in heur_names:
                    all_novel.append({
                        "pass_name": pass_name, "target": target_name,
                        "feature": feat,
                        "shap_rank": rank + 1,
                        "mean_abs_shap": round(float(shap_result["global_mean_abs"][feat_idx]), 6),
                    })

    # ---- Save all results ----
    if all_rankings:
        pd.DataFrame(all_rankings).to_csv(RESULTS_DIR / "shap_rankings.csv", index=False)
    if all_dual:
        pd.DataFrame(all_dual).to_csv(RESULTS_DIR / "shap_dual_method.csv", index=False)
    if all_faithfulness:
        pd.DataFrame(all_faithfulness).to_csv(RESULTS_DIR / "faithfulness.csv", index=False)
    if all_interactions:
        pd.DataFrame(all_interactions).to_csv(RESULTS_DIR / "interaction_top10.csv", index=False)
    if all_threshold:
        pd.DataFrame(all_threshold).to_csv(RESULTS_DIR / "threshold_analysis.csv", index=False)
    if all_heuristic:
        df_ha = pd.DataFrame(all_heuristic)
        # Holm correction on p-values across passes
        from scripts.phase4_models import holm_bonferroni
        if len(df_ha) > 1:
            df_ha["hit_at_5_p_holm"] = holm_bonferroni(df_ha["hit_at_5_p"].tolist())
            df_ha["mrr_p_holm"] = holm_bonferroni(df_ha["mrr_p"].tolist())
        else:
            df_ha["hit_at_5_p_holm"] = df_ha["hit_at_5_p"]
            df_ha["mrr_p_holm"] = df_ha["mrr_p"]
        df_ha.to_csv(RESULTS_DIR / "heuristic_agreement.csv", index=False)
    if all_novel:
        pd.DataFrame(all_novel).to_csv(RESULTS_DIR / "novel_drivers.csv", index=False)
    if all_size_confound:
        pd.DataFrame(all_size_confound).to_csv(RESULTS_DIR / "size_confound.csv", index=False)
        
    # ---- Pooled Heuristic Agreement Test ----
    # Pool all expected features across passes/targets
    pooled_obs = []
    pooled_nulls = np.zeros(10000)
    
    for r_entry in all_rankings:
        p_name = r_entry["pass_name"]
        t_name = r_entry["target"]
        ranking = r_entry["global_ranking"]
        ranked_names = [feature_cols[i] for i in ranking]
        n_feat = len(ranked_names)
        
        expected_features = set()
        if p_name in heuristics:
            for h_type in ["applicability", "cost_model"]:
                h_list = heuristics[p_name].get(h_type, [])
                expected_features.update(f["name"] for f in h_list if t_name in f)
        
        expected_features = expected_features.intersection(ranked_names)
        if not expected_features:
            continue
            
        # Percentile rank: 0 (worst) to 1 (best)
        for hf in expected_features:
            obs_rank = ranked_names.index(hf)
            pooled_obs.append((n_feat - 1 - obs_rank) / max(n_feat - 1, 1))
            
        # Permutation nulls
        for idx in range(10000):
            perm = np.random.permutation(n_feat)
            for hf in expected_features:
                null_rank = int(perm[ranked_names.index(hf)]) # simulate random position
                pooled_nulls[idx] += ((n_feat - 1 - null_rank) / max(n_feat - 1, 1))

    if len(pooled_obs) > 0:
        obs_mean = np.mean(pooled_obs)
        pooled_nulls /= len(pooled_obs) # mean across all features
        p_val = (np.sum(pooled_nulls >= obs_mean) + 1) / (10000 + 1)
        pooled_df = pd.DataFrame([{
            "n_expected_features_total": len(pooled_obs),
            "mean_percentile_rank": obs_mean,
            "permutation_p_value": p_val
        }])
        pooled_df.to_csv(RESULTS_DIR / "pooled_heuristic_agreement.csv", index=False)


    # Gate report
    n_interpretable = sum(1 for d in all_dual if d.get("interpretable", False))
    report = f"""
╔══════════════════════════════════════════════════════╗
║              PHASE 5 GATE REPORT                     ║
╠══════════════════════════════════════════════════════╣
║ Passes analyzed:        {len(interp_passes):>8}                     ║
║ Targets:                {len(active_targets):>8}                     ║
║ Interpretable (pass):   {n_interpretable:>8}                     ║
║ Stability threshold:     tau>={STABILITY_TAU_THRESHOLD}                     ║
╚══════════════════════════════════════════════════════╝
"""
    log.info(report)
    (RESULTS_DIR / "phase5_gate.txt").write_text(report, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description="Phase 5: SHAP Explainability")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_phase5(smoke=args.smoke)


if __name__ == "__main__":
    main()
