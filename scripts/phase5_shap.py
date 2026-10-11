#!/usr/bin/env python3
"""
phase5_shap.py — Why does the model predict what it predicts, and does that
match LLVM's own rules?

Only (pass, target) pairs where XGBoost beat the majority baseline in
Phase 4 are explained.  For each of them:

  A. TreeSHAP on HELD-OUT folds only (the same 15 folds as Phase 4, refitted
     with the hyperparameters chosen there).
  B. Checks that the explanation can be trusted:
       stability        Kendall tau between the fold rankings (gate: >= 0.6)
       dual method      path-dependent vs interventional SHAP
       permutation      SHAP vs permutation importance
       shuffle control  with shuffled labels the model must fall to chance
       size confound    ranking with and without log_inst_count
       faithfulness     removing top SHAP features must hurt more than
                        removing random ones
  C. Comparison with heuristics.yaml (frozen before any result existed):
       hit@5, mean reciprocal rank, permutation null, Holm correction,
       direction agreement, cross-pass control, pooled test
  D. Threshold analysis for LLVM's cost-model options, interaction pairs,
     threshold sensitivity and candidate "novel" drivers.

A pair is INTERPRETED only if it beat the majority baseline (Phase 4) and its
SHAP ranking is stable.  Statistical tests of heuristic agreement use
interpreted pairs only.

No raw SHAP arrays are written to disk: only aggregated tables and figures.

Usage:
  python -m scripts.phase5_shap [--smoke] [--jobs N] [--only p1,p2] [--force]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.analysis_common import (
    MASTER_TABLE, DiskGuard, Settings, build_pair_data, cache_dir, fold_is_usable,
    get_settings, holm_correction, importance_share, load_dataset, load_tuned_params,
    make_xgb, pair_key, percentile_ci, pr_auc, ranks_from_scores, read_master_table,
    safe_n_jobs, select_passes, tree_shap, usable_folds, write_json_small,
)
from scripts.targets import DEFAULT_THRESHOLD
from scripts.utils import FIGURES_DIR, PROJECT_ROOT, RESULTS_DIR, ensure_dirs, setup_logging

log = setup_logging("phase5")

HEURISTICS_PATH = PROJECT_ROOT / "heuristics.yaml"
SIZE_FEATURE = "log_inst_count"
# Loop-size features are 0 for functions without loops; a threshold on them
# only means something where a loop exists.
LOOP_SIZE_FEATURES = {"max_loop_header_size", "min_loop_header_size", "innermost_loop_body_size"}
SENSITIVITY_THRESHOLDS = (0.005, 0.01, 0.02)
TOP_N = 5


# ============================================================================
# heuristics.yaml
# ============================================================================

def load_heuristics(feature_names: List[str]) -> dict:
    """
    Read heuristics.yaml and stop if it names a feature that does not exist.
    Returns {pass: {"applicability": [...], "cost_model": [...]}}.
    """
    import yaml
    with open(HEURISTICS_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    heuristics = {}
    unknown = []
    for pass_name, entry in raw.items():
        lists = {}
        for kind in ("applicability", "cost_model"):
            lists[kind] = list(entry.get(kind) or [])
            for item in lists[kind]:
                if item["name"] not in feature_names:
                    unknown.append(f"{pass_name}/{kind}/{item['name']}")
        heuristics[pass_name] = lists
    if unknown:
        raise ValueError(f"heuristics.yaml names features that do not exist: {unknown}")
    return heuristics


def expected_direction(item: dict, target: str) -> Optional[str]:
    """
    'high' or 'low' for an applicability entry.  The 'fired' target uses the
    direction written for 'beneficial': the applicability list says what a
    pass needs in order to do anything.
    """
    key = "harmful" if target == "harmful" else "beneficial"
    value = item.get(key)
    return value if value in ("high", "low") else None


# ============================================================================
# One (pass, target) pair — runs in a worker process
# ============================================================================

def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import kendalltau
    value = kendalltau(a, b).statistic
    return float(value) if value == value else float("nan")


def mean_pairwise_tau(vectors: List[np.ndarray]) -> float:
    """Average Kendall tau over every pair of importance vectors."""
    taus = [kendall_tau(vectors[i], vectors[j])
            for i in range(len(vectors)) for j in range(i + 1, len(vectors))]
    taus = [t for t in taus if t == t]
    return float(np.mean(taus)) if taus else float("nan")


def mean_topk_overlap(vectors: List[np.ndarray], k: int = 10) -> float:
    """Average Jaccard overlap of the top-k feature sets over every pair of folds."""
    tops = [set(np.argsort(-v)[:k]) for v in vectors]
    overlaps = [len(tops[i] & tops[j]) / len(tops[i] | tops[j])
                for i in range(len(tops)) for j in range(i + 1, len(tops))]
    return float(np.mean(overlaps)) if overlaps else float("nan")


def explain_pair(pass_name: str, target: str, cfg: Settings) -> dict:
    """Everything in sections A, B and D of the module docstring for one pair."""
    from scipy.stats import spearmanr

    start = time.time()
    features, labels = load_dataset()
    tuned_file = load_tuned_params()
    feature_names = tuned_file["feature_names"]
    tuned_pair = tuned_file["pairs"][pair_key(pass_name, target)]
    data = build_pair_data(features, labels, pass_name, target, feature_names=feature_names)
    heuristics = load_heuristics(feature_names)
    X, y = data.X, data.y
    n_features = X.shape[1]
    notes = []

    # ------------------------------------------------------------------ A
    # TreeSHAP on every held-out fold
    fold_shares = []                       # one importance vector per fold
    fold_ids = []
    shap_oof = np.full(X.shape, np.nan, dtype=np.float32)   # repeat 0 only
    first_repeat = []                      # (train_idx, test_idx, params, seed, model)

    for repeat, fold, train_idx, test_idx, params, seed in usable_folds(data, tuned_pair, cfg):
        model = make_xgb(params, y[train_idx], seed)
        model.fit(X[train_idx], y[train_idx])
        values = tree_shap(model, X[test_idx])
        fold_shares.append(importance_share(values))
        fold_ids.append(f"r{repeat}f{fold}")
        if repeat == 0:
            shap_oof[test_idx] = values
            first_repeat.append((train_idx, test_idx, params, seed, model))

    if len(fold_shares) < 2:
        return {"pass_name": pass_name, "target": target, "failed": "fewer than 2 usable folds"}

    fold_shares_arr = np.vstack(fold_shares)
    share = fold_shares_arr.mean(axis=0)            # global importance share
    rank = ranks_from_scores(share)

    # Direction: does a HIGHER feature value push the prediction UP?
    covered = ~np.isnan(shap_oof[:, 0])
    direction = np.full(n_features, np.nan)
    for j in range(n_features):
        column = X[covered, j]
        if np.ptp(column) > 0 and np.ptp(shap_oof[covered, j]) > 0:
            direction[j] = spearmanr(column, shap_oof[covered, j]).statistic

    # ------------------------------------------------------------------ B
    stability_tau = mean_pairwise_tau(fold_shares)
    top10_overlap = mean_topk_overlap(fold_shares, k=10)

    # The remaining checks use the folds of repeat 0
    path_r0, interventional, permutation = [], [], []
    shuffle_pr, shuffle_tau, shuffle_vectors, chance_level = [], [], [], []
    size_free_vectors = []
    faith_rows = []
    rng = np.random.RandomState(cfg.base_seed)
    # Size confound: log_inst_count AND every feature that moves with it
    # (for example ret_density, which is about 1 / instruction count) are
    # removed together; removing only one of them would leave a size proxy.
    size_index = feature_names.index(SIZE_FEATURE) if SIZE_FEATURE in feature_names else None
    cluster_of = feature_clusters(features, feature_names)
    size_columns = ([j for j in range(n_features) if cluster_of[j] == cluster_of[size_index]]
                    if size_index is not None else [])
    other_columns = [j for j in range(n_features) if j not in size_columns]
    interventional_note = ""

    for fold_number, (train_idx, test_idx, params, seed, model) in enumerate(first_repeat):
        X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
        full_score = pr_auc(y_test, model.predict_proba(X_test)[:, 1])

        # -- dual method: interventional SHAP on a subsample of the test fold
        sub = rng.choice(len(test_idx), size=min(300, len(test_idx)), replace=False)
        path_r0.append(importance_share(tree_shap(model, X_test[sub])))
        try:
            import shap
            background = X_train[rng.choice(len(train_idx),
                                            size=min(cfg.background_size, len(train_idx)),
                                            replace=False)]
            explainer = shap.TreeExplainer(model, data=background,
                                           feature_perturbation="interventional")
            values = np.asarray(explainer.shap_values(X_test[sub], check_additivity=False))
            if values.ndim == 3:            # some versions return one slice per class
                values = values[..., -1]
            interventional.append(importance_share(values))
        except Exception as error:          # recorded in the output, never hidden
            interventional_note = f"interventional SHAP failed: {type(error).__name__}: {error}"[:200]

        # -- permutation importance on the test fold
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(model, X_test, y_test, scoring="average_precision",
                                      n_repeats=5, random_state=seed, n_jobs=1)
        permutation.append(kendall_tau(fold_shares[fold_number], perm.importances_mean))

        # -- label-shuffle control: same model, labels shuffled in the training fold
        shuffled = np.random.RandomState(seed).permutation(y_train)
        control = make_xgb(params, shuffled, seed)
        control.fit(X_train, shuffled)
        shuffle_pr.append(pr_auc(y_test, control.predict_proba(X_test)[:, 1]))
        chance_level.append(float(y_test.mean()))
        control_share = importance_share(tree_shap(control, X_test))
        shuffle_vectors.append(control_share)
        shuffle_tau.append(kendall_tau(fold_shares[fold_number], control_share))

        # -- size confound: refit without the size features
        if size_index is not None:
            no_size = make_xgb(params, y_train, seed)
            no_size.fit(X_train[:, other_columns], y_train)
            size_free_vectors.append(importance_share(tree_shap(no_size, X_test[:, other_columns])))

        # -- faithfulness: rank features on the TRAINING fold, remove the top k
        train_sub = rng.choice(len(train_idx), size=min(500, len(train_idx)), replace=False)
        train_order = np.argsort(-importance_share(tree_shap(model, X_train[train_sub])))

        def score_without(removed: np.ndarray) -> float:
            keep = [j for j in range(n_features) if j not in set(removed.tolist())]
            pruned = make_xgb(params, y_train, seed)
            pruned.fit(X_train[:, keep], y_train)
            return pr_auc(y_test, pruned.predict_proba(X_test[:, keep])[:, 1])

        for k in cfg.faithfulness_k:
            if k >= n_features:
                continue
            drop_top = full_score - score_without(train_order[:k])
            drop_random = np.mean([
                full_score - score_without(rng.choice(n_features, size=k, replace=False))
                for _ in range(cfg.n_random_draws)])
            faith_rows.append({"fold": fold_number, "k": k, "full_pr_auc": full_score,
                               "drop_top_shap": drop_top, "drop_random": float(drop_random)})

    dual_tau = (kendall_tau(np.mean(path_r0, axis=0), np.mean(interventional, axis=0))
                if len(interventional) == len(path_r0) and interventional else float("nan"))

    faithfulness = []
    if faith_rows:
        faith_df = pd.DataFrame(faith_rows)
        for k, group in faith_df.groupby("k"):
            faithfulness.append({
                "k": int(k), "n_folds": len(group),
                "full_pr_auc": round(group["full_pr_auc"].mean(), 4),
                "drop_top_shap": round(group["drop_top_shap"].mean(), 4),
                "drop_random": round(group["drop_random"].mean(), 4),
                "folds_where_shap_hurts_more": int((group["drop_top_shap"] > group["drop_random"]).sum()),
            })

    # Size confound: how do the ranks of the other features move?
    size_confound = []
    size_tau = float("nan")
    rank_without_size = {}
    if size_free_vectors:
        with_size = np.mean([v[other_columns] for v in fold_shares[:len(size_free_vectors)]], axis=0)
        without_size = np.mean(size_free_vectors, axis=0)
        size_tau = kendall_tau(with_size, without_size)
        rank_with = ranks_from_scores(with_size)
        rank_without = ranks_from_scores(without_size)
        rank_without_size = {feature_names[j]: int(rank_without[i]) for i, j in enumerate(other_columns)}
        for i in np.argsort(rank_with)[:10]:
            size_confound.append({
                "feature": feature_names[other_columns[i]],
                "rank_with_size_features": int(rank_with[i]),
                "rank_without_size_features": int(rank_without[i]),
            })

    # ------------------------------------------------------------------ D
    # Interaction pairs, from the first fold's model
    interactions = []
    if first_repeat:
        import xgboost
        train_idx, test_idx, params, seed, model = first_repeat[0]
        sub = rng.choice(len(test_idx), size=min(cfg.interaction_samples, len(test_idx)), replace=False)
        matrix = model.get_booster().predict(xgboost.DMatrix(X[test_idx][sub]), pred_interactions=True)
        strength = np.abs(matrix[:, :-1, :-1]).mean(axis=0)
        pairs = [(2 * strength[i, j], i, j) for i in range(n_features) for j in range(i + 1, n_features)]
        for value, i, j in sorted(pairs, reverse=True)[:10]:
            interactions.append({"feature_1": feature_names[i], "feature_2": feature_names[j],
                                 "mean_abs_interaction": round(float(value), 6)})

    # Dependence of the SHAP value on each cost-model feature of this pass
    dependence, threshold_rows = [], []
    for item in heuristics.get(pass_name, {}).get("cost_model", []):
        if item.get("direction") != "threshold":
            continue
        j = feature_names.index(item["name"])
        mask = covered.copy()
        if item["name"] in LOOP_SIZE_FEATURES:
            mask &= X[:, j] > 0                  # only functions that have a loop
        values, contributions = X[mask, j], shap_oof[mask, j]
        row = {"feature": item["name"], "option": item.get("option", ""),
               "llvm_threshold": item.get("threshold"), "n_samples": int(mask.sum()),
               "feature_rank": int(rank[j]), "crossover": None, "sign_changes": None,
               "mean_shap_at_or_below_threshold": None, "mean_shap_above_threshold": None,
               "n_at_or_below": None, "n_above": None}
        if mask.sum() >= 30 and np.ptp(values) > 0:
            # Up to 20 bins with about the same number of samples each
            edges = np.unique(np.quantile(values, np.linspace(0, 1, 21)))
            bins = []
            for low, high in zip(edges[:-1], edges[1:]):
                inside = (values >= low) & ((values < high) | (high == edges[-1]))
                if inside.sum() >= 10:
                    bins.append({"bin_low": float(low), "bin_high": float(high),
                                 "n": int(inside.sum()),
                                 "mean_value": float(values[inside].mean()),
                                 "mean_shap": float(contributions[inside].mean())})
            for b in bins:
                dependence.append({"feature": item["name"], **b})
            signs = [np.sign(b["mean_shap"]) for b in bins]
            changes = [i for i in range(1, len(signs)) if signs[i] != signs[i - 1] and signs[i] != 0]
            row["sign_changes"] = len(changes)
            if changes:
                row["crossover"] = bins[changes[0]]["bin_low"]   # where the sign first flips
            threshold = float(item["threshold"])
            below, above = values <= threshold, values > threshold
            row["n_at_or_below"], row["n_above"] = int(below.sum()), int(above.sum())
            if below.sum():
                row["mean_shap_at_or_below_threshold"] = float(contributions[below].mean())
            if above.sum():
                row["mean_shap_above_threshold"] = float(contributions[above].mean())
        threshold_rows.append(row)

    # Threshold sensitivity of the 'beneficial' label (0.5 %, 1 %, 2 %)
    sensitivity = []
    if target == "beneficial" and tuned_pair.get("modal_params"):
        from scripts.analysis_common import outer_splits
        reference_share = None
        for threshold in (DEFAULT_THRESHOLD,) + tuple(t for t in SENSITIVITY_THRESHOLDS
                                                      if t != DEFAULT_THRESHOLD):
            alt = build_pair_data(features, labels, pass_name, target, threshold=threshold,
                                  feature_names=feature_names)
            vectors = []
            for fold, (tr, te) in enumerate(outer_splits(alt.y, alt.groups, 0, cfg)):
                if not fold_is_usable(alt.y[tr], alt.y[te])[0]:
                    continue
                m = make_xgb(tuned_pair["modal_params"], alt.y[tr], cfg.base_seed + fold)
                m.fit(alt.X[tr], alt.y[tr])
                vectors.append(importance_share(tree_shap(m, alt.X[te])))
            entry = {"threshold": threshold, "positive_rate": round(float(alt.y.mean()), 4),
                     "n_folds": len(vectors), "top5": "", "tau_vs_1pct": None}
            if vectors:
                mean_share = np.mean(vectors, axis=0)
                if threshold == DEFAULT_THRESHOLD:
                    reference_share = mean_share
                entry["top5"] = "; ".join(feature_names[j] for j in np.argsort(-mean_share)[:TOP_N])
                if reference_share is not None:
                    entry["tau_vs_1pct"] = round(kendall_tau(reference_share, mean_share), 4)
            sensitivity.append(entry)
        sensitivity.sort(key=lambda e: e["threshold"])

    # ------------------------------------------------------------- figures
    figure_note = make_figures(pass_name, target, feature_names, X[covered], shap_oof[covered],
                               heuristics.get(pass_name, {}).get("cost_model", []), dependence)
    if figure_note:
        notes.append(figure_note)
    if interventional_note:
        notes.append(interventional_note)

    return {
        "pass_name": pass_name, "target": target, "failed": "",
        "n_folds": len(fold_shares), "fold_ids": fold_ids,
        "feature_names": feature_names,
        "fold_shares": np.round(fold_shares_arr, 6).tolist(),
        "share": np.round(share, 6).tolist(),
        "rank": rank.tolist(),
        "direction": [None if d != d else round(float(d), 4) for d in direction],
        "stability_tau": stability_tau, "top10_overlap": top10_overlap,
        "dual_method_tau": dual_tau, "n_interventional_folds": len(interventional),
        "permutation_tau": float(np.nanmean(permutation)) if permutation else float("nan"),
        "shuffle_pr_auc": float(np.mean(shuffle_pr)) if shuffle_pr else float("nan"),
        "chance_pr_auc": float(np.mean(chance_level)) if chance_level else float("nan"),
        "shuffle_tau_vs_real": float(np.nanmean(shuffle_tau)) if shuffle_tau else float("nan"),
        "shuffle_stability_tau": mean_pairwise_tau(shuffle_vectors) if len(shuffle_vectors) > 1 else float("nan"),
        "size_tau": size_tau, "size_feature_rank": int(rank[size_index]) if size_index is not None else None,
        "size_features_removed": [feature_names[j] for j in size_columns],
        "size_confound": size_confound, "rank_without_size": rank_without_size,
        "faithfulness": faithfulness, "interactions": interactions,
        "threshold_rows": threshold_rows, "dependence": dependence,
        "sensitivity": sensitivity, "notes": notes,
        "seconds": round(time.time() - start, 1),
    }


# ============================================================================
# Figures (written by the worker; at most two per pair)
# ============================================================================

def make_figures(pass_name, target, feature_names, X, shap_values, cost_model, dependence) -> str:
    """
    1. A SHAP summary (beeswarm) plot of the held-out samples.
    2. For each cost-model feature: its SHAP value against its value, with
       LLVM's default threshold marked.
    Returns a note if a figure could not be made (the tables do not depend on it).
    """
    from scripts.plot_style import INK_MUTED, NEUTRAL_MARK, SERIES_COLORS, save_figure, use_style
    problems = []
    plt = use_style()
    rng = np.random.RandomState(0)
    sub = rng.choice(len(X), size=min(1500, len(X)), replace=False)

    try:
        import shap
        plt.figure()
        shap.summary_plot(shap_values[sub], X[sub], feature_names=feature_names,
                          max_display=12, show=False, plot_size=(7, 4.5))
        plt.title(f"{pass_name} / {target}: SHAP values on held-out samples")
        save_figure(plt.gcf(), FIGURES_DIR / "shap" / f"beeswarm_{pass_name}_{target}.png")
    except Exception as error:
        plt.close("all")
        problems.append(f"beeswarm figure failed: {type(error).__name__}")

    for item in cost_model:
        if item.get("direction") != "threshold":
            continue
        name = item["name"]
        j = feature_names.index(name)
        mask = X[:, j] > 0 if name in LOOP_SIZE_FEATURES else np.ones(len(X), dtype=bool)
        bins = [d for d in dependence if d["feature"] == name]
        if mask.sum() < 30 or not bins:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.scatter(X[mask, j], shap_values[mask, j], s=9, color=NEUTRAL_MARK, alpha=0.35,
                   linewidths=0, label="one held-out sample")
        ax.plot([b["mean_value"] for b in bins], [b["mean_shap"] for b in bins],
                color=SERIES_COLORS[0], marker="o", label="mean per value bin")
        ax.axhline(0, color=INK_MUTED, linewidth=0.8)
        ax.axvline(float(item["threshold"]), color=INK_MUTED, linewidth=1.2, linestyle="--")
        ax.annotate(f"LLVM default {item.get('option', '')} = {item['threshold']}",
                    xy=(float(item["threshold"]), 1.0), xycoords=("data", "axes fraction"),
                    xytext=(4, -12), textcoords="offset points", fontsize=9, color=INK_MUTED)
        if X[mask, j].max() / max(X[mask, j].min(), 1e-9) > 50:
            ax.set_xscale("log")
        ax.set_xlabel(name)
        ax.set_ylabel("SHAP value (log-odds)")
        ax.set_title(f"{pass_name} / {target}: effect of {name}")
        ax.legend(loc="lower left")
        save_figure(fig, FIGURES_DIR / "threshold" / f"dependence_{pass_name}_{target}_{name}.png")
    return "; ".join(problems)


# ============================================================================
# Feature clusters
# ============================================================================

def feature_clusters(features: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
    """
    Group features that carry nearly the same information (Spearman |rho|
    above 0.8 on average).  SHAP splits credit between such features, so
    heuristic agreement is scored on clusters: a hit on any member counts once.
    Returns one cluster number per feature.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    rho = features[feature_names].corr(method="spearman").abs().to_numpy()
    distance = 1.0 - rho
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2
    tree = linkage(squareform(distance, checks=False), method="average")
    return fcluster(tree, t=0.2, criterion="distance")


# ============================================================================
# Heuristic agreement (parent process)
# ============================================================================

def agreement_scores(expected_ranks: np.ndarray, n_clusters: int) -> Dict[str, float]:
    """
    expected_ranks: the ranks (1 = most important) of the clusters that
    heuristics.yaml expects to matter.
      hit_at_5        share of expected clusters ranked in the top 5
      mrr             mean of 1 / rank over the expected clusters
      mean_percentile mean of (rank - 1) / (n_clusters - 1); 0 is best
    """
    ranks = np.asarray(expected_ranks, dtype=float)
    return {
        "hit_at_5": float(np.mean(ranks <= TOP_N)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_percentile": float(np.mean((ranks - 1) / max(n_clusters - 1, 1))),
    }


def null_ranks(n_clusters: int, n_expected: int, n_permutations: int,
               rng: np.random.RandomState) -> np.ndarray:
    """
    Ranks the expected clusters would get under RANDOM rankings: for each
    permutation, n_expected distinct ranks drawn from 1..n_clusters.
    Vectorized: one random matrix, no Python loop.
    """
    return rng.random((n_permutations, n_clusters)).argsort(axis=1)[:, :n_expected] + 1


def heuristic_agreement(explained: Dict[tuple, dict], interpreted: set, heuristics: dict,
                        cluster_of: np.ndarray, feature_names: List[str], cfg: Settings):
    """Sections C of the module docstring.  Returns a dict of DataFrames."""
    rng = np.random.RandomState(cfg.base_seed)
    cluster_ids = np.unique(cluster_of)
    n_clusters = len(cluster_ids)
    # membership[c, f] = 1 if feature f belongs to cluster c
    membership = (cluster_of[None, :] == cluster_ids[:, None]).astype(float)
    cluster_index = {c: i for i, c in enumerate(cluster_ids)}
    feature_index = {name: j for j, name in enumerate(feature_names)}

    def expected_clusters(pass_name: str, kind: str) -> List[int]:
        items = heuristics.get(pass_name, {}).get(kind, [])
        return sorted({cluster_index[cluster_of[feature_index[i["name"]]]] for i in items})

    agreement_rows, direction_rows, cross_rows, novel_rows = [], [], [], []
    pooled_inputs = []      # (target, kind, n_expected, observed percentile sum)

    for (pass_name, target), result in explained.items():
        is_interpreted = (pass_name, target) in interpreted
        share = np.asarray(result["share"])
        cluster_share = membership @ share
        cluster_rank = ranks_from_scores(cluster_share)
        fold_cluster_share = np.asarray(result["fold_shares"]) @ membership.T

        for kind in ("applicability", "cost_model"):
            expected = expected_clusters(pass_name, kind)
            row = {"pass_name": pass_name, "target": target, "list": kind,
                   "interpreted": is_interpreted, "n_expected_features":
                   len(heuristics.get(pass_name, {}).get(kind, [])),
                   "n_expected_clusters": len(expected), "n_clusters": n_clusters}
            if not expected:
                row["note"] = "heuristics.yaml lists no feature here"
                agreement_rows.append(row)
                continue
            observed = agreement_scores(cluster_rank[expected], n_clusters)
            null = null_ranks(n_clusters, len(expected), cfg.n_permutations, rng)
            null_hit = (null <= TOP_N).mean(axis=1)
            null_mrr = (1.0 / null).mean(axis=1)
            # Bootstrap over folds: resample the fold rankings, rebuild the ranking
            n_boot = min(cfg.n_bootstrap, 1000)
            picks = rng.randint(len(fold_cluster_share), size=(n_boot, len(fold_cluster_share)))
            boot_share = fold_cluster_share[picks].mean(axis=1)
            boot_rank = (-boot_share).argsort(axis=1).argsort(axis=1)[:, expected] + 1
            hit_ci = percentile_ci((boot_rank <= TOP_N).mean(axis=1), cfg.alpha)
            mrr_ci = percentile_ci((1.0 / boot_rank).mean(axis=1), cfg.alpha)
            row.update({
                "expected_ranks": " ".join(str(int(r)) for r in cluster_rank[expected]),
                "hit_at_5": round(observed["hit_at_5"], 4),
                "hit_at_5_ci_low": round(hit_ci[0], 4), "hit_at_5_ci_high": round(hit_ci[1], 4),
                "hit_at_5_null_mean": round(float(null_hit.mean()), 4),
                "hit_at_5_p": float((1 + (null_hit >= observed["hit_at_5"]).sum()) / (1 + len(null))),
                "mrr": round(observed["mrr"], 4),
                "mrr_ci_low": round(mrr_ci[0], 4), "mrr_ci_high": round(mrr_ci[1], 4),
                "mrr_null_mean": round(float(null_mrr.mean()), 4),
                "mrr_p": float((1 + (null_mrr >= observed["mrr"]).sum()) / (1 + len(null))),
                "mean_percentile": round(observed["mean_percentile"], 4),
                "note": "",
            })
            agreement_rows.append(row)
            if is_interpreted:
                pooled_inputs.append((target, kind, cluster_rank[expected]))

            # Cross-pass control: does this ranking fit ITS OWN list better
            # than the lists written for other passes?
            if kind == "applicability":
                others = []
                for other_pass in heuristics:
                    other_expected = expected_clusters(other_pass, kind)
                    if other_pass != pass_name and other_expected and other_expected != expected:
                        others.append(agreement_scores(cluster_rank[other_expected], n_clusters)["mrr"])
                if others:
                    cross_rows.append({
                        "pass_name": pass_name, "target": target, "interpreted": is_interpreted,
                        "own_mrr": round(observed["mrr"], 4),
                        "mean_mrr_other_lists": round(float(np.mean(others)), 4),
                        "n_other_lists": len(others),
                        "own_list_fits_better": bool(observed["mrr"] > np.mean(others)),
                    })

        # Direction agreement for the applicability entries
        for item in heuristics.get(pass_name, {}).get("applicability", []):
            wanted = expected_direction(item, target)
            rho = result["direction"][feature_index[item["name"]]]
            if wanted is None:
                verdict = "no direction given"
            elif rho is None or abs(rho) < 0.05:
                verdict = "undetermined"
            else:
                verdict = "agree" if (rho > 0) == (wanted == "high") else "disagree"
            direction_rows.append({
                "pass_name": pass_name, "target": target, "interpreted": is_interpreted,
                "feature": item["name"], "expected": wanted, "shap_vs_value_rho": rho,
                "verdict": verdict, "yaml_verified": bool(item.get("verified", False))})

        # Candidate novel drivers: top clusters that heuristics.yaml does not expect
        known = set(expected_clusters(pass_name, "applicability")) | set(
            expected_clusters(pass_name, "cost_model"))
        for position in np.argsort(cluster_rank)[:TOP_N]:
            if position in known:
                continue
            members = [feature_names[j] for j in np.where(membership[position] == 1)[0]]
            top_member = max(members, key=lambda name: share[feature_index[name]])
            novel_rows.append({
                "pass_name": pass_name, "target": target, "interpreted": is_interpreted,
                "cluster_rank": int(cluster_rank[position]),
                "cluster_share": round(float(cluster_share[position]), 4),
                "features": " + ".join(members),
                "contains_size_feature": SIZE_FEATURE in members,
                "top_member_rank_without_size_features":
                    result["rank_without_size"].get(top_member),
            })

    agreement = pd.DataFrame(agreement_rows)
    # Holm correction across INTERPRETED pairs, per list and per metric
    for metric in ("hit_at_5_p", "mrr_p"):
        if metric in agreement:
            agreement[metric + "_holm"] = np.nan
            for kind, group in agreement[agreement["interpreted"]
                                         & agreement[metric].notna()].groupby("list"):
                agreement.loc[group.index, metric + "_holm"] = holm_correction(group[metric].tolist())

    # Pooled test: over all interpreted pairs, are the expected clusters
    # ranked higher than chance?  Rankings are permuted within each pair.
    pooled_rows = []
    for kind in ("applicability", "cost_model"):
        for target_filter in ("beneficial", "harmful", "fired", "all"):
            chosen = [ranks for target, k, ranks in pooled_inputs
                      if k == kind and target_filter in (target, "all")]
            if not chosen:
                continue
            total = sum(len(r) for r in chosen)
            observed = sum(((r - 1) / max(n_clusters - 1, 1)).sum() for r in chosen) / total
            null_sum = np.zeros(cfg.n_permutations)
            for ranks in chosen:
                null = null_ranks(n_clusters, len(ranks), cfg.n_permutations, rng)
                null_sum += ((null - 1) / max(n_clusters - 1, 1)).sum(axis=1)
            null_mean = null_sum / total
            pooled_rows.append({
                "list": kind, "targets": target_filter, "n_pairs": len(chosen),
                "n_expected_clusters": total,
                "observed_mean_percentile": round(float(observed), 4),
                "null_mean_percentile": round(float(null_mean.mean()), 4),
                "p_value": float((1 + (null_mean <= observed).sum()) / (1 + cfg.n_permutations)),
            })

    return {
        "agreement": agreement, "direction": pd.DataFrame(direction_rows),
        "cross_pass": pd.DataFrame(cross_rows), "novel": pd.DataFrame(novel_rows),
        "pooled": pd.DataFrame(pooled_rows),
    }


# ============================================================================
# Main
# ============================================================================

def run_phase5(smoke: bool = False, n_jobs: int = 2, only: Optional[str] = None,
               force: bool = False):
    """Execute Phase 5."""
    ensure_dirs()
    cfg = get_settings(smoke)
    guard = DiskGuard()
    start = time.time()

    master = read_master_table()
    tuned_file = load_tuned_params()
    if tuned_file["settings"] != cfg.name:
        raise RuntimeError(f"Phase 4 was run with '{tuned_file['settings']}' settings, "
                           f"but this run uses '{cfg.name}'. Run Phase 4 again first.")
    feature_names = tuned_file["feature_names"]
    heuristics = load_heuristics(feature_names)
    passes = select_passes(smoke, only)

    candidates = master[(master["model_status"] == "beats majority")
                        & master["pass_name"].isin(passes)]
    pairs = list(zip(candidates["pass_name"], candidates["target"]))
    log.info("Settings: %s | pairs that beat the majority baseline: %d", cfg.name, len(pairs))

    # ---- Explain each pair (resume cache: one small JSON file per pair) ----
    cache = cache_dir("phase5", cfg)
    explained, todo = {}, []
    for pass_name, target in pairs:
        path = cache / f"{pass_name}__{target}.json"
        if path.exists() and not force:
            explained[(pass_name, target)] = json.loads(path.read_text(encoding="utf-8"))
        else:
            todo.append((pass_name, target))
    log.info("Pairs: %d cached, %d to run.", len(explained), len(todo))

    if todo:
        jobs = safe_n_jobs(n_jobs)
        if jobs == 1:
            finished = (explain_pair(p, t, cfg) for p, t in todo)
        else:
            from joblib import Parallel, delayed
            finished = Parallel(n_jobs=jobs, return_as="generator_unordered", max_nbytes=None)(
                delayed(explain_pair)(p, t, cfg) for p, t in todo)
        run_start = time.time()
        for done_count, result in enumerate(finished, start=1):
            key = (result["pass_name"], result["target"])
            explained[key] = result
            write_json_small(cache / f"{key[0]}__{key[1]}.json", result, max_mb=1.0)
            guard.check()
            elapsed = time.time() - run_start
            log.info("[%d/%d] %s %s done in %.0fs | elapsed %.0fs | remaining ~%.0fs",
                     done_count, len(todo), key[0], key[1], result.get("seconds", 0),
                     elapsed, elapsed / done_count * (len(todo) - done_count))

    failed = {key: r["failed"] for key, r in explained.items() if r.get("failed")}
    explained = {key: explained[key] for key in pairs if key in explained and not explained[key].get("failed")}

    # ---- Interpretation gate: stable SHAP ranking ----
    interpreted = {key for key, r in explained.items()
                   if r["stability_tau"] == r["stability_tau"]
                   and r["stability_tau"] >= cfg.stability_tau_min}

    # ---- Per-pair tables ----
    features, _ = load_dataset()
    cluster_of = feature_clusters(features, feature_names)
    cluster_table = pd.DataFrame({"feature": feature_names, "cluster": cluster_of})
    cluster_table["cluster_size"] = cluster_table.groupby("cluster")["feature"].transform("size")
    cluster_table.sort_values(["cluster", "feature"]).to_csv(
        RESULTS_DIR / "feature_clusters.csv", index=False)

    ranking_rows, check_rows, size_rows, faith_rows = [], [], [], []
    interaction_rows, threshold_rows, dependence_rows, sensitivity_rows = [], [], [], []
    for (pass_name, target), r in explained.items():
        base = {"pass_name": pass_name, "target": target,
                "interpreted": (pass_name, target) in interpreted}
        for j, name in enumerate(feature_names):
            ranking_rows.append({**base, "feature": name, "cluster": int(cluster_of[j]),
                                 "importance_share": r["share"][j], "rank": r["rank"][j],
                                 "shap_vs_value_rho": r["direction"][j]})
        check_rows.append({
            **base, "n_folds": r["n_folds"],
            "stability_tau": round(r["stability_tau"], 4),
            "top10_overlap": round(r["top10_overlap"], 4),
            "dual_method_tau": r["dual_method_tau"],
            "n_interventional_folds": r["n_interventional_folds"],
            "permutation_importance_tau": r["permutation_tau"],
            "shuffled_labels_pr_auc": r["shuffle_pr_auc"],
            "chance_pr_auc": r["chance_pr_auc"],
            "shuffled_vs_real_ranking_tau": r["shuffle_tau_vs_real"],
            "shuffled_stability_tau": r["shuffle_stability_tau"],
            "size_feature_rank": r["size_feature_rank"],
            "size_features_removed": " + ".join(r["size_features_removed"]),
            "ranking_tau_without_size_features": r["size_tau"],
            "notes": " | ".join(r["notes"]),
        })
        size_rows += [{**base, **row} for row in r["size_confound"]]
        faith_rows += [{**base, **row} for row in r["faithfulness"]]
        interaction_rows += [{**base, **row} for row in r["interactions"]]
        threshold_rows += [{**base, **row} for row in r["threshold_rows"]]
        dependence_rows += [{**base, **row} for row in r["dependence"]]
        sensitivity_rows += [{**base, **row} for row in r["sensitivity"]]

    outputs = {
        "shap_rankings.csv": ranking_rows, "shap_checks.csv": check_rows,
        "size_confound.csv": size_rows, "faithfulness.csv": faith_rows,
        "interaction_top10.csv": interaction_rows, "threshold_analysis.csv": threshold_rows,
        "threshold_dependence.csv": dependence_rows,
        "threshold_sensitivity.csv": sensitivity_rows,
    }
    for file_name, rows in outputs.items():
        pd.DataFrame(rows).to_csv(RESULTS_DIR / file_name, index=False)

    # ---- Heuristic agreement ----
    tables = heuristic_agreement(explained, interpreted, heuristics, cluster_of, feature_names, cfg)
    tables["agreement"].to_csv(RESULTS_DIR / "heuristic_agreement.csv", index=False)
    tables["direction"].to_csv(RESULTS_DIR / "heuristic_direction.csv", index=False)
    tables["cross_pass"].to_csv(RESULTS_DIR / "heuristic_cross_pass.csv", index=False)
    tables["pooled"].to_csv(RESULTS_DIR / "pooled_heuristic_agreement.csv", index=False)
    tables["novel"].to_csv(RESULTS_DIR / "novel_drivers.csv", index=False)

    # ---- Master table: final interpretation status for every pass x target ----
    def text(value) -> str:
        """Empty cells come back from the CSV as NaN; turn them into ''."""
        return "" if value is None or value != value else str(value)

    statuses, reasons, taus = [], [], []
    for row in master.itertuples():
        key = (row.pass_name, row.target)
        tau = explained[key]["stability_tau"] if key in explained else None
        taus.append(None if tau is None else round(tau, 4))
        if row.pass_name not in passes:
            # not part of this run (--only): keep what an earlier run wrote
            statuses.append(text(getattr(row, "interpretation_status", "")))
            reasons.append(text(getattr(row, "interpretation_reason", "")))
        elif row.keep_status != "kept":
            statuses.append("dropped"); reasons.append(text(row.keep_reason))
        elif row.model_status != "beats majority":
            statuses.append("not interpreted"); reasons.append(text(row.model_reason))
        elif key in failed:
            statuses.append("not interpreted"); reasons.append(failed[key])
        elif key in interpreted:
            statuses.append("interpreted"); reasons.append("")
        else:
            statuses.append("not interpreted")
            reasons.append(f"SHAP rank stability {tau:.2f} is below {cfg.stability_tau_min}")
    master["shap_stability_tau"] = taus
    master["interpretation_status"] = statuses
    master["interpretation_reason"] = reasons
    master.to_csv(MASTER_TABLE, index=False)

    # ---- Gate report ----
    agreement = tables["agreement"]
    lines = [
        "PHASE 5 GATE REPORT",
        "=" * 70,
        f"Settings:                              {cfg.name}"
        + ("   (SMOKE: these numbers are not results)" if smoke else ""),
        f"Pairs that beat the majority baseline: {len(pairs)}",
        f"Pairs explained:                       {len(explained)}",
        f"Pairs INTERPRETED (stability tau >= {cfg.stability_tau_min}): {len(interpreted)}",
        f"Feature clusters (|rho| > 0.8):        {len(np.unique(cluster_of))} from "
        f"{len(feature_names)} features",
        f"Run time this session:                 {time.time() - start:.0f} s",
        "",
        pd.DataFrame(check_rows)[["pass_name", "target", "interpreted", "stability_tau",
                                  "top10_overlap", "dual_method_tau",
                                  "shuffled_labels_pr_auc", "chance_pr_auc"]]
        .to_string(index=False) if check_rows else "(no pair was explained)",
        "",
        "Pooled heuristic agreement (interpreted pairs only):",
        tables["pooled"].to_string(index=False) if len(tables["pooled"]) else "(none)",
    ]
    if len(agreement) and "hit_at_5" in agreement:
        lines += ["", "Heuristic agreement per pair:",
                  agreement[["pass_name", "target", "list", "interpreted", "n_expected_clusters",
                             "hit_at_5", "mrr", "mrr_p"]].to_string(index=False)]
    report = "\n".join(lines) + "\n"
    (RESULTS_DIR / "phase5_gate.txt").write_text(report, encoding="utf-8")
    log.info("\n%s", report)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 5: SHAP and heuristic agreement")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_phase5(smoke=args.smoke, n_jobs=args.jobs, only=args.only, force=args.force)


if __name__ == "__main__":
    main()
