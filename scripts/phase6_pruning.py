#!/usr/bin/env python3
"""
phase6_pruning.py — Can the explanation be used to build a smaller model?

For every INTERPRETED (pass, target) pair, and for each outer fold of the
first repeat:

  1. Rank the features using the TRAINING fold only, in four ways:
       shap     mean |SHAP| of the full model on training rows
       gain     XGBoost's own split-gain importance
       mutual   mutual information between each feature and the label
       random   random order (several draws)
  2. Keep the top k features (k = 5, 10, 15, 20), refit with the same
     hyperparameters, and score on the test fold.

Selecting features inside the training fold matters: choosing them with the
test rows in view would make every pruned model look better than it is.

Output:
  results/phase6_pruning.csv          one row per pair, fold, method, k
  results/phase6_pruning_summary.csv  mean over folds, and share of the
                                      full model's PR-AUC that is retained
  results/figures/pruning/pruning_<pass>.png
  results/phase6_gate.txt

Usage:
  python -m scripts.phase6_pruning [--smoke] [--jobs N] [--only p1,p2] [--force]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.analysis_common import (
    DiskGuard, Settings, build_pair_data, cache_dir, get_settings, importance_share,
    load_dataset, load_tuned_params, make_xgb, pair_key, pr_auc, read_master_table,
    safe_n_jobs, select_passes, tree_shap, usable_folds, write_json_small,
)
from scripts.utils import FIGURES_DIR, RESULTS_DIR, ensure_dirs, setup_logging

log = setup_logging("phase6")

METHODS = ["shap", "gain", "mutual", "random"]
METHOD_LABELS = {"shap": "SHAP", "gain": "XGBoost gain",
                 "mutual": "mutual information", "random": "random"}


def prune_pair(pass_name: str, target: str, cfg: Settings) -> dict:
    """All pruning experiments of one (pass, target) pair (runs in a worker)."""
    from sklearn.feature_selection import mutual_info_classif

    start = time.time()
    features, labels = load_dataset()
    tuned_file = load_tuned_params()
    feature_names = tuned_file["feature_names"]
    tuned_pair = tuned_file["pairs"][pair_key(pass_name, target)]
    data = build_pair_data(features, labels, pass_name, target, feature_names=feature_names)
    X, y = data.X, data.y
    n_features = X.shape[1]
    rows = []

    def fit_and_measure(columns, X_train, y_train, X_test, y_test, params, seed) -> dict:
        """Fit on the chosen columns; return score, model size and prediction time."""
        model = make_xgb(params, y_train, seed)
        model.fit(X_train[:, columns], y_train)
        timings = []
        for _ in range(3):
            t0 = time.perf_counter()
            probability = model.predict_proba(X_test[:, columns])[:, 1]
            timings.append(time.perf_counter() - t0)
        return {
            "pr_auc": pr_auc(y_test, probability),
            "model_bytes": len(model.get_booster().save_raw("ubj")),
            "predict_us_per_sample": 1e6 * float(np.median(timings)) / len(y_test),
            "model": model,
        }

    for repeat, fold, train_idx, test_idx, params, seed in usable_folds(
            data, tuned_pair, cfg, repeats=[0]):
        X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
        rng = np.random.RandomState(seed)
        base = {"pass_name": pass_name, "target": target, "fold": fold}

        # Full model: the reference every pruned model is compared with
        full = fit_and_measure(list(range(n_features)), X_train, y_train, X_test, y_test, params, seed)
        rows.append({**base, "method": "full", "k": n_features, "draw": 0,
                     **{key: full[key] for key in ("pr_auc", "model_bytes", "predict_us_per_sample")}})

        # ---- Feature orders, from the training fold only ----
        train_sub = rng.choice(len(train_idx), size=min(500, len(train_idx)), replace=False)
        orders = {"shap": [np.argsort(-importance_share(tree_shap(full["model"], X_train[train_sub])))]}

        gain = np.zeros(n_features)
        for name, value in full["model"].get_booster().get_score(importance_type="gain").items():
            gain[int(name[1:])] = value          # XGBoost names the columns f0, f1, ...
        orders["gain"] = [np.argsort(-gain, kind="mergesort")]

        mutual = mutual_info_classif(X_train, y_train, random_state=seed)
        orders["mutual"] = [np.argsort(-mutual, kind="mergesort")]

        orders["random"] = [rng.permutation(n_features) for _ in range(cfg.n_random_draws)]

        # ---- Refit with the top k features of each order ----
        for k in cfg.pruning_k:
            if k >= n_features:
                continue
            for method in METHODS:
                for draw, order in enumerate(orders[method]):
                    result = fit_and_measure(sorted(order[:k].tolist()), X_train, y_train,
                                             X_test, y_test, params, seed)
                    rows.append({**base, "method": method, "k": k, "draw": draw,
                                 **{key: result[key] for key in
                                    ("pr_auc", "model_bytes", "predict_us_per_sample")}})

    for row in rows:
        row["pr_auc"] = round(float(row["pr_auc"]), 6)
        row["predict_us_per_sample"] = round(float(row["predict_us_per_sample"]), 3)
    return {"pass_name": pass_name, "target": target, "rows": rows,
            "seconds": round(time.time() - start, 1)}


def make_figure(pass_name: str, summary: pd.DataFrame):
    """One figure per pass: PR-AUC against the number of features kept."""
    from scripts.plot_style import INK_MUTED, SERIES_COLORS, SERIES_MARKERS, save_figure, use_style
    plt = use_style()
    targets = list(summary["target"].unique())
    fig, axes = plt.subplots(1, len(targets), figsize=(4.4 * len(targets), 4.2),
                             squeeze=False, constrained_layout=True)
    for ax, target in zip(axes[0], targets):
        part = summary[summary["target"] == target]
        full_score = part.loc[part["method"] == "full", "pr_auc_mean"].iloc[0]
        ax.axhline(full_score, color=INK_MUTED, linestyle="--", linewidth=1.2)
        ax.annotate("all features", xy=(1.0, full_score), xycoords=("axes fraction", "data"),
                    xytext=(-4, 4), textcoords="offset points", ha="right",
                    fontsize=9, color=INK_MUTED)
        for index, method in enumerate(METHODS):
            line = part[part["method"] == method].sort_values("k")
            ax.plot(line["k"], line["pr_auc_mean"], color=SERIES_COLORS[index],
                    marker=SERIES_MARKERS[index], label=METHOD_LABELS[method])
        ax.set_xlabel("features kept")
        ax.set_title(f"{pass_name} / {target}")
        ax.set_xticks(sorted(part.loc[part["method"] != "full", "k"].unique()))
    axes[0][0].set_ylabel("PR-AUC on held-out folds")
    # One legend for the whole figure, below the panels, so it never covers a line
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, title="features chosen by", loc="outside lower center",
               ncol=len(METHODS))
    save_figure(fig, FIGURES_DIR / "pruning" / f"pruning_{pass_name}.png")


def run_phase6(smoke: bool = False, n_jobs: int = 2, only: Optional[str] = None,
               force: bool = False):
    """Execute Phase 6."""
    ensure_dirs()
    cfg = get_settings(smoke)
    guard = DiskGuard()
    start = time.time()

    master = read_master_table()
    if "interpretation_status" not in master:
        raise RuntimeError("master_table.csv has no interpretation status. Run Phase 5 first.")
    passes = select_passes(smoke, only)
    chosen = master[(master["interpretation_status"] == "interpreted")
                    & master["pass_name"].isin(passes)]
    pairs = list(zip(chosen["pass_name"], chosen["target"]))
    log.info("Settings: %s | interpreted pairs to prune: %d", cfg.name, len(pairs))

    cache = cache_dir("phase6", cfg)
    done, todo = {}, []
    for pass_name, target in pairs:
        path = cache / f"{pass_name}__{target}.json"
        if path.exists() and not force:
            done[(pass_name, target)] = json.loads(path.read_text(encoding="utf-8"))
        else:
            todo.append((pass_name, target))

    if todo:
        jobs = safe_n_jobs(n_jobs)
        if jobs == 1:
            finished = (prune_pair(p, t, cfg) for p, t in todo)
        else:
            from joblib import Parallel, delayed
            finished = Parallel(n_jobs=jobs, return_as="generator_unordered", max_nbytes=None)(
                delayed(prune_pair)(p, t, cfg) for p, t in todo)
        run_start = time.time()
        for count, result in enumerate(finished, start=1):
            key = (result["pass_name"], result["target"])
            done[key] = result
            write_json_small(cache / f"{key[0]}__{key[1]}.json", result, max_mb=1.0)
            guard.check()
            elapsed = time.time() - run_start
            log.info("[%d/%d] %s %s done in %.0fs | elapsed %.0fs | remaining ~%.0fs",
                     count, len(todo), key[0], key[1], result["seconds"], elapsed,
                     elapsed / count * (len(todo) - count))

    rows = [row for key in pairs for row in done[key]["rows"]]
    detail = pd.DataFrame(rows, columns=["pass_name", "target", "fold", "method", "k", "draw",
                                         "pr_auc", "model_bytes", "predict_us_per_sample"])
    detail.to_csv(RESULTS_DIR / "phase6_pruning.csv", index=False)

    summary = pd.DataFrame()
    if len(detail):
        # Average the random draws within a fold first, then average over folds
        per_fold = detail.groupby(["pass_name", "target", "method", "k", "fold"], sort=False)[
            ["pr_auc", "model_bytes", "predict_us_per_sample"]].mean().reset_index()
        summary = per_fold.groupby(["pass_name", "target", "method", "k"], sort=False).agg(
            n_folds=("pr_auc", "size"), pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
            model_bytes=("model_bytes", "mean"),
            predict_us_per_sample=("predict_us_per_sample", "mean")).reset_index()
        full = summary[summary["method"] == "full"].set_index(["pass_name", "target"])["pr_auc_mean"]
        summary["share_of_full_pr_auc"] = [
            row.pr_auc_mean / full[(row.pass_name, row.target)] for row in summary.itertuples()]
        summary = summary.round({"pr_auc_mean": 4, "pr_auc_std": 4, "model_bytes": 0,
                                 "predict_us_per_sample": 3, "share_of_full_pr_auc": 4})
        for pass_name, part in summary.groupby("pass_name", sort=False):
            make_figure(pass_name, part)
    summary.to_csv(RESULTS_DIR / "phase6_pruning_summary.csv", index=False)

    lines = ["PHASE 6 GATE REPORT", "=" * 70,
             f"Settings:                 {cfg.name}"
             + ("   (SMOKE: these numbers are not results)" if smoke else ""),
             f"Interpreted pairs pruned: {len(pairs)}",
             f"Feature counts tried:     {list(cfg.pruning_k)}",
             f"Run time this session:    {time.time() - start:.0f} s", ""]
    if len(summary):
        wide = summary[summary["method"] != "full"].pivot_table(
            index=["pass_name", "target", "k"], columns="method", values="share_of_full_pr_auc")
        lines += ["Share of the full model's PR-AUC that is kept:", wide.to_string()]
    else:
        lines.append("No interpreted pair: nothing to prune.")
    report = "\n".join(lines) + "\n"
    (RESULTS_DIR / "phase6_gate.txt").write_text(report, encoding="utf-8")
    log.info("\n%s", report)


def main():
    parser = argparse.ArgumentParser(description="Phase 6: explanation-guided feature pruning")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_phase6(smoke=args.smoke, n_jobs=args.jobs, only=args.only, force=args.force)


if __name__ == "__main__":
    main()
