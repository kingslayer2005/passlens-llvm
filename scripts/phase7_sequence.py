#!/usr/bin/env python3
"""
phase7_sequence.py — Do the per-pass models add up to a useful pass sequence?

  1. Hold out 20 % of the benchmark programs (seeded sample, taken within
     each suite).  Their functions are never used for training here.
  2. Train one XGBoost model per pass on the OTHER programs, for the
     "beneficial" target, using the hyperparameters Phase 4 chose most often.
     Only passes whose model beat the majority baseline are used.
  3. For every held-out function build pass sequences in five ways and
     compare the final instruction count with the starting count:
        model     apply the pass with the highest predicted probability,
                  repeat (up to 8 steps, stop when nothing is above 0.5)
        oracle    try every pass, apply the one that shrinks the code most
                  (an upper bound for any greedy choice of single passes)
        random    random passes, same length as the model's sequence
        -Oz, -O2  LLVM's standard pipelines
  4. Report the geometric mean size ratio per method, with programs as the
     unit and a bootstrap confidence interval over programs.

The objective is code size (instruction count), not running time.

Output:
  results/phase7_sequence.csv   one row per held-out function and method
  results/phase7_program.csv    geometric mean per program and method
  results/phase7_summary.csv    overall geometric mean with CI
  results/phase7_holdout.txt    held-out programs and the leakage check
  results/figures/sequence/sequence_summary.png
  results/phase7_gate.txt

Usage:
  python -m scripts.phase7_sequence [--smoke] [--jobs N]
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.analysis_common import (
    DiskGuard, Settings, build_pair_data, get_settings, load_dataset, load_tuned_params,
    make_xgb, pair_key, percentile_ci, read_master_table, safe_n_jobs,
)
from scripts.phase2_features import extract_features_from_ir_text
from scripts.phase3_labels import PASS_PIPELINES, run_pipeline
from scripts.utils import (
    DATA_DIR, FIGURES_DIR, PASS_LIST, PROJECT_ROOT, RESULTS_DIR, count_instructions,
    ensure_dirs, find_llvm_tools, hash_ir, read_ir_file, require_pinned_llvm, setup_logging,
)

log = setup_logging("phase7")

METHODS = ["model", "oracle", "random", "Oz", "O2"]
METHOD_LABELS = {"model": "model-guided (ours)", "oracle": "oracle greedy",
                 "random": "random passes", "Oz": "LLVM -Oz", "O2": "LLVM -O2"}
STANDARD_PIPELINES = {"Oz": "default<Oz>", "O2": "default<O2>"}
PROBABILITY_THRESHOLD = 0.5


# ============================================================================
# Hold-out and training
# ============================================================================

def choose_holdout_programs(index: pd.DataFrame, cfg: Settings) -> List[str]:
    """A seeded sample of programs, drawn separately inside each suite."""
    rng = np.random.RandomState(cfg.base_seed)
    held_out = []
    for suite, group in index.groupby("suite", sort=True):
        programs = sorted(group["program"].unique())
        n_pick = max(1, int(round(cfg.sequence_holdout_fraction * len(programs))))
        held_out.extend(rng.choice(programs, size=n_pick, replace=False).tolist())
    return sorted(held_out)


def train_models(held_out: List[str], master: pd.DataFrame, cfg: Settings) -> Tuple[dict, list, list]:
    """
    One 'beneficial' model per usable pass, trained on programs that are NOT
    held out.  Returns (models, feature_names, audit_lines).
    """
    features, labels = load_dataset()
    tuned_file = load_tuned_params()
    feature_names = tuned_file["feature_names"]
    usable = master[(master["target"] == "beneficial")
                    & (master["model_status"] == "beats majority")]["pass_name"].tolist()
    models, audit = {}, []
    for pass_name in usable:
        data = build_pair_data(features, labels, pass_name, "beneficial", feature_names=feature_names)
        train = ~np.isin(data.groups, held_out)
        overlap = len(set(data.groups[train]) & set(held_out))
        audit.append(f"{'PASS' if overlap == 0 else 'FAIL'} {pass_name} beneficial "
                     f"n_train={int(train.sum())} train_programs={len(set(data.groups[train]))} "
                     f"program_overlap_with_holdout={overlap}")
        if overlap:
            raise RuntimeError(f"Held-out programs leaked into the training set of {pass_name}.")
        if len(np.unique(data.y[train])) < 2:
            audit.append(f"SKIP {pass_name}: training rows have a single class")
            continue
        params = tuned_file["pairs"][pair_key(pass_name, "beneficial")]["modal_params"]
        model = make_xgb(params, data.y[train], cfg.base_seed)
        model.fit(data.X[train], data.y[train])
        models[pass_name] = model
    return models, feature_names, audit


# ============================================================================
# Sequences for one function
# ============================================================================

def apply_pass_checked(ir_text: str, pass_name: str, tools: dict, func_name: str) -> Optional[str]:
    """Apply a pass; return the new IR, or None if opt failed or nothing changed."""
    output, _ = run_pipeline(ir_text, PASS_PIPELINES[pass_name], tools)
    if output is None or hash_ir(output, func_name) == hash_ir(ir_text, func_name):
        return None
    return output


def model_sequence(ir_text, func_name, models, feature_names, tools, cfg) -> Tuple[str, List[str]]:
    """Greedy sequence guided by the models' predicted probabilities."""
    sequence = []
    for _ in range(cfg.sequence_max_steps):
        feats = extract_features_from_ir_text(ir_text, tools, func_name=func_name)
        if not feats:
            break
        row = np.array([[feats[name] for name in feature_names]], dtype=np.float32)
        ranked = sorted(((float(model.predict_proba(row)[0, 1]), pass_name)
                         for pass_name, model in models.items()), reverse=True)
        applied = False
        for probability, pass_name in ranked:
            if probability < PROBABILITY_THRESHOLD:
                break
            output = apply_pass_checked(ir_text, pass_name, tools, func_name)
            if output is not None:           # the pass really changed the IR
                ir_text = output
                sequence.append(pass_name)
                applied = True
                break
        if not applied:
            break
    return ir_text, sequence


def oracle_sequence(ir_text, func_name, tools, cfg) -> Tuple[str, List[str]]:
    """Greedy sequence that looks at the true result of every pass first."""
    sequence = []
    for _ in range(cfg.sequence_max_steps):
        current = count_instructions(ir_text)
        best_count, best_ir, best_pass = current, None, None
        for pass_name in PASS_LIST:
            output, _ = run_pipeline(ir_text, PASS_PIPELINES[pass_name], tools)
            if output is None:
                continue
            size = count_instructions(output)
            if size < best_count:
                best_count, best_ir, best_pass = size, output, pass_name
        if best_ir is None:
            break
        ir_text = best_ir
        sequence.append(best_pass)
    return ir_text, sequence


def evaluate_function(func_info: dict, models: dict, feature_names: list, tools: dict,
                      cfg: Settings) -> List[dict]:
    """All five methods on one held-out function."""
    func_name = func_info["function"]
    start_ir = read_ir_file(PROJECT_ROOT / func_info["path"])
    start_count = count_instructions(start_ir)
    base = {"func_id": func_info["func_id"], "suite": func_info["suite"],
            "program": func_info["program"], "inst_start": start_count}
    rows = []

    def add(method: str, final_count: Optional[int], sequence: str, note: str = ""):
        rows.append({**base, "method": method, "inst_final": final_count,
                     "size_ratio": None if final_count is None else final_count / start_count,
                     "sequence": sequence, "note": note})

    final_ir, sequence = model_sequence(start_ir, func_name, models, feature_names, tools, cfg)
    add("model", count_instructions(final_ir), ";".join(sequence))
    model_length = len(sequence)

    final_ir, sequence = oracle_sequence(start_ir, func_name, tools, cfg)
    add("oracle", count_instructions(final_ir), ";".join(sequence))

    # Random sequences of the same length as the model's (average of several)
    rng = np.random.RandomState(int(func_info["ir_hash"][:8], 16) % (2 ** 31))
    ratios = []
    for _ in range(cfg.n_random_draws):
        ir_text = start_ir
        for pass_name in rng.choice(PASS_LIST, size=model_length):
            output, _ = run_pipeline(ir_text, PASS_PIPELINES[pass_name], tools)
            if output is not None:
                ir_text = output
        ratios.append(count_instructions(ir_text) / start_count)
    mean_ratio = float(np.mean(ratios)) if ratios else 1.0
    rows.append({**base, "method": "random", "inst_final": round(mean_ratio * start_count, 2),
                 "size_ratio": mean_ratio, "sequence": f"{model_length} random passes",
                 "note": f"mean of {cfg.n_random_draws} draws"})

    for method, pipeline in STANDARD_PIPELINES.items():
        output, reason = run_pipeline(start_ir, pipeline, tools)
        if output is None:
            add(method, None, pipeline, note=reason)      # recorded, not guessed
        else:
            add(method, count_instructions(output), pipeline)
    return rows


# ============================================================================
# Aggregation
# ============================================================================

def geometric_mean(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.exp(np.mean(np.log(values)))) if len(values) else float("nan")


def summarise(detail: pd.DataFrame, cfg: Settings) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Geometric mean size ratio per program, then over programs.
    A ratio of 0 cannot go into a geometric mean, so an (unusual) function
    reduced to nothing is counted as 1 instruction.
    """
    ok = detail.dropna(subset=["size_ratio"]).copy()
    ok["size_ratio"] = ok["size_ratio"].clip(lower=1.0 / ok["inst_start"])
    per_program = ok.groupby(["suite", "program", "method"], sort=False)["size_ratio"].agg(
        geo_mean_ratio=geometric_mean, n_functions="size").reset_index()

    wide = per_program.pivot(index="program", columns="method", values="geo_mean_ratio")
    programs = wide.index.to_numpy()
    rng = np.random.RandomState(cfg.base_seed)
    draws = rng.randint(len(programs), size=(cfg.n_bootstrap, len(programs)))
    rows = []
    for method in METHODS:
        if method not in wide:
            continue
        log_ratio = np.log(wide[method].to_numpy())
        boot = np.exp(np.nanmean(log_ratio[draws], axis=1))
        low, high = percentile_ci(boot, cfg.alpha)
        row = {"method": method, "n_programs": int(np.sum(~np.isnan(log_ratio))),
               "geo_mean_ratio": round(float(np.exp(np.nanmean(log_ratio))), 4),
               "ci_low": round(low, 4), "ci_high": round(high, 4),
               "model_vs_method_ratio": None, "model_vs_method_ci_low": None,
               "model_vs_method_ci_high": None}
        if method != "model" and "model" in wide:
            # Paired comparison on the same programs: below 1 = the model's
            # sequences end smaller than this method's
            difference = np.log(wide["model"].to_numpy()) - log_ratio
            boot = np.exp(np.nanmean(difference[draws], axis=1))
            low, high = percentile_ci(boot, cfg.alpha)
            row.update({"model_vs_method_ratio": round(float(np.exp(np.nanmean(difference))), 4),
                        "model_vs_method_ci_low": round(low, 4),
                        "model_vs_method_ci_high": round(high, 4)})
        rows.append(row)
    return per_program, pd.DataFrame(rows)


def make_figure(summary: pd.DataFrame):
    """Geometric mean size ratio per method with its confidence interval."""
    from scripts.plot_style import INK, INK_MUTED, SERIES_COLORS, save_figure, use_style
    plt = use_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    order = summary.sort_values("geo_mean_ratio", ascending=False).reset_index(drop=True)
    for position, row in order.iterrows():
        ax.plot([row["ci_low"], row["ci_high"]], [position, position],
                color=SERIES_COLORS[0], linewidth=2)
        ax.plot(row["geo_mean_ratio"], position, marker="o", color=SERIES_COLORS[0])
        ax.annotate(f"{row['geo_mean_ratio']:.2f}", xy=(row["geo_mean_ratio"], position),
                    xytext=(0, 9), textcoords="offset points", ha="center",
                    fontsize=9, color=INK)
    ax.axvline(1.0, color=INK_MUTED, linewidth=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in order["method"]])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("final size / starting size (geometric mean over programs, lower is smaller)")
    ax.set_title("Code size after each kind of pass sequence, held-out programs")
    ax.grid(axis="y", visible=False)
    save_figure(fig, FIGURES_DIR / "sequence" / "sequence_summary.png")


# ============================================================================
# Main
# ============================================================================

def run_phase7(smoke: bool = False, n_jobs: int = 2):
    """Execute Phase 7."""
    ensure_dirs()
    cfg = get_settings(smoke)
    guard = DiskGuard()
    start = time.time()
    tools = find_llvm_tools()
    require_pinned_llvm(tools)

    master = read_master_table()
    index = pd.read_csv(DATA_DIR / "function_index.csv")
    held_out = choose_holdout_programs(index, cfg)
    models, feature_names, audit = train_models(held_out, master, cfg)

    functions = index[index["program"].isin(held_out)]
    if len(functions) > cfg.sequence_max_functions:
        functions = functions.sample(n=cfg.sequence_max_functions, random_state=cfg.base_seed)
    functions = functions.sort_values("func_id").to_dict("records")

    (RESULTS_DIR / "phase7_holdout.txt").write_text(
        "# Held-out programs (never used to train the Phase 7 models)\n"
        + "\n".join(held_out) + "\n\n# Training-set check, one line per model\n"
        + "\n".join(audit) + "\n", encoding="utf-8")
    log.info("Held-out programs: %d | functions evaluated: %d | models: %s",
             len(held_out), len(functions), sorted(models))
    if not models:
        report = ("PHASE 7 GATE REPORT\n" + "=" * 70 + "\nNo 'beneficial' model beat the "
                  "majority baseline, so there is no model to guide a sequence. NOT RUN.\n")
        (RESULTS_DIR / "phase7_gate.txt").write_text(report, encoding="utf-8")
        log.warning(report)
        return

    # Threads, not processes: the time goes into `opt` subprocesses, and the
    # trained models are shared instead of being copied to every worker.
    from joblib import Parallel, delayed
    jobs = safe_n_jobs(n_jobs)
    results = Parallel(n_jobs=jobs, prefer="threads")(
        delayed(evaluate_function)(f, models, feature_names, tools, cfg) for f in functions)
    guard.check()

    detail = pd.DataFrame([row for rows in results for row in rows])
    detail.to_csv(RESULTS_DIR / "phase7_sequence.csv", index=False)
    per_program, summary = summarise(detail, cfg)
    per_program.round(4).to_csv(RESULTS_DIR / "phase7_program.csv", index=False)
    summary.to_csv(RESULTS_DIR / "phase7_summary.csv", index=False)
    if len(summary):
        make_figure(summary)

    failures = detail[detail["size_ratio"].isna()]
    lines = ["PHASE 7 GATE REPORT", "=" * 70,
             f"Settings:               {cfg.name}"
             + ("   (SMOKE: these numbers are not results)" if smoke else ""),
             f"Held-out programs:      {len(held_out)} of {index['program'].nunique()}",
             f"Functions evaluated:    {len(functions)}",
             f"Models used (passes):   {', '.join(sorted(models))}",
             f"Failed pipeline runs:   {len(failures)}",
             f"Run time this session:  {time.time() - start:.0f} s", "",
             "Final size / starting size (geometric mean over programs; 1.00 = unchanged):",
             summary.to_string(index=False)]
    report = "\n".join(lines) + "\n"
    (RESULTS_DIR / "phase7_gate.txt").write_text(report, encoding="utf-8")
    log.info("\n%s", report)


def main():
    parser = argparse.ArgumentParser(description="Phase 7: sequence-level evaluation")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    run_phase7(smoke=args.smoke, n_jobs=args.jobs)


if __name__ == "__main__":
    main()
