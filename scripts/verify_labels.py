#!/usr/bin/env python3
"""
verify_labels.py — Independent checks on the Phase 3 output.

1. Structure: every state has exactly one row per pass, state hashes are
   unique, and no hash or function appears in two programs.
2. Arithmetic: rel_reduction, outcome, beneficial and harmful are recomputed
   from inst_before / inst_after and must match the stored values.
3. Parser on TRANSFORMED IR: for a random sample of (state, pass) rows the
   state is rebuilt, the pass is applied, and the instruction and block
   counts of the OUTPUT are compared with LLVM's own analysis
   (`print<func-properties>`).  Phase 1 already checks this on baseline IR;
   this checks that pass output (new block names, new constructs) is parsed
   correctly too.

Usage:
  python -m scripts.verify_labels [--samples 300]
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import (
    PROJECT_ROOT, DATA_DIR, FEATURES_DIR, LABELS_DIR, RESULTS_DIR,
    find_llvm_tools, require_pinned_llvm, run_tool,
    count_instructions, hash_ir, read_ir_file, PASS_LIST,
)
from scripts.phase2_features import reachable_counts
from scripts.phase3_labels import (
    PASS_PIPELINES, BENEFICIAL_THRESHOLD, apply_pass_sequence, run_pipeline,
)


def llvm_counts_for_single_function(ir_text: str, tools: dict):
    """LLVM's (TotalInstructionCount, BasicBlockCount) for a one-function module."""
    cmd = [tools["opt"], "-passes=print<func-properties>", "-disable-output"]
    result = run_tool(cmd, timeout=30, check=True, input_data=ir_text)
    text = result.stderr + result.stdout
    insts = re.findall(r"TotalInstructionCount:\s*(\d+)", text)
    blocks = re.findall(r"BasicBlockCount:\s*(\d+)", text)
    if len(insts) != 1 or len(blocks) != 1:
        raise RuntimeError(f"Expected exactly one function, got {len(insts)}")
    return int(insts[0]), int(blocks[0])


def main():
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=300)
    args = parser.parse_args()

    tools = find_llvm_tools()
    require_pinned_llvm(tools)

    labels = pd.read_csv(LABELS_DIR / "labels.csv", keep_default_na=False, na_values=[""],
                         dtype={"prefix_passes": str})
    labels["prefix_passes"] = labels["prefix_passes"].fillna("")
    features = pd.read_csv(FEATURES_DIR / "features.csv")
    index = pd.read_csv(DATA_DIR / "function_index.csv")
    index_by_id = {r["func_id"]: r for r in index.to_dict("records")}

    lines = []
    problems = 0

    def check(name, ok, detail=""):
        nonlocal problems
        if not ok:
            problems += 1
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    # ---------------- 1. Structure ----------------
    rows_per_state = labels.groupby("ir_hash")["pass_name"].agg(["count", "nunique"])
    check("every state has one row for each of the 18 passes",
          bool((rows_per_state["count"] == len(PASS_LIST)).all()
               and (rows_per_state["nunique"] == len(PASS_LIST)).all()),
          f"{len(rows_per_state)} states")
    check("state hashes are unique in features.csv", bool(features["ir_hash"].is_unique))
    check("features.csv and labels.csv hold the same states",
          set(features["ir_hash"]) == set(labels["ir_hash"]))
    programs_per_hash = features.groupby("ir_hash")["program"].nunique()
    check("no state hash appears in two programs", bool((programs_per_hash == 1).all()))
    programs_per_func = features.groupby("func_id")["program"].nunique()
    check("no function appears in two programs", bool((programs_per_func == 1).all()))
    check("no missing feature values", not bool(features.isna().any().any()))
    feature_cols = [c for c in features.columns
                    if c not in ("func_id", "suite", "program", "function", "ir_hash", "state")]
    raw_counts = [c for c in feature_cols if c.endswith("_count") and c != "log_inst_count"]
    check("no raw count features", not raw_counts, f"{len(feature_cols)} features")
    constant_cols = [c for c in feature_cols if features[c].nunique() <= 1]
    lines.append(f"       features with a single value in the whole dataset: {constant_cols}")

    # ---------------- 2. Arithmetic ----------------
    ok_rows = labels[labels["outcome"] != "error"].copy()
    before = ok_rows["inst_before"].astype(int)
    after = ok_rows["inst_after"].astype(int)
    rel = (before - after) / before
    check("rel_reduction matches inst_before / inst_after",
          bool(((rel - ok_rows["rel_reduction"].astype(float)).abs() < 1e-5).all()))
    outcome = pd.Series("equal", index=ok_rows.index)
    outcome[after < before] = "decreased"
    outcome[after > before] = "increased"
    check("outcome matches the counts", bool((outcome == ok_rows["outcome"]).all()))
    check("beneficial == (rel_reduction >= 1%)",
          bool(((rel >= BENEFICIAL_THRESHOLD).astype(int) == ok_rows["beneficial"].astype(int)).all()))
    check("harmful == (outcome is increased)",
          bool(((after > before).astype(int) == ok_rows["harmful"].astype(int)).all()))
    # A pass that did not change the IR cannot have changed the count
    not_changed = ok_rows[ok_rows["changed_ir"].astype(int) == 0]
    check("rows with unchanged IR have an unchanged count",
          bool((not_changed["inst_before"].astype(int) == not_changed["inst_after"].astype(int)).all()),
          f"{len(not_changed)} rows")
    check("baseline inst_before equals the Phase 1 index",
          bool(all(index_by_id[r.func_id]["inst_count"] == int(r.inst_before)
                   for r in ok_rows[ok_rows["state"] == "baseline"]
                   .drop_duplicates("func_id").itertuples())))

    # ---------------- 3. Parser on transformed IR ----------------
    sample = ok_rows.sample(n=min(args.samples, len(ok_rows)), random_state=2026)
    n_checked = 0
    n_bad = 0
    n_changed = 0
    for row in sample.itertuples():
        func_info = index_by_id[row.func_id]
        ir_text = read_ir_file(PROJECT_ROOT / func_info["path"])
        prefix = row.prefix_passes.split(";") if row.prefix_passes else []
        if prefix:
            ir_text, reason = apply_pass_sequence(ir_text, prefix, tools)
            if ir_text is None:
                n_bad += 1
                continue
        output_ir, reason = run_pipeline(ir_text, PASS_PIPELINES[row.pass_name], tools)
        if output_ir is None:
            n_bad += 1
            continue
        n_checked += 1
        ours_total = count_instructions(output_ir)
        ours_reachable = reachable_counts(output_ir, row.function)
        llvm = llvm_counts_for_single_function(output_ir, tools)
        same_as_stored = (ours_total == int(row.inst_after)
                          and count_instructions(ir_text) == int(row.inst_before)
                          and hash_ir(ir_text, row.function) == row.ir_hash)
        if ours_reachable != llvm or not same_as_stored:
            n_bad += 1
            lines.append(f"       MISMATCH {row.func_id} state={row.state} pass={row.pass_name} "
                         f"ours={ours_reachable} llvm={llvm} stored_after={row.inst_after}")
        if int(row.changed_ir) == 1:
            n_changed += 1
    check("pass OUTPUT: our counts equal LLVM's counts, and equal the stored labels",
          n_bad == 0, f"{n_checked} random rows rebuilt from scratch, "
                      f"{n_changed} of them with IR changed by the pass")

    lines.append("")
    lines.append(f"RESULT: {'ALL CHECKS PASSED' if problems == 0 else f'{problems} CHECK(S) FAILED'}")
    report = "\n".join(lines) + "\n"
    print(report)
    (RESULTS_DIR / "phase3_verification.txt").write_text(report, encoding="utf-8")
    sys.exit(0 if problems == 0 else 1)


if __name__ == "__main__":
    main()
