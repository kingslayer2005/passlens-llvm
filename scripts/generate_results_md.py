#!/usr/bin/env python3
"""
generate_results_md.py — Auto-generate RESULTS.md from CSV tables and figures.

Reads all CSV result files and generates a comprehensive markdown report.
NO hand-typed numbers — everything comes from the data files.

Output:
  results/RESULTS.md

Usage:
  python -m scripts.generate_results_md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import RESULTS_DIR, FIGURES_DIR, LABELS_DIR, setup_logging

log = setup_logging("results_md")


def generate():
    """Generate RESULTS.md from CSV files in results/."""
    import pandas as pd

    sections = []

    # ---- Header ----
    sections.append("""# PRELIMINARY: smoke run on 52 programs, not a research result

> **Auto-generated** — all numbers come from CSV files in `results/`.
> Do not edit manually; re-run `python -m scripts.generate_results_md` instead.

---
""")

    # ---- Phase 1: Data summary ----
    gate1 = RESULTS_DIR / "phase1_gate.txt"
    if gate1.exists():
        sections.append("## Phase 1 — Data Collection\n")
        sections.append(f"```\n{gate1.read_text(encoding='utf-8')}\n```\n")

    # ---- Phase 3: Label distribution ----
    dist_path = LABELS_DIR / "label_distribution.csv"
    if dist_path.exists():
        df_dist = pd.read_csv(dist_path)
        sections.append("## Phase 3 — Label Distribution\n")
        sections.append(df_dist.to_markdown(index=False))
        sections.append("\n")

    gate3 = RESULTS_DIR / "phase3_gate.txt"
    if gate3.exists():
        sections.append(f"\n```\n{gate3.read_text(encoding='utf-8')}\n```\n")

    # ---- Phase 4: Model results ----
    summary_path = RESULTS_DIR / "phase4_summary.csv"
    if summary_path.exists():
        df_summary = pd.read_csv(summary_path)
        sections.append("## Phase 4 — Model Performance\n")
        sections.append("### Per-Pass Summary (Mean ± 95% CI over programs)\n")
        sections.append(df_summary.to_markdown(index=False))
        sections.append("\n")

    wilcoxon_path = RESULTS_DIR / "phase4_wilcoxon.csv"
    if wilcoxon_path.exists():
        df_wilcoxon = pd.read_csv(wilcoxon_path)
        sections.append("### Wilcoxon Signed-Rank Test (Holm-corrected)\n")
        sections.append(df_wilcoxon.to_markdown(index=False))
        sections.append("\n")

    thresh_path = RESULTS_DIR / "threshold_sensitivity.csv"
    if thresh_path.exists():
        df_thresh = pd.read_csv(thresh_path)
        sections.append("### Threshold Sensitivity\n")
        sections.append(df_thresh.to_markdown(index=False))
        sections.append("\n")

    gate4 = RESULTS_DIR / "phase4_gate.txt"
    if gate4.exists():
        sections.append(f"\n```\n{gate4.read_text(encoding='utf-8')}\n```\n")

    # ---- Phase 5: SHAP results ----
    dual_path = RESULTS_DIR / "shap_dual_method.csv"
    if dual_path.exists():
        df_dual = pd.read_csv(dual_path)
        sections.append("## Phase 5 — Explainability (SHAP)\n")
        sections.append("### Rigor & Stability\n")
        sections.append(df_dual.to_markdown(index=False))
        sections.append("\n")

    rankings_path = RESULTS_DIR / "shap_rankings.csv"
    if rankings_path.exists():
        df_rank = pd.read_csv(rankings_path)
        sections.append("### Top 10 Features per Interpretable Pass\n")

        for key, group in df_rank.groupby(["pass_name", "target"]):
            pass_name, target = key
            pass_top = group.head(10)
            sections.append(f"\n#### {pass_name} (target: {target})\n")
            sections.append(pass_top[["rank", "feature", "mean_abs_shap"]].to_markdown(index=False))
            sections.append("\n")

            # Embed beeswarm plot if it exists
            beeswarm = FIGURES_DIR / "shap" / f"shap_beeswarm_{pass_name}_{target}.png"
            if beeswarm.exists():
                sections.append(f"\n![SHAP Beeswarm — {pass_name} ({target})]({beeswarm})\n")

    faith_path = RESULTS_DIR / "faithfulness.csv"
    if faith_path.exists():
        df_faith = pd.read_csv(faith_path)
        sections.append("### Faithfulness Checks\n")
        sections.append(df_faith.to_markdown(index=False))
        sections.append("\n")

    interact_path = RESULTS_DIR / "interaction_top10.csv"
    if interact_path.exists():
        df_interact = pd.read_csv(interact_path)
        sections.append("### Top 10 Feature Interactions\n")
        sections.append(df_interact.to_markdown(index=False))
        sections.append("\n")

    size_path = RESULTS_DIR / "size_confound.csv"
    if size_path.exists():
        df_size = pd.read_csv(size_path)
        sections.append("### Size Confound Analysis\n")
        sections.append(df_size.to_markdown(index=False))
        sections.append("\n")

    agree_path = RESULTS_DIR / "heuristic_agreement.csv"
    if agree_path.exists():
        df_agree = pd.read_csv(agree_path)
        sections.append("### Heuristic Agreement (Permutation Test)\n")
        sections.append(df_agree.to_markdown(index=False))
        sections.append("\n")

    novel_path = RESULTS_DIR / "novel_drivers.csv"
    if novel_path.exists():
        df_novel = pd.read_csv(novel_path)
        sections.append("### Candidate Novel Drivers\n")
        sections.append(df_novel.to_markdown(index=False))
        sections.append("\n")

    pooled_path = RESULTS_DIR / "pooled_heuristic_agreement.csv"
    if pooled_path.exists():
        df_pooled = pd.read_csv(pooled_path)
        sections.append("### Pooled Heuristic Agreement Test\n")
        sections.append(df_pooled.to_markdown(index=False))
        sections.append("\n")

    thresh_analysis = RESULTS_DIR / "threshold_analysis.csv"
    if thresh_analysis.exists():
        df_ta = pd.read_csv(thresh_analysis)
        sections.append("### Threshold Analysis\n")
        sections.append(df_ta.to_markdown(index=False))
        sections.append("\n")
        # Embed dependence plots
        shap_dir = FIGURES_DIR / "shap"
        if shap_dir.exists():
            for row in df_ta.itertuples():
                img = shap_dir / f"shap_dependence_{row.pass_name}_{row.target}_{row.feature}.png"
                if img.exists():
                    sections.append(f"\n![SHAP Dependence — {row.pass_name} ({row.target}) {row.feature}]({img})\n")

    # ---- Phase 5 Gating Summary ----
    if dist_path.exists() and dual_path.exists():
        df_d = pd.read_csv(dist_path)
        df_du = pd.read_csv(dual_path)
        gate_rows = []
        for _, r in df_d.iterrows():
            p = r["pass_name"]
            for t in ["beneficial", "harmful"]:
                pos_rate = r.get(f"{t}_rate", 0)
                if pos_rate < 0.05 or pos_rate > 0.95:
                    status = "dropped"
                    reason = "Pos rate out of bounds [5%, 95%]"
                else:
                    match = df_du[(df_du["pass_name"] == p) & (df_du["target"] == t)]
                    if match.empty:
                        status = "dropped"
                        reason = "Failed model training / XGate"
                    else:
                        m = match.iloc[0]
                        if m["interpretable"]:
                            status = "interpreted"
                            reason = "Stable & beats majority"
                        else:
                            status = "not interpreted"
                            reasons = []
                            if not m["stable"]: reasons.append("tau < 0.6")
                            if not m["beats_majority"]: reasons.append("fails wilcoxon")
                            reason = " and ".join(reasons)
                gate_rows.append({"pass_name": p, "target": t, "status": status, "reason": reason})
        
        df_gate = pd.DataFrame(gate_rows)
        sections.append("### Phase 5 Gating Summary\n")
        sections.append(df_gate.to_markdown(index=False))
        sections.append("\n")

    gate5 = RESULTS_DIR / "phase5_gate.txt"
    if gate5.exists():
        sections.append(f"\n```\n{gate5.read_text(encoding='utf-8')}\n```\n")

    # ---- Phase 6: Pruning ----
    pruning_path = RESULTS_DIR / "phase6_pruning_summary.csv"
    if pruning_path.exists():
        df_prune = pd.read_csv(pruning_path)
        sections.append("## Phase 6 — Feature Pruning\n")
        sections.append(df_prune.to_markdown(index=False))
        sections.append("\n")

        # Embed pruning plots
        pruning_dir = FIGURES_DIR / "pruning"
        if pruning_dir.exists():
            for img in sorted(pruning_dir.glob("*.png")):
                pass_name = img.stem.replace("pruning_", "")
                sections.append(f"\n![Pruning Curve — {pass_name}]({img})\n")

    # ---- Phase 7: Sequence ----
    seq_summary_path = RESULTS_DIR / "phase7_summary.csv"
    if seq_summary_path.exists():
        df_seq = pd.read_csv(seq_summary_path)
        sections.append("## Phase 7 — Sequence-Level Evaluation\n")
        sections.append(df_seq.to_markdown(index=False))
        sections.append("\n")

        # Embed comparison plot
        comp_plot = FIGURES_DIR / "sequence" / "sequence_comparison.png"
        if comp_plot.exists():
            sections.append(f"\n![Sequence Comparison]({comp_plot})\n")
        box_plot = FIGURES_DIR / "sequence" / "sequence_boxplot.png"
        if box_plot.exists():
            sections.append(f"\n![Reduction Distribution]({box_plot})\n")

    gate7 = RESULTS_DIR / "phase7_gate.txt"
    if gate7.exists():
        sections.append(f"\n```\n{gate7.read_text(encoding='utf-8')}\n```\n")

    # ---- Write RESULTS.md ----
    results_md = RESULTS_DIR / "RESULTS.md"
    results_md.write_text("\n".join(sections), encoding="utf-8")
    log.info("Generated %s", results_md)


def main():
    generate()


if __name__ == "__main__":
    main()
