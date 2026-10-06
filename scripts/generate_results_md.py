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
from scripts.utils import RESULTS_DIR, FIGURES_DIR, setup_logging

log = setup_logging("results_md")


def generate():
    """Generate RESULTS.md from CSV files in results/."""
    import pandas as pd

    sections = []

    # ---- Header ----
    sections.append("""# Results

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
    dist_path = RESULTS_DIR.parent / "data" / "labels" / "label_distribution.csv"
    if dist_path.exists():
        df_dist = pd.read_csv(dist_path)
        sections.append("## Phase 3 — Label Distribution\n")
        sections.append(df_dist.to_markdown(index=False))
        sections.append("\n")

        dropped = df_dist[df_dist["dropped"] == True]["pass_name"].tolist()
        if dropped:
            sections.append(f"\n**Dropped passes** (positive rate outside [5%, 95%]): "
                            f"{', '.join(dropped)}\n")

    gate3 = RESULTS_DIR / "phase3_gate.txt"
    if gate3.exists():
        sections.append(f"\n```\n{gate3.read_text(encoding='utf-8')}\n```\n")

    # ---- Phase 4: Model results ----
    summary_path = RESULTS_DIR / "phase4_summary.csv"
    if summary_path.exists():
        df_summary = pd.read_csv(summary_path)
        sections.append("## Phase 4 — Model Performance\n")
        sections.append("### Per-Pass Summary (Mean ± Std over folds)\n")
        sections.append(df_summary.to_markdown(index=False))
        sections.append("\n")

    cross_path = RESULTS_DIR / "phase4_cross_suite.csv"
    if cross_path.exists():
        df_cross = pd.read_csv(cross_path)
        sections.append("### Cross-Suite Evaluation\n")
        sections.append(df_cross.to_markdown(index=False))
        sections.append("\n")

    gate4 = RESULTS_DIR / "phase4_gate.txt"
    if gate4.exists():
        sections.append(f"\n```\n{gate4.read_text(encoding='utf-8')}\n```\n")

    # ---- Phase 5: SHAP results ----
    rankings_path = RESULTS_DIR / "shap_rankings.csv"
    if rankings_path.exists():
        df_rank = pd.read_csv(rankings_path)
        sections.append("## Phase 5 — Explainability (SHAP)\n")
        sections.append("### Top 10 Features per Pass\n")

        for pass_name in df_rank["pass_name"].unique():
            pass_top = df_rank[df_rank["pass_name"] == pass_name].head(10)
            sections.append(f"\n#### {pass_name}\n")
            sections.append(pass_top[["rank", "feature", "mean_abs_shap"]].to_markdown(index=False))
            sections.append("\n")

            # Embed beeswarm plot if it exists
            beeswarm = FIGURES_DIR / "shap" / f"shap_beeswarm_{pass_name}.png"
            if beeswarm.exists():
                sections.append(f"\n![SHAP Beeswarm — {pass_name}]({beeswarm})\n")

    # Faithfulness
    faith_path = RESULTS_DIR / "faithfulness.csv"
    if faith_path.exists():
        df_faith = pd.read_csv(faith_path)
        sections.append("### Faithfulness Checks\n")
        sections.append(df_faith.to_markdown(index=False))
        sections.append("\n")

    # Heuristic agreement
    agree_path = RESULTS_DIR / "heuristic_agreement.csv"
    if agree_path.exists():
        df_agree = pd.read_csv(agree_path)
        sections.append("### Heuristic Agreement\n")
        sections.append(df_agree.to_markdown(index=False))
        sections.append("\n")

    # Novel drivers
    novel_path = RESULTS_DIR / "novel_drivers.csv"
    if novel_path.exists():
        df_novel = pd.read_csv(novel_path)
        sections.append("### Candidate Novel Drivers\n")
        sections.append(df_novel.to_markdown(index=False))
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
