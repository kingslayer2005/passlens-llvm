# Results

Phase 1-4 pipeline was run successfully in **full mode**, yielding the dataset metrics and baseline evaluation results.
Phase 5-8 outputs were attempted using --smoke mode, but due to severe execution time bottlenecks in size_confound_analysis during target computations, these phases were aborted and are marked as **NOT RUN**.

## Honest Assessment of Outputs

- **Phases 1-4**: Fully completed. See STATUS.md for compilation stats, dataset sizes (2953 samples, 1038 functions), and Phase 4 PR-AUC metrics for XGBoost vs Baseline.
- **Phases 5-8**: **NOT RUN**. Any pre-existing outputs in the repository for these phases are stale (derived from the buggy instruction counter logic prior to the fix) and should not be cited.

## Target Status and Pass Performance
- Leakage audits confirm **no leakage** between train/test folds (skipped_folds and leakage audits are clean).
- XGBoost statistically outperforms the majority baseline in classifying whether a pass will be eneficial or ired for multiple passes like instcombine, simplifycfg, arly-cse, gvn, jump-threading, etc. (See exact p-values and PR-AUC in esults/phase4_wilcoxon.csv).
- However, explainability insights (SHAP feature importance, interactions, faithfulness) and stretch goals (distillation) cannot be reported as Phase 5-8 were aborted due to timeout constraints on the full pipeline rerun.
