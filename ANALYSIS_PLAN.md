# Analysis Plan (Pre-Specified)

> Written and committed **before** the full pipeline run.
> Every threshold, test, and decision criterion is listed here.
> No post-hoc modifications are permitted.

---

## 1. Labelling Thresholds

| Parameter | Value | Justification |
|---|---|---|
| Primary beneficial threshold | 1 % relative instruction reduction | Filters noise while keeping meaningful gains |
| Sensitivity thresholds | 0.5 %, 1 %, 2 % | Span half-decade around primary |
| Harmful threshold | outcome == "increased" (inst_after > inst_before) | Second target |
| Pass drop criterion | positive rate outside [5 %, 95 %] | Insufficient variance for learning |

## 2. Targets Per Pass

| Target | Label column | Positive class |
|---|---|---|
| `beneficial` | `rel_reduction >= threshold` | 1 |
| `harmful` | `outcome == "increased"` | 1 |

A target is **kept** only if its positive rate is in [5 %, 95 %].
Both targets use the same features and the same cross-validation scheme.

## 3. Cross-Validation

| Parameter | Value |
|---|---|
| Outer CV | Repeated StratifiedGroupKFold, 3 repeats x 5 folds |
| Grouping variable | `program` |
| Inner CV (nested tuning) | StratifiedGroupKFold, 3 folds |
| Tuning budget | At most 10 random configurations |
| Imbalance handling | `scale_pos_weight = n_neg / n_pos` |
| CI method | 95 % bootstrap CI resampling programs (10 000 bootstrap samples) |

## 4. Models

| Model | Role |
|---|---|
| Majority-class | Baseline |
| Logistic Regression (max_iter=1000) | Baseline |
| Random Forest (100 trees, max_depth=10) | Baseline |
| XGBoost (tuned via nested CV) | Primary |

### XGBoost Hyperparameter Search Space

| Hyperparameter | Distribution |
|---|---|
| `n_estimators` | {100, 200, 300} |
| `max_depth` | {4, 6, 8} |
| `learning_rate` | {0.05, 0.1, 0.2} |
| `subsample` | {0.7, 0.8, 0.9} |
| `colsample_bytree` | {0.7, 0.8, 0.9} |

Select the configuration with the best mean PR-AUC across 3 inner folds.

## 5. Statistical Comparison

| Test | Scope | Correction |
|---|---|---|
| Wilcoxon signed-rank | XGBoost vs each baseline per target | Holm-Bonferroni across passes |
| Significance level | alpha = 0.05 (two-sided) | After correction |

The test statistic is the outer-fold PR-AUC vector (length = 15, i.e. 3 x 5).

## 6. Leakage Audit

For every fold in every repeat:
- Assert no `program` appears in **both** train and test.
- Assert no `ir_hash` appears in **both** train and test.
- Write pass/fail per fold to `results/leakage_audit.txt`.

## 7. SHAP Analysis

### 7a. Explainer Variants

| Variant | Method |
|---|---|
| `tree_path_dependent` | `shap.TreeExplainer(model)` (default) |
| `interventional` | `shap.TreeExplainer(model, data=background)`, background <= 200 training samples |

Report Kendall tau between the two rankings per pass.

### 7b. Stability

| Metric | Criterion |
|---|---|
| Kendall tau across folds x repeats | Interpret pass only if tau >= 0.6 |
| Model must beat majority baseline | After Holm correction |

### 7c. Interaction Values

- Compute on held-out data, at most 500 samples.
- Keep only the top 10 feature pairs by mean |interaction|.
- **Do not** save raw arrays; save aggregated table only.

### 7d. Faithfulness (Feature-removal test)

| k (features removed) | Source |
|---|---|
| 1, 3, 5, 10 | Top-k SHAP features |
| 1, 3, 5, 10 | k random features (20 random draws each) |

Metric: PR-AUC drop relative to full-feature model.

### 7e. Size Confound

- Remove `log_inst_count` from the feature set.
- Recompute SHAP ranking.
- Report rank changes for top-10 features.

## 8. Heuristic Agreement

| Test | Detail |
|---|---|
| Permutation null | 10 000 random rankings per pass |
| Correction | Holm-Bonferroni across passes |
| Metrics | hit@5, MRR with bootstrap 95 % CI |

## 9. Sequence Evaluation

| Metric | Definition |
|---|---|
| Geometric-mean size ratio | `exp(mean(log(method_count / baseline_count)))` per program |
| Baselines | -Oz, -O2, random (same length), oracle-greedy |
| CI | 95 % bootstrap CI resampling programs (10 000 samples) |

## 10. Threshold Sensitivity

Recompute the beneficial label at thresholds 0.5 %, 1 %, 2 % from the stored
`rel_reduction` column (no new opt calls). Report:
1. Label positive rate at each threshold.
2. Top-5 SHAP features at each threshold (train a fresh model at each threshold).

---

## Decision Rules (Stop Criteria)

1. If unique functions < 1 000 after Phase 1 -> **STOP**.
2. If a pass has positive rate outside [5 %, 95 %] -> **drop** that pass for that target.
3. If XGBoost does not beat majority after Holm correction -> **do not interpret** that pass.
4. If SHAP stability tau < 0.6 -> **do not interpret** that pass.
