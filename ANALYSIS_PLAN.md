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
| Grouping variable | `program` (capped at 100 functions per program, seeded sample) |
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
| Split categories | Applicability (what the pass needs to fire) vs Cost_Model (profitability thresholds) |
| Feature correlation | Clustered using Spearman \|rho\| > 0.8; agreement checks treat correlated features as equivalent hits |
| Permutation null | 10 000 random rankings per pass |
| Correction | Holm-Bonferroni across passes |
| Metrics | hit@5, MRR with bootstrap 95 % CI |
| Cross-pass control | Score each pass's SHAP ranking against *every other pass's* expected list (matching target direction: beneficial vs harmful). Report if agreement with its own list is higher than the mean agreement with other passes' lists. |
| Pooled test | Across all passes and targets, compute the mean percentile rank of expected features vs 10,000 permutations within each (pass, target). |
| Threshold analysis | For cost_model entries with `direction: threshold`, compute the feature value where mean SHAP changes sign (crossover) and plot SHAP dependence. |

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
3. If XGBoost does not beat majority after Holm correction -> **do not interpret** that target. (Gating is per target).
4. If SHAP stability tau < 0.6 -> **do not interpret** that target. (Gating is per target).

---

## Amendment 1 (2026-10-08, before any model was fitted on the full data)

The data pipeline (Phases 1-3) was corrected after defects were found in the
instruction counter and the feature extractor (listed in `STATUS.md`). No
model, SHAP value or heuristic-agreement score had been computed on the full
data when this amendment was written. The sections above are unchanged
except where stated here.

### A1. Instruction count
One text parser (`utils.iter_function_body`) defines what an instruction is,
for labels and for features. It is validated against LLVM's
`print<func-properties>` on every compiled function.

### A2. Features (40)
Densities (count / instruction count) for 32 opcode and CFG classes, now
including `bitwise_density` and `shift_density`; `log_inst_count`;
`mean_insts_per_bb`, `max_insts_per_bb`; and five loop features taken from
LLVM's LoopInfo: `loop_count_density`, `max_loop_depth`,
`max_loop_header_size`, `min_loop_header_size`, `innermost_loop_body_size`.
No raw counts.

### A3. Samples
A sample is a state: the baseline function or the result of a seeded random
prefix of 1-3 passes. States with IR identical to an earlier state are
dropped. `ir_hash` is the hash of the normalized function body with the
function's own name replaced, so it is unique per state in the dataset.

### A4. Grouping
`program` is the benchmark program (PolyBench kernel folder; MiBench
`<category>/<benchmark>`). Duplicate bodies are removed over the whole
dataset first, then each program is capped at 100 functions (seeded).

### A5. Third target: `fired`
`fired` = the pass output differs from its control run. Kept under the same
5 %-95 % rule. It is scored against the applicability list of
`heuristics.yaml`.

### A6. Control run for loop passes
For `licm`, `loop-rotate`, `indvars`, `loop-deletion`, `loop-idiom` and
`loop-unroll` the control is the same loop adaptor with `no-op-loop`
(LoopSimplify + LCSSA only). `inst_control` is the instruction count after
the control; for all other passes it equals `inst_before`.

### A7. Target definition for the six loop passes (decided 2026-10-09)
For `licm`, `loop-rotate`, `indvars`, `loop-deletion`, `loop-idiom` and
`loop-unroll`, `beneficial` and `harmful` are computed against
`inst_control`, not `inst_before`:

- beneficial = (inst_control - inst_after) / inst_control >= threshold
- harmful    = inst_after > inst_control

Reason: loop canonicalization alone adds instructions in 725 of 2953 states;
measured against `inst_before`, `loop-deletion` would be "harmful" in 724
states although the pass itself never increased the size. All other passes
keep `inst_before` as the reference. The one implementation is
`scripts/targets.py`. This replaces the definitions in Section 1 and 2 for
these six passes; the kept/dropped list is recomputed with it.

---

## Amendment 2 (2026-10-11, before any valid model result existed)

A Phase 4 run pushed on 2026-10-09 did not follow this plan and was withdrawn
(see `STATUS.md`); its numbers were deleted and are not used. Phases 4-7 were
then rewritten. The settings below were fixed before the rewritten code was
run on the full data. Where they differ from Sections 3-9, they replace them.

### B1. Targets and keep rule
Three targets per pass: `beneficial`, `harmful`, `fired` (A5-A7). A target is
kept when its positive rate is in [5 %, 95 %]; the rule is applied before any
model is fitted, and nothing is fitted for a dropped target. Every pass x
target appears in `results/master_table.csv` with its status and reason.

### B2. Features
All 40 columns of A2 except columns with a single value in the dataset
(`invoke_density`), which are dropped and recorded.

### B3. Cross-validation and models (replaces Sections 3 and 4 where different)
| Item | Value |
|---|---|
| Outer CV | 3 repeats x 5 folds, StratifiedGroupKFold, shuffled, grouped by program |
| Inner CV | 3 folds, same grouping, inside the training fold only |
| Tuning budget | 6 random configurations from the Section 4 space, the same 6 for every fold of a pair |
| XGBoost | `tree_method="hist"`, `scale_pos_weight = negatives / positives` |
| Baselines | majority (predicts the training positive rate), logistic regression on standardized features, random forest (100 trees, depth 10); both with balanced class weights |
| Unusable fold | a fold with one class on either side is not scored and is listed in `skipped_folds.txt` |
| Cross-suite check | train on one suite, test on the other, with the most frequently chosen configuration |

### B4. When does XGBoost "beat the majority baseline"? (replaces Section 5)
All three must hold:
1. at least 10 of the 15 outer folds are usable;
2. Wilcoxon signed-rank test on the fold PR-AUCs, two-sided, Holm-corrected
   across all kept pairs, p < 0.05, with a positive mean difference;
3. the 95 % bootstrap interval of the PR-AUC difference, resampling whole
   programs 2,000 times on the out-of-fold predictions, lies above 0.

Folds of a cross-validation share training data, so test 2 alone is
optimistic; test 3 uses programs as the independent unit.

### B5. SHAP (replaces Section 7 where different)
- Values: path-dependent TreeSHAP computed by XGBoost (`pred_contribs`), on
  held-out folds only, for all 15 folds, refitted with each fold's tuned
  hyperparameters. A unit test checks that it equals `shap.TreeExplainer`.
- Importance of a feature: mean |SHAP|, scaled to sum to 1 within a fold,
  averaged over folds.
- Stability: mean Kendall tau over all pairs of fold importance vectors.
  **A pair is interpreted only if B4 holds and this tau is at least 0.6.**
- On the 5 folds of the first repeat: interventional SHAP (background of 100
  training rows, 300 test rows), permutation importance, a label-shuffle
  control, a refit without the size features, and faithfulness.
- Size features: `log_inst_count` and every feature in its correlation
  cluster are removed together for the size-confound check.
- Faithfulness: features are ranked by SHAP on the training fold; the top k
  (1, 3, 5, 10) are removed and the model refitted; compared with 5 random
  sets of k features.
- Interaction values: 300 held-out rows of the first fold; top 10 pairs.
- Threshold sensitivity (Section 10): first repeat, most frequent configuration.

### B6. Heuristic agreement (replaces Section 8 where different)
- Features are clustered on the whole feature table (Spearman, average
  linkage, cut at 1 - |rho| = 0.2). A cluster's importance is the sum of its
  members'. Agreement is scored on cluster ranks.
- hit@5 = share of expected clusters ranked in the top 5.
  MRR = mean of 1 / rank over the expected clusters.
- Null: 10,000 random rankings. p = (1 + number of null values at least as
  large as the observed one) / 10,001. Holm correction across interpreted
  pairs, separately for the applicability and the cost-model list.
- Confidence intervals: 1,000 resamples of the fold importance vectors.
- For `fired`, the applicability list is used with the direction written
  under `beneficial`. Direction agreement: sign of the Spearman correlation
  between a feature's value and its SHAP value on held-out rows; |rho| below
  0.05 counts as undetermined.
- Cross-pass control: a pair's ranking is scored against the applicability
  lists of the other passes (MRR); it should fit its own list better.
- Pooled test: mean percentile rank of all expected clusters over interpreted
  pairs, against rankings permuted within each pair; reported per target and
  for all targets (pairs of one pass are not independent).
- Threshold analysis: for each `cost_model` entry, the feature's rank, the
  mean SHAP value at or below and above LLVM's default, and the first value
  bin where the mean SHAP value changes sign (20 equal-count bins, held-out
  rows; loop-size features on functions with loops only).
- `heuristics.yaml` is frozen by its SHA-256 in `env.json`; `run_all.py`
  stops if the file changes.

### B7. Pruning (Phase 6)
Interpreted pairs, 5 folds of the first repeat, each fold's tuned
hyperparameters. Features are ranked on the training fold by SHAP, XGBoost
gain, mutual information, or at random (5 draws); k = 5, 10, 15, 20.

### B8. Sequences (replaces Section 9 where different)
- Held out: a seeded 20 % sample of programs within each suite; at most 200
  of their functions.
- Models: `beneficial` target, passes that satisfy B4, trained only on the
  other programs with the most frequently chosen configuration.
- Methods: model-guided greedy (at most 8 steps, probability at least 0.5,
  a pass is kept only if it changes the IR), oracle greedy, random passes of
  the same length (5 draws), `default<Oz>`, `default<O2>`.
- Metric: geometric mean of final / starting instruction count per program,
  then over programs; 95 % bootstrap interval over programs (2,000).

### B9. Smoke settings
`--smoke` uses 1 repeat x 3 folds, 2 configurations and three passes. It
only shows that the code runs. Smoke output is labelled and is not a result.

