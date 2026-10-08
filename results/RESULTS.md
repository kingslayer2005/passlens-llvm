# PRELIMINARY: smoke run on 52 programs, not a research result

> **Auto-generated** — all numbers come from CSV files in `results/`.
> Do not edit manually; re-run `python -m scripts.generate_results_md` instead.

---

## Phase 1 — Data Collection

```

╔══════════════════════════════════════════════════╗
║            PHASE 1 GATE REPORT                   ║
╠══════════════════════════════════════════════════╣
║ Programs processed:         52                   ║
║ Total functions (≥10i):    854                   ║
║ Unique functions:          819                   ║
║ Duplicates removed:         35                   ║
║ Compilation failures:      742                   ║
║ Extraction failures:         0                   ║
╚══════════════════════════════════════════════════╝

```

## Phase 3 — Label Distribution

| pass_name              |   total_samples |   decreased |   equal |   increased |   beneficial_rate |   harmful_rate | drop_beneficial   | drop_harmful   |
|:-----------------------|----------------:|------------:|--------:|------------:|------------------:|---------------:|:------------------|:---------------|
| sroa                   |              40 |           0 |      40 |           0 |             0     |          0     | True              | True           |
| instcombine            |              40 |          35 |       5 |           0 |             0.875 |          0     | False             | True           |
| simplifycfg            |              40 |          26 |      14 |           0 |             0.65  |          0     | False             | True           |
| early-cse              |              40 |          26 |      14 |           0 |             0.575 |          0     | False             | True           |
| gvn                    |              40 |          36 |       4 |           0 |             0.9   |          0     | False             | True           |
| sccp                   |              40 |           0 |      40 |           0 |             0     |          0     | True              | True           |
| adce                   |              40 |           0 |      40 |           0 |             0     |          0     | True              | True           |
| dse                    |              40 |           0 |      40 |           0 |             0     |          0     | True              | True           |
| reassociate            |              40 |           0 |      40 |           0 |             0     |          0     | True              | True           |
| jump-threading         |              40 |          24 |      16 |           0 |             0.6   |          0     | False             | True           |
| correlated-propagation |              40 |           0 |      40 |           0 |             0     |          0     | True              | True           |
| licm                   |              40 |           0 |      39 |           1 |             0     |          0.025 | True              | True           |
| loop-rotate            |              40 |           0 |      13 |          27 |             0     |          0.675 | True              | False          |
| indvars                |              40 |           8 |      15 |          17 |             0.2   |          0.425 | False             | False          |
| loop-deletion          |              40 |           0 |      40 |           0 |             0     |          0     | True              | True           |
| loop-idiom             |              40 |           0 |      39 |           1 |             0     |          0.025 | True              | True           |
| loop-unroll            |              40 |           0 |      39 |           1 |             0     |          0.025 | True              | True           |
| tailcallelim           |              40 |           0 |      40 |           0 |             0     |          0     | True              | True           |



```

╔══════════════════════════════════════════════════════╗
║              PHASE 3 GATE REPORT                     ║
╠══════════════════════════════════════════════════════╣
║ Total samples:               720                     ║
║ Functions:                    10                     ║
║ Passes (active):              18                     ║
║ Passes (kept):                 7                     ║
║ Passes (dropped):             11                     ║
╠══════════════════════════════════════════════════════╣
║ Dropped passes: sroa, sccp, adce, dse, reassociate, correlated-propagation, licm, loop-deletion, loop-idiom, loop-unroll, tailcallelim║
╚══════════════════════════════════════════════════════╝

```

## Phase 4 — Model Performance

### Per-Pass Summary (Mean ± 95% CI over programs)

| pass_name      | target     | model               |   pr_auc_mean |   pr_auc_std |   pr_auc_ci_lo |   pr_auc_ci_hi |   f1_mean |   mcc_mean |
|:---------------|:-----------|:--------------------|--------------:|-------------:|---------------:|---------------:|----------:|-----------:|
| simplifycfg    | beneficial | majority            |        0.6042 |       0.164  |         0.375  |         0.75   |    0.7391 |     0      |
| simplifycfg    | beneficial | logistic_regression |        0.8687 |       0.1025 |         0.75   |         1      |    0.8646 |     0.7264 |
| simplifycfg    | beneficial | random_forest       |        0.8952 |       0.106  |         0.75   |         1      |    0.9379 |     0.877  |
| simplifycfg    | beneficial | xgboost             |        0.9091 |       0.1129 |         0.75   |         1      |    0.8654 |     0.7846 |
| early-cse      | beneficial | majority            |        0.625  |       0.1768 |         0.5    |         0.875  |    0.4445 |     0      |
| early-cse      | beneficial | logistic_regression |        0.9162 |       0.0836 |         0.8021 |         1      |    0.7758 |     0.5109 |
| early-cse      | beneficial | random_forest       |        0.9457 |       0.0447 |         0.8906 |         1      |    0.8674 |     0.7093 |
| early-cse      | beneficial | xgboost             |        0.9457 |       0.0447 |         0.8906 |         1      |    0.798  |     0.4593 |
| gvn            | beneficial | majority            |        0.875  |       0      |         0.875  |         0.875  |    0.9333 |     0      |
| gvn            | beneficial | logistic_regression |        0.8913 |       0.0312 |         0.8601 |         0.9226 |    0.9333 |     0      |
| gvn            | beneficial | random_forest       |        0.9122 |       0.0104 |         0.9018 |         0.9226 |    0.9333 |     0      |
| gvn            | beneficial | xgboost             |        0.875  |       0      |         0.875  |         0.875  |    0.7778 |     0      |
| jump-threading | beneficial | majority            |        0.5625 |       0.135  |         0.375  |         0.6875 |    0.7098 |     0      |
| jump-threading | beneficial | logistic_regression |        0.8799 |       0.0956 |         0.75   |         0.9773 |    0.8694 |     0.7314 |
| jump-threading | beneficial | random_forest       |        0.8799 |       0.0956 |         0.75   |         0.9773 |    0.9076 |     0.7921 |
| jump-threading | beneficial | xgboost             |        0.8508 |       0.0946 |         0.75   |         0.9773 |    0.7665 |     0.6364 |
| loop-rotate    | harmful    | majority            |        0.6458 |       0.1062 |         0.5    |         0.75   |    0.7795 |     0      |
| loop-rotate    | harmful    | logistic_regression |        0.9924 |       0.0107 |         0.9773 |         1      |    0.9855 |     0.9521 |
| loop-rotate    | harmful    | random_forest       |        0.9785 |       0.0304 |         0.9356 |         1      |    0.9855 |     0.9521 |
| loop-rotate    | harmful    | xgboost             |        0.9785 |       0.0304 |         0.9356 |         1      |    0.9123 |     0.8015 |
| indvars        | beneficial | majority            |        0.25   |       0      |         0.25   |         0.25   |    0      |     0      |
| indvars        | beneficial | logistic_regression |        1      |       0      |         1      |         1      |    0.25   |     0.1666 |
| indvars        | beneficial | random_forest       |        1      |       0      |         1      |         1      |    0.5    |     0.5    |
| indvars        | beneficial | xgboost             |        0.75   |       0.25   |         0.5    |         1      |    0.8333 |     0.7887 |
| indvars        | harmful    | majority            |        0.4167 |       0.0589 |         0.375  |         0.5    |    0      |     0      |
| indvars        | harmful    | logistic_regression |        0.6944 |       0.0786 |         0.5833 |         0.75   |    0.5524 |     0.4507 |
| indvars        | harmful    | random_forest       |        0.8333 |       0.1179 |         0.75   |         1      |    0.9047 |     0.8497 |
| indvars        | harmful    | xgboost             |        0.8333 |       0.1179 |         0.75   |         1      |    0.819  |     0.7406 |


### Wilcoxon Signed-Rank Test (Holm-corrected)

| pass_name      | target     | baseline            |   xgb_mean |   bl_mean |    p_value |   statistic |   p_holm | significant   |
|:---------------|:-----------|:--------------------|-----------:|----------:|-----------:|------------:|---------:|:--------------|
| simplifycfg    | beneficial | majority            |     0.9091 |    0.6042 | 0.00195312 |          45 | 0.041016 | True          |
| early-cse      | beneficial | majority            |     0.9457 |    0.625  | 0.00195312 |          45 | 0.041016 | True          |
| gvn            | beneficial | majority            |     0.875  |    0.875  | 1          |           0 | 1        | False         |
| jump-threading | beneficial | majority            |     0.8508 |    0.5625 | 0.00195312 |          45 | 0.041016 | True          |
| loop-rotate    | harmful    | majority            |     0.9785 |    0.6458 | 0.00195312 |          45 | 0.041016 | True          |
| indvars        | harmful    | majority            |     0.8333 |    0.4167 | 0.00195312 |          45 | 0.041016 | True          |
| indvars        | beneficial | majority            |     0.75   |    0.25   | 0.015625   |          21 | 0.25     | False         |
| simplifycfg    | beneficial | logistic_regression |     0.9091 |    0.8687 | 0.125      |           6 | 1        | False         |
| early-cse      | beneficial | logistic_regression |     0.9457 |    0.9162 | 0.125      |           6 | 1        | False         |
| gvn            | beneficial | logistic_regression |     0.875  |    0.8914 | 0.84375    |           6 | 1        | False         |
| jump-threading | beneficial | logistic_regression |     0.8508 |    0.8799 | 1          |           0 | 1        | False         |
| loop-rotate    | harmful    | logistic_regression |     0.9785 |    0.9924 | 1          |           0 | 1        | False         |
| indvars        | harmful    | logistic_regression |     0.8333 |    0.6944 | 0.125      |           6 | 1        | False         |
| indvars        | beneficial | logistic_regression |     0.75   |    1      | 1          |           0 | 1        | False         |
| simplifycfg    | beneficial | random_forest       |     0.9091 |    0.8952 | 0.125      |           6 | 1        | False         |
| early-cse      | beneficial | random_forest       |     0.9457 |    0.9457 | 1          |           0 | 1        | False         |
| gvn            | beneficial | random_forest       |     0.875  |    0.9122 | 1          |           0 | 1        | False         |
| jump-threading | beneficial | random_forest       |     0.8508 |    0.8799 | 1          |           0 | 1        | False         |
| loop-rotate    | harmful    | random_forest       |     0.9785 |    0.9785 | 1          |           0 | 1        | False         |
| indvars        | harmful    | random_forest       |     0.8333 |    0.8333 | 1          |           0 | 1        | False         |
| indvars        | beneficial | random_forest       |     0.75   |    1      | 1          |           0 | 1        | False         |


### Threshold Sensitivity

| pass_name      |   threshold |   pos_rate |   n_samples | top5_shap                                                                             |
|:---------------|------------:|-----------:|------------:|:--------------------------------------------------------------------------------------|
| instcombine    |       0.005 |      0.875 |          40 | log_inst_count; load_density; gep_density; const_operands_density; icmp_density       |
| simplifycfg    |       0.005 |      0.65  |          40 | log_inst_count; icmp_density; int_arith_density; fp_arith_density; fcmp_density       |
| early-cse      |       0.005 |      0.65  |          40 | gep_density; icmp_density; const_operands_density; int_arith_density; log_inst_count  |
| gvn            |       0.005 |      0.9   |          40 | log_inst_count; int_arith_density; fp_arith_density; icmp_density; fcmp_density       |
| jump-threading |       0.005 |      0.6   |          40 | log_inst_count; icmp_density; cast_density; load_density; const_operands_density      |
| loop-rotate    |       0.005 |      0     |          40 | nan                                                                                   |
| indvars        |       0.005 |      0.2   |          40 | log_inst_count; int_arith_density; fp_arith_density; icmp_density; fcmp_density       |
| instcombine    |       0.01  |      0.875 |          40 | log_inst_count; load_density; gep_density; const_operands_density; icmp_density       |
| simplifycfg    |       0.01  |      0.65  |          40 | log_inst_count; icmp_density; int_arith_density; fp_arith_density; fcmp_density       |
| early-cse      |       0.01  |      0.575 |          40 | int_arith_density; const_operands_density; icmp_density; cast_density; log_inst_count |
| gvn            |       0.01  |      0.9   |          40 | log_inst_count; int_arith_density; fp_arith_density; icmp_density; fcmp_density       |
| jump-threading |       0.01  |      0.6   |          40 | log_inst_count; icmp_density; cast_density; load_density; const_operands_density      |
| loop-rotate    |       0.01  |      0     |          40 | nan                                                                                   |
| indvars        |       0.01  |      0.2   |          40 | log_inst_count; int_arith_density; fp_arith_density; icmp_density; fcmp_density       |
| instcombine    |       0.02  |      0.675 |          40 | int_arith_density; icmp_density; log_inst_count; fp_arith_density; fcmp_density       |
| simplifycfg    |       0.02  |      0.65  |          40 | log_inst_count; icmp_density; int_arith_density; fp_arith_density; fcmp_density       |
| early-cse      |       0.02  |      0.55  |          40 | int_arith_density; icmp_density; load_density; log_inst_count; const_operands_density |
| gvn            |       0.02  |      0.9   |          40 | log_inst_count; int_arith_density; fp_arith_density; icmp_density; fcmp_density       |
| jump-threading |       0.02  |      0.6   |          40 | log_inst_count; icmp_density; cast_density; load_density; const_operands_density      |
| loop-rotate    |       0.02  |      0     |          40 | nan                                                                                   |
| indvars        |       0.02  |      0.2   |          40 | log_inst_count; int_arith_density; fp_arith_density; icmp_density; fcmp_density       |



```

╔══════════════════════════════════════════════════════╗
║              PHASE 4 GATE REPORT                     ║
╠══════════════════════════════════════════════════════╣
║ Passes evaluated:              7                     ║
║ Targets per pass:              2                     ║
║ Repeats x folds:        3x5 =   15                    ║
║ XGB beats majority:            5                     ║
║ Flagged passes:                2                     ║
╚══════════════════════════════════════════════════════╝

```

## Phase 5 — Explainability (SHAP)

### Rigor & Stability

| pass_name      | target     |   stability_tau |   dual_method_tau | beats_majority   | stable   | interpretable   |
|:---------------|:-----------|----------------:|------------------:|:-----------------|:---------|:----------------|
| simplifycfg    | beneficial |          0.7329 |               nan | True             | True     | True            |
| early-cse      | beneficial |          0.6591 |               nan | True             | True     | True            |
| jump-threading | beneficial |          0.704  |               nan | True             | True     | True            |


### Top 10 Features per Interpretable Pass


#### early-cse (target: beneficial)

|   rank | feature                |   mean_abs_shap |
|-------:|:-----------------------|----------------:|
|      1 | gep_count              |        0.600701 |
|      2 | int_arith_density      |        0.474263 |
|      3 | icmp_density           |        0.413014 |
|      4 | gep_density            |        0.26916  |
|      5 | const_operands_density |        0.170599 |
|      6 | const_operands         |        0.128084 |
|      7 | cond_br_density        |        0.087797 |
|      8 | cast_density           |        0.084407 |
|      9 | blocks_2_pred_density  |        0.075509 |
|     10 | int_arith_count        |        0.072937 |



![SHAP Beeswarm — early-cse (beneficial)](C:\Users\Aarush Gupta\Downloads\LLVM Project\results\figures\shap\shap_beeswarm_early-cse_beneficial.png)


#### jump-threading (target: beneficial)

|   rank | feature         |   mean_abs_shap |
|-------:|:----------------|----------------:|
|      1 | int_arith_count |        1.48964  |
|      2 | load_count      |        0.320793 |
|      3 | icmp_density    |        0.259435 |
|      4 | gep_count       |        0.172325 |
|      5 | const_operands  |        0.166847 |
|      6 | store_count     |        0.162641 |
|      7 | phi_count       |        0.1545   |
|      8 | cast_density    |        0.149208 |
|      9 | icmp_count      |        0.13453  |
|     10 | load_density    |        0.063055 |



![SHAP Beeswarm — jump-threading (beneficial)](C:\Users\Aarush Gupta\Downloads\LLVM Project\results\figures\shap\shap_beeswarm_jump-threading_beneficial.png)


#### simplifycfg (target: beneficial)

|   rank | feature           |   mean_abs_shap |
|-------:|:------------------|----------------:|
|      1 | int_arith_count   |        1.39581  |
|      2 | icmp_density      |        0.1718   |
|      3 | gep_count         |        0.145613 |
|      4 | icmp_count        |        0.136055 |
|      5 | const_operands    |        0.128369 |
|      6 | load_count        |        0.120744 |
|      7 | phi_count         |        0.087388 |
|      8 | int_arith_density |        0.063829 |
|      9 | cond_br_density   |        0.044848 |
|     10 | load_density      |        0.040442 |



![SHAP Beeswarm — simplifycfg (beneficial)](C:\Users\Aarush Gupta\Downloads\LLVM Project\results\figures\shap\shap_beeswarm_simplifycfg_beneficial.png)

### Faithfulness Checks

| pass_name      | target     |   k |   full_prauc |   shap_removed_prauc |   shap_drop |   random_removed_prauc_mean |   random_removed_prauc_std |   random_drop_mean |
|:---------------|:-----------|----:|-------------:|---------------------:|------------:|----------------------------:|---------------------------:|-------------------:|
| simplifycfg    | beneficial |   1 |       0.7727 |               0.9148 |     -0.142  |                      0.7838 |                     0.0347 |            -0.0111 |
| simplifycfg    | beneficial |   3 |       0.7727 |               0.9773 |     -0.2045 |                      0.798  |                     0.052  |            -0.0253 |
| simplifycfg    | beneficial |   5 |       0.7727 |               0.9148 |     -0.142  |                      0.802  |                     0.0529 |            -0.0293 |
| simplifycfg    | beneficial |  10 |       0.7727 |               0.8523 |     -0.0795 |                      0.8233 |                     0.0701 |            -0.0506 |
| early-cse      | beneficial |   1 |       0.9464 |               0.9464 |      0      |                      0.9464 |                     0      |            -0      |
| early-cse      | beneficial |   3 |       0.9464 |               0.9464 |      0      |                      0.9464 |                     0      |            -0      |
| early-cse      | beneficial |   5 |       0.9464 |               0.8214 |      0.125  |                      0.9464 |                     0      |            -0      |
| early-cse      | beneficial |  10 |       0.9464 |               0.9464 |      0      |                      0.9464 |                     0      |            -0      |
| jump-threading | beneficial |   1 |       0.9356 |               0.9773 |     -0.0417 |                      0.9335 |                     0.0207 |             0.0021 |
| jump-threading | beneficial |   3 |       0.9356 |               0.9773 |     -0.0417 |                      0.9335 |                     0.0308 |             0.0021 |
| jump-threading | beneficial |   5 |       0.9356 |               0.9773 |     -0.0417 |                      0.9377 |                     0.0246 |            -0.0021 |
| jump-threading | beneficial |  10 |       0.9356 |               0.8523 |      0.0833 |                      0.9273 |                     0.0408 |             0.0083 |


### Top 10 Feature Interactions

| pass_name      | target     | feature_1       | feature_2              |   mean_abs_interaction |
|:---------------|:-----------|:----------------|:-----------------------|-----------------------:|
| simplifycfg    | beneficial | int_arith_count | fp_arith_count         |               0        |
| simplifycfg    | beneficial | int_arith_count | icmp_count             |               0        |
| simplifycfg    | beneficial | int_arith_count | fcmp_count             |               0        |
| simplifycfg    | beneficial | int_arith_count | load_count             |               0        |
| simplifycfg    | beneficial | int_arith_count | store_count            |               0        |
| simplifycfg    | beneficial | int_arith_count | gep_count              |               0        |
| simplifycfg    | beneficial | int_arith_count | alloca_count           |               0        |
| simplifycfg    | beneficial | int_arith_count | phi_count              |               0        |
| simplifycfg    | beneficial | int_arith_count | call_count             |               0        |
| simplifycfg    | beneficial | int_arith_count | cast_count             |               0        |
| early-cse      | beneficial | load_count      | const_operands_density |               0.109857 |
| early-cse      | beneficial | gep_count       | const_operands_density |               0.046986 |
| early-cse      | beneficial | int_arith_count | fp_arith_count         |               0        |
| early-cse      | beneficial | int_arith_count | icmp_count             |               0        |
| early-cse      | beneficial | int_arith_count | fcmp_count             |               0        |
| early-cse      | beneficial | int_arith_count | load_count             |               0        |
| early-cse      | beneficial | int_arith_count | store_count            |               0        |
| early-cse      | beneficial | int_arith_count | gep_count              |               0        |
| early-cse      | beneficial | int_arith_count | alloca_count           |               0        |
| early-cse      | beneficial | int_arith_count | phi_count              |               0        |
| jump-threading | beneficial | int_arith_count | icmp_density           |               0.022178 |
| jump-threading | beneficial | int_arith_count | icmp_count             |               0.02093  |
| jump-threading | beneficial | int_arith_count | fp_arith_count         |               0        |
| jump-threading | beneficial | int_arith_count | fcmp_count             |               0        |
| jump-threading | beneficial | int_arith_count | load_count             |               0        |
| jump-threading | beneficial | int_arith_count | store_count            |               0        |
| jump-threading | beneficial | int_arith_count | gep_count              |               0        |
| jump-threading | beneficial | int_arith_count | alloca_count           |               0        |
| jump-threading | beneficial | int_arith_count | phi_count              |               0        |
| jump-threading | beneficial | int_arith_count | call_count             |               0        |


### Size Confound Analysis

| pass_name      | target     | feature                |   rank_without_size |   rank_with_size |
|:---------------|:-----------|:-----------------------|--------------------:|-----------------:|
| simplifycfg    | beneficial | int_arith_count        |                   1 |                1 |
| simplifycfg    | beneficial | icmp_density           |                   2 |                2 |
| simplifycfg    | beneficial | gep_count              |                   3 |                3 |
| simplifycfg    | beneficial | icmp_count             |                   4 |                4 |
| simplifycfg    | beneficial | fcmp_count             |                   5 |              nan |
| simplifycfg    | beneficial | load_count             |                   6 |                6 |
| simplifycfg    | beneficial | store_count            |                   7 |              nan |
| simplifycfg    | beneficial | fp_arith_count         |                   8 |              nan |
| simplifycfg    | beneficial | phi_count              |                   9 |                7 |
| simplifycfg    | beneficial | call_count             |                  10 |              nan |
| early-cse      | beneficial | int_arith_density      |                   1 |                2 |
| early-cse      | beneficial | const_operands_density |                   2 |                5 |
| early-cse      | beneficial | cast_density           |                   3 |                8 |
| early-cse      | beneficial | gep_count              |                   4 |                1 |
| early-cse      | beneficial | load_count             |                   5 |              nan |
| early-cse      | beneficial | store_count            |                   6 |              nan |
| early-cse      | beneficial | fcmp_count             |                   7 |              nan |
| early-cse      | beneficial | icmp_count             |                   8 |              nan |
| early-cse      | beneficial | phi_count              |                   9 |              nan |
| early-cse      | beneficial | call_count             |                  10 |              nan |
| jump-threading | beneficial | int_arith_count        |                   1 |                1 |
| jump-threading | beneficial | icmp_density           |                   2 |                3 |
| jump-threading | beneficial | cast_density           |                   3 |                8 |
| jump-threading | beneficial | load_count             |                   4 |                2 |
| jump-threading | beneficial | gep_count              |                   5 |                4 |
| jump-threading | beneficial | icmp_count             |                   6 |                9 |
| jump-threading | beneficial | fcmp_count             |                   7 |              nan |
| jump-threading | beneficial | fp_arith_count         |                   8 |              nan |
| jump-threading | beneficial | phi_count              |                   9 |                7 |
| jump-threading | beneficial | call_count             |                  10 |              nan |


### Heuristic Agreement (Permutation Test)

| pass_name      | target     | heuristic_type   |   hit_at_5 |   hit_at_5_ci_lo |   hit_at_5_ci_hi |   hit_at_5_p |    mrr |   mrr_ci_lo |   mrr_ci_hi |   mrr_p |   cross_pass_mean_hit5 |   cross_pass_mean_mrr | beats_others   |   n_heuristic_features |   hit_at_5_p_holm |   mrr_p_holm |
|:---------------|:-----------|:-----------------|-----------:|-----------------:|-----------------:|-------------:|-------:|------------:|------------:|--------:|-----------------------:|----------------------:|:---------------|-----------------------:|------------------:|-------------:|
| simplifycfg    | beneficial | applicability    |          0 |                0 |                0 |            1 | 0.0614 |      0.0204 |      0.1111 |  0.3188 |                 0.0588 |                0.0799 | False          |                      3 |                 1 |       0.9564 |
| early-cse      | beneficial | applicability    |          0 |                0 |                0 |            1 | 0.0714 |      0.0714 |      0.0714 |  0.2079 |                 0.1765 |                0.1106 | False          |                      1 |                 1 |       0.855  |
| early-cse      | beneficial | cost_model       |          0 |                0 |                0 |            1 | 0.0159 |      0.0159 |      0.0159 |  0.9264 |                 0      |                0.03   | False          |                      1 |                 1 |       1      |
| jump-threading | beneficial | applicability    |          0 |                0 |                0 |            1 | 0.0833 |      0.0833 |      0.0833 |  0.171  |                 0.0588 |                0.063  | True           |                      1 |                 1 |       0.855  |
| jump-threading | beneficial | cost_model       |          0 |                0 |                0 |            1 | 0.0159 |      0.0159 |      0.0159 |  0.9264 |                 0      |                0.0289 | False          |                      1 |                 1 |       1      |


### Candidate Novel Drivers

| pass_name      | target     | feature                |   shap_rank |   mean_abs_shap |
|:---------------|:-----------|:-----------------------|------------:|----------------:|
| simplifycfg    | beneficial | int_arith_count        |           1 |        1.39581  |
| simplifycfg    | beneficial | icmp_density           |           2 |        0.1718   |
| simplifycfg    | beneficial | gep_count              |           3 |        0.145613 |
| simplifycfg    | beneficial | icmp_count             |           4 |        0.136055 |
| simplifycfg    | beneficial | const_operands         |           5 |        0.128369 |
| simplifycfg    | beneficial | load_count             |           6 |        0.120744 |
| simplifycfg    | beneficial | phi_count              |           7 |        0.087388 |
| simplifycfg    | beneficial | int_arith_density      |           8 |        0.063829 |
| simplifycfg    | beneficial | load_density           |          10 |        0.040442 |
| early-cse      | beneficial | gep_count              |           1 |        0.600701 |
| early-cse      | beneficial | int_arith_density      |           2 |        0.474263 |
| early-cse      | beneficial | icmp_density           |           3 |        0.413014 |
| early-cse      | beneficial | gep_density            |           4 |        0.26916  |
| early-cse      | beneficial | const_operands_density |           5 |        0.170599 |
| early-cse      | beneficial | const_operands         |           6 |        0.128084 |
| early-cse      | beneficial | cond_br_density        |           7 |        0.087797 |
| early-cse      | beneficial | cast_density           |           8 |        0.084407 |
| early-cse      | beneficial | blocks_2_pred_density  |           9 |        0.075509 |
| early-cse      | beneficial | int_arith_count        |          10 |        0.072937 |
| jump-threading | beneficial | int_arith_count        |           1 |        1.48964  |
| jump-threading | beneficial | load_count             |           2 |        0.320793 |
| jump-threading | beneficial | icmp_density           |           3 |        0.259435 |
| jump-threading | beneficial | gep_count              |           4 |        0.172325 |
| jump-threading | beneficial | const_operands         |           5 |        0.166847 |
| jump-threading | beneficial | store_count            |           6 |        0.162641 |
| jump-threading | beneficial | phi_count              |           7 |        0.1545   |
| jump-threading | beneficial | cast_density           |           8 |        0.149208 |
| jump-threading | beneficial | icmp_count             |           9 |        0.13453  |
| jump-threading | beneficial | load_density           |          10 |        0.063055 |


### Phase 5 Gating Summary

| pass_name              | target     | status      | reason                           |
|:-----------------------|:-----------|:------------|:---------------------------------|
| sroa                   | beneficial | dropped     | Pos rate out of bounds [5%, 95%] |
| sroa                   | harmful    | dropped     | Pos rate out of bounds [5%, 95%] |
| instcombine            | beneficial | dropped     | Failed model training / XGate    |
| instcombine            | harmful    | dropped     | Pos rate out of bounds [5%, 95%] |
| simplifycfg            | beneficial | interpreted | Stable & beats majority          |
| simplifycfg            | harmful    | dropped     | Pos rate out of bounds [5%, 95%] |
| early-cse              | beneficial | interpreted | Stable & beats majority          |
| early-cse              | harmful    | dropped     | Pos rate out of bounds [5%, 95%] |
| gvn                    | beneficial | dropped     | Failed model training / XGate    |
| gvn                    | harmful    | dropped     | Pos rate out of bounds [5%, 95%] |
| sccp                   | beneficial | dropped     | Pos rate out of bounds [5%, 95%] |
| sccp                   | harmful    | dropped     | Pos rate out of bounds [5%, 95%] |
| adce                   | beneficial | dropped     | Pos rate out of bounds [5%, 95%] |
| adce                   | harmful    | dropped     | Pos rate out of bounds [5%, 95%] |
| dse                    | beneficial | dropped     | Pos rate out of bounds [5%, 95%] |
| dse                    | harmful    | dropped     | Pos rate out of bounds [5%, 95%] |
| reassociate            | beneficial | dropped     | Pos rate out of bounds [5%, 95%] |
| reassociate            | harmful    | dropped     | Pos rate out of bounds [5%, 95%] |
| jump-threading         | beneficial | interpreted | Stable & beats majority          |
| jump-threading         | harmful    | dropped     | Pos rate out of bounds [5%, 95%] |
| correlated-propagation | beneficial | dropped     | Pos rate out of bounds [5%, 95%] |
| correlated-propagation | harmful    | dropped     | Pos rate out of bounds [5%, 95%] |
| licm                   | beneficial | dropped     | Pos rate out of bounds [5%, 95%] |
| licm                   | harmful    | dropped     | Pos rate out of bounds [5%, 95%] |
| loop-rotate            | beneficial | dropped     | Pos rate out of bounds [5%, 95%] |
| loop-rotate            | harmful    | dropped     | Failed model training / XGate    |
| indvars                | beneficial | dropped     | Failed model training / XGate    |
| indvars                | harmful    | dropped     | Failed model training / XGate    |
| loop-deletion          | beneficial | dropped     | Pos rate out of bounds [5%, 95%] |
| loop-deletion          | harmful    | dropped     | Pos rate out of bounds [5%, 95%] |
| loop-idiom             | beneficial | dropped     | Pos rate out of bounds [5%, 95%] |
| loop-idiom             | harmful    | dropped     | Pos rate out of bounds [5%, 95%] |
| loop-unroll            | beneficial | dropped     | Pos rate out of bounds [5%, 95%] |
| loop-unroll            | harmful    | dropped     | Pos rate out of bounds [5%, 95%] |
| tailcallelim           | beneficial | dropped     | Pos rate out of bounds [5%, 95%] |
| tailcallelim           | harmful    | dropped     | Pos rate out of bounds [5%, 95%] |



```

╔══════════════════════════════════════════════════════╗
║              PHASE 5 GATE REPORT                     ║
╠══════════════════════════════════════════════════════╣
║ Passes analyzed:               3                     ║
║ Targets:                       2                     ║
║ Interpretable (pass):          3                     ║
║ Stability threshold:     tau>=0.6                     ║
╚══════════════════════════════════════════════════════╝

```

## Phase 6 — Feature Pruning

| pass_name      | method      |   k |   pr_auc_mean |   pr_auc_std |   model_size_mean |   inference_ms_mean |
|:---------------|:------------|----:|--------------:|-------------:|------------------:|--------------------:|
| early-cse      | mutual_info |   5 |      0.839767 |    0.232662  |               200 |            0.616667 |
| early-cse      | mutual_info |  10 |      0.9092   |    0.114045  |               200 |            0.616667 |
| early-cse      | mutual_info |  15 |      0.9092   |    0.114045  |               200 |            0.586667 |
| early-cse      | mutual_info |  20 |      0.9092   |    0.114045  |               200 |            0.556667 |
| early-cse      | mutual_info |  30 |      0.945667 |    0.0547037 |               200 |            0.503333 |
| early-cse      | mutual_info |  68 |      0.945667 |    0.0547037 |               200 |            0.513333 |
| early-cse      | random      |   5 |      0.867533 |    0.116467  |               200 |            0.56     |
| early-cse      | random      |  10 |      0.857133 |    0.128774  |               200 |            0.61     |
| early-cse      | random      |  15 |      0.867533 |    0.116467  |               200 |            0.526667 |
| early-cse      | random      |  20 |      0.9092   |    0.114045  |               200 |            0.576667 |
| early-cse      | random      |  30 |      0.9092   |    0.114045  |               200 |            0.543333 |
| early-cse      | random      |  68 |      0.945667 |    0.0547037 |               200 |            0.516667 |
| early-cse      | shap        |   5 |      0.945667 |    0.0547037 |               200 |            0.97     |
| early-cse      | shap        |  10 |      0.945667 |    0.0547037 |               200 |            0.536667 |
| early-cse      | shap        |  15 |      0.945667 |    0.0547037 |               200 |            0.53     |
| early-cse      | shap        |  20 |      0.945667 |    0.0547037 |               200 |            0.526667 |
| early-cse      | shap        |  30 |      0.945667 |    0.0547037 |               200 |            0.573333 |
| early-cse      | shap        |  68 |      0.945667 |    0.0547037 |               200 |            0.596667 |
| early-cse      | xgb_gain    |   5 |      0.945667 |    0.0547037 |               200 |            0.506667 |
| early-cse      | xgb_gain    |  10 |      0.945667 |    0.0547037 |               200 |            0.563333 |
| early-cse      | xgb_gain    |  15 |      0.945667 |    0.0547037 |               200 |            0.51     |
| early-cse      | xgb_gain    |  20 |      0.945667 |    0.0547037 |               200 |            0.59     |
| early-cse      | xgb_gain    |  30 |      0.945667 |    0.0547037 |               200 |            0.69     |
| early-cse      | xgb_gain    |  68 |      0.945667 |    0.0547037 |               200 |            0.656667 |
| jump-threading | mutual_info |   5 |      0.863267 |    0.113652  |               200 |            0.626667 |
| jump-threading | mutual_info |  10 |      0.850767 |    0.11582   |               200 |            0.656667 |
| jump-threading | mutual_info |  15 |      0.850767 |    0.11582   |               200 |            0.67     |
| jump-threading | mutual_info |  20 |      0.866033 |    0.101149  |               200 |            0.66     |
| jump-threading | mutual_info |  30 |      0.866033 |    0.101149  |               200 |            0.6      |
| jump-threading | mutual_info |  68 |      0.838267 |    0.0821539 |               200 |            0.64     |
| jump-threading | random      |   5 |      0.879933 |    0.117097  |               200 |            0.596667 |
| jump-threading | random      |  10 |      0.838267 |    0.0821539 |               200 |            0.653333 |
| jump-threading | random      |  15 |      0.8216   |    0.0622168 |               200 |            0.563333 |
| jump-threading | random      |  20 |      0.8216   |    0.0622168 |               200 |            0.58     |
| jump-threading | random      |  30 |      0.8216   |    0.0622168 |               200 |            0.623333 |
| jump-threading | random      |  68 |      0.838267 |    0.0821539 |               200 |            0.596667 |
| jump-threading | shap        |   5 |      0.838267 |    0.0821539 |               200 |            0.65     |
| jump-threading | shap        |  10 |      0.838267 |    0.0821539 |               200 |            0.616667 |
| jump-threading | shap        |  15 |      0.838267 |    0.0821539 |               200 |            0.59     |
| jump-threading | shap        |  20 |      0.838267 |    0.0821539 |               200 |            0.6      |
| jump-threading | shap        |  30 |      0.838267 |    0.0821539 |               200 |            0.613333 |
| jump-threading | shap        |  68 |      0.838267 |    0.0821539 |               200 |            0.676667 |
| jump-threading | xgb_gain    |   5 |      0.838267 |    0.0821539 |               200 |            0.653333 |
| jump-threading | xgb_gain    |  10 |      0.838267 |    0.0821539 |               200 |            0.62     |
| jump-threading | xgb_gain    |  15 |      0.838267 |    0.0821539 |               200 |            0.56     |
| jump-threading | xgb_gain    |  20 |      0.838267 |    0.0821539 |               200 |            0.643333 |
| jump-threading | xgb_gain    |  30 |      0.838267 |    0.0821539 |               200 |            0.666667 |
| jump-threading | xgb_gain    |  68 |      0.838267 |    0.0821539 |               200 |            0.753333 |
| simplifycfg    | mutual_info |   5 |      0.888267 |    0.127095  |               200 |            0.713333 |
| simplifycfg    | mutual_info |  10 |      0.888267 |    0.127095  |               200 |            0.713333 |
| simplifycfg    | mutual_info |  15 |      0.9091   |    0.138251  |               200 |            0.736667 |
| simplifycfg    | mutual_info |  20 |      0.9091   |    0.138251  |               200 |            0.726667 |
| simplifycfg    | mutual_info |  30 |      0.867433 |    0.125685  |               200 |            0.716667 |
| simplifycfg    | mutual_info |  68 |      0.867433 |    0.125685  |               200 |            0.79     |
| simplifycfg    | random      |   5 |      0.8952   |    0.129804  |               200 |            0.706667 |
| simplifycfg    | random      |  10 |      0.8409   |    0.138251  |               200 |            0.77     |
| simplifycfg    | random      |  15 |      0.867433 |    0.125685  |               200 |            0.706667 |
| simplifycfg    | random      |  20 |      0.867433 |    0.125685  |               200 |            0.74     |
| simplifycfg    | random      |  30 |      0.867433 |    0.125685  |               200 |            0.68     |
| simplifycfg    | random      |  68 |      0.867433 |    0.125685  |               200 |            0.74     |
| simplifycfg    | shap        |   5 |      0.867433 |    0.125685  |               200 |            0.716667 |
| simplifycfg    | shap        |  10 |      0.867433 |    0.125685  |               200 |            0.733333 |
| simplifycfg    | shap        |  15 |      0.867433 |    0.125685  |               200 |            0.703333 |
| simplifycfg    | shap        |  20 |      0.867433 |    0.125685  |               200 |            0.7      |
| simplifycfg    | shap        |  30 |      0.867433 |    0.125685  |               200 |            0.713333 |
| simplifycfg    | shap        |  68 |      0.867433 |    0.125685  |               200 |            0.7      |
| simplifycfg    | xgb_gain    |   5 |      0.867433 |    0.125685  |               200 |            0.706667 |
| simplifycfg    | xgb_gain    |  10 |      0.867433 |    0.125685  |               200 |            0.693333 |
| simplifycfg    | xgb_gain    |  15 |      0.867433 |    0.125685  |               200 |            0.723333 |
| simplifycfg    | xgb_gain    |  20 |      0.867433 |    0.125685  |               200 |            0.706667 |
| simplifycfg    | xgb_gain    |  30 |      0.867433 |    0.125685  |               200 |            0.703333 |
| simplifycfg    | xgb_gain    |  68 |      0.867433 |    0.125685  |               200 |            0.803333 |



![Pruning Curve — early-cse](C:\Users\Aarush Gupta\Downloads\LLVM Project\results\figures\pruning\pruning_early-cse.png)


![Pruning Curve — jump-threading](C:\Users\Aarush Gupta\Downloads\LLVM Project\results\figures\pruning\pruning_jump-threading.png)


![Pruning Curve — simplifycfg](C:\Users\Aarush Gupta\Downloads\LLVM Project\results\figures\pruning\pruning_simplifycfg.png)

## Phase 7 — Sequence-Level Evaluation

| method   |   geomean_size_ratio |   ci_lo |   ci_hi |   mean_reduction |
|:---------|---------------------:|--------:|--------:|-----------------:|
| greedy   |               0.8159 |  0.8159 |  0.8159 |           0.1789 |
| Oz       |               0.8023 |  0.8023 |  0.8023 |           0.1868 |
| O2       |               0.8544 |  0.8544 |  0.8544 |           0.0832 |
| random   |               0.8495 |  0.8495 |  0.8495 |           0.1413 |
| oracle   |               0.7651 |  0.7651 |  0.7651 |           0.228  |



![Sequence Comparison](C:\Users\Aarush Gupta\Downloads\LLVM Project\results\figures\sequence\sequence_comparison.png)


![Reduction Distribution](C:\Users\Aarush Gupta\Downloads\LLVM Project\results\figures\sequence\sequence_boxplot.png)


```

╔══════════════════════════════════════════════════════╗
║              PHASE 7 GATE REPORT                     ║
╠══════════════════════════════════════════════════════╣
║ Held-out functions:            4                     ║
║ Greedy geomean ratio:     0.8159                     ║
║ -Oz geomean ratio:        0.8023                     ║
║ -O2 geomean ratio:        0.8544                     ║
║ Random geomean ratio:     0.8495                     ║
║ Oracle geomean ratio:     0.7651                     ║
╚══════════════════════════════════════════════════════╝

```
