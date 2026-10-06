# Results

> **Auto-generated** — all numbers come from CSV files in `results/`.
> Do not edit manually; re-run `python -m scripts.generate_results_md` instead.

---

## Phase 1 — Data Collection

```

╔══════════════════════════════════════════════════╗
║            PHASE 1 GATE REPORT                   ║
╠══════════════════════════════════════════════════╣
║ Programs processed:          3                   ║
║ Total functions (≥10i):     12                   ║
║ Unique functions:           12                   ║
║ Duplicates removed:          0                   ║
║ Compilation failures:        0                   ║
║ Extraction failures:         0                   ║
╚══════════════════════════════════════════════════╝

```

## Phase 3 — Label Distribution

| pass_name              |   total_samples |   beneficial |   not_beneficial |   positive_rate | dropped   |
|:-----------------------|----------------:|-------------:|-----------------:|----------------:|:----------|
| sroa                   |              40 |            0 |               40 |           0     | True      |
| instcombine            |              40 |           35 |                5 |           0.875 | False     |
| simplifycfg            |              40 |           26 |               14 |           0.65  | False     |
| early-cse              |              40 |           23 |               17 |           0.575 | False     |
| gvn                    |              40 |           36 |                4 |           0.9   | False     |
| sccp                   |              40 |            0 |               40 |           0     | True      |
| adce                   |              40 |            0 |               40 |           0     | True      |
| dse                    |              40 |            0 |               40 |           0     | True      |
| reassociate            |              40 |            0 |               40 |           0     | True      |
| jump-threading         |              40 |           24 |               16 |           0.6   | False     |
| correlated-propagation |              40 |            0 |               40 |           0     | True      |
| licm                   |              40 |            0 |               40 |           0     | True      |
| loop-rotate            |              40 |            0 |               40 |           0     | True      |
| indvars                |              40 |            8 |               32 |           0.2   | False     |
| loop-deletion          |              40 |            0 |               40 |           0     | True      |
| loop-idiom             |              40 |            0 |               40 |           0     | True      |
| loop-unroll            |              40 |            0 |               40 |           0     | True      |
| tailcallelim           |              40 |            0 |               40 |           0     | True      |



**Dropped passes** (positive rate outside [5%, 95%]): sroa, sccp, adce, dse, reassociate, correlated-propagation, licm, loop-rotate, loop-deletion, loop-idiom, loop-unroll, tailcallelim


```

╔══════════════════════════════════════════════════════╗
║              PHASE 3 GATE REPORT                     ║
╠══════════════════════════════════════════════════════╣
║ Total samples:               720                     ║
║ Functions:                    10                     ║
║ Passes (active):              18                     ║
║ Passes (kept):                 6                     ║
║ Passes (dropped):             12                     ║
╠══════════════════════════════════════════════════════╣
║ Dropped passes: sroa, sccp, adce, dse, reassociate, correlated-propagation, licm, loop-rotate, loop-deletion, loop-idiom, loop-unroll, tailcallelim║
╚══════════════════════════════════════════════════════╝

```

## Phase 4 — Model Performance

### Per-Pass Summary (Mean ± Std over folds)

| pass_name      | model               |   pr_auc_mean |   pr_auc_std |   f1_mean |   f1_std |   mcc_mean |   mcc_std |
|:---------------|:--------------------|--------------:|-------------:|----------:|---------:|-----------:|----------:|
| simplifycfg    | majority            |        0.6042 |       0.2009 |    0.7391 |   0.169  |     0      |    0      |
| simplifycfg    | logistic_regression |        0.8952 |       0.1298 |    0.9379 |   0.0732 |     0.877  |    0.1141 |
| simplifycfg    | random_forest       |        0.9091 |       0.1383 |    0.9379 |   0.0732 |     0.877  |    0.1141 |
| simplifycfg    | xgboost             |        0.8674 |       0.1257 |    0.7968 |   0.2391 |     0.7213 |    0.3089 |
| early-cse      | majority            |        0.625  |       0.2165 |    0.4445 |   0.3849 |     0      |    0      |
| early-cse      | logistic_regression |        0.7773 |       0.1828 |    0.7758 |   0.042  |     0.5109 |    0.1151 |
| early-cse      | random_forest       |        0.9457 |       0.0547 |    0.8008 |   0.0739 |     0.5685 |    0.1862 |
| early-cse      | xgboost             |        0.9457 |       0.0547 |    0.8008 |   0.0739 |     0.5685 |    0.1862 |
| gvn            | majority            |        0.875  |       0      |    0.9333 |   0      |     0      |    0      |
| gvn            | logistic_regression |        0.8631 |       0.0042 |    0.9333 |   0      |     0      |    0      |
| gvn            | random_forest       |        0.8914 |       0.0442 |    0.9333 |   0      |     0      |    0      |
| gvn            | xgboost             |        0.875  |       0      |    0.9333 |   0      |     0      |    0      |
| jump-threading | majority            |        0.5625 |       0.1654 |    0.7098 |   0.1441 |     0      |    0      |
| jump-threading | logistic_regression |        0.8369 |       0.0934 |    0.9076 |   0.0497 |     0.7921 |    0.0575 |
| jump-threading | random_forest       |        0.8799 |       0.1171 |    0.8694 |   0.0352 |     0.7314 |    0.0516 |
| jump-threading | xgboost             |        0.8369 |       0.0934 |    0.7665 |   0.2036 |     0.6364 |    0.2146 |
| indvars        | majority            |        0.25   |       0      |    0      |   0      |     0      |    0      |
| indvars        | logistic_regression |        0.75   |       0.3536 |    0.8334 |   0.2357 |     0.7887 |    0.2988 |
| indvars        | random_forest       |        1      |       0      |    0.3334 |   0.4714 |     0.2887 |    0.4083 |
| indvars        | xgboost             |        1      |       0      |    0.5    |   0.7071 |     0.5    |    0.7071 |



```

╔══════════════════════════════════════════════════════╗
║              PHASE 4 GATE REPORT                     ║
╠══════════════════════════════════════════════════════╣
║ Passes evaluated:              6                     ║
║ Passes where XGB > Maj:        5                     ║
║ Flagged passes:                1                     ║
╠══════════════════════════════════════════════════════╣
║ Flagged: gvn                                         ║
║ (excluded from Phase 5 interpretation)               ║
╚══════════════════════════════════════════════════════╝

```

## Phase 5 — Explainability (SHAP)

### Top 10 Features per Pass


#### simplifycfg

|   rank | feature                |   mean_abs_shap |
|-------:|:-----------------------|----------------:|
|      1 | int_arith_count        |        1.58415  |
|      2 | icmp_density           |        0.234677 |
|      3 | gep_count              |        0.20706  |
|      4 | icmp_count             |        0.155582 |
|      5 | load_count             |        0.154019 |
|      6 | phi_count              |        0.129409 |
|      7 | const_operands         |        0.1134   |
|      8 | max_loop_depth_density |        0.091271 |
|      9 | load_density           |        0.044967 |
|     10 | call_count             |        0.04238  |



![SHAP Beeswarm — simplifycfg](C:\Users\Aarush Gupta\Downloads\LLVM Project\results\figures\shap\shap_beeswarm_simplifycfg.png)


#### early-cse

|   rank | feature                |   mean_abs_shap |
|-------:|:-----------------------|----------------:|
|      1 | gep_count              |        0.852431 |
|      2 | icmp_density           |        0.44273  |
|      3 | int_arith_density      |        0.314605 |
|      4 | gep_density            |        0.260874 |
|      5 | const_operands_density |        0.173871 |
|      6 | const_operands         |        0.160278 |
|      7 | int_arith_count        |        0.129164 |
|      8 | log_inst_count         |        0.087668 |
|      9 | cast_density           |        0.078623 |
|     10 | cond_br_density        |        0.073601 |



![SHAP Beeswarm — early-cse](C:\Users\Aarush Gupta\Downloads\LLVM Project\results\figures\shap\shap_beeswarm_early-cse.png)


#### jump-threading

|   rank | feature                |   mean_abs_shap |
|-------:|:-----------------------|----------------:|
|      1 | int_arith_count        |        1.58252  |
|      2 | max_loop_depth_density |        0.359054 |
|      3 | icmp_density           |        0.270337 |
|      4 | load_count             |        0.257256 |
|      5 | gep_count              |        0.180554 |
|      6 | store_count            |        0.164423 |
|      7 | icmp_count             |        0.16438  |
|      8 | phi_count              |        0.144543 |
|      9 | cast_density           |        0.095449 |
|     10 | const_operands         |        0.086655 |



![SHAP Beeswarm — jump-threading](C:\Users\Aarush Gupta\Downloads\LLVM Project\results\figures\shap\shap_beeswarm_jump-threading.png)


#### indvars

|   rank | feature         |   mean_abs_shap |
|-------:|:----------------|----------------:|
|      1 | load_count      |        0.556449 |
|      2 | gep_count       |        0.372808 |
|      3 | cast_density    |        0.266279 |
|      4 | store_count     |        0.25544  |
|      5 | icmp_count      |        0.246063 |
|      6 | load_density    |        0.066507 |
|      7 | icmp_density    |        0.064146 |
|      8 | int_arith_count |        0.062666 |
|      9 | cast_count      |        0.061151 |
|     10 | const_operands  |        0.037972 |



![SHAP Beeswarm — indvars](C:\Users\Aarush Gupta\Downloads\LLVM Project\results\figures\shap\shap_beeswarm_indvars.png)

### Faithfulness Checks

| pass_name      |   kendall_tau_mean |   kendall_tau_std |   top10_overlap_mean |   perm_kendall_tau |   shuffle_shap_max |   shuffle_shap_mean |
|:---------------|-------------------:|------------------:|---------------------:|-------------------:|-------------------:|--------------------:|
| simplifycfg    |           0.605962 |         0.0601247 |                 0.79 |             0.826  |            1.06215 |            0.038984 |
| early-cse      |           0.593846 |         0.0699643 |                 0.9  |             0.8779 |            1.07622 |            0.032276 |
| jump-threading |           0.638269 |         0.0623783 |                 0.88 |             0.775  |            0.93302 |            0.036441 |
| indvars        |           0.720577 |         0.0595536 |                 0.81 |             0.925  |            0.74366 |            0.02555  |


### Heuristic Agreement

| pass_name      |   hit_at_5 |    mrr |   direction_agreement |   random_hit_at_5 |   random_mrr |   n_heuristic_features |
|:---------------|-----------:|-------:|----------------------:|------------------:|-------------:|-----------------------:|
| simplifycfg    |     0      | 0.0338 |                0.3333 |            0.0615 |       0.0303 |                      4 |
| early-cse      |     0.3333 | 0.1389 |                0      |            0.0462 |       0.0303 |                      3 |
| jump-threading |     0      | 0.0352 |                0.5    |            0.0462 |       0.0303 |                      3 |
| indvars        |     0      | 0.0609 |                0      |            0.0462 |       0.0303 |                      3 |


### Candidate Novel Drivers

| pass_name      | feature                |   shap_rank |   mean_abs_shap |
|:---------------|:-----------------------|------------:|----------------:|
| simplifycfg    | int_arith_count        |           1 |        1.58415  |
| simplifycfg    | icmp_density           |           2 |        0.234677 |
| simplifycfg    | gep_count              |           3 |        0.20706  |
| simplifycfg    | icmp_count             |           4 |        0.155582 |
| simplifycfg    | load_count             |           5 |        0.154019 |
| simplifycfg    | phi_count              |           6 |        0.129409 |
| simplifycfg    | const_operands         |           7 |        0.1134   |
| simplifycfg    | max_loop_depth_density |           8 |        0.091271 |
| simplifycfg    | load_density           |           9 |        0.044967 |
| simplifycfg    | call_count             |          10 |        0.04238  |
| early-cse      | gep_count              |           1 |        0.852431 |
| early-cse      | icmp_density           |           2 |        0.44273  |
| early-cse      | gep_density            |           4 |        0.260874 |
| early-cse      | const_operands_density |           5 |        0.173871 |
| early-cse      | const_operands         |           6 |        0.160278 |
| early-cse      | int_arith_count        |           7 |        0.129164 |
| early-cse      | log_inst_count         |           8 |        0.087668 |
| early-cse      | cast_density           |           9 |        0.078623 |
| early-cse      | cond_br_density        |          10 |        0.073601 |
| jump-threading | int_arith_count        |           1 |        1.58252  |
| jump-threading | max_loop_depth_density |           2 |        0.359054 |
| jump-threading | icmp_density           |           3 |        0.270337 |
| jump-threading | load_count             |           4 |        0.257256 |
| jump-threading | gep_count              |           5 |        0.180554 |
| jump-threading | store_count            |           6 |        0.164423 |
| jump-threading | icmp_count             |           7 |        0.16438  |
| jump-threading | phi_count              |           8 |        0.144543 |
| jump-threading | cast_density           |           9 |        0.095449 |
| jump-threading | const_operands         |          10 |        0.086655 |
| indvars        | load_count             |           1 |        0.556449 |
| indvars        | gep_count              |           2 |        0.372808 |
| indvars        | cast_density           |           3 |        0.266279 |
| indvars        | store_count            |           4 |        0.25544  |
| indvars        | icmp_count             |           5 |        0.246063 |
| indvars        | load_density           |           6 |        0.066507 |
| indvars        | int_arith_count        |           8 |        0.062666 |
| indvars        | cast_count             |           9 |        0.061151 |
| indvars        | const_operands         |          10 |        0.037972 |



```

╔══════════════════════════════════════════════════════╗
║              PHASE 5 GATE REPORT                     ║
╠══════════════════════════════════════════════════════╣
║ Passes analyzed:               5                     ║
║ Feature clusters:             24                     ║
║ Novel driver candidates:      38                     ║
╚══════════════════════════════════════════════════════╝

```

## Phase 6 — Feature Pruning

| pass_name      | method      |   k |   pr_auc_mean |   pr_auc_std |   model_size_mean |   inference_ms_mean |
|:---------------|:------------|----:|--------------:|-------------:|------------------:|--------------------:|
| early-cse      | mutual_info |   5 |      0.7842   |    0.19153   |               200 |            0.536667 |
| early-cse      | mutual_info |  10 |      0.8901   |    0.0565517 |               200 |            0.486667 |
| early-cse      | mutual_info |  15 |      0.8901   |    0.0565517 |               200 |            0.61     |
| early-cse      | mutual_info |  20 |      0.8901   |    0.0565517 |               200 |            0.583333 |
| early-cse      | mutual_info |  30 |      0.806767 |    0.195529  |               200 |            0.6      |
| early-cse      | mutual_info |  65 |      0.945667 |    0.0547037 |               200 |            0.546667 |
| early-cse      | random      |   5 |      0.686967 |    0.243224  |               200 |            0.55     |
| early-cse      | random      |  10 |      0.654    |    0.264241  |               200 |            0.673333 |
| early-cse      | random      |  15 |      0.945667 |    0.0547037 |               200 |            0.67     |
| early-cse      | random      |  20 |      0.945667 |    0.0547037 |               200 |            0.613333 |
| early-cse      | random      |  30 |      0.945667 |    0.0547037 |               200 |            0.506667 |
| early-cse      | random      |  65 |      0.945667 |    0.0547037 |               200 |            0.733333 |
| early-cse      | shap        |   5 |      0.945667 |    0.0547037 |               200 |            0.49     |
| early-cse      | shap        |  10 |      0.945667 |    0.0547037 |               200 |            0.543333 |
| early-cse      | shap        |  15 |      0.945667 |    0.0547037 |               200 |            0.51     |
| early-cse      | shap        |  20 |      0.945667 |    0.0547037 |               200 |            0.566667 |
| early-cse      | shap        |  30 |      0.945667 |    0.0547037 |               200 |            0.67     |
| early-cse      | shap        |  65 |      0.945667 |    0.0547037 |               200 |            0.683333 |
| early-cse      | xgb_gain    |   5 |      0.945667 |    0.0547037 |               200 |            0.636667 |
| early-cse      | xgb_gain    |  10 |      0.945667 |    0.0547037 |               200 |            0.586667 |
| early-cse      | xgb_gain    |  15 |      0.945667 |    0.0547037 |               200 |            0.543333 |
| early-cse      | xgb_gain    |  20 |      0.945667 |    0.0547037 |               200 |            0.586667 |
| early-cse      | xgb_gain    |  30 |      0.945667 |    0.0547037 |               200 |            0.63     |
| early-cse      | xgb_gain    |  65 |      0.945667 |    0.0547037 |               200 |            0.71     |
| indvars        | mutual_info |   5 |      1        |    0         |               200 |            0.57     |
| indvars        | mutual_info |  10 |      1        |    0         |               200 |            0.605    |
| indvars        | mutual_info |  15 |      0.75     |    0.353553  |               200 |            0.515    |
| indvars        | mutual_info |  20 |      0.5      |    0         |               200 |            0.535    |
| indvars        | mutual_info |  30 |      1        |    0         |               200 |            0.545    |
| indvars        | mutual_info |  65 |      1        |    0         |               200 |            0.59     |
| indvars        | random      |   5 |      0.75     |    0.353553  |               200 |            0.55     |
| indvars        | random      |  10 |      0.75     |    0.353553  |               200 |            0.605    |
| indvars        | random      |  15 |      0.66665  |    0.471428  |               200 |            0.52     |
| indvars        | random      |  20 |      0.66665  |    0.471428  |               200 |            0.49     |
| indvars        | random      |  30 |      1        |    0         |               200 |            0.47     |
| indvars        | random      |  65 |      1        |    0         |               200 |            0.615    |
| indvars        | shap        |   5 |      1        |    0         |               200 |            0.49     |
| indvars        | shap        |  10 |      1        |    0         |               200 |            0.495    |
| indvars        | shap        |  15 |      1        |    0         |               200 |            0.52     |
| indvars        | shap        |  20 |      1        |    0         |               200 |            0.49     |
| indvars        | shap        |  30 |      1        |    0         |               200 |            0.455    |
| indvars        | shap        |  65 |      1        |    0         |               200 |            0.58     |
| indvars        | xgb_gain    |   5 |      1        |    0         |               200 |            0.54     |
| indvars        | xgb_gain    |  10 |      1        |    0         |               200 |            0.58     |
| indvars        | xgb_gain    |  15 |      1        |    0         |               200 |            0.49     |
| indvars        | xgb_gain    |  20 |      1        |    0         |               200 |            0.485    |
| indvars        | xgb_gain    |  30 |      1        |    0         |               200 |            0.445    |
| indvars        | xgb_gain    |  65 |      1        |    0         |               200 |            0.595    |
| jump-threading | mutual_info |   5 |      0.879933 |    0.117097  |               200 |            0.55     |
| jump-threading | mutual_info |  10 |      0.879933 |    0.117097  |               200 |            0.573333 |
| jump-threading | mutual_info |  15 |      0.879933 |    0.117097  |               200 |            0.573333 |
| jump-threading | mutual_info |  20 |      0.866033 |    0.101149  |               200 |            0.63     |
| jump-threading | mutual_info |  30 |      0.866033 |    0.101149  |               200 |            0.57     |
| jump-threading | mutual_info |  65 |      0.838267 |    0.0821539 |               200 |            0.543333 |
| jump-threading | random      |   5 |      0.8327   |    0.0824016 |               200 |            0.493333 |
| jump-threading | random      |  10 |      0.8313   |    0.0949136 |               200 |            0.516667 |
| jump-threading | random      |  15 |      0.822967 |    0.0989541 |               200 |            0.66     |
| jump-threading | random      |  20 |      0.822967 |    0.0989541 |               200 |            0.566667 |
| jump-threading | random      |  30 |      0.8091   |    0.052971  |               200 |            0.616667 |
| jump-threading | random      |  65 |      0.838267 |    0.0821539 |               200 |            0.626667 |
| jump-threading | shap        |   5 |      0.838267 |    0.0821539 |               200 |            0.5      |
| jump-threading | shap        |  10 |      0.838267 |    0.0821539 |               200 |            0.56     |
| jump-threading | shap        |  15 |      0.838267 |    0.0821539 |               200 |            0.576667 |
| jump-threading | shap        |  20 |      0.838267 |    0.0821539 |               200 |            0.633333 |
| jump-threading | shap        |  30 |      0.838267 |    0.0821539 |               200 |            0.593333 |
| jump-threading | shap        |  65 |      0.838267 |    0.0821539 |               200 |            0.603333 |
| jump-threading | xgb_gain    |   5 |      0.838267 |    0.0821539 |               200 |            0.563333 |
| jump-threading | xgb_gain    |  10 |      0.838267 |    0.0821539 |               200 |            0.73     |
| jump-threading | xgb_gain    |  15 |      0.838267 |    0.0821539 |               200 |            0.683333 |
| jump-threading | xgb_gain    |  20 |      0.838267 |    0.0821539 |               200 |            0.586667 |
| jump-threading | xgb_gain    |  30 |      0.838267 |    0.0821539 |               200 |            0.57     |
| jump-threading | xgb_gain    |  65 |      0.838267 |    0.0821539 |               200 |            0.633333 |
| simplifycfg    | mutual_info |   5 |      0.9091   |    0.138251  |               200 |            0.623333 |
| simplifycfg    | mutual_info |  10 |      0.9091   |    0.138251  |               200 |            0.536667 |
| simplifycfg    | mutual_info |  15 |      0.9091   |    0.138251  |               200 |            0.546667 |
| simplifycfg    | mutual_info |  20 |      0.8952   |    0.129804  |               200 |            0.563333 |
| simplifycfg    | mutual_info |  30 |      0.867433 |    0.125685  |               200 |            0.516667 |
| simplifycfg    | mutual_info |  65 |      0.867433 |    0.125685  |               200 |            0.503333 |
| simplifycfg    | random      |   5 |      0.850367 |    0.132081  |               200 |            4.31     |
| simplifycfg    | random      |  10 |      0.867433 |    0.125685  |               200 |            5.00667  |
| simplifycfg    | random      |  15 |      0.867433 |    0.125685  |               200 |            0.593333 |
| simplifycfg    | random      |  20 |      0.867433 |    0.125685  |               200 |            0.603333 |
| simplifycfg    | random      |  30 |      0.867433 |    0.125685  |               200 |            0.55     |
| simplifycfg    | random      |  65 |      0.867433 |    0.125685  |               200 |            0.633333 |
| simplifycfg    | shap        |   5 |      0.867433 |    0.125685  |               200 |            5.54     |
| simplifycfg    | shap        |  10 |      0.867433 |    0.125685  |               200 |            6.69667  |
| simplifycfg    | shap        |  15 |      0.867433 |    0.125685  |               200 |            0.836667 |
| simplifycfg    | shap        |  20 |      0.867433 |    0.125685  |               200 |            0.59     |
| simplifycfg    | shap        |  30 |      0.867433 |    0.125685  |               200 |            3.63667  |
| simplifycfg    | shap        |  65 |      0.867433 |    0.125685  |               200 |            0.556667 |
| simplifycfg    | xgb_gain    |   5 |      0.867433 |    0.125685  |               200 |            0.713333 |
| simplifycfg    | xgb_gain    |  10 |      0.867433 |    0.125685  |               200 |            0.633333 |
| simplifycfg    | xgb_gain    |  15 |      0.867433 |    0.125685  |               200 |            0.593333 |
| simplifycfg    | xgb_gain    |  20 |      0.867433 |    0.125685  |               200 |            0.706667 |
| simplifycfg    | xgb_gain    |  30 |      0.867433 |    0.125685  |               200 |            5.14     |
| simplifycfg    | xgb_gain    |  65 |      0.867433 |    0.125685  |               200 |            0.593333 |



![Pruning Curve — early-cse](C:\Users\Aarush Gupta\Downloads\LLVM Project\results\figures\pruning\pruning_early-cse.png)


![Pruning Curve — indvars](C:\Users\Aarush Gupta\Downloads\LLVM Project\results\figures\pruning\pruning_indvars.png)


![Pruning Curve — jump-threading](C:\Users\Aarush Gupta\Downloads\LLVM Project\results\figures\pruning\pruning_jump-threading.png)


![Pruning Curve — simplifycfg](C:\Users\Aarush Gupta\Downloads\LLVM Project\results\figures\pruning\pruning_simplifycfg.png)

## Phase 7 — Sequence-Level Evaluation

| method   |   mean_reduction |   median_reduction |   mean_final_count |
|:---------|-----------------:|-------------------:|-------------------:|
| greedy   |         0.178875 |            0.1678  |              47    |
| Oz       |         0.1868   |            0.21645 |              46.25 |
| O2       |         0.0832   |            0.20635 |              52    |
| random   |         0.141275 |            0.1254  |              49.25 |
| oracle   |         0.227975 |            0.25885 |              43.75 |



![Sequence Comparison](C:\Users\Aarush Gupta\Downloads\LLVM Project\results\figures\sequence\sequence_comparison.png)


![Reduction Distribution](C:\Users\Aarush Gupta\Downloads\LLVM Project\results\figures\sequence\sequence_boxplot.png)


```

╔══════════════════════════════════════════════════════╗
║              PHASE 7 GATE REPORT                     ║
╠══════════════════════════════════════════════════════╣
║ Held-out functions:            4                     ║
║ Greedy mean reduction:     17.9%%                    ║
║ -Oz mean reduction:        18.7%%                    ║
║ -O2 mean reduction:         8.3%%                    ║
║ Random mean reduction:     14.1%%                    ║
║ Oracle mean reduction:     22.8%%                    ║
╚══════════════════════════════════════════════════════╝

```
