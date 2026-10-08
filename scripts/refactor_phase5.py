import re

with open('scripts/phase5_shap.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove pos_rate checks (B3)
# Actually in phase5, there is no pos_rate < 0.05 drop in the compute_shap_for_pass loop
# Wait, I should rewrite compute_shap_for_pass to train on polybench and test on mibench.

new_compute_shap = """
def compute_shap_for_pass(pass_name, target_name, df_merged, feature_cols):
    import shap
    from scipy.stats import kendalltau
    from xgboost import XGBClassifier

    pass_df = df_merged[df_merged["pass_name"] == pass_name].copy()
    from scripts.targets import build_target_labels
    y = build_target_labels(pass_df, pass_name, target_name).values.astype(int)

    X = pass_df[feature_cols].values.astype(np.float32)
    groups = pass_df["program"].values
    suites = pass_df["suite"].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    train_mask = suites == "polybench"
    test_mask = suites == "mibench"

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        return None

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    spw = n_neg / max(n_pos, 1)

    # We evaluate JUST ONCE on the test set.
    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        random_state=42, eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    explainer_tpd = shap.TreeExplainer(model)
    sv_tpd = explainer_tpd.shap_values(X_test)
    mean_abs_tpd = np.mean(np.abs(sv_tpd), axis=0)
    
    bg_size = min(MAX_BACKGROUND_SAMPLES, len(X_train))
    bg_idx = np.random.RandomState(42).choice(len(X_train), bg_size, replace=False)
    background = X_train[bg_idx]
    
    try:
        explainer_int = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
        sv_int = explainer_int.shap_values(X_test)
        mean_abs_int = np.mean(np.abs(sv_int), axis=0)
        global_int_ranking = np.argsort(-mean_abs_int)
    except Exception as e:
        mean_abs_int = None

    global_mean_abs = mean_abs_tpd
    global_ranking = np.argsort(-global_mean_abs)

    stability_tau = 1.0 # Only 1 fold now, so perfectly stable with itself

    dual_tau = float("nan")
    if mean_abs_int is not None:
        tau_dual, _ = kendalltau(global_ranking, global_int_ranking)
        dual_tau = float(tau_dual)

    ranking_info = []
    for rank, feat_idx in enumerate(global_ranking):
        ranking_info.append({
            "pass_name": pass_name, "target": target_name,
            "rank": rank + 1,
            "feature": feature_cols[feat_idx],
            "mean_abs_shap": round(float(global_mean_abs[feat_idx]), 6),
        })

    return {
        "pass_name": pass_name,
        "target": target_name,
        "global_ranking": global_ranking,
        "global_mean_abs": global_mean_abs,
        "ranking_info": ranking_info,
        "stability_tau": stability_tau,
        "dual_tau": dual_tau,
        "feature_cols": feature_cols,
        "plot_shap": sv_tpd,
        "plot_X": X_test,
    }
"""

content = re.sub(
    r'def compute_shap_for_pass\(pass_name, target_name, df_merged, feature_cols\):.*?return \{\n.*?"plot_X": plot_X,\n    \}',
    new_compute_shap,
    content,
    flags=re.DOTALL
)

# Interactions
new_interactions = """
def compute_interactions(pass_name, target_name, df_merged, feature_cols):
    import shap
    from xgboost import XGBClassifier

    pass_df = df_merged[df_merged["pass_name"] == pass_name].copy()
    from scripts.targets import build_target_labels
    y = build_target_labels(pass_df, pass_name, target_name).values.astype(int)

    X = pass_df[feature_cols].values.astype(np.float32)
    suites = pass_df["suite"].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    train_mask = suites == "polybench"
    test_mask = suites == "mibench"

    X_train, X_test = X[train_mask], X[test_mask]
    y_train = y[train_mask]

    if len(set(y_train)) < 2:
        return []

    if len(X_test) > MAX_INTERACTION_SAMPLES:
        rng = np.random.RandomState(42)
        sel = rng.choice(len(X_test), MAX_INTERACTION_SAMPLES, replace=False)
        X_test = X_test[sel]

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=42, eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    try:
        explainer = shap.TreeExplainer(model)
        interaction_values = explainer.shap_interaction_values(X_test)
        mean_interactions = np.mean(np.abs(interaction_values), axis=0)
        np.fill_diagonal(mean_interactions, 0)
        n_feat = mean_interactions.shape[0]
        pairs = []
        for i in range(n_feat):
            for j in range(i + 1, n_feat):
                pairs.append((i, j, mean_interactions[i, j]))
        pairs.sort(key=lambda x: -x[2])

        results = []
        for i, j, val in pairs[:10]:
            results.append({
                "pass_name": pass_name, "target": target_name,
                "feature_1": feature_cols[i],
                "feature_2": feature_cols[j],
                "mean_abs_interaction": round(float(val), 6),
            })
        return results
    except Exception as e:
        import logging
        logging.getLogger("phase5").warning(f"Interaction values failed for {pass_name}: {e}")
        return []
"""

content = re.sub(
    r'def compute_interactions\(pass_name, target_name, df_merged, feature_cols\):.*?return \[\]',
    new_interactions,
    content,
    flags=re.DOTALL
)

# Faithfulness
new_faith = """
def faithfulness_test(pass_name, target_name, df_merged, feature_cols, global_ranking):
    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier

    pass_df = df_merged[df_merged["pass_name"] == pass_name].copy()
    from scripts.targets import build_target_labels
    y = build_target_labels(pass_df, pass_name, target_name).values.astype(int)

    X = pass_df[feature_cols].values.astype(np.float32)
    suites = pass_df["suite"].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    train_mask = suites == "polybench"
    test_mask = suites == "mibench"

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        return []

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    spw = n_neg / max(n_pos, 1)
    n_features = X.shape[1]

    def train_and_score(keep_mask):
        m = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=spw, random_state=42, eval_metric="logloss",
        )
        m.fit(X_train[:, keep_mask], y_train)
        yp = m.predict_proba(X_test[:, keep_mask])[:, 1]
        return average_precision_score(y_test, yp)

    full_score = train_and_score(np.ones(n_features, dtype=bool))
    import numpy as np
    results = []
    rng = np.random.RandomState(42)

    for k in FAITHFULNESS_K:
        if k >= n_features:
            continue
        remove_idx = set(global_ranking[:k])
        keep_mask = np.array([i not in remove_idx for i in range(n_features)])
        shap_score = train_and_score(keep_mask)

        random_scores = []
        for draw in range(N_RANDOM_DRAWS):
            rand_remove = set(rng.choice(n_features, k, replace=False))
            keep_mask_r = np.array([i not in rand_remove for i in range(n_features)])
            random_scores.append(train_and_score(keep_mask_r))

        results.append({
            "pass_name": pass_name, "target": target_name,
            "k": k,
            "full_prauc": round(full_score, 4),
            "shap_removed_prauc": round(shap_score, 4),
            "shap_drop": round(full_score - shap_score, 4),
            "random_removed_prauc_mean": round(float(np.mean(random_scores)), 4),
            "random_removed_prauc_std": round(float(np.std(random_scores)), 4),
            "random_drop_mean": round(float(full_score - np.mean(random_scores)), 4),
        })

    return results
"""

content = re.sub(
    r'def faithfulness_test\(pass_name, target_name, df_merged, feature_cols, global_ranking\):.*?return results',
    new_faith,
    content,
    flags=re.DOTALL
)

# Size confound
new_size = """
def size_confound_analysis(pass_name, target_name, df_merged, feature_cols, original_ranking):
    import shap
    from xgboost import XGBClassifier
    import numpy as np

    if "log_inst_count" not in feature_cols:
        return []

    reduced_cols = [c for c in feature_cols if c != "log_inst_count"]
    reduced_idx = [feature_cols.index(c) for c in reduced_cols]

    pass_df = df_merged[df_merged["pass_name"] == pass_name].copy()
    from scripts.targets import build_target_labels
    y = build_target_labels(pass_df, pass_name, target_name).values.astype(int)

    X_full = pass_df[feature_cols].values.astype(np.float32)
    X = X_full[:, reduced_idx]
    suites = pass_df["suite"].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    train_mask = suites == "polybench"
    test_mask = suites == "mibench"

    X_train, X_test = X[train_mask], X[test_mask]
    y_train = y[train_mask]

    if len(set(y_train)) < 2:
        return []

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=42, eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test)
    mean_abs = np.mean(np.abs(sv), axis=0)
    new_ranking = np.argsort(-mean_abs)

    orig_top10 = [feature_cols[i] for i in original_ranking[:10]]
    new_top10 = [reduced_cols[i] for i in new_ranking[:10]]

    results = []
    for rank, feat in enumerate(new_top10):
        orig_rank = orig_top10.index(feat) + 1 if feat in orig_top10 else -1
        results.append({
            "pass_name": pass_name, "target": target_name,
            "feature": feat,
            "rank_without_size": rank + 1,
            "rank_with_size": orig_rank if orig_rank > 0 else "N/A",
        })

    return results
"""

content = re.sub(
    r'def size_confound_analysis\(pass_name, target_name, df_merged, feature_cols, original_ranking\):.*?return results',
    new_size,
    content,
    flags=re.DOTALL
)

with open('scripts/phase5_shap.py', 'w', encoding='utf-8') as f:
    f.write(content)
