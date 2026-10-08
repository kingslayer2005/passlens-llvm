import re

with open('scripts/phase6_pruning.py', 'r', encoding='utf-8') as f:
    content = f.read()

new_evaluate_pruning = """
def evaluate_pruning(pass_name, df_merged, feature_cols, rankings):
    import pandas as pd
    import numpy as np
    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier
    from scripts.targets import build_target_labels

    pass_df = df_merged[df_merged["pass_name"] == pass_name].copy()
    X = pass_df[feature_cols].values.astype(np.float32)
    y = build_target_labels(pass_df, pass_name, "beneficial").values.astype(int)
    suites = pass_df["suite"].values
    groups = pass_df["program"].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n_features = X.shape[1]
    k_values = [k for k in K_VALUES if k < n_features] + [n_features]
    results = []

    train_mask = suites == "polybench"
    test_mask = suites == "mibench"

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    groups_test = groups[test_mask]

    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        return []

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    spw = n_neg / max(n_pos, 1)

    rng = np.random.RandomState(42)

    for method, ranking in rankings.items():
        if not ranking:
            continue
        
        for k in k_values:
            top_k = ranking[:k]
            keep_mask = np.array([feature_cols.index(f) in [feature_cols.index(c) for c in top_k] for f in feature_cols])

            m = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                scale_pos_weight=spw, random_state=42, eval_metric="logloss",
            )
            m.fit(X_train[:, keep_mask], y_train)
            yp = m.predict_proba(X_test[:, keep_mask])[:, 1]
            pr_auc = average_precision_score(y_test, yp)

            # Bootstrapped CI for PR-AUC
            boot_prauc = []
            unique_test_programs = np.unique(groups_test)
            if len(unique_test_programs) >= 2 and not pd.isna(pr_auc):
                for _ in range(15):
                    sampled = rng.choice(unique_test_programs, size=len(unique_test_programs), replace=True)
                    boot_idx = []
                    for p in sampled:
                        boot_idx.extend(np.where(groups_test == p)[0])
                    if len(set(y_test[boot_idx])) >= 2:
                        boot_prauc.append(average_precision_score(y_test[boot_idx], yp[boot_idx]))
            
            mean_val = np.mean(boot_prauc) if boot_prauc else pr_auc
            std_val = np.std(boot_prauc) if boot_prauc else 0.0

            results.append({
                "pass_name": pass_name,
                "method": method,
                "k": k,
                "pr_auc": round(float(pr_auc), 4),
                "boot_pr_auc_mean": round(float(mean_val), 4),
                "boot_pr_auc_std": round(float(std_val), 4),
            })

    return results
"""

content = re.sub(
    r'def evaluate_pruning\(pass_name, df_merged, feature_cols, rankings\):.*?return results',
    new_evaluate_pruning,
    content,
    flags=re.DOTALL
)

with open('scripts/phase6_pruning.py', 'w', encoding='utf-8') as f:
    f.write(content)
