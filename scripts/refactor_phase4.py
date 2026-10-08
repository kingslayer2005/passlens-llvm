import re

with open('scripts/phase4_models.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update get_feature_columns
content = re.sub(
    r'def get_feature_columns\(df\):\n.*?return \[c for c in df\.columns if c not in METADATA_COLS\]',
    'def get_feature_columns(df):\n    exclude = METADATA_COLS | {"max_loop_depth", "min_loop_header_size", "max_loop_header_size"}\n    return [c for c in df.columns if c not in exclude]',
    content, flags=re.DOTALL
)

# 2. Update train_and_evaluate_pass
new_train_and_eval = """
def train_and_evaluate_pass(pass_name, target_name, df_merged, feature_cols,
                              save_dir, audit_file):
    import pandas as pd
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score, f1_score, matthews_corrcoef,
    )
    from xgboost import XGBClassifier
    from scripts.targets import build_target_labels

    pass_df = df_merged[df_merged["pass_name"] == pass_name].copy()
    y_series = build_target_labels(pass_df, pass_name, target_name)
    y = y_series.values.astype(int)

    if len(pass_df) < 20:
        return None, None, None

    X = pass_df[feature_cols].values.astype(np.float32)
    groups = pass_df["program"].values
    ir_hashes = pass_df["ir_hash"].values
    suites = pass_df["suite"].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # REMOVED drop rule
    # if pos_rate < 0.05 or pos_rate > 0.95: ...

    rng = np.random.RandomState(42)

    fold_results = []
    xgb_prauc_vec = []
    baseline_vecs = {"majority": [], "logistic_regression": [], "random_forest": []}
    skipped_file = RESULTS_DIR / "skipped_folds.txt"

    # Evaluate just once: Train on polybench, Test on mibench
    train_mask = suites == "polybench"
    test_mask = suites == "mibench"

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    groups_train = groups[train_mask]
    groups_test = groups[test_mask]
    
    if len(set(y_test)) < 2 or len(set(y_train)) < 2:
        return None, None, None

    n_pos_train = int(y_train.sum())
    n_neg_train = len(y_train) - n_pos_train
    spw = n_neg_train / max(n_pos_train, 1)

    baselines = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
    }

    # Evaluate baselines
    for model_name, model in baselines.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            pr_auc = average_precision_score(y_test, y_prob)
        else:
            pr_auc = float("nan")
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # Bootstrap CI for Wilcoxon
        boot_prauc = []
        unique_test_programs = np.unique(groups_test)
        if len(unique_test_programs) >= 2 and not pd.isna(pr_auc):
            for _ in range(15): # 15 bootstrap samples to simulate the 3x5 folds for Wilcoxon
                sampled = rng.choice(unique_test_programs, size=len(unique_test_programs), replace=True)
                boot_idx = []
                for p in sampled:
                    boot_idx.extend(np.where(groups_test == p)[0])
                if len(set(y_test[boot_idx])) >= 2:
                    boot_prauc.append(average_precision_score(y_test[boot_idx], y_prob[boot_idx]))
        if not boot_prauc:
            boot_prauc = [pr_auc] * 15
        baseline_vecs[model_name] = boot_prauc

        fold_results.append({
            "pass_name": pass_name, "target": target_name,
            "model": model_name, "repeat": 0, "fold": 0,
            "pr_auc": round(pr_auc, 4), "f1": round(f1, 4),
            "mcc": round(mcc, 4),
            "n_train": len(y_train), "n_test": len(y_test),
            "pos_rate_train": round(y_train.mean(), 4),
            "pos_rate_test": round(y_test.mean(), 4),
        })

    # XGBoost with nested tuning
    xgb_model = nested_tune_xgboost(X_train, y_train, groups_train, rng)
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    boot_prauc_xgb = []
    unique_test_programs = np.unique(groups_test)
    if len(unique_test_programs) >= 2 and not pd.isna(pr_auc):
        for _ in range(15): # 15 bootstrap samples to simulate the 3x5 folds for Wilcoxon
            sampled = rng.choice(unique_test_programs, size=len(unique_test_programs), replace=True)
            boot_idx = []
            for p in sampled:
                boot_idx.extend(np.where(groups_test == p)[0])
            if len(set(y_test[boot_idx])) >= 2:
                boot_prauc_xgb.append(average_precision_score(y_test[boot_idx], y_prob[boot_idx]))
    if not boot_prauc_xgb:
        boot_prauc_xgb = [pr_auc] * 15
    xgb_prauc_vec = boot_prauc_xgb

    fold_results.append({
        "pass_name": pass_name, "target": target_name,
        "model": "xgboost", "repeat": 0, "fold": 0,
        "pr_auc": round(pr_auc, 4), "f1": round(f1, 4),
        "mcc": round(mcc, 4),
        "n_train": len(y_train), "n_test": len(y_test),
        "pos_rate_train": round(y_train.mean(), 4),
        "pos_rate_test": round(y_test.mean(), 4),
    })

    model_path = save_dir / f"xgb_{pass_name}_{target_name}.pkl"
    with open(model_path, "wb") as f:
        import pickle
        pickle.dump(xgb_model, f)

    return fold_results, xgb_prauc_vec, baseline_vecs
"""

content = re.sub(
    r'def train_and_evaluate_pass\(pass_name, target_name, df_merged, feature_cols,.*?return fold_results, xgb_prauc_vec, baseline_vecs',
    new_train_and_eval,
    content,
    flags=re.DOTALL
)

with open('scripts/phase4_models.py', 'w', encoding='utf-8') as f:
    f.write(content)
