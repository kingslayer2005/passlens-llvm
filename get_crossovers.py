import sys
import pandas as pd
import numpy as np
import json
import shap
import pickle
from pathlib import Path

DATA_DIR = Path("data")
MODELS_DIR = DATA_DIR / "models"
LABELS_DIR = DATA_DIR / "labels"
FEATURES_DIR = DATA_DIR / "features"

with open("heuristics.yaml") as f:
    import yaml
    heuristics = yaml.safe_load(f)

with open(FEATURES_DIR / "feature_columns.json") as f:
    feature_cols = json.load(f)

interp_passes = ["instcombine", "simplifycfg", "early-cse", "gvn", "sccp", "adce", "reassociate", "jump-threading", "correlated-propagation", "licm", "loop-rotate", "indvars", "loop-deletion", "loop-idiom", "loop-unroll", "tailcallelim"]

features = pd.read_csv(FEATURES_DIR / "features.csv")
labels = pd.read_csv(LABELS_DIR / "labels.csv")
df_merged = pd.merge(labels, features, on=["ir_hash", "suite", "program", "function"], how="inner")

crossover_values = []

for pass_name in interp_passes:
    if pass_name not in heuristics: continue
    
    target_name = "beneficial"
    model_path = MODELS_DIR / f"xgb_{pass_name}_{target_name}.pkl"
    if not model_path.exists(): continue
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    pass_df = df_merged[df_merged["pass_name"] == pass_name]
    X_full = pass_df[feature_cols].values.astype(np.float32)
    X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)
    
    explainer = shap.TreeExplainer(model)
    
    # Subsample if large
    if len(X_full) > 500:
        rng = np.random.RandomState(42)
        sel = rng.choice(len(X_full), 500, replace=False)
        X = X_full[sel]
    else:
        X = X_full
        
    shap_vals = explainer.shap_values(X)
    
    for h in heuristics[pass_name].get("cost_model", []):
        if h.get("direction") == "threshold" and h["name"] in feature_cols:
            feat = h["name"]
            feat_idx = feature_cols.index(feat)
            
            x_vals = X[:, feat_idx]
            s_vals = shap_vals[:, feat_idx]
            
            df_t = pd.DataFrame({"x": x_vals, "shap": s_vals})
            df_t = df_t.sort_values("x")
            
            # Simple smoothing
            df_t["smoothed_shap"] = df_t["shap"].rolling(window=max(1, len(df_t)//20), center=True).mean()
            df_t = df_t.dropna()
            
            if len(df_t) < 2: continue
            
            signs = np.sign(df_t["smoothed_shap"].values)
            sign_changes = np.where(signs[:-1] != signs[1:])[0]
            if len(sign_changes) > 0:
                idx = sign_changes[0]
                crossover = df_t.iloc[idx]["x"]
                crossover_values.append({
                    "pass_name": pass_name,
                    "feature": feat,
                    "crossover": float(crossover)
                })

print("CROSSOVERS:")
for c in crossover_values:
    print(c)
